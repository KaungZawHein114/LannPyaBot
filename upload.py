import os
import json
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_path

# --------------------
# Load environment variables
# --------------------
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

# --------------------
# Helpers
# --------------------
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def extract_text_from_pdf(pdf_path, use_ocr=True):
    text = ""
    if not use_ocr:
        try:
            import fitz
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text("text")
        except Exception as e:
            print(f"⚠️ PyMuPDF failed for {pdf_path}: {e}")
            use_ocr = True

    if use_ocr:
        print("🔍 Using OCR for PDF...")
        pages = convert_from_path(pdf_path, dpi=300)
        for page in pages:
            page_text = pytesseract.image_to_string(page, lang="mya+eng")
            text += page_text + "\n"

    return text.strip()

def batch_upsert(entries, source):
    """entries = [(id, text)]"""
    if not entries:
        return

    texts = [text for _, text in entries]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    vectors = []
    for (entry_id, text), emb in zip(entries, response.data):
        vectors.append((entry_id, emb.embedding, {"text": text, "source": source}))

    index.upsert(vectors)
    print(f"✅ Upserted {len(vectors)} entries from {source}")

# --------------------
# Clear index
# --------------------
index.delete(delete_all=True)
print("🧹 Cleared all previous entries from the index.")

# --------------------
# Step 1: Upload JSON
# --------------------
json_file = "knowledge_base.json"
if os.path.exists(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    batch = []
    for topic, entries in data.items():
        for i, entry in enumerate(entries, start=1):
            entry_id = f"{topic.replace(' ', '-')}-{i}"
            batch.append((entry_id, entry))

            if len(batch) >= 50:
                batch_upsert(batch, "json")
                batch = []

    if batch:
        batch_upsert(batch, "json")
else:
    print("⚠️ No knowledgebase.json found, skipping JSON upload.")

# --------------------
# Step 2: Upload PDFs
# --------------------
pdf_folder = "knowledgebase"
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        pdf_name = os.path.splitext(filename)[0].replace(" ", "-")

        print(f"\n📄 Processing PDF: {filename}...")
        text = extract_text_from_pdf(pdf_path, use_ocr=True)
        chunks = chunk_text(text, chunk_size=1000, overlap=200)

        batch = []
        for i, chunk in enumerate(chunks, start=1):
            entry_id = f"{pdf_name}-{i}"
            batch.append((entry_id, chunk))

            if len(batch) >= 50:
                batch_upsert(batch, filename)
                batch = []

        if batch:
            batch_upsert(batch, filename)

print("\n🎉 All knowledgebase data uploaded successfully!")
