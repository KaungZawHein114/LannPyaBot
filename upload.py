import os
import json
import re
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
    """entries = [(id, text, metadata)]"""
    if not entries:
        return

    texts = [text for _, text, _ in entries]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    vectors = []
    for (entry_id, text, metadata), emb in zip(entries, response.data):
        vectors.append((entry_id, emb.embedding, {"text": text, **metadata}))

    index.upsert(vectors)
    print(f"✅ Upserted {len(vectors)} entries from {source}")

# --------------------
# Burmese number normalization
# --------------------
BURMESE_NUMS = "၀၁၂၃၄၅၆၇၈၉"
ARABIC_NUMS = "0123456789"
TRANS_TABLE = str.maketrans(BURMESE_NUMS, ARABIC_NUMS)

def normalize_numbers(text):
    return text.translate(TRANS_TABLE)

# --------------------
# Sanitize vector IDs
# --------------------
def sanitize_id(text):
    # Convert Burmese numbers to Arabic numbers
    text = normalize_numbers(text)
    # Replace non-ASCII characters with dash
    return re.sub(r'[^\x00-\x7F]+', '-', text)

# --------------------
# Parser for Cybersecurity Law.txt
# --------------------
def parse_law_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    chapter = None
    law_number = None
    buffer = []
    entries = []

    def save_entry():
        if law_number and buffer:
            text = "\n".join(buffer).strip()
            # Chunk long texts
            chunks = chunk_text(text, chunk_size=1000, overlap=200)
            for idx, chunk in enumerate(chunks, start=1):
                entries.append((
                    f"{law_number}-part{idx}",
                    chunk,
                    {
                        "chapter": chapter,
                        "law_number_burmese": law_number,
                        "law_number_arabic": normalize_numbers(law_number),
                        "source": os.path.basename(filepath)
                    }
                ))

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect chapters (e.g. အခန်း (၁))
        if line.startswith("အခန်း"):
            chapter = line
            continue

        # Detect law sections (e.g. ၃၆။ )
        match = re.match(r"^([၀၁၂၃၄၅၆၇၈၉]+)။", line)
        if match:
            save_entry()  # save previous
            law_number = match.group(1)  # Burmese number only
            buffer = [line]
        else:
            buffer.append(line)

    save_entry()
    return entries

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
            entry_id = sanitize_id(f"{topic}-{i}")
            batch.append((entry_id, entry, {"source": "json"}))

            if len(batch) >= 50:
                batch_upsert(batch, "json")
                batch = []

    if batch:
        batch_upsert(batch, "json")
else:
    print("⚠️ No knowledge_base.json found, skipping JSON upload.")

# --------------------
# Step 2: Upload PDFs and TXTs
# --------------------
pdf_folder = "knowledgebase"
for filename in os.listdir(pdf_folder):
    file_path = os.path.join(pdf_folder, filename)
    file_name = sanitize_id(os.path.splitext(filename)[0].replace(" ", "-"))

    if filename.endswith(".pdf"):
        print(f"\n📄 Processing PDF: {filename}...")
        text = extract_text_from_pdf(file_path, use_ocr=True)
        chunks = chunk_text(text, chunk_size=1000, overlap=200)

        batch = []
        for i, chunk in enumerate(chunks, start=1):
            entry_id = sanitize_id(f"{file_name}-{i}")
            batch.append((entry_id, chunk, {"source": filename}))

            if len(batch) >= 50:
                batch_upsert(batch, filename)
                batch = []

        if batch:
            batch_upsert(batch, filename)

    elif filename.endswith(".txt") and "Cybersecurity" in filename:
        print(f"\n📄 Processing TXT Law File: {filename}...")
        entries = parse_law_text(file_path)

        batch = []
        for i, (entry_id, text, metadata) in enumerate(entries, start=1):
            full_id = sanitize_id(f"{file_name}-{entry_id}")
            batch.append((full_id, text, metadata))

            if len(batch) >= 50:
                batch_upsert(batch, filename)
                batch = []

        if batch:
            batch_upsert(batch, filename)

print("\n🎉 All knowledgebase data uploaded successfully!")
