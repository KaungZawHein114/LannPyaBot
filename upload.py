import json
import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# 1️⃣ Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2️⃣ Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX")
index = pc.Index(index_name)

# 0️⃣ Clear the whole index first
index.delete(delete_all=True)
print("🧹 Cleared all previous entries from the index.")

# 3️⃣ Load JSON knowledge base
with open("knowledge_base.json", "r", encoding="utf-8") as f:
    data = json.load(f)  # dict: category -> list of texts

# 4️⃣ Upload each entry with category-based IDs
for category, texts in data.items():
    prefix = category[:3].upper()  # first three letters of category
    for i, text in enumerate(texts, start=1):
        entry_id = f"{prefix}-{i}"

        # create embedding
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        vector = emb.data[0].embedding

        # upsert to Pinecone
        index.upsert([(entry_id, vector, {"text": text, "category": category})])
        print(f"✅ Upserted {entry_id}: {text[:40]}...")  # preview first 40 chars

print("🎉 Knowledge base uploaded successfully!")
