# import openai
# import os
# from dotenv import load_dotenv

# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

# print("=== OpenAI Library Version ===")
# print(openai.__version__)

# print("\n=== Checking API Key and Available Models ===")
# try:
#     # List models using the new API
#     models = openai.models.list()
#     print("✅ API key is valid!")
#     print("\n=== Available Models ===")
#     for m in models.data:
#         print("-", m.id)
# except Exception as e:
#     print("❌ Error:", e)

import pinecone
import os
from dotenv import load_dotenv

load_dotenv()
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

index = pinecone.Index(os.getenv("PINECONE_INDEX"))

