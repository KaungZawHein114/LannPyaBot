import os
import openai
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone (new way)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX")
index = pc.Index(index_name)

def ask_bot(user_input, top_k=3):
    """
    RAG-enabled chatbot for Burmese + English answers
    """
    # 1️⃣ Embed user query
    try:
        emb_resp = openai.embeddings.create(
            model="text-embedding-3-small",
            input=user_input
        )
        query_vector = emb_resp.data[0].embedding
    except Exception as e:
        return f"Error creating embedding: {str(e)}"

    # 2️⃣ Query Pinecone for top_k matches
    try:
        result = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        context_texts = [match.metadata["text"] for match in result.matches]
    except Exception as e:
        return f"Error querying Pinecone: {str(e)}"

    # 3️⃣ System prompt with context
    system_prompt = (
        "You are LannPya Bot, a friendly cybersecurity guide for non-tech-savvy users in Myanmar. "
        "Always respond in Burmese using short and clear sentences. "
        "Explain in simple terms. "
        "Use the following context from your knowledge base:\n\n"
        f"{chr(10).join(context_texts)}\n\n"
        "Only use your knowledgebase"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    # 4️⃣ OpenAI ChatCompletion
    try:
        chat_resp = openai.chat.completions.create(
            model="gpt-5-chat-latest",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        return chat_resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"
