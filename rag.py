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
    General RAG chatbot for Burmese + English answers
    """
    return _query_with_context(
        user_input=user_input,
        feature="general",
        top_k=top_k
    )


def content_checker(user_content, top_k=5):
    """
    Content Checker Feature:
    Analyze whether pasted article/post/blog is valid or suspicious
    Uses the `content_checker` key (or namespace) in Pinecone DB
    """
    return _query_with_context(
        user_input=user_content,
        feature="content_checker",
        top_k=top_k
    )


def _query_with_context(user_input, feature="general", top_k=3):
    """
    Internal helper: query Pinecone for different features
    `feature` can be "general" or "content_checker"
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

    # 2️⃣ Query Pinecone (with namespace depending on feature)
    try:
        namespace = "content_checker" if feature == "content_checker" else None
        result = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        context_texts = [match.metadata["text"] for match in result.matches]
    except Exception as e:
        return f"Error querying Pinecone: {str(e)}"

    # 3️⃣ Build system prompt
    if feature == "general":
        system_prompt = (
            "You are LannPya Bot, a friendly cybersecurity guide for non-tech-savvy users in Myanmar. "
            "Always respond in Burmese using short and clear sentences. "
            "Explain in simple terms. "
            "Use the following context from your knowledge base:\n\n"
            f"{chr(10).join(context_texts)}\n\n"
            "Only use your knowledgebase."
        )
    else:  # content_checker
        system_prompt = (
            "You are LannPya Bot’s Content Checker. "
            "Your job is to check if the given content (article, blog, post) looks valid or suspicious. "
            "Always respond in Burmese using short and clear sentences. "
            "Explain why it is safe or unsafe. "
            "Show examples if needed. "
            "Use the following rules from your knowledge base:\n\n"
            f"{chr(10).join(context_texts)}\n\n"
            "Only use your knowledgebase for judgment."
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
            max_tokens=800
        )
        return chat_resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"
    
def ask_bot_content_checker(content, poster, date, platform, top_k=3):
    """
    Content checker using RAG + metadata
    """
    try:
        emb_resp = openai.embeddings.create(
            model="text-embedding-3-small",
            input=content
        )
        query_vector = emb_resp.data[0].embedding
    except Exception as e:
        return f"Error creating embedding: {str(e)}"

    try:
        result = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter={"category": "content_checker"}  # only search content checker KB
        )
        context_texts = [match.metadata["text"] for match in result.matches]
    except Exception as e:
        return f"Error querying Pinecone: {str(e)}"

    # system prompt includes metadata
    system_prompt = (
        "You are LannPya Bot, a content checker for Burmese users. "
        "Analyze whether the content is good, reliable, or suspicious. "
        "Take into account the following metadata:\n"
        f"Poster: {poster}\nDate: {date}\nPlatform: {platform}\n\n"
        "Use this context from the knowledge base:\n"
        f"{chr(10).join(context_texts)}\n\n"
        "Respond clearly in Burmese and give reasoning."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]

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

