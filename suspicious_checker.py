import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def check_suspicious(text):
    """
    Check if a message, link, or app name is potentially dangerous.
    Returns a short analysis in Burmese and English.
    """
    prompt = f"""
    You are LannPya Bot, a friendly bilingual cybersecurity guide.
    Analyze the following content for scams, phishing, or malicious intent.
    
    Content: {text}

    Respond clearly in two short paragraphs:
    1. Risk Level: Low / Medium / High
    2. Reason: Simple explanation in Burmese and English for non-tech-savvy users.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        analysis = response.choices[0].message.content.strip()
        return analysis
    except Exception as e:
        return f"Error: {str(e)}"
