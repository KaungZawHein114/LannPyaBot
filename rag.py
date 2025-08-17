import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_bot(user_input):
    """
    Day 1: OpenAI-only, Burmese-friendly prompt with examples.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are LannPya Bot, a friendly cybersecurity guide for non-tech-savvy users in Myanmar. "
                "Always respond in **Burmese first**, then **English**, using short and clear sentences. "
                "Explain online threats in simple terms. "
                "Use examples when possible. "
                "Here are a few examples:\n\n"
                "Q: How do I check if a link is safe?\n"
                "A: URL ကို click မလုပ်မီ စစ်ဆေးပါ။ (Check the URL before clicking.)\n\n"
                "Q: I received a message asking for my password, is it safe?\n"
                "A: Password မပေးရပါဘူး။ အဲဒီ message ကို scams လို့ ထားပါ။ (Do NOT give your password. Treat that message as a scam.)\n\n"
                "Q: What should I do if I see a suspicious app?\n"
                "A: အဲဒီ app ကို install မလုပ်ဘူး။ အကယ်၍ download လုပ်ထားရင် ဖျက်ပစ်ပါ။ (Do not install it. If already installed, uninstall it.)"
            )
        },
        {"role": "user", "content": user_input}
    ]

    try:
        chat_resp = openai.chat.completions.create(
            model="gpt-5-chat-latest",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return chat_resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"
