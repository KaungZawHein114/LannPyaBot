import os
import openai
from pinecone import Pinecone
from dotenv import load_dotenv
import json
import random

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone (new way)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX")
index = pc.Index(index_name)

# AI-only helper
def ai_only(prompt: str, max_tokens=600):
    try:
        chat_resp = openai.chat.completions.create(
            model="gpt-5-chat-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_tokens
        )
        return chat_resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error in ai_only: {str(e)}"

def ask_bot(user_input, top_k=3):
    """
    General RAG chatbot for Burmese + English answers.
    Always uses default Pinecone namespace.
    """
    return _query_with_context(user_input=user_input, feature="general", top_k=top_k)


def content_checker(content, poster="", date="", platform="", top_k=3):
    """
    Content Checker chatbot.
    Adds metadata (poster, date, platform) to the system prompt.
    Uses default Pinecone namespace.
    """
    # Include metadata in user input to influence GPT reasoning
    user_input_with_meta = f"Poster: {poster}\nDate: {date}\nPlatform: {platform}\n\nContent:\n{content}"
    
    return _query_with_context(user_input=user_input_with_meta, feature="content_checker", top_k=top_k)


def _query_with_context(user_input, feature="general", top_k=3):
    """
    Internal helper to query Pinecone and build GPT prompt.
    Uses default namespace for all features.
    """
    # 1️⃣ Embed user input
    try:
        emb_resp = openai.embeddings.create(
            model="text-embedding-3-small",
            input=user_input
        )
        query_vector = emb_resp.data[0].embedding
    except Exception as e:
        return f"Error creating embedding: {str(e)}"

    # 2️⃣ Query Pinecone
    try:
        result = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )
        context_texts = [match.metadata["text"] for match in result.matches]
    except Exception as e:
        context_texts = []
        print(f"Warning: Pinecone query failed: {str(e)}")

    # 3️⃣ Build system prompt
    if feature == "general":
        system_prompt = (
            "You are LannPya Bot, a friendly cybersecurity guide for non-tech-savvy users in Myanmar. "
            "Always respond in Burmese using short and clear sentences. Explain in simple terms.\n"
            "Always use consistent pronouns: Refer to yourself as ကျွန်တော် and the user as သင်\n"
            "Use consistent pronouns in Burmese: ကျွန်တော် for I (subject). ကျွန်တော့် for me (object). ကျွန်တော့်ရဲ့ for my/mine (possessive). ကျွန်တော်ကိုယ်တိုင် for myself. သင် for you (subject). သင့် for you (object). သင့်ရဲ့  for your/yours (possessive). သင်ကိုယ်တိုင် for yourself.\n"
            "Appreciate the user if he knows or can share specific data to help solve the problem.\n"
            "Always use an encouraging and supportive tone.\n"
            "Remind the user that they are doing well and that learning step by step is normal.\n"
            "Use polite particles like 'ပါ', 'မယ်', 'တယ်' to keep a respectful and friendly tone.\n"
            "Sometimes use 'ကျွန်တော်တို့' (we) to make the user feel included, e.g., 'ကျွန်တော်တို့ အတူတူ လေ့လာကြမယ်။'.\n"
            "Handle unclear inputs gently: If the user types something confusing, suggest politely what they might mean instead of rejecting directly.\n"
            "Use simple encouragements like 'အဆင်ပြေမယ်', 'မပူပါနဲ့', 'ကောင်းပါပြီ', 'လုပ်နိုင်မယ်' to motivate the user.\n"
            "Occasionally use helpful emojis (✅, 🔐, 📱, 👍) to make responses friendlier, but never overuse them.\n"
            "Separate lines for better understanding and clear format.\n"
            "Bold the important keywords.\n"
            "Use Burmese words as much as possible but use English words for tech terms.\n"
            "All of your response must be either in English or in Burmses. Do not put any other language symbols.\n"
            "If user shares their name, start every response by calling their name.\n"
            "Conversation rules:\n"
            "1. If the user only says short acknowledgments like 'yes', 'ok', 'thanks', 'thank you', "
            "or emoji responses, respond politely but do not assume they are asking a new question.\n"
            "2. Only provide detailed guidance if the user asks a clear question about cybersecurity or privacy.\n"
            "3. If the user asks questions outside your scope (e.g., 'What's life?', general knowledge, or unrelated topics), "
            "politely refuse and remind them that you only provide guidance on cybersecurity and privacy.\n"
            "4. Keep answers short and to the point when the message is clearly an acknowledgment.\n"
            "Knowledge about laws:\n"
            "1. You have access to Myanmar Cybersecurity Law and related regulations uploaded in your knowledgebase.\n"
            "2. Always explain legal information accurately in Burmese using simple sentences.\n"
            "3. Highlight important **law numbers**, **chapters**, and **key points** in bold.\n"
            "4. If the user asks for a law reference, provide the **law number** and **chapter** clearly.\n"
            "5. Avoid giving legal advice beyond what the law text says; always stick to factual legal content.\n\n"
            "Knowledge base context:\n"
            + ("\n".join(context_texts) if context_texts else "[No relevant knowledgebase context]") +
            "\n\nUse this knowledgebase to analyze the content. "
            "If the answer is not found well enough in the knowledgebase, use your own knowledge."
        )
    else:  # content_checker
        system_prompt = (
            "You are LannPya Bot’s Content Checker. "
            "Analyze whether the given content is trustworthy or suspicious. "
            "Respond in Burmese with short, clear sentences. "
            "Greeting should always start with 'မင်္ဂလာပါ' and polite tone in Burmese. "
            "Separate lines for better understanding and clear format. "
            "Your tone must sound as an educator or guide. "
            "Bold the important keywords. "
            "Use Burmese words as much but use English words for tech terms. "
            "If user shares his name, start every response by calling his name."
            "Respond only with analysis; do NOT ask follow-up questions or continue the conversation."
            "Do NOT ask any follow-up questions, suggest extra steps, or engage in conversation.\n"
            "Knowledge about laws:\n"
            "- You have access to Myanmar Cybersecurity Law and related regulations uploaded in your knowledgebase.\n"
            "- If the content is related to legal matters, mention the relevant **law numbers**, **chapters**, or **key points**.\n"
            "- Always explain legal information accurately in Burmese using simple, clear sentences.\n"
            "- Avoid giving legal advice; stick to factual information from the law texts.\n\n"
            "Knowledge base context:\n"
            + ("\n".join(context_texts) if context_texts else "[No relevant knowledgebase context]") +
            "Use this knowledgebase to analyze the content. "
            "If the answer is not found well enough in the knowledgebase, use your own knowledge."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    # 4️⃣ GPT completion
    try:
        chat_resp = openai.chat.completions.create(
            model="gpt-5-chat-latest",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        return chat_resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error in GPT response: {str(e)}"

    
def ask_bot_content_checker(content, poster, date, platform, top_k=3):
    """
    Content Checker using RAG + GPT.
    1️⃣ Query Pinecone for top_k most relevant context.
    2️⃣ Include metadata (poster/platform/date) in reasoning.
    3️⃣ GPT uses knowledgebase first, falls back to its own knowledge if needed.
    """
    # 1️⃣ Embed user content
    try:
        emb_resp = openai.embeddings.create(
            model="text-embedding-3-small",
            input=content
        )
        query_vector = emb_resp.data[0].embedding
    except Exception as e:
        return f"Error creating embedding: {str(e)}"

    # 2️⃣ Query Pinecone (default namespace)
    try:
        result = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )
        context_texts = [match.metadata["text"] for match in result.matches]
    except Exception as e:
        context_texts = []
        print(f"Warning: Pinecone query failed: {str(e)}")

# 3️⃣ Build system prompt for Content Checker (English)
    system_prompt = (
        "You are LannPya Bot’s Content Checker. "
        "Analyze whether the given content is trustworthy or suspicious. "
        "Respond in Burmese with short, clear sentences. "
        "Greeting should always start with 'မင်္ဂလာပါ' and polite tone in Burmese. "
        "Separate lines for better understanding and clear format. "
        "Your tone must sound as an educator or guide. "
        "Bold the important keywords. "
        "Use Burmese words as much but use English words for tech terms. "
        "If user shares his name, start every response by calling his name."
        "- Respond only with analysis; do NOT ask follow-up questions or continue the conversation."
        "- Do NOT ask any follow-up questions, suggest extra steps, or engage in conversation.\n"
        "Knowledge about laws:\n"
        "- You have access to Myanmar Cybersecurity Law and related regulations uploaded in your knowledgebase.\n"
        "- If the content is related to legal matters, mention the relevant **law numbers**, **chapters**, or **key points**.\n"
        "- Always explain legal information accurately in Burmese using simple, clear sentences.\n"
        "- Avoid giving legal advice; stick to factual information from the law texts.\n\n"
        "Knowledge base context:\n"
        + (
            "\n".join(context_texts)
            if context_texts
            else "[No relevant knowledgebase context]"
        )
        + 
        "Use this knowledgebase to analyze the content. "
        "If the answer is not found well enough in the knowledgebase, use your own knowledge."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]

    # 4️⃣ GPT completion
    try:
        chat_resp = openai.chat.completions.create(
            model="gpt-5-chat-latest",
            messages=messages,
            temperature=0.6,
            max_tokens=800
        )
        return chat_resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error in GPT response: {str(e)}"
    
PREDEFINED_QUESTIONS = {
    "phishing": [
        "သင့်ဆီကို မသိသောလူမှ အီးမေးလ် (သို့) မက်ဆေ့ချ် ရရှိခဲ့သလား?",
        "ထိုမက်ဆေ့ချ်တွင် လင့်ခ်တစ်ခုကို နှိပ်ရန် (သို့) တစ်ခုခု ဒေါင်းလုဒ်လုပ်ရန် တောင်းဆိုထားသလား?",
        "၎င်းတွင် သင့်စကားဝှက်၊ ဘဏ်အချက်အလက် (သို့) ကိုယ်ရေးကိုယ်တာ အချက်အလက်တွေကို တောင်းဆိုထားသလား?",
        "ပို့သူ၏ လိပ်စာ (သို့) လင့်ခ်သည် သံသယဖြစ်ဖွယ် (သို့) ထူးခြားနေသလား?",
        "သင် အချက်အလက်တစ်ခုခု မျှဝေမိပြီးလား (သို့) လင့်ခ်ကို နှိပ်မိပြီးလား?"
    ],
    "scam": [
        "တစ်စုံတစ်ယောက်က ငွေကြေး (သို့) ဆုလာဘ်အကြောင်း မျှော်လင့်မထားဘဲ ဆက်သွယ်လာသလား?",
        "သင့်ကို မြန်မြန်ဆန်ဆန် လုပ်ဆောင်ရန် ဖိအားပေးခဲ့သလား?",
        "ထူးခြားသောနည်းလမ်းများဖြင့် (ဆုကဒ်၊ ကရစ်ပတိုငွေကြေး) ငွေပေးချေရန် တောင်းဆိုခဲ့သလား?",
        "ထိုအကြောင်းကြားချက် (သို့) ခေါ်ဆိုသူသည် တရားဝင်မှုရှိမရှိ သင် စစ်ဆေးခဲ့သလား?",
        "သင် ငွေပေးချေပြီးသွားပြီလား (သို့) ကိုယ်ရေးကိုယ်တာ အချက်အလက်များ မျှဝေမိပြီးလား?"
    ],
    "hack": [
        "သင့်အကောင့်များတွင် ပုံမှန်မဟုတ်သော လှုပ်ရှားမှုများ ရှိမရှိ သတိထားမိသလား?",
        "မကြာသေးမီက သင့်စကားဝှက်ကို ပြန်သုံးမိသလား (သို့) မျှဝေမိသလား?",
        "သင့်ပိုင်ဆိုင်သော အကောင့်တစ်ခုခုသို့ ဝင်ရောက်၍ မရတော့သလား?",
        "ထူးဆန်းသော တည်နေရာများမှ အကောင့်ဝင်ရောက်မှုအကြောင်း သတိပေးချက်များ ရရှိခဲ့သလား?",
        "သင့်ထံခွင့်မတောင်းဘဲ ဖိုင်များ (သို့) ဆက်တင်များ ပြောင်းလဲသွားသလား?"
    ],
    "fraud": [
        "သင့်ငွေကြေး (သို့) ကိုယ်ရေးကိုယ်တာ အချက်အလက်များကို ခွင့်ပြုချက်မရှိဘဲ အသုံးပြုခံရသလား?",
        "သင့်အကောင့်ထဲတွင် မသိသော ငွေလွှဲမှုများ ရှိနေသလား?",
        "တစ်စုံတစ်ယောက်က အွန်လိုင်း (သို့) အော့ဖ်လိုင်းတွင် သင့်ကိုယ်စား ဟန်ဆောင်နေသလား?",
        "သံသယဖြစ်ဖွယ် လုပ်ငန်း (သို့) ဝက်ဘ်ဆိုက်တစ်ခုထံ အချက်အလက်များ ပေးခဲ့မိသလား?",
        "ထိုကဲ့သို့သော ပြဿနာ ယခင်က ဖြစ်ဖူးပါသလား?"
    ],
    "threat": [
        "သင့်ကို အွန်လိုင်း (သို့) အော့ဖ်လိုင်းမှ တိုက်ရိုက်ခြိမ်းခြောက်မှု ရရှိခဲ့သလား?",
        "ထိုမက်ဆေ့ချ်တွင် သင့်ကိုယ်တိုင်၊ မိသားစု (သို့) ပိုင်ဆိုင်မှုများကို အန္တရာယ်ပြုမည်ဟု ဖော်ပြထားသလား?",
        "ငွေကြေး (သို့) တုံ့ပြန်မှုတစ်စုံတစ်ရာအတွက် ၎င်းတို့က တောင်းဆိုထားသလား?",
        "ထိုခြိမ်းခြောက်မှုသည် ထပ်ကာထပ်ကာ ရရှိသလား (သို့) နေရာအမျိုးမျိုးမှ ရရှိသလား?",
        "ထိုခြိမ်းခြောက်မှုကို အာဏာပိုင်တစ်ခုခုသို့ သတင်းပို့တင်ပြခဲ့ပါသလား?"
    ],
    "malware": [
        "သင့်စက်ပစ္စည်း ပုံမှန်ထက် နှေးကွေးနေသလား (သို့) ပုံမှန်မဟုတ်ဘဲ လုပ်ဆောင်နေသလား?",
        "အတည်မပြုရသေးသော ဆော့ဖ်ဝဲတစ်ခုခုကို ဒေါင်းလုဒ်လုပ် (သို့) ထည့်သွင်းခဲ့သလား?",
        "သင် ထည့်သွင်းခဲ့ခြင်းမရှိသော ပေါ့အပ်များ (သို့) ပရိုဂရမ်များ မြင်နေရသလား?",
        "ဗိုင်းရပ်စ်ကာကွယ်ရေးဆော့ဖ်ဝဲမှ တစ်ခုခုကို မကြာသေးမီက စစ်ဆေးတွေ့ရှိ (သို့) ပိတ်ဆို့ထားသလား?",
        "ဖိုင်များ ပျောက်ဆုံး (သို့) စာဝှက်သွားခံရသလား?"
    ]
}

def generate_followup_questions(topic, user_answers):
    """
    Given first 5 answers, generate 5 more questions dynamically using OpenAI.
    """
    combined_text = "\n".join([f"Q: {q} → A: {a}" for q, a in user_answers.items()])
    prompt = f"""
    Based on the user's answers to the following core cybersecurity questions:
    {combined_text}

    Generate 5 follow-up diagnostic questions for the same topic in Burmese.
    Output as a numbered list, short questions only.
    """
    followup_qs = ai_only(prompt).split("\n")
    questions = []
    for q in followup_qs:
        q = q.strip()
        if q:
            questions.append({"question": q, "type": "mcq_text"})  # mcq + optional textbox
    return questions[:5]  # only 5

# Scenario Simulation in rag.py
def get_scenario_questions(topic, stage=1, user_answers=None):
    """
    Returns questions for scenario simulation.
    - stage=1: first 5 core questions
    - stage=2: generate 5 follow-up questions based on first 5 answers
    - stage=others: fully AI-generated
    Each question is a dict: {"question": str, "type": "mcq_text"/"text"}
    """
    questions = []

    if topic in PREDEFINED_QUESTIONS:
        if stage == 1:
            for q in PREDEFINED_QUESTIONS[topic]:
                questions.append({"question": q, "type": "mcq_text"})
        elif stage == 2 and user_answers:
            questions = generate_followup_questions(topic, user_answers)
        # Always add free-text at the end
        questions.append({"question": "Any additional details?", "type": "text"})
    else:
        # Fully AI-generated 10 questions (text input)
        prompt = "Generate 10 short cybersecurity diagnostic questions in Burmese for a general unknown threat."
        text_qs = ai_only(prompt).split("\n")[:10]
        for q in text_qs:
            questions.append({"question": q, "type": "text"})
        questions.append({"question": "Any additional details?", "type": "text"})
    return questions


def analyze_scenario_responses(topic, user_answers):
    """
    Analyze user answers:
    1️⃣ Pinecone retrieval
    2️⃣ OpenAI analysis
    Returns risks + solutions
    """
    # Prepare combined text
    combined_text = "\n".join([f"Q: {q} → A: {a}" for q, a in user_answers.items()])

    # Step 1: Pinecone + OpenAI
    pinecone_context = _query_with_context(combined_text, feature="general", top_k=5)

    # Step 2: Refined OpenAI analysis
    final_prompt = f"""
    You are LannPya Bot, a Scenario Simulation Assistant.

    User Topic: {topic}

    User's Responses:
    {combined_text}

    Pinecone Context:
    {pinecone_context}

    Instructions:
    - Analyze the above information and summarize the main **RISKS** and actionable **SOLUTIONS**.
    - If any risks or solutions are related to legal requirements, mention the relevant **Myanmar Cybersecurity Law numbers**, **chapters**, or **key points**.
    - Respond in **Burmese**, clearly and concisely.
    - Use line breaks to separate each risk and solution for clarity.
    - Bold the most important keywords.
    - Use Burmese words as much as possible, and English words for technical terms.
    - Respond only with analysis; do NOT ask follow-up questions or continue the conversation.
    - Do NOT ask any follow-up questions, suggest extra steps, or engage in conversation.
    - Polite, professional, and educator-like tone.
    - Use few emojis to emphasize meanings of keywords (not too many).

    Output Format Example:
    **RISKS (အန္တရာယ်):**
    1. <risk description>
    2. <risk description>

    **SOLUTIONS (ဖြေရှင်းချက်):**
    1. <solution description>
    2. <solution description>

    **LEGAL REFERENCES (ဥပဒေညွှန်ကြားချက်) [if relevant]:**
    - Mention relevant **law numbers** and **chapters** from the Myanmar Cybersecurity Law as needed.
    - Only include factual information from the law; do not give legal advice.
    """
    
    final_result = ai_only(final_prompt, max_tokens=1000)
    return final_result

def load_quiz_data(path="your_data.json"):
    """Load quiz data JSON safely."""
    with open(path, "r", encoding="utf-8") as f:
        data = f.read().strip()
        if not data:
            raise ValueError("JSON file is empty!")
        return json.loads(data)


def parse_quiz_text(ai_text):
    """
    Converts AI output into a structured list of questions.
    AI should return something like:
    1. Question?
       a) Option1
       b) Option2
       c) Option3
       d) Option4
       Answer: a
    """
    import re
    questions = []
    if not ai_text or not ai_text.strip():
        return questions

    try:
        text = ai_text.replace("\r\n", "\n").replace("\r", "\n").strip()

        # Primary split by blank lines or lines of dashes
        raw_blocks = [b for b in re.split(r"\n\s*(?:\n|[-=]{3,,}\n)+", text) if b.strip()]
        for block in raw_blocks:
            try:
                lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
                if not lines:
                    continue

                q_line = re.sub(r"^\s*\d+[\.)]\s*", "", lines[0]).strip()
                question_text = q_line

                option_pattern = re.compile(r"^([a-dA-D])\s*[\)\.:\-]\s*(.+)$")
                label_to_option = {}
                answer_letter = None
                answer_text = None

                for ln in lines[1:]:
                    if re.match(r"(?i)^answer\s*[:\-]", ln):
                        m_ans_letter = re.search(r"(?i)answer\s*[:\-]\s*([a-dA-D])\b", ln)
                        if m_ans_letter:
                            answer_letter = m_ans_letter.group(1).lower()
                        else:
                            m_ans_text = re.search(r"(?i)answer\s*[:\-]\s*(.+)$", ln)
                            if m_ans_text:
                                answer_text = m_ans_text.group(1).strip()
                        continue

                    m = option_pattern.match(ln)
                    if m:
                        label = m.group(1).lower()
                        text_val = m.group(2).strip()
                        if label not in label_to_option:
                            label_to_option[label] = text_val

                if not all(k in label_to_option for k in ["a", "b", "c", "d"]):
                    continue

                if answer_letter and answer_letter in label_to_option:
                    correct_option_text = label_to_option[answer_letter]
                elif answer_text:
                    norm = answer_text.strip().lower()
                    correct_option_text = next((opt for opt in label_to_option.values() if opt.strip().lower() == norm), None)
                else:
                    correct_option_text = None

                if not correct_option_text:
                    for opt_txt in label_to_option.values():
                        if re.search(r"(?i)\b(correct|true|right)\b", opt_txt):
                            correct_option_text = opt_txt
                            break

                if not correct_option_text:
                    continue

                ordered_options = [label_to_option[k] for k in ["a", "b", "c", "d"]]
                questions.append({
                    "question": question_text,
                    "options": ordered_options,
                    "answer": correct_option_text
                })
            except Exception as inner_e:
                print("Warning: Skipping malformed quiz block:", inner_e)

    except Exception as e:
        print("Error parsing AI quiz text:", e)

    return questions[:10]

def generate_quiz_from_topic(topic_name, difficulty: str | None = None):
    """
    Generate quiz from JSON topic using OpenAI.
    """
    with open("knowledge_base.json", "r", encoding="utf-8") as f:
        quiz_data = json.load(f)
        
    if topic_name not in quiz_data:
        print(f"Topic '{topic_name}' not found in JSON.")
        return []

    topic_text = "\n".join(quiz_data[topic_name])  # all lines for that topic
    diff_text = (difficulty or "medium").lower()
    if diff_text not in {"easy", "medium", "hard"}:
        diff_text = "medium"

    base_prompt = f"""
    You are an expert quiz creator.
    Create 10 multiple-choice questions (1 correct + 3 wrong answers each) from the following text:

    {topic_text}

    Difficulty: {diff_text}

    Return questions in EXACTLY this format, with a blank line between questions, no extra commentary:
    1. Question text in Burmese?
       a) Option 1
       b) Option 2
       c) Option 3
       d) Option 4
       Answer: a
    """

    prompts = [
        base_prompt,
        base_prompt + "\nOnly output the items in the specified format. Do not add explanations, headings, or anything else.",
        base_prompt + "\nAbsolutely do not include any text outside the 10 questions in the specified format.",
    ]

    for attempt, prompt in enumerate(prompts, start=1):
        try:
            response = openai.chat.completions.create(
                model="gpt-5-chat-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1500
            )
            ai_text = response.choices[0].message.content.strip()
            questions = parse_quiz_text(ai_text)
            if questions:
                return questions
            else:
                print(f"Warning: attempt {attempt} parsed 0 questions.")
        except Exception as e:
            print(f"Error generating quiz on attempt {attempt}:", e)

    # If still nothing, return empty list (caller may decide how to handle)
    return []

def generate_random_tip():
    """
    Returns a random cybersecurity tip using AI from the knowledge base.
    """
    # Load knowledge base once
    with open("knowledge_base.json", "r", encoding="utf-8") as f:
        kb = json.load(f)
    
    topic = random.choice(list(kb.keys()))
    content = "\n".join(kb[topic])

    prompt = (
        f"Create a short, practical cybersecurity tip from the following information:\n"
        f"{content}\n"
        f"Keep it under 10 words."
        "Use Burmese but you can use English technical terms where needed."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-5-chat-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        tip = response.choices[0].message.content.strip()
    except Exception as e:
        print("Error generating tip:", e)
        tip = "Stay safe online! (Could not generate AI tip.)"

    return {"tip": tip, "topic": topic}