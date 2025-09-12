from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rag import ask_bot_content_checker, get_scenario_questions, analyze_scenario_responses, ai_only, _query_with_context, generate_quiz_from_topic, generate_random_tip

app = Flask(__name__)
CORS(app)  # allow frontend to call API

@app.route("/")
def home():
    return render_template("index.html")  # frontend file

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/content-check")
def content_checker_page():
    return render_template("contentChecker.html")

@app.route("/scenario/start")
def scenario_simulation_page():
    return render_template("scenarioSimulation.html")

@app.route("/generate-quiz")
def interactive_quiz_page():
    return render_template("interactiveQuiz.html")

# Keep in-memory conversation history per session (for example purposes; for production, use Redis or DB)
conversation_history = []

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    
    # 1️⃣ Add user message to history
    conversation_history.append({"role": "user", "content": user_message})
    
    # 2️⃣ Get RAG context from Pinecone
    rag_context = _query_with_context(user_input=user_message, feature="general", top_k=3)
    
    # 3️⃣ Build GPT prompt with conversation + RAG context
    messages = []
    system_msg = {
        "role": "system",
        "content": (
            "You are LannPya Bot, a friendly cybersecurity guide for non-tech-savvy users in Myanmar. "
            "Always use consistent pronouns: Refer to yourself as ကျွန်တော် and the user as သင်\n"
            "Use consistent pronouns in Burmese: ကျွန်တော် for I (subject). ကျွန်တော့် for me (object). ကျွန်တော့်ရဲ့ for my/mine (possessive). ကျွန်တော်ကိုယ်တိုင် for myself. သင် for you (subject). သင့် for you (object). သင့်ရဲ့  for your/yours (possessive). သင်ကိုယ်တိုင် for yourself.\n"
            "Always respond in Burmese using short and clear sentences. Explain in simple terms.\n"
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
            "Never use any other language except English and Burmese.\n"
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
            f"Knowledge base context:\n{rag_context if rag_context else '[No relevant context]'}\n\n"
            "Use this knowledgebase to answer the user's question. "
            "If the answer is not found well enough in the knowledgebase, use your own knowledge—but only for cybersecurity and privacy questions. "
            "Politely refuse all other questions."
        )
    }
    messages.append(system_msg)
    
    # Add entire conversation history
    for turn in conversation_history:
        messages.append(turn)
    
    # 4️⃣ Call GPT
    bot_response = ai_only(prompt="", max_tokens=800)  # empty prompt, GPT reads `messages`
    
    # Actually, ai_only currently expects a string, so we can make a small helper:
    def ai_with_messages(messages, max_tokens=800):
        import openai
        resp = openai.chat.completions.create(
            model="gpt-5-chat-latest",
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()
    
    bot_reply = ai_with_messages(messages)
    
    # 5️⃣ Add bot reply to conversation
    conversation_history.append({"role": "assistant", "content": bot_reply})
    
    return jsonify({"reply": bot_reply})

@app.route("/content-check", methods=["GET", "POST"])
def content_check():
    data = request.json
    content = data.get("content", "")
    poster = data.get("poster", "")
    date = data.get("date", "")
    platform = data.get("platform", "")
    result = ask_bot_content_checker(content, poster, date, platform)
    return jsonify({"result": result})


@app.route("/scenario/start", methods=["GET", "POST"])
def scenario_start():
    data = request.json
    topic = data.get("topic", "others")
    # Stage 1: return first 5 core questions
    questions = get_scenario_questions(topic, stage=1)
    return jsonify({"topic": topic, "questions": questions})

@app.route("/scenario/next", methods=["GET", "POST"])
def scenario_next():
    data = request.json
    topic = data.get("topic", "others")
    first_answers = data.get("answers", {})
    # Stage 2: generate 5 follow-up questions
    questions = get_scenario_questions(topic, stage=2, user_answers=first_answers)
    return jsonify({"topic": topic, "questions": questions})

@app.route("/scenario/analyze", methods=["GET", "POST"])
def scenario_analyze():
    data = request.json
    topic = data.get("topic", "others")
    answers = data.get("answers", {})  # expects dict {question: answer, ...}
    result = analyze_scenario_responses(topic, answers)
    return jsonify({"result": result})

@app.route("/scenario/others", methods=["GET", "POST"])
def scenario_others():
    data = request.json
    history = data.get("history", [])  # list of {q, a}
    step = data.get("step", 0)

    # Step 0 → always return the fixed first question
    if step == 0:
        return jsonify({
            "question": "အကြောင်းအရာကို အနည်းငယ်ရှင်းပြပါ",
            "step": 1,
            "done": False,
            "history": []
        })

    # If less than 10 → generate next question using ALL history
    if step < 10:
        combined = "\n".join([f"Q: {h['q']} → A: {h['a']}" for h in history])
        prompt = f"""
        The user has answered these so far:
        {combined}

        Based on everything so far, generate the NEXT most relevant cybersecurity diagnostic question in Burmese.
        Keep it short and clear. Only output the question.
        """
        next_q = ai_only(prompt)
        return jsonify({
            "question": next_q.strip(),
            "step": step + 1,
            "done": False,
            "history": history
        })

    # After 10 → final analysis
    combined = "\n".join([f"Q: {h['q']} → A: {h['a']}" for h in history])
    final_prompt = f"""
    The user answered 10 cybersecurity diagnostic questions:
    {combined}

    Summarize the main RISKS and possible SOLUTIONS for the user.
    Respond in Burmese, clearly and concisely and produce risks and solutions clearly.
    """
    result = ai_only(final_prompt, max_tokens=800)

    return jsonify({
        "done": True,
        "result": result,
        "history": history
    })

@app.route("/generate-quiz", methods=["POST"])
def generate_quiz():
    data = request.get_json()
    topic = data.get("topic")  # user selected topic
    if not topic:
        return jsonify({"error": "No topic provided"}), 400

    questions = generate_quiz_from_topic(topic)
    if not questions:
        return jsonify({"error": "Failed to generate quiz"}), 500

    return jsonify({"questions": questions})

@app.route("/random-tip")
def random_tip():
    return jsonify(generate_random_tip())

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
