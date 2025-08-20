from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from rag import ask_bot, ask_bot_content_checker
import json
import random

app = Flask(__name__)
CORS(app)  # allow frontend to call API

# Load daily tips
with open("tips.json", "r", encoding="utf-8") as f:
    tips = json.load(f)

@app.route("/")
def home():
    return render_template("index.html")  # frontend file

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    bot_response = ask_bot(user_input)
    return jsonify({"reply": bot_response})

@app.route("/content-check", methods=["POST"])
def content_check():
    data = request.json
    content = data.get("content", "")
    poster = data.get("poster", "")
    date = data.get("date", "")
    platform = data.get("platform", "")

    # call your rag function and pass metadata
    result = ask_bot_content_checker(content, poster, date, platform)
    return jsonify({"result": result})


@app.route("/daily-tip", methods=["GET"])
def daily_tip():
    tip = random.choice(tips)
    return jsonify({"tip": tip})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
