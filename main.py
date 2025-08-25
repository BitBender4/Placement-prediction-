from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)

# Replace this with your actual OpenAI API key
OPENAI_API_KEY = "your-api-key"

@app.route("/")
def index():
    return render_template("chatbot.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    try:
        data = request.get_json()
        user_input = data.get("msg", "").strip()

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gpt-4o-mini",   # ✅ OpenAI model name
            "messages": [{"role": "user", "content": user_input}]
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",  # ✅ OpenAI endpoint
            headers=headers,
            json=payload
        )

        if response.status_code == 401:
            return jsonify({"response": "❌ Unauthorized: Please check your OpenAI API key."})

        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]

        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
