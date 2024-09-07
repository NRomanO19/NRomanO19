from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient

app = Flask(__name__)

client = InferenceClient(
    "mattshumer/Reflection-Llama-3.1-70B",
    token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    response = ""

    for message in client.chat_completion(
        messages=[{"role": "user", "content": user_message}],
        max_tokens=500,
        stream=True,
    ):
        response += message.choices[0].delta.content

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
