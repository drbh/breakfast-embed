# This is a functional example of a locally running LLM server. We do not provide the 
# checkpoint for the model, but you can use your own checkpoint or any other generative 
# model that you want to use.

from transformers import pipeline
from flask import request, jsonify, Flask

checkpoint = "./LaMini-Flan-T5-783M"

model = pipeline("text2text-generation", model=checkpoint)

app = Flask(__name__)


def get_prompt(question, context):
    return f"""
Please answer the following question based on the given text:

[QUESTION]
{question}

[FACTS]
{context}
"""


@app.route("/api", methods=["POST"])
def api():
    data = request.get_json()
    input_prompt = data["input_prompt"]
    context = data["context"]
    full_prompt = get_prompt(input_prompt, context)
    out = model(full_prompt, max_length=512, do_sample=True)
    generated_text = out[0]["generated_text"]
    return jsonify({"response": generated_text})


if __name__ == "__main__":
    print("Starting Flask Server")
    app.run(debug=False, port=5000)
