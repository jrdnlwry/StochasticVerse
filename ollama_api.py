from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain import PromptTemplate, LLMChain

app = Flask(__name__)

# Set up Ollama + template
llm = Ollama(model="mistral")
template = """
You are an editor for a poetry publishing house.

Your task is to clean up the following poem by correcting grammar, adding punctuation, and improving coherence — but without altering the structure or line breaks.

Return **only** the revised version of the poem. Do not include any commentary, introductions, explanations, or formatting outside the poem itself.

Poem:
{poem}
"""
prompt = PromptTemplate.from_template(template)
chain = LLMChain(llm=llm, prompt=prompt)

@app.route('/clean', methods=['POST'])
def clean_poem():
    data = request.json
    poem = data.get("poem")
    if not poem:
        return jsonify({"error": "Missing poem"}), 400
    try:
        result = chain.run(poem=poem)
        return jsonify({"cleaned_poem": result.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
