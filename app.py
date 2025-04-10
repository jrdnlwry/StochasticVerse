import streamlit as st
import random
import requests
import numpy as np
import pandas as pd
import re
import copy
from collections import Counter
# from langchain_community.llms import Ollama
# from langchain import PromptTemplate, LLMChain

# --- Utility Functions ---
def clean_normalize(text):
    text_copy = copy.deepcopy(text)
    text_sub = re.sub(r"[^a-zA-Z0-9\s]", "", text_copy)
    text_clean = text_sub.lower().strip()
    return text_clean

def vocab_mapping(text):
    unique_words = set(text.split())
    token_reference = {idx: word for idx, word in enumerate(unique_words)}
    bigrams = list(zip(text.split()[:-1], text.split()[1:]))
    return bigrams, token_reference

def agg_dataframe(bigrams):
    df = pd.DataFrame(bigrams, columns=["prev", "next"])
    grouped = df.groupby(["prev", "next"]).size().reset_index(name='count')
    return grouped

def row_summation(df):
    rowSum = df.groupby("prev")["count"].sum().reset_index()
    merged = pd.merge(rowSum, df, on="prev")
    merged = merged.rename(columns={"count_x": "row_sum", "count_y": "bigram_freq"})
    return merged

def normalization(df):
    df["prob"] = df["bigram_freq"] / df["row_sum"]
    return df

def generate_start_token(token_reference):
    idx = random.randint(0, len(token_reference) - 1)
    return token_reference[idx]

def sentence_generator(start_token, norm_df):
    generated_text = start_token
    current_word = start_token
    for _ in range(10):
        options = norm_df[norm_df["prev"] == current_word]
        if options.empty:
            break
        next_word = np.random.choice(options["next"], p=options["prob"])
        generated_text += " " + next_word
        current_word = next_word
    return generated_text

def clean_poem_with_api(poem: str) -> str:
    #api_url = st.secrets.get("OLLAMA_API_URL", "http://localhost:5000/clean")
    #api_key = st.secrets.get("OLLAMA_API_KEY", "supersecret")
    api_url = st.secrets["OLLAMA_API_URL"]
    api_key = st.secrets["OLLAMA_API_KEY"]

    try:
        response = requests.post(
            api_url,
            json={"poem": poem},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=45
        )
        if response.status_code == 200:
            return response.json()["cleaned_poem"]
        else:
            return f"Error: {response.json().get('error', 'Unknown error')}"
    except Exception as e:
        return f"Connection error: {str(e)}"





# --- Streamlit UI ---
st.title("🎙️ StochasticVerse Poetry Generator")

# Session state setup
if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""
if "reset_flag" not in st.session_state:
    st.session_state["reset_flag"] = False

# Two buttons side by side
col1, col2 = st.columns([1, 1])

with col1:
    generate_clicked = st.button("Generate Poetry")

with col2:
    reset_clicked = st.button("Reset Text")

# Handle reset logic
if reset_clicked:
    st.session_state["text_input"] = ""
    st.session_state["reset_flag"] = True

# Text area input (conditional reset)
user_input = st.text_area(
    "Paste raw text below to train your generator:",
    height=300,
    value="" if st.session_state["reset_flag"] else st.session_state["text_input"]
)

# Clear the reset flag and store input
st.session_state["reset_flag"] = False
st.session_state["text_input"] = user_input

# Generation logic
if generate_clicked:
    if not user_input.strip():
        st.warning("Please paste some text before generating.")
    else:
        clean_text = clean_normalize(user_input)
        biGram_df, token_reference = vocab_mapping(clean_text)
        agg_df = agg_dataframe(biGram_df)
        summed_df = row_summation(agg_df)
        norm_df = normalization(summed_df)

        paragraph = ""
        for _ in range(4):
            start = generate_start_token(token_reference)
            line = sentence_generator(start, norm_df)
            paragraph += line + "\n"

        st.subheader("📝 Raw Generated Text")
        st.text(paragraph)

        st.subheader("🤖 Cleaned Final Poem")
        with st.spinner("Polishing with LLM..."):
            # llm = Ollama(model="mistral")
            # template = """
            #     You are an editor for a poetry publishing house.

            #     Your task is to clean up the following poem by correcting grammar, adding punctuation, and improving coherence — but without altering the structure or line breaks.

            #     Return **only** the revised version of the poem. Do not include any commentary, introductions, explanations, or formatting outside the poem itself.

            #     Poem:
            #     {poem}
            # """
            # prompt = PromptTemplate.from_template(template)
            # chain = LLMChain(llm=llm, prompt=prompt)
            # response = chain.run(poem=paragraph)

            # Replace the old chain.run() call with this:
            response = clean_poem_with_api(paragraph)

            st.success("Done!")
            st.text_area("Final Output:", response.strip(), height=300)
