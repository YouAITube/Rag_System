

import os
import subprocess
from pathlib import Path
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import streamlit as st
from sentence_transformers import SentenceTransformer
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_code_embedding(code, model_name="microsoft/codebert-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding


def search_similar_code(query_embedding, top_k=5):
    index = faiss.read_index("code_embeddings.index")
    query_embedding = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query_embedding, top_k)
    return indices[0]

def get_plagiarism_response(user_code, similar_code_files):
    # Construct a prompt for the LLM
    prompt = f"""
    You are a plagiarism detection assistant. Your task is to determine whether the given code is plagiarized.

    Here is the user's code:
    {user_code}

    Below are similar code files found in the database:
    {similar_code_files}

    Based on the above, is the user's code plagiarized? Respond with only "Yes" or "No". If "Yes", include the references to the code files from the database as context.
    """

    client = openai.OpenAI()

    response = client.completions.create(
    model="gpt-4",  # Use "gpt-4" or another available model
    messages=[{"role": "system", "content": "You are a plagiarism detection assistant."},
              {"role": "user", "content": prompt}],
    max_tokens=50,
    temperature=0
)
    return response.choices[0].text.strip()

# Streamlit UI
def streamlit_ui():
    st.title("Code Plagiarism Detection")

    code_input = st.text_area("Enter your code snippet:")

    if st.button("Check for Plagiarism"):
        if code_input:
            # Get the embedding for the user's code snippet
            query_embedding = get_code_embedding(code_input)

            # Search for similar code in the FAISS index
            similar_indices = search_similar_code(query_embedding)

            # Retrieve the paths of the similar code files
            similar_code_files = [f"Code File: {code_file}" for code_file in similar_indices]

            # Get LLM's response on plagiarism
            response = get_plagiarism_response(code_input, "\n".join(similar_code_files))

            # Display the response (whether the code is plagiarized or not)
            if response.lower() == "yes":
                st.write("The code appears to be plagiarized. Here are the references:")
                st.write("\n".join(similar_code_files))
            else:
                st.write("The code does not appear to be plagiarized.")
        else:
            st.error("Please enter a code snippet.")

def main():
    # Streamlit UI
    streamlit_ui()

if __name__ == '__main__':
    main()
