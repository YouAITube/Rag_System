import os
import subprocess
from pathlib import Path
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import streamlit as st
from sentence_transformers import SentenceTransformer

# Assuming you have a Hugging Face model for text generation
# Replace "your_huggingface_model" with the actual model name
generation_model_name = "google/flan-t5-large" # Or another suitable model

generation_pipeline = pipeline("text2text-generation", model=generation_model_name)

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
    # Construct a prompt for the Hugging Face model
    prompt = f"""
    You are a plagiarism detection assistant. Your task is to determine whether the given code is plagiarized.

    Here is the user's code:
    {user_code}

    Below are similar code files found in the database:
    {similar_code_files}

    Based on the above, is the user's code plagiarized? Respond with only "Yes" or "No". If "Yes", include the references to the code files from the database as context.
    """

    # Generate response using the Hugging Face model
    response = generation_pipeline(prompt, max_length=150, temperature=0.1)[0]['generated_text']

    return response.strip()

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

            # Get Hugging Face model's response on plagiarism
            response = get_plagiarism_response(code_input, "\n".join(similar_code_files))

            # Display the response (whether the code is plagiarized or not)
            if "yes" in response.lower():
                st.write("The code appears to be plagiarized. Here are the references:")
                st.write("\n".join(similar_code_files))
            elif "no" in response.lower():
                st.write("The code does not appear to be plagiarized.")
            else:
                st.write(f"Response from model: {response}") #if the response is not yes or no.
        else:
            st.error("Please enter a code snippet.")

def main():
    # Streamlit UI
    streamlit_ui()

if __name__ == '__main__':
    main()
