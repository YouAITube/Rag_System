import os
import subprocess
from pathlib import Path
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import streamlit as st
from sentence_transformers import SentenceTransformer

# openai.api_key = os.getenv("OPENAI_API_KEY") # No longer needed

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

def get_plagiarism_response(user_code, similar_code_files, model_name="mistralai/Mistral-7B-Instruct-v0.2"): # Choose a suitable Hugging Face model
    # Construct a prompt for the LLM
    prompt = f"""
    You are a plagiarism detection assistant. Your task is to determine whether the given code is plagiarized.

    Here is the user's code:
    {user_code}

    Below are similar code files found in the database:
    {similar_code_files}

    Based on the above, is the user's code plagiarized? Respond with only "Yes" or "No". If "Yes", include the references to the code files from the database as context.
    """

    # Initialize the Hugging Face pipeline
    pipe = pipeline("text-generation", model=model_name, device=0 if torch.cuda.is_available() else -1) # Use GPU if available

    # Generate the response
    response = pipe(
        prompt,
        max_length=100, # Adjust as needed
        num_return_sequences=1,
        temperature=0.1, # Adjust for more deterministic output
        eos_token_id=pipe.tokenizer.eos_token_id,
    )[0]['generated_text']

    # Extract the relevant part of the response
    # This might need adjustment based on the specific model's output format
    if "Yes" in response:
        return "Yes" + response.split("Yes", 1)[1].strip()
    elif "No" in response:
        return "No"
    else:
        return "Could not determine plagiarism." # Handle cases where the model doesn't give a clear "Yes" or "No"

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
            similar_code_files = [f"Code File: {idx}" for idx in similar_indices] # Using index as placeholder, replace with actual filenames

            # Get Hugging Face model's response on plagiarism
            response = get_plagiarism_response(code_input, "\n".join(similar_code_files))

            # Display the response (whether the code is plagiarized or not)
            if response.lower().startswith("yes"):
                st.write("The code appears to be plagiarized. Here are the references:")
                st.write(response.split("Yes", 1)[1].strip())
            elif response.lower() == "no":
                st.write("The code does not appear to be plagiarized.")
            else:
                st.write(response) # Display the "Could not determine" message
        else:
            st.error("Please enter a code snippet.")

def main():
    # Streamlit UI
    streamlit_ui()

if __name__ == '__main__':
    # Create a dummy index file for demonstration if it doesn't exist
    if not Path("code_embeddings.index").exists():
        print("Creating a dummy code_embeddings.index for demonstration...")
        d = 768  # Assuming CodeBERT embedding dimension
        index = faiss.IndexFlatL2(d)
        faiss.write_index(index, "code_embeddings.index")
        with open("code_files.txt", "w") as f:
            f.write("sample_code1.py\n")
            f.write("sample_code2.py\n")
            f.write("sample_code3.py\n")
            f.write("sample_code4.py\n")
            f.write("sample_code5.py\n")

    main()

