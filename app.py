import os
import faiss
import torch
import numpy as np
import streamlit as st
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModel

# Streamlit UI Header
st.title("🔍 Code Plagiarism Detection System")

# Keep the original get_code_embedding function unchanged
def get_code_embedding(code, model_name="microsoft/codebert-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# Function to search similar code in FAISS index
def search_similar_code(query_embedding, top_k=5):
    index_path = "code_embeddings.index"

    if not Path(index_path).exists():
        st.error("❌ FAISS index file not found! Please generate embeddings first.")
        return []

    try:
        index = faiss.read_index(index_path)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        distances, indices = index.search(query_embedding, top_k)
        return indices[0]
    except Exception as e:
        st.error(f"Error searching FAISS index: {str(e)}")
        return []

# Function to check for plagiarism using an LLM
def get_plagiarism_response(user_code, similar_code_files, model_name="tiiuae/falcon-7b-instruct"):
    prompt = f"""
    You are a plagiarism detection assistant. Analyze the provided code snippet and compare it with similar code files.

    **User Code:**
    ```
    {user_code}
    ```

    **Similar Code Files Found:**
    {similar_code_files}

    Is this code plagiarized? Respond with only "Yes" or "No". If "Yes", provide references to similar code files.
    """

    try:
        # Initialize Hugging Face pipeline for text generation
        pipe = pipeline("text-generation", model=model_name, device=0 if torch.cuda.is_available() else -1)
        
        # Generate the response
        response = pipe(prompt, max_length=100, num_return_sequences=1, temperature=0.1)[0]['generated_text']

        # Extract Yes/No answer from response
        if "Yes" in response:
            return "Yes" + response.split("Yes", 1)[1].strip()
        elif "No" in response:
            return "No"
        else:
            return "Could not determine plagiarism."

    except Exception as e:
        return f"❌ Error loading model: {str(e)}"

# Streamlit UI for user input
code_input = st.text_area("📝 Enter your code snippet:")

if st.button("🔍 Check for Plagiarism"):
    if code_input.strip():
        with st.spinner("Processing..."):
            # Generate embedding for input code
            query_embedding = get_code_embedding(code_input)

            # Search FAISS index for similar code
            similar_indices = search_similar_code(query_embedding)

            if similar_indices is not None and len(similar_indices) > 0:
                similar_code_files = [f"Code File {idx}" for idx in similar_indices]
                
                # Get plagiarism check response
                response = get_plagiarism_response(code_input, "\n".join(similar_code_files))

                # Display results
                if response.lower().startswith("yes"):
                    st.error("🚨 Plagiarism Detected! References:")
                    st.write(response.split("Yes", 1)[1].strip())
                elif response.lower() == "no":
                    st.success("✅ No plagiarism detected.")
                else:
                    st.warning(response)  # Display error or undetermined message
            else:
                st.warning("⚠️ No similar code found in the database.")
    else:
        st.error("⚠️ Please enter a code snippet.")

# FAISS index creation (for demo purposes)
if __name__ == "__main__":
    index_path = "code_embeddings.index"
    
    if not Path(index_path).exists():
        print("Creating a dummy FAISS index for demonstration...")
        d = 768  # Embedding size for CodeBERT
        index = faiss.IndexFlatL2(d)
        faiss.write_index(index, index_path)

        with open("code_files.txt", "w") as f:
            f.write("sample_code1.py\nsample_code2.py\nsample_code3.py\nsample_code4.py\nsample_code5.py\n")

    st.write("🚀 Ready for plagiarism detection!")
