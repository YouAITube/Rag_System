
import os
import subprocess
from pathlib import Path
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import streamlit as st
from sentence_transformers import SentenceTransformer

def clone_repositories(repo_urls, base_dest_path):
    """
    Clones multiple Git repositories to the specified destination directory.

    Args:
        repo_urls (list): List of Git repository URLs.
        base_dest_path (str): Base directory where repositories will be cloned.
    """
    os.makedirs(base_dest_path, exist_ok=True)
    cloned_paths = []
    for repo_url in repo_urls:
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        dest_path = os.path.join(base_dest_path, repo_name)
        if not os.path.exists(dest_path):
            subprocess.run(["git", "clone", repo_url, dest_path], check=True)
        cloned_paths.append(dest_path)
    return cloned_paths

def load_repository(repo_path):
    """
    Scans the given repository path and collects paths of code files.

    Args:
        repo_path (str): Path to the repository.

    Returns:
        List[str]: List of code file paths.
    """
    code_extensions = {'.py', '.java', '.c', '.sh', '.cpp', '.js', '.ts'}
    code_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if Path(file).suffix in code_extensions:
                code_files.append(os.path.join(root, file))
    return code_files


def load_multiple_repositories(repo_paths):
    """
    Loads multiple repositories and extracts code file paths.

    Args:
        repo_paths (list): List of repository paths.

    Returns:
        List[str]: List of all code file paths from all repositories.
    """
    all_code_files = []
    for repo_path in repo_paths:
        all_code_files.extend(load_repository(repo_path))
    return all_code_files


def get_code_embedding(file_path, model_name="microsoft/codebert-base"):
    """
    Generates an embedding for a given code file using a model from Hugging Face.

    Args:
        file_path (str): Path to the code file.
        model_name (str): Name of the Hugging Face model.

    Returns:
        np.array: Vector representation of the code.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()

    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

def store_embeddings(embeddings):
    """
    Stores the embeddings in a FAISS vector database.

    Args:
        embeddings (dict): Dictionary where keys are file paths and values are embeddings.
    """
    embedding_dim = len(next(iter(embeddings.values())))
    index = faiss.IndexFlatL2(embedding_dim)

    vectors = np.array(list(embeddings.values()), dtype=np.float32)
    index.add(vectors)

    faiss.write_index(index, "code_embeddings.index")


def main():
    repo_urls = ["https://github.com/YouAITube/Streamlit_App.git", "https://github.com/ArmaanSeth/ChatPDF.git"]
    base_dest_path = "./repositories"
    repo_paths = clone_repositories(repo_urls, base_dest_path)
    code_files = load_multiple_repositories(repo_paths)
    embeddings = {file: get_code_embedding(file) for file in code_files}
    store_embeddings(embeddings)



if __name__ == '__main__':
    main()
