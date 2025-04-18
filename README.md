![image](https://github.com/user-attachments/assets/dd541625-cba8-4889-8e1f-ad7b9eb76715)



Code Plagiarism Detection App

# Overview

This project is a Code Plagiarism Detection System built using Streamlit, FAISS, and Hugging Face Transformers. The system allows users to input a code snippet and checks for similar code files stored in a vector database using FAISS. It then queries an LLM  to determine if the submitted code is plagiarized.

# Features:

Clone multiple Git repositories and extract code files.

Generate vector embeddings for code files using microsoft/codebert-base.

Store embeddings in FAISS for efficient similarity search.

Compare new code snippets against stored embeddings.

Use GPT-based analysis to determine plagiarism.

Streamlit UI for easy user interaction.


### How to Use  

1. Clone this repository.  
2. Install dependencies from `requirements.txt`.  
3. Run `app_store.py` to process repositories.  
4. Start the Streamlit UI with:  

   ```bash
   streamlit run app_main.py

```
code-plagiarism-app/
│── app_main.py       # Streamlit UI for plagiarism detection
│── app_store.py      # Clones repositories, extracts code, generates embeddings
│── requirements.txt  # Dependencies
│── Dockerfile        # Docker configuration
│── README.md         # Documentation
```


