Code Plagiarism Detection App

Overview

This project is a Code Plagiarism Detection System built using Streamlit, FAISS, and Hugging Face Transformers. The system allows users to input a code snippet and checks for similar code files stored in a vector database using FAISS. It then queries an LLM (GPT-4) to determine if the submitted code is plagiarized.

Features:
Clone multiple Git repositories and extract code files.

Generate vector embeddings for code files using microsoft/codebert-base.

Store embeddings in FAISS for efficient similarity search.

Compare new code snippets against stored embeddings.

Use GPT-based analysis to determine plagiarism.

Streamlit UI for easy user interaction.

\documentclass{article}
\usepackage{xcolor}
\usepackage{listings}

\begin{document}

\section*{\textbf{Running the Application}}

\subsection*{\textbf{Run with Python}}

\subsubsection*{\textbf{Step 1: Process Repositories}}
\noindent
\begin{lstlisting}[language=bash, backgroundcolor=\color{black}, basicstyle=\ttfamily\color{white}]
python app_store.py
\end{lstlisting}

\subsubsection*{\textbf{Step 2: Start the Streamlit App}}
\noindent
\begin{lstlisting}[language=bash, backgroundcolor=\color{black}, basicstyle=\ttfamily\color{white}]
streamlit run app_main.py
\end{lstlisting}

\end{document}

