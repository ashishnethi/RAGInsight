# 🤖 Intelligent Document Query Agent

A Retrieval-Augmented Generation (RAG) agent that lets you ask intelligent questions about documents with **full pipeline transparency**.
Built using **Streamlit, LangChain, and Qdrant**, this project is designed to be **completely free to use**, leveraging free APIs and local resources so anyone can learn and test RAG without cost.



## 🔹 Features

* **Free to Run**: Uses HuggingFace embeddings and Qdrant (local or free tier). No paid services required.
* **Flexible Document Input**: Upload PDF, TXT, or Markdown files, or provide a document URL.
* **Smart Chunking & Indexing**: Automatically splits documents into smaller chunks and stores them in Qdrant for fast retrieval.
* **Transparent RAG Pipeline**:

  * 🔍 **Retrieve**: Finds the most relevant chunks.
  * 📊 **Grade**: Selects the top 3 chunks.
  * 🛠️ **Refine**: Improves the query for better results.
  * 🤖 **Generate**: Produces clear, context-based answers.
* **Modern UI / UX**: Gradient backgrounds, styled chunk cards, collapsible pipeline steps, and clean typography.
* **Downloadable Answers**: Save responses as `.txt` for offline reference.
* **API Flexibility**: Works with OpenRouter (optional) or free local embeddings.



## 🔹 How It Works

1. **Document Loading**

   * Upload a file or provide a URL.
   * Content is automatically extracted.

2. **Document Splitting**

   * Content is chunked using **Recursive Character Text Splitter**.
   * Chunks are stored in **Qdrant** for similarity search.

3. **Query Processing**

   * Enter a question about the document.
   * Pipeline retrieves, grades, refines, and generates an answer.

4. **Answer Display**

   * Pipeline steps are shown in expandable sections.
   * Final answer is presented in a styled box with download option.



## 🔹 User Experience

* **Interactive Sidebar**: Add API keys (optional) for OpenRouter and Qdrant.
* **Tabbed Input**: Choose between **Upload Document** or **Document URL**.
* **Pipeline Transparency**: Every step is visible to the user.
* **Clean Outputs**: Chunk previews, refined queries, and generated answers with a modern look.

---

## 🔹 Benefits

* **Zero-Cost Learning**: Built to run free with open resources.
* **Efficient Retrieval**: Quickly extract knowledge from long documents.
* **Full Transparency**: See exactly how answers are generated.
* **Scalable & Flexible**: Switch between local and remote APIs as needed.
* **User-Friendly**: Designed for clarity and ease of use.


## 🔹 Applications

* **Research**: Summarize or query academic papers and reports.
* **Business**: Extract insights from company documents.
* **Education**: Learn from study material with quick Q&A.
* **Legal / Compliance**: Retrieve critical details from lengthy policies or contracts.


## 🌟 Note

This project emphasizes **free usage** by default. All core components (HuggingFace embeddings, Qdrant, Streamlit) are open-source or free-tier.
Optional paid APIs (e.g., OpenRouter) can be plugged in, but they are not required.

<img width="1915" height="847" alt="image" src="https://github.com/user-attachments/assets/2bbafe9d-9d38-4e82-a1af-9a92dc35536e" />

<img width="1522" height="652" alt="image" src="https://github.com/user-attachments/assets/2a0196bb-2522-424b-8246-52111b8f0abf" />

