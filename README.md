🤖 Intelligent Document Query Agent

An advanced Retrieval-Augmented Generation (RAG) agent that allows you to ask questions about documents intelligently, providing answers with full transparency on how they were generated. Built using Streamlit, LangChain, and Qdrant, this agent combines modern AI with an intuitive, user-friendly interface.

🔹 Features

Document Input Flexibility: Upload PDF, TXT, or Markdown files, or provide a document URL.

Smart Chunking & Indexing: Automatically splits documents into manageable chunks and indexes them in Qdrant for fast retrieval.

Transparent RAG Pipeline:

🔍 Retrieve: Finds relevant document chunks.

📊 Grade: Selects the top 3 most relevant chunks.

🛠️ Query Refinement: Improves the question for better results.

🤖 Answer Generation: Produces a concise, context-based answer using LLMs.

Modern UI / UX: Gradient backgrounds, styled chunk cards, collapsible pipeline steps, and clean typography for clarity.

Downloadable Answers: Save responses as a .txt file for offline reference.

API Integration: Works with OpenRouter for LLMs and HuggingFace embeddings or remote embeddings, making it flexible for low-resource environments.

🔹 How It Works

Document Loading:

User uploads a document or provides a URL.

The system automatically loads and reads the content.

Document Splitting:

Documents are split into small chunks using Recursive Character Text Splitter.

Each chunk is indexed in Qdrant, allowing efficient similarity searches.

Query Processing:

User enters a question about the document.

The RAG pipeline retrieves relevant chunks, grades them, refines the query, and generates a contextual answer.

Answer Display:

Full pipeline steps are displayed in expandable sections, showing transparency at every stage.

The answer is displayed in a styled box with an option to download.

🔹 User Experience

Interactive Sidebar: Enter API keys for OpenRouter and Qdrant.

Tabs for Input: Switch between Upload Document and Document URL seamlessly.

Pipeline Transparency: Each stage of the RAG process is visible for the user.

Styled Outputs: Chunk previews, refined queries, and generated answers are presented clearly with modern UI components.

🔹 Benefits

Efficient Knowledge Retrieval: Quickly find information from long documents.

Transparency: Users can see exactly which parts of the document were used to generate answers.

Flexible & Scalable: Works with local embeddings or remote APIs; can handle multiple document types.

User-Friendly: Clean, visually appealing interface designed for effortless interaction.

🔹 Applications

Research Assistance: Ask questions about papers, reports, or manuals.

Business Intelligence: Quickly extract insights from long reports or documents.

Education: Study documents efficiently and get concise answers.

Legal / Compliance: Search and retrieve key information from legal or policy documents.
<img width="1915" height="847" alt="image" src="https://github.com/user-attachments/assets/2bbafe9d-9d38-4e82-a1af-9a92dc35536e" />

<img width="1522" height="652" alt="image" src="https://github.com/user-attachments/assets/2a0196bb-2522-424b-8246-52111b8f0abf" />

