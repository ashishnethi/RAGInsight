import streamlit as st
import os
import tempfile
import base64
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from typing import Dict, TypedDict

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="RAG Document Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        body { background-color: #0d1117; color: #e6edf3; }
        .stApp { background: linear-gradient(135deg, #0d1117, #161b22); color: #fff; }
        h1, h2, h3 { color: #58a6ff; }
        .stButton button { background: linear-gradient(90deg, #58a6ff, #1f6feb); color: white; font-size: 16px; border-radius: 10px; padding: 10px 20px; border: none; transition: 0.3s; }
        .stButton button:hover { background: linear-gradient(90deg, #1f6feb, #58a6ff); transform: scale(1.05); }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .answer-box { background: #161b22; border: 1px solid #30363d; padding: 20px; border-radius: 12px; font-size: 16px; line-height: 1.6; color: #e6edf3; margin-bottom: 1rem; }
        .download-link { background: #238636; padding: 10px 20px; border-radius: 8px; text-decoration: none; font-weight: bold; color: white; transition: 0.3s; }
        .download-link:hover { background: #2ea043; }
        .chunk-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; margin-bottom: 10px; }
        .expander-header { font-weight: bold; color: #58a6ff !important; }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=120)
    st.title("⚙️ Configuration")
    with st.expander("🔑 OpenRouter API Key"):
        st.session_state.openrouter_api_key = st.text_input(
            "Enter OpenRouter API Key",
            type="password",
            key="openrouter_api_key_input",
            value=st.session_state.get("openrouter_api_key", "")
        )
    with st.expander("🔑 Qdrant API Key"):
        st.session_state.qdrant_api_key = st.text_input(
            "Enter Qdrant API Key",
            type="password",
            key="qdrant_api_key_input",
            value=st.session_state.get("qdrant_api_key", "")
        )
    with st.expander("🔗 Qdrant URL"):
        st.session_state.qdrant_url = st.text_input(
            "Enter Qdrant URL",
            key="qdrant_url_input",
            value=st.session_state.get("qdrant_url", ""),
            help="e.g. http://localhost:6333"
        )
    st.caption("⚡ Secure session-based usage. Keys not stored.")

if not all([st.session_state.get("openrouter_api_key"), st.session_state.get("qdrant_api_key"), st.session_state.get("qdrant_url")]):
    st.sidebar.warning("⚠️ Please provide all API keys and Qdrant URL to continue.")
    st.stop()

openrouter_api_key = st.session_state.openrouter_api_key
qdrant_api_key = st.session_state.qdrant_api_key
qdrant_url = st.session_state.qdrant_url

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
VECTOR_SIZE = 384
client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
collection_name = "rag-qdrant"

try:
    client.delete_collection(collection_name)
except:
    pass

client.recreate_collection(
    collection_name=collection_name,
    vectors_config={"size": VECTOR_SIZE, "distance": "Cosine"}
)

retriever = None

def load_documents(file_or_url: str, is_url: bool = True):
    try:
        if is_url:
            loader = WebBaseLoader(file_or_url)
            loader.requests_per_second = 1
        else:
            ext = os.path.splitext(file_or_url)[1].lower()
            if ext == '.pdf':
                loader = PyPDFLoader(file_or_url)
            elif ext in ['.txt', '.md']:
                loader = TextLoader(file_or_url)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        return loader.load()
    except Exception as e:
        st.error(f"🚫 Error loading document: {str(e)}")
        return []

st.title("🤖 Intelligent Document Query Agent")
tab_upload, tab_url = st.tabs(["📂 Upload Document", "🌐 From URL"])
docs = None

with tab_upload:
    uploaded_file = st.file_uploader("Upload PDF, TXT, or MD", type=['pdf', 'txt', 'md'])
    if uploaded_file:
        st.success(f"✅ Uploaded: `{uploaded_file.name}`")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        docs = load_documents(tmp_file_path, is_url=False)
        os.unlink(tmp_file_path)

with tab_url:
    doc_url = st.text_input("Paste document URL here")
    if doc_url:
        docs = load_documents(doc_url, is_url=True)
        if docs:
            st.success("✅ Document loaded from URL")

if docs:
    with st.spinner("⚡ Splitting documents and populating vector DB..."):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)
        all_splits = text_splitter.split_documents(docs)
        vectorstore = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings
        )
        vectorstore.add_documents(all_splits)
        retriever = vectorstore.as_retriever()
    st.markdown(f"✅ Document split into **{len(all_splits)}** chunks and indexed.")

class GraphState(TypedDict):
    keys: Dict[str, any]

def retrieve(state):
    state_dict = state["keys"]
    question = state_dict["question"]
    if retriever is None:
        return {"keys": {"documents": [], "question": question}}
    docs = retriever.get_relevant_documents(question)
    return {"keys": {"documents": docs, "question": question}}

def grade_documents(state):
    state_dict = state["keys"]
    docs = state_dict["documents"]
    graded_docs = docs[:3]
    return {"keys": {"documents": graded_docs, "question": state_dict["question"]}}

def transform_query(state):
    state_dict = state["keys"]
    question = state_dict["question"].strip()
    return {"keys": {"documents": state_dict["documents"], "question": question}}

def generate(state):
    from langchain_openai import ChatOpenAI
    state_dict = state["keys"]
    question, documents = state_dict["question"], state_dict["documents"]
    try:
        prompt = PromptTemplate(
            template="""Answer the question concisely using ONLY the context.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:""",
            input_variables=["context", "question"]
        )
        llm = ChatOpenAI(
            model="nousresearch/hermes-4-405b",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=openrouter_api_key,
            temperature=0,
            max_tokens=700
        )
        context = "\n\n".join(f"Chunk {i+1}: {doc.page_content}" for i, doc in enumerate(documents))
        rag_chain = (
            {"context": lambda x: context, "question": lambda x: question}
            | prompt
            | llm
            | StrOutputParser()
        )
        generation = rag_chain.invoke({})
        return {"keys": {"documents": documents, "question": question, "generation": generation}}
    except Exception as e:
        st.error(f"🚫 Error during generation: {str(e)}")
        return {"keys": {"documents": documents, "question": question, "generation": "Error during generation."}}

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents", "transform_query")
workflow.add_edge("transform_query", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()

st.markdown("### ❓ Ask a question about your document")
user_question = st.text_area("Type your question here", height=120)

if st.button("🚀 Submit Query", use_container_width=True):
    if not docs:
        st.warning("⚠️ Upload or load a document first.")
    elif not user_question.strip():
        st.warning("⚠️ Enter a question.")
    else:
        with st.spinner("⚡ Processing your question..."):
            inputs = {"keys": {"question": user_question}}
            for step_outputs in app.stream(inputs):
                for step_name, value in step_outputs.items():
                    step_labels = {
                        "retrieve": "🔍 Retrieval - Relevant Chunks",
                        "grade_documents": "📊 Grading - Top Evidence Selected",
                        "transform_query": "🛠️ Query Refinement",
                        "generate": "🤖 Generated Answer"
                    }
                    step_label = step_labels.get(step_name, step_name)

                    with st.expander(step_label, expanded=True):
                        if step_name == "retrieve":
                            docs_ = value['keys']['documents']
                            if not docs_:
                                st.info("No relevant chunks found.")
                            else:
                                for i, doc in enumerate(docs_, 1):
                                    st.markdown(f'<div class="chunk-card"><b>Chunk {i} Preview:</b><br>{doc.page_content[:400]}{"..." if len(doc.page_content) > 400 else ""}</div>', unsafe_allow_html=True)
                        elif step_name == "grade_documents":
                            st.info("Selected top 3 relevant chunks.")
                        elif step_name == "transform_query":
                            st.write(f"Refined Question: `{value['keys']['question']}`")
                        elif step_name == "generate":
                            answer_text = value['keys']['generation']
                            st.markdown(f'<div class="answer-box">{answer_text}</div>', unsafe_allow_html=True)
                            b64 = base64.b64encode(answer_text.encode()).decode()
                            href = f'<a href="data:file/txt;base64,{b64}" download="rag_answer.txt" class="download-link">⬇ Download Answer</a>'
                            st.markdown(href, unsafe_allow_html=True)

st.markdown("---")
st.caption("✨ Built with Streamlit, LangChain, Qdrant, and OpenRouter | Chatbot by Ashish ✨")
