import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from typing import Dict, TypedDict

### Streamlit sidebar config - API keys
with st.sidebar:
    st.subheader("API Configuration")
    openrouter_api_key = st.text_input("OpenRouter API Key", type="password")
    qdrant_api_key = st.text_input("Qdrant API Key", type="password")
    qdrant_url = st.text_input("Qdrant URL")
    # Remove default doc URL
    doc_url = st.text_input("Document URL (optional)")

    if not all([openrouter_api_key, qdrant_api_key, qdrant_url]):
        st.warning("Please provide the required API keys and Qdrant URL")
        st.stop()

### Embeddings - HuggingFace, no key needed
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

### Qdrant setup - vector size = 384
client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key
)

collection_name = "rag-qdrant"
# Recreate collection
try:
    client.delete_collection(collection_name)
except Exception:
    pass

client.recreate_collection(
    collection_name=collection_name,
    vectors_config={"size": 384, "distance": "Cosine"}
)

retriever = None

def load_documents(file_or_url: str, is_url: bool = True) -> list:
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
        st.error(f"Error loading document: {str(e)}")
        return []

### Document input
st.subheader("Document Input")
input_option = st.radio("Choose input method:", ["URL", "File Upload"])
docs = None  
if input_option == "URL":
    url = st.text_input("Enter document URL:", value=doc_url)
    if url:
        docs = load_documents(url, is_url=True)
else:
    uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'txt', 'md'])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            docs = load_documents(tmp_file.name, is_url=False)
        os.unlink(tmp_file.name)

### Chunk + Add to Qdrant
if docs:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    all_splits = text_splitter.split_documents(docs)
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    vectorstore.add_documents(all_splits)
    retriever = vectorstore.as_retriever()

### State & Pipeline Logic (LangGraph)
class GraphState(TypedDict):
    keys: Dict[str, any]

def retrieve(state):
    state_dict = state["keys"]
    question = state_dict["question"]
    if retriever is None:
        return {"keys": {"documents": [], "question": question}}
    documents = retriever.get_relevant_documents(question)
    return {"keys": {"documents": documents, "question": question}}

def grade_documents(state):
    """Simple relevance grading: keep top 3 most relevant chunks"""
    state_dict = state["keys"]
    docs = state_dict["documents"]
    graded_docs = docs[:3]  # For demo, top 3
    return {"keys": {"documents": graded_docs, "question": state_dict["question"]}}

def transform_query(state):
    """Optional query enhancement"""
    state_dict = state["keys"]
    question = state_dict["question"]
    # For demo, just strip whitespace
    enhanced_question = question.strip()
    return {"keys": {"documents": state_dict["documents"], "question": enhanced_question}}

def generate(state):
    from langchain_openai import ChatOpenAI

    state_dict = state["keys"]
    question, documents = state_dict["question"], state_dict["documents"]
    try:
        prompt = PromptTemplate(
            template="""Based on the following context, answer concisely.
            Context: {context}
            Question: {question}
            Answer:""", 
            input_variables=["context", "question"]
        )
        llm = ChatOpenAI(
            model="nousresearch/hermes-4-405b",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=openrouter_api_key,
            temperature=0,
            max_tokens=1000
        )
        context = "\n\n".join(doc.page_content for doc in documents)
        rag_chain = (
            {"context": lambda x: context, "question": lambda x: question}
            | prompt
            | llm
            | StrOutputParser()
        )
        generation = rag_chain.invoke({})
        return {
            "keys": {
                "documents": documents,
                "question": question,
                "generation": generation
            }
        }
    except Exception as e:
        st.error(f"Error in generate function: {str(e)}")
        return {"keys": {"documents": documents, "question": question, "generation": "Error during generation."}}

### Minimal graph: Retrieve → Grade → Transform → Generate
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

### Streamlit UI
st.title("🔄 Corrective RAG Agent (Phase 1 - Free APIs)")
st.text("Query example: What are the experiment results and ablation studies in this research paper?")

user_question = st.text_input("Enter your question:")

if user_question and docs:
    inputs = {"keys": {"question": user_question}}
    for output in app.stream(inputs):
        for step_name, value in output.items():
            with st.expander(f"🔹 Step: {step_name}", expanded=True):
                if step_name == "retrieve":
                    st.write(f"📂 Retrieved {len(value['keys']['documents'])} document chunks.")
                    for i, doc in enumerate(value['keys']['documents'], 1):
                        st.text(f"Chunk {i} preview:\n{doc.page_content[:300]}...")
                elif step_name == "grade_documents":
                    st.write("📝 Document grading done. Relevant docs selected.")
                elif step_name == "transform_query":
                    st.write(f"✏️ Transformed query: {value['keys']['question']}")
                elif step_name == "generate":
                    st.write("✅ Generated Answer Preview")
                    st.text(value['keys']['generation'][:300] + "...")
                    with st.expander("🎯 Final Answer", expanded=True):
                        st.write(value['keys']['generation'])
elif user_question:
    st.warning("Document loading required first!")
