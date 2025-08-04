import os
import asyncio
import time
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from whoosh import index, fields, qparser
import tempfile
import shutil

# === Ensure event loop exists ===
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# ==== Init Environment ====
os.environ['GOOGLE_API_KEY'] = "Your API KEY"

# === Streamlit Title ===
st.title("Ask me what you want ?")

# === File uploader for the PDF ===
uploaded_file = st.file_uploader("Upload your PDF here", type=["pdf"])

# ==== Utility: Save uploaded PDF to a temporary file ====
def save_uploaded_file(uploaded_file):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path, temp_dir

if uploaded_file:  # If a PDF was uploaded
    pdf_path, tmp_uploaded_dir = save_uploaded_file(uploaded_file)
else:
    pdf_path = None
    tmp_uploaded_dir = None

# ==== Step 1: Load & Split PDF (cached) ====
@st.cache_resource(show_spinner=False)
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # keep chunks reasonably large for better semantics
        chunk_overlap=300
    )
    return splitter.split_documents(pages)

# ==== Step 2: Build FAISS Vector Store (cached) ====
@st.cache_resource(show_spinner=False)
def build_vector_store(_docs):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.environ['GOOGLE_API_KEY']
    )
    vectors = FAISS.from_documents(_docs, embeddings)
    return vectors

# ==== Step 3: Build Whoosh Keyword Index (cached) ====
@st.cache_resource(show_spinner=False)
def build_whoosh_index(_docs):
    tmpdir = tempfile.mkdtemp()
    schema = fields.Schema(page=fields.NUMERIC(stored=True),
                           content=fields.TEXT(stored=True))
    idx = index.create_in(tmpdir, schema)
    writer = idx.writer()
    for doc in _docs:
        page_num = doc.metadata.get("page", 0)
        writer.add_document(page=page_num, content=doc.page_content)
    writer.commit()
    return idx, tmpdir

# ==== Only proceed if a PDF is available ====
if pdf_path:
    documents = load_and_split_pdf(pdf_path)
    vectors = build_vector_store(documents)
    retriever = vectors.as_retriever(search_kwargs={"k": 10})

    # Whoosh index for fallback keyword search
    whoosh_idx, whoosh_dir = build_whoosh_index(documents)
    parser = qparser.QueryParser("content", schema=whoosh_idx.schema)
else:
    documents = None
    retriever = None
    whoosh_idx = None
    parser = None
    whoosh_dir = None

# ==== LLM & Prompts ====
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    temperature=0,
    google_api_key=os.environ['GOOGLE_API_KEY']
)

# Sub-question decomposition prompt
decomp_prompt = ChatPromptTemplate.from_template(
    """  You are a helpful assistant that generates multiple sub-questions related to an input question.
    Break down the following question into 3 sub-questions/search queries:
    {question}
    Output only the queries (one per line)"""
)

# RAG retrieval prompt
rag_prompt = hub.pull("rlm/rag-prompt")

# Final synthesis prompt
synth_prompt = ChatPromptTemplate.from_template(
    "Given these Q&A pairs:\n{context}\nSynthesize an answer to: {question}"
)

# ==== Helper Functions ====


def decompose(question: str):
    raw = (decomp_prompt | llm | StrOutputParser()).invoke({"question": question})
    return [q.strip() for q in raw.splitlines() if q.strip()]

def keyword_fallback_search(query: str, top_n: int = 5):
    # Uses Whoosh for simple keyword search if vector retrieval fails
    with whoosh_idx.searcher() as searcher:
        parsed = parser.parse(query)
        hits = searcher.search(parsed, limit=top_n)
        return [{"page": hit["page"], "content": hit["content"]} for hit in hits]

def retrieve_context(query: str):
    # 1. Semantic retrieval first
    docs = retriever.get_relevant_documents(query)
    if any("tenure" in d.page_content.lower() for d in docs):
        return docs
    # 2. Fallback: Exact search using keywords
    hits = keyword_fallback_search(query)
    return [type("D", (), {"page_content": h["content"]}) for h in hits]

def answer_with_rag(question: str):
    subqs = decompose(question)
    qa_pairs = []
    for sq in subqs:
        docs = retrieve_context(sq)
        context = "\n\n".join(d.page_content for d in docs)
        ans = (rag_prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "question": sq
        })
        qa_pairs.append((sq, ans))
    # Synthesize final answer
    context_str = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in qa_pairs)
    final = (synth_prompt | llm | StrOutputParser()).invoke({
        "context": context_str,
        "question": question
    })
    return final, qa_pairs

# ==== Streamlit UI for Q&A ====
query = st.text_input("Ask a question about the PDF:")

if query:
    if documents is None:
        st.error("Please upload a PDF before asking questions.")
    else:
        start = time.time()
        answer, sub_answers = answer_with_rag(query)
        st.markdown("### Synthesized Answer")
        st.write(answer)
        st.caption(f"Generated in {time.time() - start:.2f}s")

        with st.expander("Show Sub-Questions & Answers"):
            for i, (sq, ans) in enumerate(sub_answers, 1):
                st.markdown(f"**Sub-question {i}:** {sq}")
                st.markdown(f"**Answer {i}:** {ans}")
                st.write("---")

# ==== Cleanup on Exit ====
import atexit
def cleanup():
    if whoosh_dir and os.path.exists(whoosh_dir):
        shutil.rmtree(whoosh_dir, ignore_errors=True)
    if tmp_uploaded_dir and os.path.exists(tmp_uploaded_dir):
        shutil.rmtree(tmp_uploaded_dir, ignore_errors=True)
atexit.register(cleanup)
