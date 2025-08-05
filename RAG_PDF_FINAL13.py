import os
import os
import asyncio
import threading

# === Ensure event loop exists ===
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import loader

import streamlit as st
import time

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader



# ==== Init Environment ====
os.environ['GOOGLE_API_KEY'] = "YOUR_GEMINI_API_KEY"
### if want to user upload any pdf
# ==== App UI ====
st.title("📄 Policy Analyzer ")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# choose LLM models
llm= ChatGoogleGenerativeAI(model="models/gemini-2.0-flash",temperature=0,google_api_key=os.environ['GOOGLE_API_KEY'])

# # app title
# st.title("Policy Analyzer ")

# choose the embedding model
if uploaded_file is not None:
    if "uploaded_file" not in st.session_state or st.session_state.uploaded_filename != uploaded_file.name:
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.environ['GOOGLE_API_KEY']
        )
        # Save uploaded file temporarily
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())
    # load pdf
    st.session_state.loader = PyPDFLoader("temp_uploaded.pdf")  # Load the PDF
    pages = st.session_state.loader.load_and_split()
    # Automatically splits it into pages
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    final_documents = splitter.split_documents(pages[:50])
    st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
    # ──────────────── Add retriever here ────────────────
    # Use k=10 to pull back up to 10 chunks per query
    st.session_state.retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 10})
    # GET retrieve from FAISS
    retriever = st.session_state.retriever

# # GET retrieve from FAISS
# retriever  = st.session_state.retriever.get_relevant_documents(user_question)

# vector store(FAISS)
# st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)


# user question
user_question = st.text_input("Ask me a question?")

# user query translation template
user_template="""
You are a helpful assistant that generates multiple sub-questions related to an input question.
Break down the following question into 3 sub-questions/search queries:
{question}
Output only the queries (one per line)
"""
query_translation  = ChatPromptTemplate.from_template(user_template)
user_chain=(
    query_translation
    | llm
    | StrOutputParser()
    | (lambda x:[q.strip() for q in x.split("\n") if q.strip()])
)

# RAG prompt
rag_prompt ="""You are a helpful and highly intelligent assistant that analyzes a user’s question in-depth and provides an accurate, complete, and well-structured answer using only the content from the provided PDF document.

Your job is to:

- Carefully read the context extracted from the PDF (provided as "context").
- Understand the user's question fully and think critically.
- Answer the question precisely using only the relevant information from the context.
- Do not guess. If the answer is not present in the context, respond with “The answer is not available in the document.”
- If the context contains related ideas, synthesize them into a coherent and expert-level explanation.
- Resolve any ambiguity in the question by interpreting it in a clear and logical way.

Speak like a professional domain expert. Avoid fluff. Be direct, confident, and helpful. The answer should leave no doubts in the user's mind.

<context>
{context}
</context>

<question>
{question}
</question>

Provide your answer below:
"""
rag_prompt_template = ChatPromptTemplate.from_template(rag_prompt)
# Retrieve and answer all sub questions
def retrieve_rag(question_text):
    sub_question= user_chain.invoke({"question":question_text})

    answers =[]
    for sub_q in sub_question:
        retrieve_docs= st.session_state.retriever.get_relevant_documents(sub_q)
        # convert all text into a context string
        context_text="\n\n".join(doc.page_content for doc in retrieve_docs)
        # run RAG PROMPT
        answer = (rag_prompt_template | llm | StrOutputParser()).invoke({
            "context": context_text,
            "question": sub_q
        })
        answers.append((sub_q, answer))

    return answers

# synthesizer Prompt
synth_template=""" Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}"""

final_synth_prompt=ChatPromptTemplate.from_template(synth_template)

def format_qa_pairs(qa_pairs):
    return "\n\n".join([f"Q:{q}\nA: {a}" for q, a in qa_pairs])

if user_question:
    start=time.time()
    # 1. Retrieve sub-questions and answers
    qa_pairs = retrieve_rag(user_question)

    # 2. Synthesize final answer
    context_str = format_qa_pairs(qa_pairs)
    final_answer = (final_synth_prompt | llm | StrOutputParser()).invoke({
        "context": context_str,
        "question": user_question
    })
    st.markdown("###  Synthesized Answer")
    st.write(final_answer)
    st.caption(f" Response in {time.time() - start:.2f}s")

    # Show sub-answers & docs (expander, if needed)
    with st.expander("See sub-questions and retrieved answers"):
        for i, (q, a) in enumerate(qa_pairs, 1):
            st.markdown(f"**Sub-question {i}:** {q}")
            st.markdown(f"**Answer {i}:** {a}")
            st.divider()


