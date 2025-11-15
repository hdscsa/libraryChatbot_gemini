import os
import sys
import streamlit as st
import nest_asyncio

nest_asyncio.apply()

# ================================
# Google Gemini API Key
# ================================
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("⚠️ GOOGLE_API_KEY를 Streamlit Secrets에 등록하세요!")
    st.stop()

# ================================
# LangChain 관련 모듈
# ================================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

from langchain_chroma import Chroma

import shutil

# ================================
# PDF 파일 설정
# ================================
PDF_PATH = r"/mount/src/librarychatbot_gemini/안전한 바다여행_최종.pdf"
PDF_NAME = os.path.splitext(os.path.basename(PDF_PATH))[0]
VECTOR_DIR = f"./chroma_db_{PDF_NAME}"

# ================================
# Streamlit 캐시 초기화
# ================================
if st.button("🔄 캐시 및 벡터DB 초기화"):
    if os.path.exists(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR)
    st.cache_resource.clear()
    st.success("초기화 완료. 새로고침하세요.")

# ================================
# PDF 로드 + 분할
# ================================
@st.cache_resource
def load_and_split_pdf(filepath):
    loader = PyPDFLoader(filepath)
    return loader.load_and_split()

# ================================
# 벡터 DB 생성 함수 (Chroma)
# ================================
@st.cache_resource
def create_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(docs)
    st.info(f"📄 {len(split_docs)}개의 문서 청크 생성됨")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma.from_documents(
        split_docs,
        embedding=embeddings,
        persist_directory=VECTOR_DIR,   # client_settings 제거!!!!
        collection_name="default"
    )

    st.success("💾 Chroma 벡터DB 생성 완료!")
    return vectorstore

# ================================
# 기존 DB 로드 또는 생성
# ================================
@st.cache_resource
def get_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if os.path.exists(VECTOR_DIR):
        st.info("📂 기존 ChromaDB 로드 중...")
        return Chroma(
            persist_directory=VECTOR_DIR,
            embedding_function=embeddings,
            collection_name="default"  # client_settings 제거!!!!
        )
    else:
        return create_vector_store(docs)

# ================================
# RAG 구성
# ================================
@st.cache_resource
def initialize_rag(model_name):
    pages = load_and_split_pdf(PDF_PATH)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # ---- 질문 재구성 프롬프트 ----
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Given the chat history and latest question, rewrite the question as a standalone question. Do not answer it."),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])

    # ---- QA 프롬프트 ----
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a Korean assistant. Use the retrieved context to answer. If unknown, say you don't know.\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.7,
        convert_system_message_to_human=True
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain

# ================================
# UI
# ================================
st.header("🌊 안전한 바다여행 Q&A 챗봇")

if not os.path.exists(VECTOR_DIR):
    st.info("🔄 첫 실행: 벡터 생성 중...")
else:
    st.info(f"📂 {PDF_NAME} 벡터DB 로드 완료")

model_choice = st.selectbox(
    "Gemini 모델 선택",
    ["gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-2.0-flash-lite"]
)

with st.spinner("챗봇 초기화 중..."):
    try:
        rag_chain = initialize_rag(model_choice)
        st.success("챗봇 준비 완료!")
    except Exception as e:
        st.error(f"초기화 오류: {str(e)}")
        st.stop()

# ================================
# 대화 히스토리
# ================================
chat_history = StreamlitChatMessageHistory(key="chat_messages")

chat_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer"
)

# ================================
# 이전 대화 출력
# ================================
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

# ================================
# 사용자 입력 처리
# ================================
if user_input := st.chat_input("질문을 입력하세요…"):
    st.chat_message("human").write(user_input)

    with st.chat_message("ai"):
        with st.spinner("답변 생성 중..."):
            response = chat_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "session1"}}
            )

            st.write(response["answer"])

            with st.expander("📘 참고 문서"):
                for d in response["context"]:
                    st.write(d.metadata.get("source", "출처 정보 없음"))
