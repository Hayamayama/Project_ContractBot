# 檔案名稱: Home.py
# (此版本已移除介紹頁，直接進入主應用程式以提升性能)

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# LangChain 元件
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# 其他工具
import pandas as pd
import json
from datetime import datetime
from difflib import get_close_matches
from streamlit_option_menu import option_menu

# --- 1. 頁面設定 ---
st.set_page_config(page_title="AI 合約動態比對工具", layout="wide")

# --- 2. 環境變數與核心設定 ---
load_dotenv()
INDEX_NAME = "contract-assistant"

# --- 3. Session State 初始化 ---
# (移除了 app_mode 相關的 state)
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None

if "processed_namespaces" not in st.session_state:
    st.session_state.processed_namespaces = []
if "selected_namespace" not in st.session_state:
    st.session_state.selected_namespace = None
if "core_points_text" not in st.session_state:
    st.session_state.core_points_text = ""

if "search_history" not in st.session_state:
    st.session_state.search_history = [
        {"id":"q-001","query":"星宇航空合約分析","timestamp":"2025-06-28T10:34:00", "results":[{"title":"Q2 摘要","snippet":"營收年增 12%","path":"reports/q2_2024.pdf"}], "tags":["finance","q2"],"pinned":True,"notes":"董事會簡報用"},
        {"id":"q-002","query":"客戶流失儀表板","timestamp":"2025-07-03T14:09:20", "results":[{"title":"Cohort 分析","snippet":"Jan–Jun","path":"dashboards/churn.html"}], "tags":["product","retention"],"pinned":False,"notes":""},
    ]

# --- 4. 核心功能函式 ---
@st.cache_resource
def get_pinecone_client():
    return Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

def fetch_pinecone_namespaces(index_name):
    pc = get_pinecone_client()
    try:
        index_stats = pc.describe_index(index_name).stats
        return list(index_stats.namespaces.keys()) if index_stats and index_stats.namespaces else []
    except Exception:
        return []

if not st.session_state.processed_namespaces:
    st.session_state.processed_namespaces = fetch_pinecone_namespaces(INDEX_NAME)

def process_and_ingest_reference_file(uploaded_file):
    namespace = uploaded_file.name
    with st.spinner(f"正在處理參考文件 '{namespace}' 並存入永久知識庫..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        PineconeVectorStore.from_documents(docs, embedding=embeddings, index_name=INDEX_NAME, namespace=namespace)
        os.remove(tmp_file_path)
    st.success(f"參考文件 '{namespace}' 已成功存入知識庫！")
    if namespace not in st.session_state.processed_namespaces:
        st.session_state.processed_namespaces.append(namespace)
    st.cache_data.clear()

@st.cache_resource
def load_and_process_pdf_for_faiss(_uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(_uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    os.remove(tmp_file_path)
    return vectorstore.as_retriever(search_kwargs={'k': 2})

def run_comparison(template_retriever, uploaded_retriever, review_points, temperature, max_tokens):
    llm = ChatOpenAI(model_name='gpt-4o', temperature=temperature, max_tokens=max_tokens)
    comparison_template = """
    你是一位頂尖的法務專家，你的任務是精準比較兩份合約中的同一條款，並以保護「我方公司」的利益為最高原則。
    **我方公司的標準範本條款:**
    ```{template_clause}```
    **待審文件的對應條款:**
    ```{uploaded_clause}```
    請針對「{topic}」這個審查重點，完成以下任務：
    1. **條款摘要**, 2. **差異分析**, 3. **風險提示與建議**。
    請用 Markdown 格式清晰地呈現你的分析報告。
    """
    prompt = PromptTemplate.from_template(comparison_template)
    chain = prompt | llm | StrOutputParser()
    results = {}
    progress_bar_main = st.progress(0, text="開始進行比對...")
    for i, topic in enumerate(review_points):
        template_clause_docs = template_retriever.invoke(topic)
        uploaded_clause_docs = uploaded_retriever.invoke(topic)
        template_clause_text = "\n---\n".join([doc.page_content for doc in template_clause_docs])
        uploaded_clause_text = "\n---\n".join([doc.page_content for doc in uploaded_clause_docs])
        result = chain.invoke({ "topic": topic, "template_clause": template_clause_text, "uploaded_clause": uploaded_clause_text })
        results[topic] = result
        progress_bar_main.progress((i + 1) / len(review_points), text=f"正在分析: {topic}")
    progress_bar_main.empty()
    return results

# --- 5. UI 渲染函式 ---

# (函式 draw_intro_page() 已被完整刪除)

def draw_main_app():
    """繪製主應用程式 (精確復刻 CB_Mainapp2.2.py)"""
    st.title("🚀 AI 合約動態比對工具")
    st.markdown("##### 您可以上傳參考文件作為永久比對基準，然後上傳待審文件進行即時分析。You may upload a reference document to serve as a permanent comparison baseline, and subsequently upload the document under review for real-time analysis.")
    
    def _find_by_id(qid):
        for it in st.session_state.search_history:
            if it["id"] == qid: return it
        return None
    def _df(items):
        return pd.DataFrame([{
            "id": it["id"], 
            "query": it["query"], 
            "timestamp": it["timestamp"], 
            "tags": ", ".join(it.get("tags", [])), "pinned": it.get("pinned", False), 
            "top_title": (it["results"][0]["title"] if it.get("results") else ""),
            "top_path": (it["results"][0]["path"] if it.get("results") else ""),} for it in items])

# --- 6. 主邏輯 ---
# (簡化主邏輯，直接執行主應用程式)
draw_main_app()