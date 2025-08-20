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
st.logo("logo.png")

# --- 2. 環境變數與核心設定 ---
load_dotenv()
INDEX_NAME = "contract-assistant"

# --- 3. Session State 初始化 ---
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None

if "processed_namespaces" not in st.session_state:
    st.session_state.processed_namespaces = []
if "selected_namespace" not in st.session_state:
    st.session_state.selected_namespace = None
if "core_points_text" not in st.session_state:
    st.session_state.core_points_text = ""

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

# --- 5. UI 渲染函式 ---

def draw_main_app():
    # ===== CSS =====
    st.markdown("""
    <style>
      .block-container {max-width: 1180px; padding-top: 0!important;}
      /* Hero */
      .hero-wrap{
        margin: 0 -2rem 1.25rem; padding: 3.5rem 2rem 2.25rem;
        background: radial-gradient(1200px 600px at 10% -10%, rgba(99,102,241,.25), transparent 60%),
                    radial-gradient(900px 600px at 110% 10%, rgba(16,185,129,.24), transparent 60%),
                    linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
        border-bottom: 1px solid rgba(255,255,255,.08);
      }
      .kicker{
        display:inline-flex; gap:.5rem; align-items:center;
        padding:.35rem .7rem; border-radius:999px;
        background: rgba(99,102,241,.15);
        border:1px solid rgba(99,102,241,.35);
        font-size:.82rem; letter-spacing:.02em;
      }
      h1.hero{margin:.6rem 0 .2rem; font-size:2.1rem; line-height:1.15;}
      p.sub{margin:.3rem 0 0; opacity:.9}
      .btn-row{display:flex; gap:.6rem; margin-top:1rem; flex-wrap:wrap}
      .btn{
        padding:.6rem .95rem; border-radius:12px; text-decoration:none;
        border:1px solid rgba(255,255,255,.14);
        background:linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
        transition:.2s;
      }
      .btn:hover{transform:translateY(-1px); border-color:rgba(255,255,255,.28)}
      .btn.primary{background:linear-gradient(180deg, rgba(99,102,241,.55), rgba(99,102,241,.35)); border-color:rgba(99,102,241,.65)}
      .btn.success{background:linear-gradient(180deg, rgba(16,185,129,.55), rgba(16,185,129,.35)); border-color:rgba(16,185,129,.65)}

      /* Cards */
      .glass{
        background: rgba(255,255,255,.05);
        border: 1px solid rgba(255,255,255,.12);
        box-shadow: 0 10px 30px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.04);
        backdrop-filter: blur(10px);
        border-radius: 18px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
      }
      .section-h {font-size:1.05rem; font-weight:600; opacity:.95; margin-bottom:.35rem}
      .chip{display:inline-flex; align-items:center; gap:.4rem; padding:.25rem .55rem; border-radius:999px; font-size:.78rem;
            border:1px solid rgba(255,255,255,.18); background: rgba(255,255,255,.06);}

      /* History rows */
      .row{padding:.5rem 0;}
      .row + .row{border-top:1px solid rgba(255,255,255,.08)}
    </style>
    """,unsafe_allow_html=True)

    # ===== MAIN =====  
    params = st.query_params   #checks query parameters for routing
    if params.get("goto") == "review":
        st.switch_page("pages/4_Review_Parameters.py")   

    st.markdown("""
        <div class="hero-wrap">
        <span class="kicker">AI 合約動態比對工具 Contract Analysis Tool</span>
            <h1 class="hero">運用創新高速 AI，加速契約分析<br>
            <span class="translation">Accelerate Contract Analysis with Innovative Turbocharged AI</span></h1>
        <p class="sub">上傳基準 ➜ 上傳審閱 ➜ 即時差異、條款風險與修訂建議。您可以上傳參考文件作為永久比對基準，然後上傳待審文件進行即時分析。</p>
        </div>
        """, unsafe_allow_html=True)

    # ===== HELPERS =====
    def _find_by_id(qid):
        for it in st.session_state.search_history:
            if it["id"] == qid: return it
        return None

    # ===== 使用步驟 INSTRUCTIONS =====
    st.markdown('<div id="how"></div>', unsafe_allow_html=True)
    with st.container():
        st.markdown("### 使用步驟 Instructions")
        st.markdown(
            """
            1. **建立基準**：上傳「參考合約」，系統會作為永久比對基準保存。
            2. **上傳草稿**：再上傳「待審文件」，執行段落級動態差異與條款語義對齊。
            3. **智慧分析**：在功能頁輸入查詢（例：`責任上限`、`解約條款`），取得差異重點與修訂建議。
            4. **歷程與釘選**：所有分析操作會被記錄，支援釘選與標籤便於審計。
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    #===== 商業價值 BUSINESS VALUE =====
    st.markdown('<div id="how"></div>', unsafe_allow_html=True)
    with st.container():
        st.markdown("### 商業價值 Business Value")
        st.markdown("""
                    <style>
                    /* Layout */
                    .main > div { padding-top: 0rem; }
                    .section { max-width: 1200px; margin: 0 auto; padding: 2rem 1rem 4rem 1rem; }

                    /* Features grid */
                        .features {
                        display: grid;
                        grid-template-columns: repeat(4, minmax(0, 1fr));
                        gap: 0.7rem;
                        margin-top: -0.4rem;
                    }
                    @media (max-width: 980px) {
                        .features { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                    }
                    @media (max-width: 560px) {
                        .features { grid-template-columns: 1fr; }
                    }
                    .card {
                        background: linear-gradient(90deg, #0B2343 0%, #0A7C4A 100%);
                        border-radius: 18px; padding: 1.35rem; height: 100%;
                        box-shadow: 0 6px 16px rgba(11,35,67,.06);
                    }
                    .card h3 {
                    margin: .6rem 0 .35rem 0;
                    font-size: 1.1rem; line-height: 1.2; color: #0b2343;
                    color: white;
                    }
                    .card p { margin: 0; font-size: 0.9rem; color: white; }

                    /* Simple icon pill */
                    .icon {
                    width: 42px; height: 42px; display:grid; place-items:center;
                    border-radius: 10px; background: #eef5ff; color: #0b2343; font-size: 1.2rem;
                    box-shadow: inset 0 0 0 1px rgba(11,35,67,.06);
                    }""", unsafe_allow_html=True)
        
    st.markdown("""
    <div class="features">
    <div class="card">
        <div class="icon">🔘</div>
        <h3>完整追蹤每項條款</h3>
        <p>可輕鬆建立自訂化 AI 模型，以追蹤您組織所關注的所有事項。</p>
    </div>

    <div class="card">
        <div class="icon">🌐</div>
        <h3>迅速取得 AI 洞見</h3>
        <p>快速啟用完整的智慧化合約資料庫。</p>
    </div>

    <div class="card">
        <div class="icon">🧭</div>
        <h3>免除人工作業</h3>
        <p>透過標準化與最佳化流程，降低錯誤、加速審核與核准，並確保文件用語符合法規與內控標準。</p>
    </div>

    <div class="card">
        <div class="icon">📈</div>
        <h3>良好內控把關驅動成效</h3>
        <p>以合規為先、以成效為本，讓每一份合約都創造更高價值。</p>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


    # ===== FOOTER =====
    st.markdown(
        """
        <div style="opacity:.5; text-align:center; padding:1.1rem 0;">
          Built with Streamlit. Designed for Legal Ops. ©2025 Ernst & Young LLP. All Rights Reserved.
        </div>
        """,
        unsafe_allow_html=True
    )

# --- 6. 主邏輯 ---
draw_main_app()
