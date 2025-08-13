<<<<<<< HEAD
=======
# 檔案名稱: Home.py
# (此版本已移除介紹頁，直接進入主應用程式以提升性能)

>>>>>>> origin/main
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

<<<<<<< HEAD
# 安永Logo
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.image("logo.png", width=100) 
    

=======
>>>>>>> origin/main
# --- 2. 環境變數與核心設定 ---
load_dotenv()
INDEX_NAME = "contract-assistant"

# --- 3. Session State 初始化 ---
<<<<<<< HEAD
=======
# (移除了 app_mode 相關的 state)
>>>>>>> origin/main
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

<<<<<<< HEAD
def draw_main_app():
    """首頁視覺升級：ChatGPT 風格 Hero、玻璃質感卡片、精緻歷程區"""

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
    """, unsafe_allow_html=True)

    # ===== HERO =====
    st.markdown("""
    <div class="hero-wrap">
      <span class="kicker">AI合約動態比對工具 Contract Analysis Tool</span>
      <h1 class="hero">上傳基準 ➜ 上傳審閱 ➜ 即時差異、條款風險與修訂建議</h1>
      <p class="sub">您可以上傳參考文件作為永久比對基準，然後上傳待審文件進行即時分析。<br>
      Upload a reference baseline and then a document under review for real-time side-by-side analysis.</p>
      <div class="btn-row">
        <a class="btn primary" href="#upload">開始上傳 / Get Started</a>
        <a class="btn" href="#how">如何運作 / How it works</a>
        <a class="btn success" href="#history">歷程 / Activity</a>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ===== HELPERS =====
=======
# (函式 draw_intro_page() 已被完整刪除)

def draw_main_app():
    """繪製主應用程式 (精確復刻 CB_Mainapp2.2.py)"""
    st.title("🚀 AI 合約動態比對工具")
    st.markdown("##### 您可以上傳參考文件作為永久比對基準，然後上傳待審文件進行即時分析。You may upload a reference document to serve as a permanent comparison baseline, and subsequently upload the document under review for real-time analysis.")
    
>>>>>>> origin/main
    def _find_by_id(qid):
        for it in st.session_state.search_history:
            if it["id"] == qid: return it
        return None
<<<<<<< HEAD

=======
>>>>>>> origin/main
    def _df(items):
        return pd.DataFrame([{
            "id": it["id"], 
            "query": it["query"], 
            "timestamp": it["timestamp"], 
<<<<<<< HEAD
            "tags": ", ".join(it.get("tags", [])),
            "pinned": it.get("pinned", False), 
            "top_title": (it["results"][0]["title"] if it.get("results") else ""),
            "top_path": (it["results"][0]["path"] if it.get("results") else ""),
        } for it in items])

    # ===== 如何運作 HOW IT WORKS =====
    st.markdown('<div id="how"></div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### 如何運作 / How it works")
        st.markdown(
            """
            1. **建立基準**：上傳「參考合約」，系統會作為永久比對基準保存。  
            2. **上傳草稿**：再上傳「待審文件」，執行**段落級動態差異**與**條款語義對齊**。  
            3. **智慧分析**：在功能頁輸入查詢（例：`責任上限`、`解約條款`），取得**差異重點**與**修訂建議**。  
            4. **歷程與釘選**：所有分析操作會被記錄，支援**釘選**與**標籤**便於審計。
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ===== FOOTER =====
    st.markdown(
        """
        <div style="opacity:.6; text-align:center; padding:1.1rem 0;">
          Built with Streamlit. Designed for Legal Ops. ©Ernst & Young LLP. All Rights Reserved.
        </div>
        """,
        unsafe_allow_html=True
    )

# --- 6. 主邏輯 ---
=======
            "tags": ", ".join(it.get("tags", [])), "pinned": it.get("pinned", False), 
            "top_title": (it["results"][0]["title"] if it.get("results") else ""),
            "top_path": (it["results"][0]["path"] if it.get("results") else ""),} for it in items])

# --- 6. 主邏輯 ---
# (簡化主邏輯，直接執行主應用程式)
>>>>>>> origin/main
draw_main_app()