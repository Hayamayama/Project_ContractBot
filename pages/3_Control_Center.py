# pages/Control_Center.py
import os, tempfile, json
from datetime import datetime
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from dotenv import load_dotenv

# LangChain / Vector DB
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# ---------- Page setup ----------
st.set_page_config(page_title="Control Center", layout="wide")
# --- 【修改】: 使用 st.logo() ---
st.logo("logo.png")

load_dotenv()
INDEX_NAME = "contract-assistant"

# ---------- Session defaults ----------
if "search_history" not in st.session_state:
    st.session_state.search_history = [
        {"id":"q-001","query":"星宇航空合約分析","timestamp":"2025-06-28T10:34:00",
         "results":[{"title":"Q2 摘要","snippet":"營收年增 12%","path":"reports/q2_2024.pdf"}],
         "tags":["finance","q2"],"pinned":True,"notes":"董事會簡報用"},
        {"id":"q-002","query":"客戶流失儀表板","timestamp":"2025-07-03T14:09:20",
         "results":[{"title":"Cohort 分析","snippet":"Jan–Jun","path":"dashboards/churn.html"}],
         "tags":["product","retention"],"pinned":False,"notes":""},
    ]
if "processed_namespaces" not in st.session_state:
    st.session_state.processed_namespaces = []
if "selected_namespace" not in st.session_state:
    st.session_state.selected_namespace = None
if "core_points_text" not in st.session_state:
    st.session_state.core_points_text = ""
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 256

# ---------- Helpers ----------
def _find_by_id(qid):
    for it in st.session_state.search_history:
        if it["id"] == qid: return it
    return None

def _df(items):
    return pd.DataFrame([{
        "id": it["id"],
        "query": it["query"],
        "timestamp": it["timestamp"],
        "tags": ", ".join(it.get("tags", [])),
        "pinned": it.get("pinned", False),
        "top_title": (it["results"][0]["title"] if it.get("results") else ""),
        "top_path": (it["results"][0]["path"] if it.get("results") else ""),
    } for it in items])

@st.cache_resource
def get_pinecone_client():
    return Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

def fetch_pinecone_namespaces(index_name):
    pc = get_pinecone_client()
    try:
        stats = pc.describe_index(index_name).stats
        return list(stats.namespaces.keys()) if stats and stats.namespaces else []
    except Exception:
        return []

if not st.session_state.processed_namespaces:
    st.session_state.processed_namespaces = fetch_pinecone_namespaces(INDEX_NAME)

def process_and_ingest_reference_file(uploaded_file):
    namespace = uploaded_file.name
    with st.spinner(f"正在處理參考文件 '{namespace}' 並存入永久知識庫..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            path = tmp.name
        loader = PyPDFLoader(path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs = splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        PineconeVectorStore.from_documents(docs, embedding=embeddings,
                                           index_name=INDEX_NAME, namespace=namespace)
        os.remove(path)
    st.success(f"參考文件 '{namespace}' 已成功存入知識庫！")
    if namespace not in st.session_state.processed_namespaces:
        st.session_state.processed_namespaces.append(namespace)
    st.cache_data.clear()

@st.cache_resource
def load_and_process_pdf_for_faiss(_uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_uploaded_file.getvalue())
        path = tmp.name
    loader = PyPDFLoader(path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(split_docs, embeddings)
    os.remove(path)
    return vs.as_retriever(search_kwargs={'k': 2})

def run_comparison(template_retriever, uploaded_retriever, review_points, temperature, max_tokens):
    llm = ChatOpenAI(model_name='gpt-4o', temperature=temperature, max_tokens=max_tokens)
    tpl = """
你是一位頂尖的法務專家，你的任務是精準比較兩份合約中的同一條款，並以保護「我方公司」的利益為最高原則。
**我方公司的標準範本條款:**
```{template_clause}```
**待審文件的對應條款:**
```{uploaded_clause}```
請針對「{topic}」這個審查重點，完成以下任務：
1. **條款摘要**, 2. **差異分析**, 3. **風險提示與建議**。
請用 Markdown 格式清晰地呈現你的分析報告。
"""
    prompt = PromptTemplate.from_template(tpl)
    chain = prompt | llm | StrOutputParser()
    results = {}
    progress = st.progress(0, text="開始進行比對...")
    for i, topic in enumerate(review_points):
        t_docs = template_retriever.invoke(topic)
        u_docs = uploaded_retriever.invoke(topic)
        t_text = "\n---\n".join([d.page_content for d in t_docs])
        u_text = "\n---\n".join([d.page_content for d in u_docs])
        results[topic] = chain.invoke({"topic": topic,
                                       "template_clause": t_text,
                                       "uploaded_clause": u_text})
        progress.progress((i + 1) / len(review_points), text=f"正在分析: {topic}")
    progress.empty()
    return results

# ---------- UI ----------
st.title("控制中心 Control Center")

pins = sum(1 for it in st.session_state.search_history if it.get("pinned"))

top = option_menu(
    None, ["Settings", "Search"],
    icons=["gear-fill", "search"],
    menu_icon="cast", default_index=1, orientation="horizontal",
)

# ----- Settings -----
if top == "Settings":
    st.subheader('模型參數 Model Parameters')
    st.session_state.temperature = st.slider("參數溫度 Temperature", 0.0, 2.0, st.session_state.temperature, 0.1)
    with st.expander("What does temperature do?"):
        st.caption("Lower = focused & predictable; higher = varied & creative.")
    st.session_state.max_tokens = st.slider("最大字元數 Max Tokens", 1, 4096, st.session_state.max_tokens)
    with st.expander("What do tokens do?"):
        st.caption("Max Tokens limits the length of AI responses.")
    st.toggle("Streaming responses", value=True)
    st.text_input("System prompt preset", placeholder="You are a helpful analyst…")

# ----- Search -----
if top == "Search":
    st.subheader("搜尋控制台 Search Console")
    sub = option_menu(
        None, ["Query History", f"Pinned ({pins})", "Tools"],
        icons=["clock-history", "pin-angle-fill", "wrench-adjustable-circle"],
        menu_icon="cast", default_index=0, orientation="horizontal",
    )

    if sub == "Query History":
        q = st.text_input("Search your past queries…", placeholder="Type to filter in real-time")
        c1, c2 = st.columns([3, 1])
        tag_filter = c1.text_input("Filter by tag", placeholder="finance / product / …")
        only_pinned = c2.toggle("Pinned only", value=False)
        items = st.session_state.search_history
        if q: items = [it for it in items if q.lower() in it["query"].lower()]
        if tag_filter: items = [it for it in items if tag_filter.lower() in [t.lower() for t in it.get("tags", [])]]
        if only_pinned: items = [it for it in items if it.get("pinned")]
        items = sorted(items, key=lambda x: x["timestamp"], reverse=True)
        if not items:
            st.info("No matching history.")
        else:
            for it in items:
                with st.expander(f"**{it['query']}** · {datetime.fromisoformat(it['timestamp']).strftime('%b %d, %Y %H:%M')}"):
                    st.toggle(
                        "Pinned", key=f"pin_{it['id']}", value=it.get("pinned"),
                        on_change=lambda _id=it["id"]: _find_by_id(_id).__setitem__("pinned", not _find_by_id(_id).get("pinned"))
                    )

    if sub.startswith("Pinned"):
        df = _df([it for it in st.session_state.search_history if it.get("pinned")])
        st.dataframe(df, use_container_width=True, hide_index=True) if not df.empty else st.info("No pinned items yet.")

    if sub == "Tools":
        all_df = _df(st.session_state.search_history)
        st.markdown("##### Quick Exports")
        c1, c2 = st.columns(2)
        c1.download_button("Export all (JSON)",
                           json.dumps(st.session_state.search_history, ensure_ascii=False, indent=2).encode("utf-8"),
                           file_name="search_history.json")
        c2.download_button("Export all (CSV)",
                           all_df.to_csv(index=False).encode("utf-8"),
                           file_name="search_history.csv", mime="text/csv")

# ----- Results -----
if st.session_state.get("comparison_results"):
    st.subheader("📜 合約比對分析報告")
    for topic, result in st.session_state.comparison_results.items():
        with st.expander(f"**審查項目：{topic}**", expanded=True):
            st.markdown(result, unsafe_allow_html=True)
    st.session_state.comparison_results = None
