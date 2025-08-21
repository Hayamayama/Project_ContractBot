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
st.header("控制中心 Control Center")
pins = sum(1 for it in st.session_state.search_history if it.get("pinned"))

# ----- Search -----

sub = option_menu(
    None, ["查詢歷史 Query History", f"釘選 Pinned ({pins})", "其他工具 Tools"],
    icons=["clock-history", "pin-angle-fill", "wrench-adjustable-circle"],
    menu_icon="cast", default_index=0, orientation="horizontal",
)

    # --- helper used by "Queue History" and "Pinned" sections -------------------------------------------
def _toggle_pin(_id: str):
    item = _find_by_id(_id)
    if not item:
        return
    item["pinned"] = not item.get("pinned", False)
    # keep the toggle UI state in sync on reruns
    st.session_state[f"pin_{_id}"] = item["pinned"]


# --- Query History -----------------------------------------------------------
if sub == "查詢歷史 Query History":
    q = st.text_input("輸入關鍵字篩選 Search using Keywords")
    c1, c2 = st.columns([3, 1])
    tag_filter = c1.text_input("依標籤篩選 Filter by Tag")
    only_pinned = c2.toggle("僅顯示釘選 Pinned Only", value=False)

    items = list(st.session_state.search_history)  

    # text filter
    if q:
        ql = q.lower()
        items = [it for it in items if ql in it.get("query", "").lower()]

    # tag filter (case-insensitive)
    if tag_filter:
        tf = tag_filter.lower()
        items = [
            it for it in items
            if tf in [str(t).lower() for t in it.get("tags", [])]
        ]

    # pinned-only filter
    if only_pinned:
        items = [it for it in items if it.get("pinned")]

    # newest first
    items = sorted(items, key=lambda x: x.get("timestamp", ""), reverse=True)

    if not items:
        st.info("No matching history.")
    else:
        for it in items:
            ts_raw = it.get("timestamp", "")
            try:
                ts_disp = datetime.fromisoformat(ts_raw).strftime("%b %d, %Y %H:%M")
            except Exception:
                ts_disp = ts_raw or "—"

            with st.expander(f"**{it.get('query','(no query)')}** · {ts_disp}"):
                st.toggle(
                    "Pinned",
                    key=f"pin_{it['id']}",
                    value=it.get("pinned", False),
                    on_change=_toggle_pin,
                    args=(it["id"],),
                )


# --- Pinned tab --------------------------------------------------------------
elif sub.startswith("釘選 Pinned"):
    pinned_items = [it for it in st.session_state.search_history if it.get("pinned")]
    if pinned_items:
        df = _df(pinned_items)
        # (optional) hide the 'pinned' column in the table
        df = df.drop(columns=["pinned"], errors="ignore")
        # show newest first if the df has a 'timestamp' column
        if "timestamp" in df.columns:
            # avoid exceptions if non-ISO strings are present
            try:
                df = df.sort_values("timestamp", ascending=False)
            except Exception:
                pass
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No pinned items yet.")

# --- Tools tab --------------------------------------------------------------
elif sub == "其他工具 Tools":
    st.markdown("##### 快速匯出 Quick Exports")

    items = list(st.session_state.search_history)
    all_df = _df(items)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    c1, c2, c3 = st.columns([1, 1, 1])  #equal width columns

    # Export ALL (JSON)
    c1.download_button(
        "Export all (JSON)",
        json.dumps(items, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"search_history_{ts}.json",
        mime="application/json",
        disabled=not items,
        use_container_width=True
    )

    # Export ALL (CSV)
    c2.download_button(
        "Export all (CSV)",
        (all_df.to_csv(index=False).encode("utf-8") if not all_df.empty else b""),
        file_name=f"search_history_{ts}.csv",
        mime="text/csv",
        disabled=all_df.empty,
        use_container_width=True
    )

    # Export PINNED (CSV)
    pinned_items = [it for it in items if it.get("pinned")]
    pinned_df = _df(pinned_items)

    c3.download_button(
        "Export pinned (CSV)",
        (pinned_df.to_csv(index=False).encode("utf-8") if not pinned_df.empty else b""),
        file_name=f"search_history_pinned_{ts}.csv",
        mime="text/csv",
        disabled=pinned_df.empty,
        use_container_width=True
    )

    st.markdown("##### Maintenance")
    m1, m2, m3 = st.columns(3)

    # Clear all history
    if m1.button("Clear all history", use_container_width=True, type="secondary", disabled=not items):
        st.session_state.search_history = []
        st.success("History cleared. Rerun to see changes.")
        st.stop()

    # Unpin all
    if m2.button("Unpin all", use_container_width=True, type="secondary", disabled=not pinned_items):
        for it in st.session_state.search_history:
            it["pinned"] = False
        st.success("All items unpinned.")

    # Small preview so users see what they'll export
    with st.expander("Preview (first 100 rows)"):
        if all_df.empty:
            st.info("No history to preview.")
        else:
            preview_df = all_df.copy()
            # Keep preview readable; hide internal flags if you like
            preview_df = preview_df.drop(columns=["pinned"], errors="ignore")
            st.dataframe(preview_df.head(100), use_container_width=True, hide_index=True)


# ----- Results -----
if st.session_state.get("comparison_results"):
    st.subheader("合約比對分析報告")
    for topic, result in st.session_state.comparison_results.items():
        with st.expander(f"**審查項目：{topic}**", expanded=True):
            st.markdown(result, unsafe_allow_html=True)
    st.session_state.comparison_results = None
