# pages/Control_Center.py

import os, tempfile
import streamlit as st
from datetime import datetime  # (kept if you log timestamps elsewhere)
from dotenv import load_dotenv

# Data / Vector stuff
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()  # ensure env vars like PINECONE_API_KEY are loaded

# -----------------------------
# Helpers
# -----------------------------
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
        results[topic] = chain.invoke({
            "topic": topic,
            "template_clause": t_text,
            "uploaded_clause": u_text
        })
        progress.progress((i + 1) / len(review_points), text=f"正在分析: {topic}")
    progress.empty()
    return results

INDEX_NAME = "contract-assistant"

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
        PineconeVectorStore.from_documents(
            docs, embedding=embeddings, index_name=INDEX_NAME, namespace=namespace
        )
        os.remove(path)
    st.success(f"參考文件 '{namespace}' 已成功存入知識庫！")
    if namespace not in st.session_state.processed_namespaces:
        st.session_state.processed_namespaces.append(namespace)
    st.cache_data.clear()

def fetch_pinecone_namespaces(index_name):
    pc = get_pinecone_client()
    try:
        stats = pc.describe_index(index_name).stats
        return list(stats.namespaces.keys()) if stats and stats.namespaces else []
    except Exception:
        return []

@st.cache_resource
def get_pinecone_client():
    return Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# -----------------------------
# UI: Only the "Review Parameters" flow
# -----------------------------

# session state guards (since we removed Settings/Search initialization)
if "processed_namespaces" not in st.session_state:
    st.session_state.processed_namespaces = fetch_pinecone_namespaces(INDEX_NAME) or []
if "selected_namespace" not in st.session_state:
    st.session_state.selected_namespace = None
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 256

st.title("可自訂的審查項目 Customizable Parameters")

CORE_REVIEW_POINTS = [
    "合約的保密期限 &nbsp;(Confidentiality Period)",
    "機密資訊的定義範圍 &nbsp;(Definition of Confidential Information)",
    "允許揭露機密資訊的例外情況 &nbsp;(Permitted Disclosures)",
    "合約的準據法與管轄法院 &nbsp;(Governing Law and Jurisdiction)",
    "資訊返還或銷毀的義務 &nbsp;(Return or Destruction of Information)",
    "針對違約行為的補救措施或賠償條款 &nbsp;(Remedies for Breach)",
    "智慧財產權的歸屬 &nbsp;(Intellectual Property Rights)",
    "違約通知與改善期限 &nbsp;(Notice of Breach and Cure Period)"
]
for point in CORE_REVIEW_POINTS:
    st.toggle(point.split(" (")[0], value=True, key=point)
st.text_area("新增審查項目（每行一個)：", key="core_points_text", height=100)
st.markdown("---")

st.header("步驟一：管理參考文件 Manage Reference Documents")
new_ref_file = st.file_uploader("選擇 PDF 作為新的比對基準", type="pdf", key="ref_uploader")
if st.button("處理並儲存至知識庫"):
    if new_ref_file:
        process_and_ingest_reference_file(new_ref_file)
    else:
        st.warning("請先選擇一個參考文件。")
st.divider()

st.header("步驟二：選擇比對基準 Select Comparison Criteria")
selected = st.selectbox(
    "請從已有的知識庫中選擇一份參考文件：",
    options=st.session_state.processed_namespaces,
    index=(
        st.session_state.processed_namespaces.index(st.session_state.selected_namespace)
        if st.session_state.selected_namespace in st.session_state.processed_namespaces
        else None
    ),
    placeholder="請選擇..."
)
if selected is not None:
    st.session_state.selected_namespace = selected

if st.button("手動同步知識庫列表"):
    st.session_state.processed_namespaces = fetch_pinecone_namespaces(INDEX_NAME)
    st.rerun()
st.divider()

st.header("步驟三：上傳待審文件並執行分析 Document Upload & Analysis")
selected_namespace = st.session_state.selected_namespace
if not selected_namespace:
    st.info("請在上方步驟二選擇一份參考文件作為比對基準。")
else:
    st.success(f"當前比對基準為： **{selected_namespace}**")

target_file = st.file_uploader("上傳您要審查的合約文件 (PDF)", type="pdf", key="target_uploader_main")
st.button(
    "開始自動比對與分析 Initialize Comparison & Analysis",
    type="primary",
    use_container_width=True,
    disabled=(not target_file),
    key="start_compare_btn"
)

if st.session_state.get("start_compare_btn") and target_file and selected_namespace:
    with st.spinner("正在準備比對環境..."):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        template_retriever = PineconeVectorStore(
            index_name=INDEX_NAME, embedding=embeddings, namespace=selected_namespace
        ).as_retriever(search_kwargs={'k': 2})
        uploaded_retriever = load_and_process_pdf_for_faiss(target_file)

    temp = st.session_state.get('temperature', 0.7)
    max_tok = st.session_state.get('max_tokens', 256)
    st.session_state.comparison_results = run_comparison(
        template_retriever,
        uploaded_retriever,
        CORE_REVIEW_POINTS,
        temp,
        max_tok
    )
    st.rerun()

# (Optional) render results if they exist
if st.session_state.comparison_results:
    st.header("分析報告 Analysis Results")
    for topic, md in st.session_state.comparison_results.items():
        with st.expander(topic, expanded=False):
            st.markdown(md)