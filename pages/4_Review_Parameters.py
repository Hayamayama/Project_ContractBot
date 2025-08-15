import os
import tempfile
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# LangChain / Vector DB
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from pinecone import Pinecone

# --- 頁面設定 ---
st.set_page_config(page_title="AI 合約初審", layout="wide", page_icon="📝")
st.logo("logo.png")

# --- 環境變數與核心設定 ---
load_dotenv()
INDEX_NAME = "contract-assistant"
LEARNING_NAMESPACE = "approved-analyses"

# -----------------------------
# Helpers
# -----------------------------

@st.cache_data(show_spinner=False)
def get_language(text_snippet: str, _llm):
    """使用 LLM 快速檢測文字語言"""
    if not text_snippet.strip():
        return "unknown"
    try:
        prompt = PromptTemplate.from_template("Detect the primary language of the following text. Respond with only the two-letter ISO 639-1 code (e.g., 'en' for English, 'zh' for Chinese). Text: ```{text}```")
        chain = prompt | _llm | StrOutputParser()
        # 取一小段文字樣本進行檢測即可，加快速度
        sample = text_snippet[:200]
        lang_code = chain.invoke({"text": sample})
        return lang_code.lower()
    except Exception:
        return "unknown"

@st.cache_data(show_spinner=False)
def translate_to_chinese(text_to_translate: str, _llm):
    """使用 LLM 將文字翻譯為繁體中文"""
    if not text_to_translate.strip():
        return ""
    try:
        prompt = PromptTemplate.from_template("Please translate the following legal text into Traditional Chinese. Only return the translated text, without any explanation or preamble. Text: ```{text}```")
        chain = prompt | _llm | StrOutputParser()
        return chain.invoke({"text": text_to_translate})
    except Exception as e:
        st.error(f"翻譯時發生錯誤: {e}")
        return text_to_translate

def run_comparison(template_retriever, approved_retriever, uploaded_retriever, review_points, temperature, max_tokens):
    """
    執行合約比對，包含自動語言偵測與條件式翻譯，並回傳詳細分析報告字典。
    """
    llm = ChatOpenAI(model_name='gpt-4o', temperature=temperature, max_tokens=max_tokens)
    
    # 最終分析用的 Prompt (不變)
    tpl = """
You are a senior legal counsel at a top-tier professional services firm. Your task is to conduct a preliminary review of a counterparty's contract clause against our company's standard template. Your analysis must be sharp, insightful, and actionable for the project team.

**Background Information - Past Excellent Analysis Examples (if any):**
```{approved_examples}```
---
**Clause A (Normalized to Traditional Chinese):**
```{clause_A}```
---
**Clause B (Normalized to Traditional Chinese):**
```{clause_B}```
---

Based on the clauses provided above for the topic "**{topic}**", please draft a concise yet comprehensive review report in Markdown format. The report MUST include the following three sections:

### 1. 核心條款摘要 (Clause Summary)
- **文件一核心**: Briefly summarize the key points of Clause A.
- **文件二核心**: Briefly summarize the key points of Clause B.

### 2. 關鍵差異與風險分析 (Key Differences & Risk Analysis)
- **差異點**: Clearly identify the material differences between the two clauses.
- **對我方風險**: Analyze the potential legal, commercial, or operational risks these differences pose to **our company**. Be specific.

### 3. 具體修訂與談判建議 (Actionable Revision & Negotiation Suggestions)
- **修訂建議**: Propose specific wording changes to the riskier clause to mitigate risks. If no change is needed, state so.
- **談判策略**: Briefly suggest a negotiation position or points to emphasize.
"""
    analysis_chain = PromptTemplate.from_template(tpl) | llm | StrOutputParser()
    
    results = {}
    progress = st.progress(0, text="AI 法務專家正在審閱合約...")

    mq_template_retriever = MultiQueryRetriever.from_llm(retriever=template_retriever, llm=llm)
    mq_uploaded_retriever = MultiQueryRetriever.from_llm(retriever=uploaded_retriever, llm=llm)

    if approved_retriever:
        mq_approved_retriever = MultiQueryRetriever.from_llm(retriever=approved_retriever, llm=llm)
        ensemble_retriever = EnsembleRetriever(retrievers=[mq_template_retriever, mq_approved_retriever], weights=[0.7, 0.3])
    else:
        ensemble_retriever = mq_template_retriever

    for i, topic in enumerate(review_points):
        display_topic = topic.split(' (')[0].replace('&nbsp;', ' ').strip()
        search_query = topic.replace('&nbsp;', ' ')

        # 1. 檢索文件
        ensemble_docs = ensemble_retriever.get_relevant_documents(search_query)
        u_docs = mq_uploaded_retriever.get_relevant_documents(search_query)

        t_docs, a_docs = [], []
        for doc in ensemble_docs:
            if doc.metadata.get('namespace') == LEARNING_NAMESPACE: a_docs.append(doc)
            else: t_docs.append(doc)

        t_text_original = "\n---\n".join([d.page_content for d in t_docs])
        u_text_original = "\n---\n".join([d.page_content for d in u_docs])
        a_text = "\n---\n".join([d.page_content for d in a_docs]) if a_docs else "無相關範例"
        
        # 2. 自動語言偵測
        lang_t = get_language(t_text_original, llm)
        lang_u = get_language(u_text_original, llm)
        
        # 3. 條件式翻譯，統一轉換為繁體中文
        with st.spinner(f"正在進行語言正規化 ({display_topic})..."):
            t_text_final = translate_to_chinese(t_text_original, llm) if 'en' in lang_t else t_text_original
            u_text_final = translate_to_chinese(u_text_original, llm) if 'en' in lang_u else u_text_original

        if not t_text_final.strip(): t_text_final = "文件中未找到相關條款"
        if not u_text_final.strip(): u_text_final = "文件中未找到相關條款"
            
        # 4. 將統一語言後的文本送入分析鏈
        report = analysis_chain.invoke({
            "topic": display_topic,
            "approved_examples": a_text,
            "clause_A": t_text_final,
            "clause_B": u_text_final
        })
        results[topic] = report
        
        progress.progress((i + 1) / len(review_points), text=f"正在分析: {display_topic}")

    progress.empty()
    return results

@st.cache_resource
def load_and_process_pdf_for_faiss(_uploaded_file):
    # ... (此函式內容不變)
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
    return vs.as_retriever(search_kwargs={'k': 3})

def process_and_ingest_reference_file(uploaded_file):
    # ... (此函式內容不變)
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

@st.cache_resource
def get_pinecone_client():
    return Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

@st.cache_data(ttl="10m")
def fetch_pinecone_namespaces(index_name):
    pc = get_pinecone_client()
    try:
        index_stats = pc.describe_index(index_name).stats
        return list(index_stats.namespaces.keys()) if index_stats and index_stats.namespaces else []
    except Exception as e:
        st.error(f"獲取 Pinecone namespaces 時發生錯誤: {e}")
        return []

# --- UI 部分 ---
if "processed_namespaces" not in st.session_state:
    st.session_state.processed_namespaces = fetch_pinecone_namespaces(INDEX_NAME) or []

st.title("AI 合約初審與風險分析")

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
with st.expander("自訂審查項目", expanded=True):
    cols = st.columns(2)
    for i, point in enumerate(CORE_REVIEW_POINTS):
        with cols[i % 2]:
            st.toggle(point.split(" (")[0], value=True, key=point)
    st.text_area("新增審查項目（每行一個)：", key="core_points_text", height=100, placeholder="例如：\n賠償責任上限 (Limitation of Liability)\n合約的可轉讓性 (Assignability)")
st.divider()

col1, col2 = st.columns(2)
# 【UI 升級】: 移除 UI 上的語言標示
with col1:
    st.header("步驟一：管理參考文件")
    new_ref_file = st.file_uploader("上傳 PDF 作為新的比對基準", type="pdf", key="ref_uploader")
    if st.button("處理並儲存至知識庫"):
        if new_ref_file: process_and_ingest_reference_file(new_ref_file)
        else: st.warning("請先選擇一個參考文件。")
with col2:
    st.header("步驟二：選擇比對基準")
    selected = st.selectbox(
        "從知識庫中選擇一份參考文件：",
        options=st.session_state.processed_namespaces,
        index=(st.session_state.processed_namespaces.index(st.session_state.get("selected_namespace"))
               if st.session_state.get("selected_namespace") in st.session_state.processed_namespaces else None),
        placeholder="請選擇..."
    )
    if selected is not None: st.session_state.selected_namespace = selected
    if st.button("手動同步知識庫列表"):
        st.session_state.processed_namespaces = fetch_pinecone_namespaces(INDEX_NAME)
        st.rerun()
st.divider()

st.header("步驟三：上傳待審文件並執行分析")
selected_namespace = st.session_state.get("selected_namespace")
if not selected_namespace:
    st.info("請在上方步驟二選擇一份參考文件作為比對基準。")
else:
    st.success(f"當前比對基準為： **{selected_namespace}**")

target_file = st.file_uploader("上傳您要審查的合約文件", type="pdf", key="target_uploader")

if target_file: st.session_state.target_file_name = target_file.name

start_button = st.button("開始 AI 深度審閱", type="primary", use_container_width=True, disabled=(not target_file or not selected_namespace))

if start_button:
    # ... (此區塊邏輯不變)
    with st.spinner("正在準備比對環境..."):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        template_retriever = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings, namespace=selected_namespace).as_retriever(search_kwargs={'k': 3})
        uploaded_retriever = load_and_process_pdf_for_faiss(target_file)
        
        approved_retriever = None
        if LEARNING_NAMESPACE in st.session_state.processed_namespaces:
            approved_retriever = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings, namespace=LEARNING_NAMESPACE).as_retriever(search_kwargs={'k': 2})
            st.info("💡 已載入過往的優質分析範例，本次分析品質將更高。")

    temp = st.session_state.get('temperature', 0.3)
    max_tok = st.session_state.get('max_tokens', 4096)

    active_review_points = [p for p in CORE_REVIEW_POINTS if st.session_state.get(p, True)]
    custom_points = [line.strip() for line in st.session_state.get("core_points_text", "").split('\n') if line.strip()]
    final_review_points = active_review_points + custom_points

    if not final_review_points:
        st.error("請至少選擇或新增一個審查項目。")
    else:
        st.session_state.comparison_results = run_comparison(template_retriever, approved_retriever, uploaded_retriever, final_review_points, temp, max_tok)
        st.rerun()

if st.session_state.get("comparison_results"):
    st.balloons()
    st.header("✅ AI 深度審閱報告已完成")
    st.info("您可以檢視下方的逐項分析報告。高品質的報告將可在下一頁歸檔，用於 AI 再學習。")
    st.page_link("pages/5_Analysis_Saving.py", label="下一步：前往「分析歸檔與學習」頁面", icon="🧠", use_container_width=True)
    st.divider()
    
    st.subheader("逐項審閱報告")
    for topic, report_md in st.session_state.comparison_results.items():
        with st.expander(f"**審查項目：{topic.split(' (')[0]}**", expanded=True):
            st.markdown(report_md, unsafe_allow_html=True)
