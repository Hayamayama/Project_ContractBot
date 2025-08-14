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
from langchain.retrievers import EnsembleRetriever  # <--- 新增導入
from pinecone import Pinecone

# --- 頁面設定 ---
st.set_page_config(page_title="可自訂的審查項目", layout="wide")
st.logo("logo.png")

load_dotenv()
INDEX_NAME = "contract-assistant"
LEARNING_NAMESPACE = "approved-analyses"  # 定義優質分析庫的名稱

# -----------------------------
# Helpers
# -----------------------------

# 【重大升級】: 更新 run_comparison 函式以使用 EnsembleRetriever 和新 Prompt
def run_comparison(template_retriever, approved_retriever, uploaded_retriever, review_points, temperature, max_tokens):
    llm = ChatOpenAI(model_name='gpt-4o', temperature=temperature, max_tokens=max_tokens)

    # 【大腦升級】: 新的 Prompt，增加了「優質分析範例」區塊
    tpl = """
你是一位頂尖的法務專家，你的任務是精準比較兩份合約中的同一條款，並以保護「我方公司」的利益為最高原則。

為了幫助你做得更好，除了原始的合約範本外，我還提供了一些過去被人類專家認可的【優質分析範例】。
請學習這些範例的分析邏輯、風險提示方式與建議的語氣，以產出同樣高品質的分析報告。

---
【優質分析範例】(如果有的話):
```{approved_examples}```
---
【我方公司的標準範本條款】:
```{template_clause}```
---
【待審文件的對應條款】:
```{uploaded_clause}```
---

請針對「{topic}」這個審查重點，完成以下任務：
1. **條款摘要**, 2. **差異分析**, 3. **風險提示與建議**。

請用 Markdown 格式清晰地呈現你的分析報告。
"""
    prompt = PromptTemplate.from_template(tpl)
    chain = prompt | llm | StrOutputParser()
    results = {}
    progress = st.progress(0, text="開始進行比對...")

    # 【引擎升級】: 建立一個能同時查找「範本」和「優質分析」的集成檢索器
    # 如果優質分析庫存在，則使用集成檢索器；否則退回舊版單一檢索器
    if approved_retriever:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[template_retriever, approved_retriever],
            weights=[0.6, 0.4]  # 權重分配，讓原始範本稍微重要一些
        )
    else:
        ensemble_retriever = template_retriever

    for i, topic in enumerate(review_points):
        # 使用集成檢索器來查找相關的「範本」和「優質分析」
        ensemble_docs = ensemble_retriever.invoke(topic)

        # 將檢索到的文件分類
        t_docs, a_docs = [], []
        for doc in ensemble_docs:
            # 假設優質分析的 namespace 中繼資料被正確設定
            if doc.metadata.get('namespace') == LEARNING_NAMESPACE:
                a_docs.append(doc)
            else:
                t_docs.append(doc)

        u_docs = uploaded_retriever.invoke(topic)

        # 組合 Prompt 需要的文字
        t_text = "\n---\n".join([d.page_content for d in t_docs])
        u_text = "\n---\n".join([d.page_content for d in u_docs])
        a_text = "\n---\n".join([d.page_content for d in a_docs]) if a_docs else "無相關範例"

        results[topic] = chain.invoke({
            "topic": topic,
            "approved_examples": a_text,
            "template_clause": t_text,
            "uploaded_clause": u_text
        })
        progress.progress((i + 1) / len(review_points), text=f"正在分析: {topic}")
    progress.empty()
    return results

# ... (其他 helper 函式 load_and_process_pdf_for_faiss, fetch_pinecone_namespaces 等保持不變) ...
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

@st.cache_resource
def get_pinecone_client():
    """快取 Pinecone 連線，避免重複初始化。"""
    return Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

def fetch_pinecone_namespaces(index_name):
    """從 Pinecone 獲取所有已存在的 Namespaces 列表。"""
    pc = get_pinecone_client()
    try:
        # 確保 pc 物件不是 None
        if pc:
            index_stats = pc.describe_index(index_name).stats
            return list(index_stats.namespaces.keys()) if index_stats and index_stats.namespaces else []
        return []
    except Exception as e:
        st.error(f"獲取 Pinecone namespaces 時發生錯誤: {e}")
        return []

# --- UI 部分 ---
if "processed_namespaces" not in st.session_state:
    st.session_state.processed_namespaces = fetch_pinecone_namespaces(INDEX_NAME) or []

# ... (UI上半部分保持不變) ...
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
        st.session_state.processed_namespaces.index(st.session_state.get("selected_namespace"))
        if st.session_state.get("selected_namespace") in st.session_state.processed_namespaces
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
selected_namespace = st.session_state.get("selected_namespace")
if not selected_namespace:
    st.info("請在上方步驟二選擇一份參考文件作為比對基準。")
else:
    st.success(f"當前比對基準為： **{selected_namespace}**")

target_file = st.file_uploader("上傳您要審查的合約文件 (PDF)", type="pdf", key="target_uploader_main")
start_button = st.button(
    "開始自動比對與分析",
    type="primary",
    use_container_width=True,
    disabled=(not target_file),
    key="start_compare_btn"
)

# 【重大升級】: 在執行分析時，同時傳入「優質分析庫」的檢索器
if start_button and target_file and selected_namespace:
    with st.spinner("正在準備比對環境..."):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # 1. 建立原始範本的檢索器
        template_retriever = PineconeVectorStore(
            index_name=INDEX_NAME, embedding=embeddings, namespace=selected_namespace
        ).as_retriever(search_kwargs={'k': 2})

        # 2. 建立待審文件的檢索器
        uploaded_retriever = load_and_process_pdf_for_faiss(target_file)

        # 3. 嘗試建立「優質分析庫」的檢索器
        approved_retriever = None
        if LEARNING_NAMESPACE in st.session_state.processed_namespaces:
            approved_retriever = PineconeVectorStore(
                index_name=INDEX_NAME, embedding=embeddings, namespace=LEARNING_NAMESPACE
            ).as_retriever(search_kwargs={'k': 2})
            st.info("💡 已載入過往的優質分析範例，本次分析品質將更高。")

    temp = st.session_state.get('temperature', 0.7)
    max_tok = st.session_state.get('max_tokens', 4096) # 增加 token 限制以容納更長的 prompt

    active_review_points = [p for p in CORE_REVIEW_POINTS if st.session_state.get(p, False)]
    custom_points = [line.strip() for line in st.session_state.get("core_points_text", "").split('\n') if line.strip()]
    final_review_points = active_review_points + custom_points

    if not final_review_points:
        st.error("請至少選擇或新增一個審查項目。")
    else:
        # 將所有檢索器傳入 run_comparison 函式
        st.session_state.comparison_results = run_comparison(
            template_retriever,
            approved_retriever,  # <--- 傳入新的檢索器
            uploaded_retriever,
            final_review_points,
            temp,
            max_tok
        )
        st.rerun()


# 在報告生成後，引導使用者前往新的歸檔頁面
if st.session_state.get("comparison_results"):
    st.balloons()
    st.header("✅ 分析報告已生成！")
    st.info("您可以初步檢視下方的分析結果。完整的歸檔與 AI 學習流程，請前往下一個頁面操作。")

    st.page_link("pages/5_Analysis_saving.py", label="下一步：前往「分析歸檔與學習」頁面", icon="🧠", use_container_width=True)

    st.divider()
    
    st.subheader("分析結果預覽")
    for topic, md in st.session_state.comparison_results.items():
        with st.expander(topic, expanded=False):
            st.markdown(md)
