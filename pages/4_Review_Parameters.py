import os
import tempfile
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# LangChain / Vector DB
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader # <--- [新增] 為了讀取暫存檔
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever

# --- [新增] 導入 S3 儲存與 Pinecone 學習的工具函式庫 ---
import storage_utils as storage
from utils import ingest_docs_to_pinecone


# --- 頁面設定 ---
st.set_page_config(page_title="AI 合約初審", layout="wide", page_icon="📝")
st.logo("logo.png")


# --- 環境變數與核心設定 ---
load_dotenv()
# --- [新增] AI 學習用的核心設定 ---
LEARNING_NAMESPACE = "approved-analyses"
INDEX_NAME = "contract-assistant"

# --- 【新增】初始化模型參數的 Session State ---
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3 # 建議的預設值
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 2048 # 建議的預設值

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
        st.markdown(f"<span style='color:white'>翻譯時發生錯誤: {e}</span>", unsafe_allow_html=True)
        return text_to_translate

def run_comparison(template_retriever, uploaded_retriever, review_points, temperature, max_tokens):
    """
    執行合約比對，並回傳包含「摘要矩陣」與「詳細報告」的字典。
    """
    llm = ChatOpenAI(model_name='gpt-4o', temperature=temperature, max_tokens=max_tokens)
    
    # --- 高品質摘要矩陣 Prompt ---
    summary_tpl = """
    你是一位頂尖律師事務所的資深法務專家，你的工作是為下方的法律條款製作一份清晰、完整且具體的「摘要分析矩陣」。

    **任務指示:**
     請嚴格確定回答不超過{st.session_state.max_tokens}tokens,避免回答不完整。
    1.  **完整分析差異**:
        * 以「繁體中文」進行比較。
        * 使用「點列式」(bullet points, e.g., `- ...`) 清晰列出兩份條款之間所有具實質影響的差異點。
        * 分析必須具體，包含期限、範圍、義務等關鍵事實的比較。

    2.  **提出具體修訂建議**:
        * 針對上述差異，對風險較高或對我方不利的條款，提出清晰、可執行的修訂建議。
        * 建議應包含「為何要改」以及「建議改成什麼內容」。
        * 同樣使用「點列式」呈現。

    3.  **嚴格格式**:
        * 你的回答「只能」是一行文字。
        * 必須嚴格使用 '|||' 作為分隔符，分隔「差異分析」與「修改建議」。
        * 在 '|||' 的兩邊，你可以自由使用 Markdown 的點列式語法 (`- `) 與換行 (`\n`)。

    **格式範例:**
    `- 對方保密期從「合約終止後」起算，我方為「生效日起算」。\n- 對方保密期長達5年，比我方的3年更長。 ||| - 建議將保密期起算點修改為「資訊揭露日起」，以確保公平。\n- 建議將期限縮短為3年，與我方標準一致，降低我方長期義務。`

    ---
    **待分析資料:**

    **主題**: {topic}

    **條款 A (我方範本)**:
    ```{clause_A}```

    **條款 B (對方文件)**:
    ```{clause_B}```
    ---

    **請立即產生符合上述所有要求的摘要內容:**
    """
    summary_chain = PromptTemplate.from_template(summary_tpl) | llm | StrOutputParser()
    
    # --- 詳細報告 Prompt ---
    tpl = """
**Role:** You are a seasoned Senior Legal Counsel at a top-tier professional services firm, acting to protect the interests of **"Our Company"**. Your review must be commercially-aware, risk-focused, and provide immediately actionable advice for a non-lawyer project team.

**Objective:** Conduct a detailed preliminary review of a counterparty's contract clause ("Clause B") against Our Company's standard template ("Clause A") on the specific topic of "**{topic}**".

---
**Context 1: Past High-Quality Analysis Examples (for style and depth reference)**
```{approved_examples}```
---
**Context 2: Clause A (Our Company's Standard Template - Normalized to Traditional Chinese)**
```{clause_A}```
---
**Context 3: Clause B (Counterparty's Draft - Normalized to Traditional Chinese)**
```{clause_B}```
---

**Task:** Generate a review report in Markdown. You MUST use the exact headings provided below and address all bullet points within each section.

### 1. 核心條款摘要 (Clause Summary)
-   **我方範本 (Clause A)**: Summarize the core purpose and mechanism of our standard clause in one sentence.
-   **對方草案 (Clause B)**: Summarize the core purpose and mechanism of their proposed clause in one sentence.

### 2. 關鍵差異與風險分析 (Key Differences & Risk Analysis)
-   **實質差異點 (Material Differences)**: Using bullet points, identify *all* significant differences in obligations, timelines, scope, or definitions between the two clauses. Be specific and quantitative (e.g., "30 days vs. 60 days," "includes affiliates vs. does not").
-   **對我方商業風險 (Business Risk to Our Company)**: For each difference identified, explain the potential legal, financial, or operational risk it poses *specifically to Our Company*. Frame it as "This exposes us to the risk of...".

### 3. 具體修訂與談判建議 (Actionable Revision & Negotiation Suggestions)
-   **建議修訂文字 (Suggested Redline)**: Provide a direct, copy-pasteable revision to Clause B to mitigate the identified risks and align it closer to our position in Clause A. If no change is needed, state "建議接受 (Acceptable as is)".
-   **談判底線與策略 (Negotiation Points & Bottom Line)**: Briefly state our primary negotiation goal (e.g., "Our main goal is to cap liability at...") and a fallback position if our primary suggestion is rejected.
"""
    analysis_chain = PromptTemplate.from_template(tpl) | llm | StrOutputParser()
    
    detailed_results = {}
    summary_points = []
    progress = st.progress(0, text="AI 法務專家正在審閱合約...")

    mq_template_retriever = MultiQueryRetriever.from_llm(retriever=template_retriever, llm=llm)
    mq_uploaded_retriever = MultiQueryRetriever.from_llm(retriever=uploaded_retriever, llm=llm)
    
    for i, topic in enumerate(review_points):
        display_topic = topic.split(' (')[0].replace('&nbsp;', ' ').strip()
        search_query = topic.replace('&nbsp;', ' ')
        progress.progress((i + 1) / len(review_points), text=f"正在分析: {display_topic}")

        t_docs = mq_template_retriever.get_relevant_documents(search_query)
        u_docs = mq_uploaded_retriever.get_relevant_documents(search_query)
        
        a_text = "無相關範例"
        t_text_original = "\n---\n".join([d.page_content for d in t_docs])
        u_text_original = "\n---\n".join([d.page_content for d in u_docs])
        
        lang_t = get_language(t_text_original, llm)
        lang_u = get_language(u_text_original, llm)
        
        with st.spinner(f"正在進行語言正規化 ({display_topic})..."):
            t_text_final = translate_to_chinese(t_text_original, llm) if 'en' in lang_t else t_text_original
            u_text_final = translate_to_chinese(u_text_original, llm) if 'en' in lang_u else u_text_original

        if not t_text_final.strip(): t_text_final = "文件中未找到相關條款"
        if not u_text_final.strip(): u_text_final = "文件中未找到相關條款"
            
        summary_raw = summary_chain.invoke({
            "topic": display_topic,
            "clause_A": t_text_final,
            "clause_B": u_text_final
        })
        try:
            parts = summary_raw.strip().split('|||')
            if len(parts) == 2:
                difference, suggestion = [p.strip() for p in parts]
            else:
                difference, suggestion = "無法生成摘要", "格式錯誤"
        except Exception:
            difference, suggestion = "摘要生成失敗", "處理時發生錯誤"

        summary_points.append({
            'topic': display_topic,
            'difference': difference,
            'suggestion': suggestion
        })

        report = analysis_chain.invoke({
            "topic": display_topic,
            "approved_examples": a_text,
            "clause_A": t_text_final,
            "clause_B": u_text_final
        })
        detailed_results[topic] = report
        
    progress.empty()
    return {"summary": summary_points, "details": detailed_results}

@st.cache_resource
def load_and_process_pdf_for_faiss(_uploaded_file):
    if not _uploaded_file:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_uploaded_file.getvalue())
        path = tmp.name
    loader = PyPDFLoader(path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)
    if not split_docs:
        os.remove(path)
        st.info(f"文件 '{_uploaded_file.name}' 中沒有可處理的文字內容。")
        return None
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(split_docs, embeddings)
    os.remove(path)
    return vs.as_retriever(search_kwargs={'k': 3})

def process_and_store_reference_file(uploaded_file):
    filename = uploaded_file.name
    with st.spinner(f"正在處理參考文件 '{filename}' 並載入記憶體..."):
        retriever = load_and_process_pdf_for_faiss(uploaded_file)
        if retriever:
            st.session_state.reference_retrievers[filename] = retriever
            st.success(f"參考文件 '{filename}' 已成功載入！")

# --- UI 部分 (前半部維持不變) ---

if "reference_retrievers" not in st.session_state:
    st.session_state.reference_retrievers = {}

st.header("AI 合約初審與風險分析 AI-Assisted Contract Preliminary Review and Risk Analysis")

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
with st.expander("自訂審查項目 Customize Review Parameters", expanded=True):
    cols = st.columns(2)
    for i, point in enumerate(CORE_REVIEW_POINTS):
        with cols[i % 2]:
            st.toggle(point.split(" (")[0], value=True, key=point)
    st.text_area("新增審查項目（每行一個)：", key="core_points_text", height=100, placeholder="例如：\n賠償責任上限 (Limitation of Liability)\n合約的可轉讓性 (Assignability)")
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.header("步驟一：上傳參考文件")
    new_ref_file = st.file_uploader("上傳 PDF 作為新的比對基準", type="pdf", key="ref_uploader_faiss")
    if st.button("處理並載入參考文件"):
        if new_ref_file:
            process_and_store_reference_file(new_ref_file)
        else:
            st.info("請先選擇一個參考文件。")
with col2:
    st.header("步驟二：選擇比對基準")
    processed_files = list(st.session_state.reference_retrievers.keys())
    selected_index = None
    if st.session_state.get("selected_namespace") in processed_files:
        selected_index = processed_files.index(st.session_state.get("selected_namespace"))
    selected = st.selectbox(
        "從已上傳的參考文件中選擇一份：",
        options=processed_files,
        index=selected_index,
        placeholder="請選擇..."
    )
    if selected is not None: 
        st.session_state.selected_namespace = selected
st.divider()

# --- 模型參數設定改為獨立的步驟三 ---
st.header("步驟三：設定 AI 分析參數")
st.session_state.temperature = st.slider(
    "參數溫度 Temperature", 0.0, 1.0, st.session_state.temperature, 0.05,
    help='數值較低，結果會更具體和一致；數值較高，結果會更有創意和多樣性。'
)
st.session_state.max_tokens = st.slider(
    "最大字元數 Max Tokens", 256, 4096, st.session_state.max_tokens, 128,
    help='限制單次 AI 回應的長度。較長的報告可能需要較高的數值。'
)
st.divider()

# --- 原步驟三改為步驟四 ---
st.header("步驟四：上傳待審文件並執行分析")
selected_namespace = st.session_state.get("selected_namespace")
if not selected_namespace:
    st.info("請在上方步驟一上傳參考文件，並在步驟二選擇一份作為比對基準。")
else:
    st.success(f"當前比對基準為： **{selected_namespace}**")

target_file = st.file_uploader("上傳您要審查的合約文件", type="pdf", key="target_uploader")

if target_file: 
    st.session_state.target_file_name = target_file.name

start_button = st.button("開始 AI 深度審閱", type="primary", use_container_width=True, disabled=(not target_file or not selected_namespace))

if start_button:
    with st.spinner("正在準備比對環境..."):
        template_retriever = st.session_state.reference_retrievers[selected_namespace]
        uploaded_retriever = load_and_process_pdf_for_faiss(target_file)
        
    if not uploaded_retriever:
        st.info("待審文件處理失敗或內容為空，請重新上傳。")
    else:
        temp = st.session_state.temperature
        max_tok = st.session_state.max_tokens
        
        active_review_points = [p for p in CORE_REVIEW_POINTS if st.session_state.get(p, True)]
        custom_points = [line.strip() for line in st.session_state.get("core_points_text", "").split('\n') if line.strip()]
        final_review_points = active_review_points + custom_points

        if not final_review_points:
            st.info("請至少選擇或新增一個審查項目。")
        else:
            st.session_state.comparison_results = run_comparison(template_retriever, uploaded_retriever, final_review_points, temp, max_tok)
            st.rerun()

# --- 整合第五頁的報告顯示與儲存功能 ---
if st.session_state.get("comparison_results"):
    st.balloons()
    st.header("✅ AI 深度審閱報告已完成")
    st.info("您可以檢視下方的摘要與報告，若您認為這份報告品質優良，可將其歸檔用於 AI 再學習。")
    st.divider()

    # --- 報告預覽 ---
    st.subheader("風險摘要總覽")
    
    summary_data = st.session_state.comparison_results['summary']
    details_data = st.session_state.comparison_results['details']

    # 建立一個完整的 Markdown 字串用於後續儲存
    full_report_md = "# AI 合約審閱報告\n\n"
    
    # 摘要表格
    summary_table_md = f"| **項目** | **主要差異** | **核心修改建議** |\n"
    summary_table_md += "|:---|:---|:---|\n"
    for item in summary_data:
        # 為了顯示和儲存，我們需要處理換行
        difference_display = item['difference'].replace('\n', '<br>')
        suggestion_display = item['suggestion'].replace('\n', '<br>')
        summary_table_md += f"| {item['topic']} | {difference_display} | {suggestion_display} |\n"
    
    st.markdown(summary_table_md, unsafe_allow_html=True)
    full_report_md += "## 風險摘要總覽\n\n" + summary_table_md.replace('<br>', '\n') + "\n\n"
    st.divider()
    
    # 詳細報告
    st.subheader("逐項審閱報告")
    full_report_md += "## 逐項審閱報告\n\n"
    for topic, report_md in details_data.items():
        with st.expander(f"**審查項目：{topic.split(' (')[0]}**", expanded=True):
            st.markdown(report_md, unsafe_allow_html=True)
        full_report_md += f"### 審查項目：{topic.split(' (')[0]}\n\n{report_md}\n\n---\n\n"
        
    st.divider()
    
    # --- [新增] 歸檔與學習功能 ---
    st.subheader("🧠 分析歸檔與 AI 再學習")
    st.markdown("若您認可這份報告的分析品質，可以點擊下方按鈕，系統會將其歸檔至 Amazon S3，並將其內容作為一個完整的「優良範例」餵給 AI 進行學習。")

    if st.button("✅ 我認可這份報告的品質，歸檔至雲端並用於 AI 學習", type="primary", use_container_width=True):
        # 準備要儲存的內容和檔名
        template_name = st.session_state.get("selected_namespace", "template").replace('.pdf', '')
        target_name = st.session_state.get("target_file_name", "target").replace('.pdf', '')
        timestamp = datetime.now().strftime('%Y%m%d')
        
        storage_filename = f"Approved_Report_{template_name}_vs_{target_name}_{timestamp}.md"
        
        # 1. 呼叫 S3 的上傳函式
        upload_success = storage.upload_report_to_storage(full_report_md, filename=storage_filename)

        # 2. 確保雲端上傳成功後才進行學習
        if upload_success:
            try:
                with st.spinner(f"正在將報告知識轉化為 AI 的長期記憶..."):
                    # 準備餵給 AI 的文字內容，加上標題以提供上下文
                    learning_content = f"【優良分析案例：合約審閱報告 - {template_name} vs {target_name}】\n\n{full_report_md}"
                    
                    # 寫入暫存檔
                    with tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8", suffix=".txt") as tmp_file:
                        tmp_file.write(learning_content)
                        tmp_file_path = tmp_file.name
                    
                    # 使用 TextLoader 載入並上傳至 Pinecone
                    loader = TextLoader(tmp_file_path)
                    documents = loader.load()
                    ingest_docs_to_pinecone(documents, INDEX_NAME, LEARNING_NAMESPACE)
                    os.remove(tmp_file_path)
                    
                    st.success(f"AI 已成功學習此份報告的分析模式！")
                    
                    # 處理完成後，清空 session_state 並顯示成功訊息
                    st.session_state.comparison_results = None
                    st.header("處理完成！優質報告已成功歸檔並用於 AI 再學習。")
                    st.info("頁面即將刷新...")
                    st.rerun()

            except Exception as e:
                st.error(f"在 AI 學習過程中發生錯誤: {e}")
