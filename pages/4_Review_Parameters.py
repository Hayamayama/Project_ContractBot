import os
import tempfile
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# LangChain / Vector DB
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
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
    st.session_state.max_tokens = 3072 # 建議的預設值，因為詳細報告需要較多空間

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

# --- [MODIFIED] 核心比對函式已完全重構 ---
def run_comparison(template_retriever, uploaded_retriever, review_points, temperature, max_tokens):
    """
    執行合約比對，採用「先分析、後摘要」的兩步驟高品質生成流程。
    """
    llm = ChatOpenAI(model_name='gpt-4o', temperature=temperature, max_tokens=max_tokens)
    
    # --- STEP 1: 優化後的高品質詳細報告 Prompt ---
    # 這個 Prompt 是整個分析的核心，專注於產生具有商業洞察的深度分析。
    tpl = """
**Role:** You are a seasoned Senior Legal Counsel at EY. Your primary duty is to protect EY's interests. Your review must be commercially-aware, risk-focused, and provide immediately actionable advice for our internal non-lawyer project teams.

**Objective:** Conduct a detailed preliminary review of a counterparty's contract clause ("Clause B") against our standard template ("Clause A") on the specific topic of "**{topic}**". Assume "Our Company" is EY.

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

**Task & Formatting Rules:**
1.  **Language:** The entire report MUST be written in the language of original clause.
2.  **Headings:** Use Markdown level 3 headings (`###`) for the two main sections (e.g., `### 1. 核心差異與對我方 (EY) 的風險`).
3.  **Bullet Points:** Use a single dash (`- `) for all bullet points. Do not use asterisks (`*`) or circles (`o`).
4.  **Content:** Address all points with insightful, concise analysis based on the provided clauses.

### 1. 核心差異與對我方 (EY) 的風險 (Key Differences & Risks to EY)
-   **核心差異點 (Material Differences)**: Directly compare Clause A and B. Instead of just listing facts, synthesize the differences.
    * *Good Example:* `對方草案將保密義務延長至合約終止後5年，而我方範本僅為2年，大幅增加了我方長期的法律遵循成本與風險。`
    * *Bad Example:* `條款A是2年，條款B是5年。`
-   **對 EY 的潛在風險 (Potential Risks to EY)**: For each key difference, explicitly state the commercial, legal, or operational risk. Frame it as "這將使我方面臨...的風險" (This exposes us to the risk of...). Be specific to EY's business model (e.g., regulatory duties, data handling, global firm structure).

### 2. 修訂與談判策略建議 (Revision & Negotiation Strategy)
-   **首選修訂建議 (Primary Redline Suggestion)**: Provide a direct, copy-pasteable revision to Clause B to mitigate the risks. If no change is truly needed, state "建議接受 (Acceptable as is)".
-   **談判策略與底線 (Negotiation Strategy & Bottom Line)**:
    * **談判目標 (Goal):** Clearly state our main goal (e.g., "主要目標是將保密期限縮短至不超過3年").
    * **理由闡述 (Rationale):** Provide a brief, commercially-sound reason we can use in negotiations (e.g., "向對方說明，2-3年是行業標準，過長的期限不符合比例原則且增加雙方管理成本").
    * **後備方案 (Fallback Position):** Offer a potential compromise if our primary suggestion is rejected (e.g., "若對方堅持5年，我方可接受，但要求增加『不包含我方為遵循法規或專業準則而必須保留的資料』之豁免條款").
"""
    analysis_chain = PromptTemplate.from_template(tpl) | llm | StrOutputParser()
    
    detailed_results = {}
    progress = st.progress(0, text="AI 法務專家正在深度審閱合約...")

    mq_template_retriever = MultiQueryRetriever.from_llm(retriever=template_retriever, llm=llm)
    mq_uploaded_retriever = MultiQueryRetriever.from_llm(retriever=uploaded_retriever, llm=llm)
    
    # --- 執行詳細分析的迴圈 ---
    for i, topic in enumerate(review_points):
        display_topic = topic.split(' (')[0].replace('&nbsp;', ' ').strip()
        search_query = topic.replace('&nbsp;', ' ')
        progress.progress((i + 0.5) / len(review_points), text=f"正在深度分析: {display_topic}")

        # --- [優化建議] 啟動 Few-Shot Learning ---
        # 這裡可以加入邏輯，從 Pinecone 的 LEARNING_NAMESPACE 中檢索與 topic 相關的優良範例
        # a_text = search_for_approved_examples(topic) 
        # 目前暫時維持原樣
        a_text = "無相關範例"

        t_docs = mq_template_retriever.get_relevant_documents(search_query)
        u_docs = mq_uploaded_retriever.get_relevant_documents(search_query)
        
        t_text_original = "\n---\n".join([d.page_content for d in t_docs])
        u_text_original = "\n---\n".join([d.page_content for d in u_docs])
        
        lang_t = get_language(t_text_original, llm)
        lang_u = get_language(u_text_original, llm)
        
        with st.spinner(f"正在進行語言正規化 ({display_topic})..."):
            t_text_final = translate_to_chinese(t_text_original, llm) if 'en' in lang_t else t_text_original
            u_text_final = translate_to_chinese(u_text_original, llm) if 'en' in lang_u else u_text_original

        if not t_text_final.strip(): t_text_final = "文件中未找到相關條款"
        if not u_text_final.strip(): u_text_final = "文件中未找到相關條款"
            
        report = analysis_chain.invoke({
            "topic": display_topic,
            "approved_examples": a_text,
            "clause_A": t_text_final,
            "clause_B": u_text_final
        })
        detailed_results[topic] = report
        
    # --- STEP 2: [NEW] 在所有詳細報告生成後，進行高品質摘要 ---
    progress.progress(1.0, text="正在提煉風險摘要總覽...")

    full_detailed_report_context = "\n\n---\n\n".join(
        f"### 審查項目：{topic.split(' (')[0]}\n\n{report}" 
        for topic, report in detailed_results.items()
    )

    # 新的、專門用於從高品質報告中生成摘要的 Prompt
    final_summary_tpl = """
    你是一位頂尖的法務協理，你的任務是閱讀下方的「逐項審閱報告全文」，並為高階主管製作一份極度精簡的「風險摘要總覽」。

    **任務指示:**
    1.  **專注於核心**: 從每一項報告中，提煉出最重要的「核心差異與風險」以及最關鍵的「首選修訂建議」。
    2.  **結果導向**: 摘要應清晰、直接，讓讀者能立刻掌握問題和解決方案。
    3.  **嚴格格式**: 你的回答「只能」是一行文字。使用 '|||' 分隔「主題」、「主要差異與風險」與「核心修改建議」。使用 ';;;' 分隔不同的審查項目。在每個欄位內部，你可以使用 Markdown 的點列式語法 (`- `) 與換行 (`\\n`)。

    **格式範例:**
    `合約的保密期限|||- 對方草案的保密期長達5年，大幅增加我方長期遵循風險。\\n- 起算點為合約終止後，對我方不利。|||- 建議將期限縮短為 EY 標準的2年。\\n- 建議修改起算點為「資訊揭露日」。;;;機密資訊的定義範圍|||- 對方定義過於寬泛，可能將公開資訊也納入。|||- 建議加入我方範本中的五大標準例外情況。`

    ---
    **逐項審閱報告全文:**
    ```{full_report}```
    ---
    **請立即產生符合上述所有要求的摘要內容:**
    """
    final_summary_prompt = PromptTemplate.from_template(final_summary_tpl)
    # 使用一個溫度較低的獨立 LLM 來確保摘要的穩定性
    summary_llm = ChatOpenAI(model_name='gpt-4o', temperature=0.1, max_tokens=2048)
    summary_chain = final_summary_prompt | summary_llm | StrOutputParser()
    
    summary_raw = summary_chain.invoke({"full_report": full_detailed_report_context})
    
    # 解析新格式的摘要
    summary_points = []
    try:
        items = summary_raw.strip().split(';;;')
        for item in items:
            if not item.strip(): continue
            parts = item.strip().split('|||')
            if len(parts) == 3:
                topic, difference, suggestion = [p.strip().replace('\\n', '\n') for p in parts]
                summary_points.append({
                    'topic': topic,
                    'difference': difference,
                    'suggestion': suggestion
                })
            else: # 如果格式不符，做個簡單的降級處理
                 summary_points.append({'topic': item, 'difference': '格式解析失敗', 'suggestion': '請查看詳細報告'})
    except Exception as e:
        st.error(f"生成摘要時發生錯誤: {e}")
        summary_points.append({'topic': '摘要生成失敗', 'difference': '無法解析 AI 回應', 'suggestion': str(e)})

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

# --- UI 部分 (維持不變) ---

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

st.header("步驟三：設定 AI 分析參數")
st.session_state.temperature = st.slider(
    "參數溫度 Temperature", 0.0, 1.0, st.session_state.temperature, 0.05,
    help='數值較低，結果會更具體和一致；數值較高，結果會更有創意和多樣性。建議使用 0.1-0.4 之間的值以獲得穩定且具洞察的分析。'
)
st.session_state.max_tokens = st.slider(
    "最大字元數 Max Tokens", 512, 4096, st.session_state.max_tokens, 128,
    help='限制單次 AI 回應的長度。由於詳細報告內容較多，建議設定在 3000 以上以避免報告被截斷。'
)
st.divider()

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

# --- 報告顯示與儲存功能 (維持不變) ---
if st.session_state.get("comparison_results"):
    st.balloons()
    st.header("✅ AI 深度審閱報告已完成")
    st.info("您可以檢視下方的摘要與報告，若您認為這份報告品質優良，可將其歸檔用於 AI 再學習。")
    st.divider()

    st.subheader("風險摘要總覽")
    
    summary_data = st.session_state.comparison_results.get('summary', [])
    details_data = st.session_state.comparison_results.get('details', {})

    full_report_md = "# AI 合約審閱報告\n\n"
    
    summary_table_md = "| **項目** | **主要差異與風險** | **核心修改建議** |\n"
    summary_table_md += "|:---|:---|:---|\n"
    for item in summary_data:
        # 為了顯示和儲存，我們需要處理換行
        topic_display = item.get('topic', 'N/A')
        difference_display = item.get('difference', '').replace('\n', '<br>')
        suggestion_display = item.get('suggestion', '').replace('\n', '<br>')
        summary_table_md += f"| {topic_display} | {difference_display} | {suggestion_display} |\n"
    
    st.markdown(summary_table_md, unsafe_allow_html=True)
    full_report_md += "## 風險摘要總覽\n\n" + summary_table_md.replace('<br>', '\n') + "\n\n"
    st.divider()
    
    st.subheader("逐項審閱報告")
    full_report_md += "## 逐項審閱報告\n\n"
    for topic, report_md in details_data.items():
        display_topic = topic.split(' (')[0].replace('&nbsp;', ' ').strip()
        with st.expander(f"**審查項目：{display_topic}**", expanded=False):
            st.markdown(report_md, unsafe_allow_html=True)
        full_report_md += f"### 審查項目：{display_topic}\n\n{report_md}\n\n---\n\n"
        
    st.divider()
    
    st.subheader("🧠 分析歸檔與 AI 再學習")
    st.markdown("若您認可這份報告的分析品質，可以點擊下方按鈕，系統會將其歸檔至 Amazon S3，並將其內容作為一個完整的「優良範例」餵給 AI 進行學習。")

    if st.button("✅ 我認可這份報告的品質，歸檔至雲端並用於 AI 學習", type="primary", use_container_width=True):
        template_name = st.session_state.get("selected_namespace", "template").replace('.pdf', '')
        target_name = st.session_state.get("target_file_name", "target").replace('.pdf', '')
        timestamp = datetime.now().strftime('%Y%m%d')
        
        storage_filename = f"Approved_Report_{template_name}_vs_{target_name}_{timestamp}.md"
        
        upload_success = storage.upload_report_to_storage(full_report_md, filename=storage_filename)

        if upload_success:
            try:
                with st.spinner(f"正在將報告知識轉化為 AI 的長期記憶..."):
                    learning_content = f"【優良分析案例：合約審閱報告 - {template_name} vs {target_name}】\n\n{full_report_md}"
                    
                    with tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8", suffix=".txt") as tmp_file:
                        tmp_file.write(learning_content)
                        tmp_file_path = tmp_file.name
                    
                    loader = TextLoader(tmp_file_path)
                    documents = loader.load()
                    ingest_docs_to_pinecone(documents, INDEX_NAME, LEARNING_NAMESPACE)
                    os.remove(tmp_file_path)
                    
                    st.success(f"AI 已成功學習此份報告的分析模式！")
                    
                    st.session_state.comparison_results = None
                    st.info("頁面即將刷新...")
                    # 使用 st.experimental_rerun() 或 st.rerun() 根據您的 Streamlit 版本
                    st.rerun()

            except Exception as e:
                st.error(f"在 AI 學習過程中發生錯誤: {e}")
