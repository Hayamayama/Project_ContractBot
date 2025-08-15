import streamlit as st
from datetime import datetime
import tempfile
import os

# 導入 S3 的儲存工具函式庫和共用函式庫
import storage_utils as storage
from utils import ingest_docs_to_pinecone
from langchain.document_loaders import TextLoader

# --- 頁面設定 ---
st.set_page_config(page_title="分析歸檔與學習", layout="wide")
st.logo("logo.png")

st.title("🧠 分析歸檔與 AI 再學習")
st.markdown("在這裡，您可以檢視先前由 AI 生成的**對照矩陣**。若您認為這份矩陣的整體分析品質優良，可以同意將其儲存，系統會將其歸檔至 Amazon S3，並將其內容作為一個完整的「優良範例」餵給 AI 進行學習。")
st.info("**操作流程**：檢視下方的完整矩陣 → 按下儲存按鈕。")
st.divider()

# --- 核心設定 ---
LEARNING_NAMESPACE = "approved-analyses"
INDEX_NAME = "contract-assistant"

# --- 檢查是否有待處理的報告 ---
# 【修改】: 檢查 comparison_results 是否存在且為非空字串
if 'comparison_results' not in st.session_state or not st.session_state.comparison_results:
    st.warning("目前沒有待處理的分析報告。")
    st.page_link("pages/4_Review_Parameters.py", label="點擊這裡前往「可自訂的審查項目」頁面產生一份新的分析報告。", icon="🚀")
else:
    # --- 報告預覽 ---
    st.subheader("待歸檔的對照矩陣")
    st.markdown(st.session_state.comparison_results, unsafe_allow_html=True)
    st.divider()

    # --- 歸檔表單 ---
    # 【修改】: 從多選項表單簡化為單一確認按鈕
    with st.form("approval_form"):
        submitted = st.form_submit_button(
            "✅ 我認可這份對照矩陣的品質，歸檔至雲端並用於 AI 學習",
            use_container_width=True,
            type="primary"
        )

    if submitted:
        # 準備要儲存的內容和檔名
        report_content = st.session_state.comparison_results
        
        # 從 session_state 獲取檔名以建立一個有意義的存檔檔名
        template_name = st.session_state.get("selected_namespace", "template").replace('.pdf', '')
        target_name = st.session_state.get("target_file_name", "target").replace('.pdf', '')
        timestamp = datetime.now().strftime('%Y%m%d')
        
        storage_filename = f"Approved_Matrix_{template_name}_vs_{target_name}_{timestamp}.md"
        
        # 1. 呼叫 S3 的上傳函式
        upload_success = storage.upload_report_to_storage(report_content, filename=storage_filename)

        # 2. 確保雲端上傳成功後才進行學習
        if upload_success:
            try:
                with st.spinner(f"正在將矩陣知識轉化為 AI 的長期記憶..."):
                    # 準備餵給 AI 的文字內容，加上標題以提供上下文
                    learning_content = f"【優良分析案例：合約對照矩陣 - {template_name} vs {target_name}】\n\n{report_content}"
                    
                    # 寫入暫存檔
                    with tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8", suffix=".txt") as tmp_file:
                        tmp_file.write(learning_content)
                        tmp_file_path = tmp_file.name
                    
                    # 使用 TextLoader 載入並上傳至 Pinecone
                    loader = TextLoader(tmp_file_path)
                    documents = loader.load()
                    ingest_docs_to_pinecone(documents, INDEX_NAME, LEARNING_NAMESPACE)
                    os.remove(tmp_file_path)
                    
                    st.success(f"AI 已成功學習此份對照矩陣的分析模式！")
                    
                    # 處理完成後，清空 session_state 並顯示成功訊息
                    st.session_state.comparison_results = None
                    st.balloons()
                    st.header("處理完成！優質報告已成功歸檔並用於 AI 再學習。")
                    if st.button("返回分析頁面"):
                        st.switch_page("pages/4_Review_Parameters.py")
                    
                    st.rerun() # 立即重新整理頁面以顯示 "沒有待處理報告" 的狀態

            except Exception as e:
                st.error(f"在 AI 學習過程中發生錯誤: {e}")
