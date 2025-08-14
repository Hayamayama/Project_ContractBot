import streamlit as st
from datetime import datetime
import tempfile
import os

# 導入專案的共用函式庫
import google_drive_utils as gdrive
from utils import ingest_docs_to_pinecone
from langchain.document_loaders import TextLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="分析歸檔與學習", layout="wide")
st.logo("logo.png")

st.title("🧠 分析歸檔與 AI 再學習")
st.markdown("在這裡，您可以檢視先前由 AI 生成的分析報告。若您認為某份報告品質優良，可以同意將其儲存，系統會將其歸檔至 Google Drive，並將其內容作為新的「優良範例」餵給 AI 進行學習，以提升未來的分析品質。")
st.info("**操作流程**：勾選您認可的優質報告 → 連接 Google Drive (若尚未連接) → 按下儲存按鈕。")
st.divider()

# --- 2. 核心設定 ---
# AI 學習後要存入的 Pinecone Namespace
LEARNING_NAMESPACE = "approved-analyses"
INDEX_NAME = "contract-assistant" # 確保與您專案中的索引名稱一致

# --- 3. 檢查 session state 中是否有待處理的分析報告 ---
if 'comparison_results' not in st.session_state or not st.session_state.comparison_results:
    st.warning("目前沒有待處理的分析報告。")
    st.page_link("pages/4_Review_Parameters.py", label="點擊這裡前往「可自訂的審查項目」頁面產生一份新的分析報告。", icon="🚀")
else:
    # --- 4. 顯示報告並提供勾選介面 ---
    st.subheader("待歸檔的分析報告")
    st.caption("請勾選您認為分析得當、可作為未來 AI 學習範本的報告項目。")

    # 使用 form 來包裹所有選項和按鈕，避免每次勾選都重新整理頁面
    with st.form("approval_form"):
        # 儲存用戶勾選的項目
        approved_topics = []
        
        # 迭代顯示所有分析結果
        for topic, result_md in st.session_state.comparison_results.items():
            # 顯示可展開的報告內容
            with st.expander(f"審查項目：{topic}", expanded=False):
                st.markdown(result_md, unsafe_allow_html=True)
            
            # 提供勾選框
            if st.checkbox(f"我認可這份「{topic.split(' (')[0]}」的分析品質，同意儲存並用於 AI 學習。", key=f"cb_{topic}"):
                approved_topics.append({
                    "topic": topic.split(' (')[0].replace('&nbsp;', ' '), # 清理 topic 名稱
                    "content": result_md
                })
        
        # Form 的提交按鈕
        submitted = st.form_submit_button("💾 歸檔選定的優質報告", use_container_width=True, type="primary")

    # --- 5. 處理表單提交事件 ---
    if submitted:
        if not approved_topics:
            st.error("您沒有選擇任何要歸檔的報告。")
        else:
            # 檢查 Google Drive 是否已授權
            if 'google_credentials' not in st.session_state:
                st.warning("在儲存之前，請先授權存取 Google Drive。")
                auth_url = gdrive.get_auth_url()
                if auth_url:
                    st.link_button("🔗 連接至 Google Drive", url=auth_url)
            else:
                st.success(f"✅ 已連接 Google Drive。準備處理 {len(approved_topics)} 份選定的報告...")
                
                success_count = 0
                for report in approved_topics:
                    topic = report["topic"]
                    content = report["content"]
                    
                    st.write(f"---")
                    st.write(f"正在處理: **{topic}**")

                    # --- 步驟 A: 歸檔至 Google Drive ---
                    drive_filename = f"Approved_Analysis_{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md"
                    drive_link = gdrive.upload_report_to_drive(content, filename=drive_filename)

                    # --- 步驟 B: 餵給 Pinecone 進行再學習 ---
                    if drive_link: # 確保 Google Drive 上傳成功後才進行學習
                        try:
                            with st.spinner(f"正在將「{topic}」的知識轉化為 AI 的長期記憶..."):
                                # 1. 將報告內容寫入一個暫存文字檔
                                with tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8", suffix=".txt") as tmp_file:
                                    tmp_file.write(f"【優良分析案例：{topic}】\n\n{content}")
                                    tmp_file_path = tmp_file.name

                                # 2. 使用 TextLoader 載入
                                loader = TextLoader(tmp_file_path)
                                documents = loader.load()

                                # 3. 使用 ingest 函式（它會自動切割、向量化並上傳）
                                ingest_docs_to_pinecone(documents, INDEX_NAME, LEARNING_NAMESPACE)
                                
                                # 4. 清理暫存檔
                                os.remove(tmp_file_path)

                                st.success(f"AI 已成功學習「{topic}」的分析模式！")
                                success_count += 1

                        except Exception as e:
                            st.error(f"在 AI 學習「{topic}」的過程中發生錯誤: {e}")

                st.write(f"---")
                if success_count > 0:
                    st.balloons()
                    st.header(f"處理完成！共 {success_count} 份優質報告已成功歸檔並用於 AI 再學習。")
                
                # 處理完畢後，清空 session state 中的舊報告，避免重複提交
                st.session_state.comparison_results = None
                if st.button("返回分析頁面"):
                    st.switch_page("pages/4_Review_Parameters.py")


# --- 授權回呼處理 ---
# 這段邏輯需要放在檔案的主體部分，確保每次頁面載入時都會檢查
if "code" in st.query_params:
    gdrive.process_oauth_callback()
