import streamlit as st
from datetime import datetime
import tempfile
import os

# 【關鍵修改】: 導入 S3 的儲存工具函式庫
import storage_utils as storage
from utils import ingest_docs_to_pinecone
from langchain.document_loaders import TextLoader

# --- 頁面設定 ---
st.set_page_config(page_title="分析歸檔與學習", layout="wide")
st.logo("logo.png")

st.title("🧠 分析歸檔與 AI 再學習")
st.markdown("在這裡，您可以檢視先前由 AI 生成的分析報告。若您認為某份報告品質優良，可以同意將其儲存，系統會將其歸檔至 Amazon S3，並將其內容作為新的「優良範例」餵給 AI 進行學習。")
st.info("**操作流程**：勾選您認可的優質報告 → 按下儲存按鈕。")
st.divider()

# --- 核心設定 ---
LEARNING_NAMESPACE = "approved-analyses"
INDEX_NAME = "contract-assistant"

# --- 檢查是否有待處理的報告 ---
if 'comparison_results' not in st.session_state or not st.session_state.comparison_results:
    st.warning("目前沒有待處理的分析報告。")
    st.page_link("pages/4_Review_Parameters.py", label="點擊這裡前往「可自訂的審查項目」頁面產生一份新的分析報告。", icon="🚀")
else:
    st.subheader("待歸檔的分析報告")
    st.caption("請勾選您認為分析得當、可作為未來 AI 學習範本的報告項目。")

    with st.form("approval_form"):
        approved_topics = []
        for topic, result_md in st.session_state.comparison_results.items():
            with st.expander(f"審查項目：{topic}", expanded=False):
                st.markdown(result_md, unsafe_allow_html=True)
            
            if st.checkbox(f"我認可這份「{topic.split(' (')[0]}」的分析品質，同意儲存並用於 AI 學習。", key=f"cb_{topic}"):
                approved_topics.append({
                    "topic": topic.split(' (')[0].replace('&nbsp;', ' '),
                    "content": result_md
                })
        
        submitted = st.form_submit_button("💾 歸檔選定的優質報告至雲端", use_container_width=True, type="primary")

    if submitted:
        if not approved_topics:
            st.error("您沒有選擇任何要歸檔的報告。")
        else:
            st.info(f"準備處理 {len(approved_topics)} 份選定的報告...")
            success_count = 0
            for report in approved_topics:
                topic = report["topic"]
                content = report["content"]
                
                st.write(f"---")
                st.write(f"正在處理: **{topic}**")

                # 【關鍵修改】: 呼叫 S3 的上傳函式
                storage_filename = f"Approved_Analysis_{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md"
                upload_success = storage.upload_report_to_storage(content, filename=storage_filename)

                # 確保雲端上傳成功後才進行學習
                if upload_success:
                    try:
                        with st.spinner(f"正在將「{topic}」的知識轉化為 AI 的長期記憶..."):
                            with tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8", suffix=".txt") as tmp_file:
                                tmp_file.write(f"【優良分析案例：{topic}】\n\n{content}")
                                tmp_file_path = tmp_file.name
                            
                            loader = TextLoader(tmp_file_path)
                            documents = loader.load()
                            ingest_docs_to_pinecone(documents, INDEX_NAME, LEARNING_NAMESPACE)
                            os.remove(tmp_file_path)
                            st.success(f"AI 已成功學習「{topic}」的分析模式！")
                            success_count += 1
                    except Exception as e:
                        st.error(f"在 AI 學習「{topic}」的過程中發生錯誤: {e}")

            st.write(f"---")
            if success_count > 0:
                st.balloons()
                st.header(f"處理完成！共 {success_count} 份優質報告已成功歸檔並用於 AI 再學習。")
            
            st.session_state.comparison_results = None
            if st.button("返回分析頁面"):
                st.switch_page("pages/4_Review_Parameters.py")
