import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 從我們的共用函式庫中導入函式
from utils import (
    extract_revisions_from_single_doc,
    extract_comments_from_docx,
    ingest_docs_to_pinecone,
    get_pinecone_client
)

st.set_page_config(page_title="管理後台", page_icon="⚙️", layout="wide")
# --- 【修改】: 使用 st.logo() ---
st.logo("logo.png")

st.header("知識庫管理後台 Knowledge Base Admin Dashboard")
st.markdown("在這裡，您可以管理 GCO 經驗或進行系統維護。")

INDEX_NAME = "contract-assistant"


# --- 功能一：上傳 GCO 經驗文件 ---
st.subheader("上傳 GCO 審閱經驗文件 (.docx)")
gco_file = st.file_uploader("選擇包含追蹤修訂或註解的 Word 檔案", type="docx", key="gco_uploader")

gco_namespace = st.text_input(
    "為這份 GCO 經驗指定一個 Namespace",
    value="gco-case-studies",
    help="建議將所有 GCO 經驗文件都存入同一個 Namespace，方便統一檢索。"
)

if st.button("從 GCO 文件提取並儲存經驗"):
    if gco_file and gco_namespace:
        with st.spinner(f"正在從 '{gco_file.name}' 提取 GCO 經驗..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                tmp_file.write(gco_file.getvalue())
                tmp_file_path = tmp_file.name

            # 提取經驗
            word_nsmap = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            revision_chunks = extract_revisions_from_single_doc(tmp_file_path, word_nsmap)
            comment_chunks = extract_comments_from_docx(tmp_file_path)
            all_wisdom_chunks = revision_chunks + comment_chunks

            os.remove(tmp_file_path)

        if all_wisdom_chunks:
            # 將提取出的經驗文字轉換為 LangChain Document 物件
            with tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8") as wisdom_txt:
                wisdom_txt.write("\n".join(all_wisdom_chunks))
                wisdom_txt_path = wisdom_txt.name

            loader = TextLoader(wisdom_txt_path)
            documents = loader.load()

            # 上傳至 Pinecone
            ingest_docs_to_pinecone(documents, INDEX_NAME, gco_namespace)
            os.remove(wisdom_txt_path)
            st.success(f"成功提取並儲存了 {len(all_wisdom_chunks)} 條 GCO 經驗！")
        else:
            st.warning("在文件中未找到可提取的追蹤修訂或註解。")
    else:
        st.error("請上傳檔案並指定 Namespace。")

st.divider()



# --- 功能二：索引管理 (危險區域) ---
with st.expander("🚨 危險區域：索引管理"):
    st.warning("警告：以下操作將會永久刪除知識庫中的資料。")

    if st.button("清空索引內所有資料 (Delete All Vectors)"):
        try:
            pc = get_pinecone_client()
            index = pc.Index(INDEX_NAME)
            index.delete(delete_all=True)
            st.success(f"已成功清空索引 '{INDEX_NAME}' 中的所有資料！")
            # 清除快取，讓主頁的下拉選單能感知到變化
            st.cache_data.clear()
        except Exception as e:
            st.error(f"清空索引時發生錯誤: {e}")
