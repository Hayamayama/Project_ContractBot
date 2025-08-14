import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io

# 定義 API 權限範圍
SCOPES = ['https://www.googleapis.com/auth/drive']

def get_drive_service():
    """
    使用儲存在 Streamlit Secrets 中的服務帳號 JSON 金鑰來建立 Google Drive API 服務。
    """
    try:
        creds_json = st.secrets["google_service_account"]
        credentials = service_account.Credentials.from_service_account_info(creds_json, scopes=SCOPES)
        service = build('drive', 'v3', credentials=credentials)
        return service
    except Exception as e:
        st.error(f"無法連接至 Google Drive，請檢查服務帳號憑證設定。錯誤訊息: {e}")
        return None

def upload_report_to_drive(report_content, filename="contract_analysis_report.md"):
    """將報告內容上傳到指定的 Google Drive 資料夾（支援共用雲端硬碟）。"""
    service = get_drive_service()
    if not service:
        return None

    try:
        folder_id = st.secrets["gdrive_config"]["target_folder_id"]
    except KeyError:
        st.error("找不到目標資料夾 ID，請在 secrets.toml 中設定 gdrive_config.target_folder_id。")
        return None

    report_bytes = report_content.encode('utf-8')
    media = io.BytesIO(report_bytes)

    file_metadata = {
        'name': filename,
        'parents': [folder_id]
    }
    media_body = MediaIoBaseUpload(media, mimetype='text/markdown', resumable=True)

    try:
        with st.spinner(f"正在上傳 '{filename}' 至共用雲端硬碟..."):
            # 【關鍵修改】: 新增 supportsAllDrives=True 參數
            # 這個參數告訴 API，我們的應用程式已經準備好處理共用雲端硬碟中的內容。
            file = service.files().create(
                body=file_metadata,
                media_body=media_body,
                fields='id, webViewLink',
                supportsAllDrives=True  # <--- 這就是魔法所在！
            ).execute()
        
        st.success(f"報告已成功儲存至共用雲端硬碟！")
        st.markdown(f"📄 **[{filename}]({file.get('webViewLink')})**")
        return file.get('webViewLink')
    except Exception as e:
        st.error(f"上傳檔案至 Google Drive 時發生錯誤: {e}")
        return None

