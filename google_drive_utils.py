import streamlit as st
import os
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaIoBaseUpload
import io

# --- 常數設定 ---
# 這個範圍表示我們的應用程式只會請求建立新檔案的權限，不會讀取或刪除用戶的其他檔案。
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def get_google_credentials_from_secrets():
    """從 st.secrets 讀取 Google 憑證"""
    return st.secrets.get("google_credentials", {})

def initialize_flow():
    """初始化 Google OAuth 流程"""
    creds = get_google_credentials_from_secrets()
    if not all(k in creds for k in ["client_id", "client_secret", "redirect_uri"]):
        st.error("Google 憑證未在 secrets.toml 中正確設定。")
        return None

    client_config = {
        "web": {
            "client_id": creds["client_id"],
            "client_secret": creds["client_secret"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/oauth2/v4/token",
            "redirect_uris": [creds["redirect_uri"]],
        }
    }
    return Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=creds["redirect_uri"])

def get_auth_url():
    """產生授權 URL，讓使用者點擊以進行授權"""
    flow = initialize_flow()
    if flow:
        auth_url, state = flow.authorization_url(access_type='offline', prompt='consent')
        st.session_state['oauth_state'] = state
        return auth_url
    return None

def process_oauth_callback():
    """處理從 Google 重新導向回來時的授權碼"""
    flow = initialize_flow()
    if not flow:
        return

    try:
        # 從 URL query params 獲取授權碼
        auth_code = st.query_params.get("code")
        if auth_code:
            # 檢查 state 以防止 CSRF 攻擊
            if st.query_params.get("state") != st.session_state.get('oauth_state'):
                st.error("State Mismatch. 授權過程可能不安全。")
                return

            flow.fetch_token(code=auth_code)
            # 將憑證儲存在 session state 中，以便後續使用
            st.session_state['google_credentials'] = {
                'token': flow.credentials.token,
                'refresh_token': flow.credentials.refresh_token,
                'token_uri': flow.credentials.token_uri,
                'client_id': flow.credentials.client_id,
                'client_secret': flow.credentials.client_secret,
                'scopes': flow.credentials.scopes
            }
            # 清除 URL 中的授權碼，避免重複觸發
            st.query_params.clear()
            st.rerun() # 重新整理頁面以更新 UI 狀態
    except Exception as e:
        st.error(f"處理授權時發生錯誤: {e}")


def get_drive_service():
    """使用儲存的憑證建立 Google Drive API 服務物件"""
    if 'google_credentials' not in st.session_state:
        return None
    creds_dict = st.session_state['google_credentials']
    credentials = Credentials.from_authorized_user_info(creds_dict, SCOPES)
    return build('drive', 'v3', credentials=credentials)

def upload_report_to_drive(report_content, filename="contract_analysis_report.md"):
    """將報告內容上傳到 Google Drive"""
    service = get_drive_service()
    if not service:
        st.warning("無法連接到 Google Drive，請先授權。")
        return None

    # 將 Markdown 報告內容轉換為位元組流
    report_bytes = report_content.encode('utf-8')
    media = io.BytesIO(report_bytes)

    file_metadata = {'name': filename}
    media_body = MediaIoBaseUpload(media, mimetype='text/markdown', resumable=True)

    try:
        with st.spinner(f"正在上傳 '{filename}' 至 Google Drive..."):
            file = service.files().create(
                body=file_metadata,
                media_body=media_body,
                fields='id, webViewLink'
            ).execute()

        st.success(f"報告已成功儲存！")
        st.markdown(f"📄 **[{filename}]({file.get('webViewLink')})**")
        return file.get('webViewLink')
    except Exception as e:
        st.error(f"上傳檔案時發生錯誤: {e}")
        return None
