import streamlit as st
import boto3
from botocore.exceptions import ClientError
import io

def init_s3_client():
    """
    使用儲存在 Streamlit Secrets 中的憑證來初始化 S3 客戶端。
    """
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
            aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
            region_name=st.secrets["aws"]["s3_region_name"]
        )
        return s3_client
    except Exception as e:
        st.error(f"無法連接至 AWS S3，請檢查憑證設定。錯誤訊息: {e}")
        return None

def upload_report_to_storage(report_content, filename="contract_analysis_report.md"):
    """將報告內容上傳到指定的 S3 儲存桶"""
    s3_client = init_s3_client()
    if not s3_client:
        return None

    try:
        bucket_name = st.secrets["aws"]["s3_bucket_name"]
    except KeyError:
        st.error("找不到 S3 儲存桶名稱，請在 secrets.toml 中設定 aws.s3_bucket_name。")
        return None

    # 將 Markdown 報告內容轉換為位元組流
    report_bytes_io = io.BytesIO(report_content.encode('utf-8'))

    try:
        with st.spinner(f"正在上傳 '{filename}' 至 Amazon S3..."):
            s3_client.upload_fileobj(
                report_bytes_io,
                bucket_name,
                filename,
                ExtraArgs={'ContentType': 'text/markdown'}
            )
        
        st.success(f"報告 '{filename}' 已成功儲存至 Amazon S3！")
        return True
    except ClientError as e:
        st.error(f"上傳檔案至 S3 時發生錯誤: {e}")
        return None
