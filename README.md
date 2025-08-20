# AI 合約審閱與知識進化平台 (ContractBot)

這是一個基於 Streamlit、LangChain 及生成式 AI 技術打造的智慧合約分析與知識管理平台。其核心設計理念不僅是比對文件差異，而是建立一個**可持續學習的智慧循環**：系統能從過往的法務審閱實戰經驗中學習，提供更精準的修訂建議，並將高品質的分析報告結構化地歸檔，反覆用於提升未來 AI 的分析品質與準確性。

## ✨ 核心功能

* **動態即時比對 (Session-based Comparison)**:
    在 `Review Parameters` 頁面，使用者可**針對單次任務**，臨時上傳一份「我方範本」與一份「待審文件」。系統利用 **FAISS** 在記憶體中快速建立向量索引，進行高效的即時比對與深度分析，無需依賴預先建立的永久知識庫，確保了分析的靈活性與即時性。

* **AI 自我進化學習迴圈 (AI Self-Evolution Loop)**:
    * **GCO 實戰經驗導入**: 在 `Admin Panel`，可上傳包含法務專家 (GCO)「追蹤修訂」和「註解」的 Word 文件。系統會自動解析並提取這些寶貴的實戰經驗，存入 **Pinecone** 作為永久的專業知識。
    * **優質報告反饋學習**: 在 `Review Parameters` 頁面產出高品質報告後，使用者可一鍵將其歸檔。系統會將報告的 Markdown 版本儲存至 **Amazon S3** 作為永久記錄，並同步將其核心內容餵給 AI 學習（存入 **Pinecone** 的特定 `approved-analyses` 命名空間），使其分析能力與日俱增。

* **整合式知識庫問答機器人 (Knowledge Base Bot)**:
    `Contract Bot` 提供一個直觀的聊天介面，讓使用者能用自然語言，對 **Pinecone** 中的永久知識庫（包含所有參考範本、GCO 實戰經驗、已學習的優質報告）進行快速問答。同時，也可以臨時上傳 PDF 檔案，在當前的對話中進行混合查詢，實現了長期知識與短期文件的無縫整合。

* **自動化風險分類 (Automated Risk Classification)**:
    在 `Risk Classification` 頁面，使用者可上傳一份合約 PDF。系統會自動將其拆解為獨立條款，並利用大型語言模型，根據預先定義的 `Risk_Knowledge.py` 知識庫，將每個條款標記為 **高風險 (HIGH)** 或 **中風險 (MEDIUM)**，並提供簡潔的判斷理由與關鍵字標籤。

* **多功能控制與管理介面**:
    * **管理後台 (Admin Panel)**: 統一的後台，用於上傳與管理存放在 **Pinecone** 的永久知識庫（GCO 經驗和參考文件），並提供清空索引等系統維護功能。
    * **控制中心 (Control Center)**: 提供模型參數（如 Temperature、Max Tokens）的即時調整，以及查詢歷史的瀏覽、篩選、釘選與多格式（JSON, CSV）匯出功能。

## ⚙️ 應用架構與運行邏輯

本專案採用了先進的 **RAG (Retrieval-Augmented Generation)** 架構，並透過創新的**雙向量資料庫策略**，完美地區分了「即時性分析」與「長期性知識管理」兩種核心需求。

#### 技術棧 (Technology Stack):

* **前端框架**: Streamlit
* **AI 核心框架**: LangChain, OpenAI (`gpt-4o`, `text-embedding-3-small`)
* **資料儲存**:
    * **FAISS (暫存/會話向量庫)**: 用於 `Review_Parameters` 頁面的核心比對功能。在使用者上傳文件後即時於記憶體中建立索引，提供高速、隔離的分析環境，任務結束後即釋放。
    * **Pinecone (永久向量庫)**: 作為系統的**長期記憶核心**。用於儲存「標準參考文件」、「GCO 審閱經驗」、以及經認可的「優質分析範例」，供 `Contract_Bot` 進行跨文件的深度知識問答。
    * **Amazon S3 (永久檔案庫)**: 用於歸檔儲存由專家認可的 Markdown 格式分析報告，作為永久、可追溯的企業知識資產。
* **文件處理**: `python-docx`, `PyPDF2`, `unstructured`, `lxml`
* **核心套件**: `pydantic` (用於 `Risk_Classification` 的資料驗證)

## 📁 專案結構

```
/contractbot/
├── 📄 Homepage.py               # 應用程式首頁/入口
├── 📄 utils.py                   # 共用函式庫 (Pinecone 連線, DOCX 處理, 資料上傳)
├── 📄 storage_utils.py           # Amazon S3 儲存相關函式庫
├── 📄 Risk_Knowledge.py          # 風險分類的知識庫與規則定義
├── 📁 pages/
│   ├── 📄 1_Admin_Panel.py      # 管理後台 (上傳與管理 Pinecone 知識)
│   ├── 📄 2_Contract_Bot.py      # 知識庫問答機器人 (查詢 Pinecone + FAISS)
│   ├── 📄 3_Control_Center.py    # 搜尋歷史與模型參數控制台
│   ├── 📄 4_Review_Parameters.py # 核心比對分析頁面 (使用 FAISS)
│   └── 📄 5_Risk_Classification.py # 合約條款風險自動分類
├── 📄 .env                        # API 金鑰與環境變數設定檔
├── 📄 requirements.txt            # Python 套件依賴列表
└── 📄 .gitignore                  # Git 忽略清單
```

## 🛠️ 安裝與設定

#### 前置需求

* Python 3.9+
* (推薦 for M1/M2/M3 Mac) [Miniforge](https://github.com/conda-forge/miniforge/releases/latest) 以確保能順利安裝 `faiss-cpu`。

#### 步驟一：建立並啟用虛擬環境

```bash
# 使用 venv (推薦)
python -m venv venv
source venv/bin/activate

# 或者使用 Conda
conda create -n contract_env python=3.10 -y
conda activate contract_env
```

#### 步驟二：安裝所有依賴套件

```bash
pip install -r requirements.txt
```

#### 步驟三：設定 API 金鑰與雲端服務

1.  **本地環境變數**: 在專案根目錄中，建立一個名為 `.env` 的檔案，並填入您的金鑰。

    **`.env` 檔案範例:**
    ```
    OPENAI_API_KEY="sk-..."
    PINECONE_API_KEY="..."
    ```

2.  **Streamlit Secrets (用於部署)**: 若要啟用 S3 歸檔功能，您需要在 Streamlit Cloud 的設定中配置 Secrets，或是在本地建立 `.streamlit/secrets.toml` 檔案。

    **`.streamlit/secrets.toml` 檔案範例:**
    ```toml
    [aws]
    aws_access_key_id = "YOUR_AWS_ACCESS_KEY_ID"
    aws_secret_access_key = "YOUR_AWS_SECRET_ACCESS_KEY"
    s3_bucket_name = "your-s3-bucket-name"
    s3_region_name = "your-s3-bucket-region"
    ```

## 🚀 如何使用

#### 1. 啟動應用程式

在終端機中，確認您位於專案根目錄並且虛擬環境已啟動，然後執行：

```bash
streamlit run Homepage.py
```

#### 2. (可選) 建立永久知識庫

* 若要使用 `Contract Bot` 問答功能，或讓比對分析時能參考過往的 GCO 經驗，請先前往 **Admin Panel** 頁面。
* 上傳「GCO 審閱經驗文件」或「標準參考文件」，這些文件將被處理並存入 **Pinecone** 作為永久知識。

#### 3. 日常使用（執行比對與學習）

1.  前往 **Review Parameters** 頁面。
2.  **步驟一**: 上傳一份當次分析要使用的「參考文件」（例如：我方標準範本）。
3.  **步驟二**: 從下拉選單中選擇剛剛上傳的文件作為比對基準。
4.  **步驟三**: 上傳一份您想要審查的「待審文件」（例如：對方草案）。
5.  點擊 **"開始 AI 深度審閱"** 按鈕，等待 AI 生成摘要總覽與逐項分析報告。
6.  報告生成後，在頁面下方找到 **"分析歸檔與 AI 再學習"** 區塊。
7.  若您認可報告品質，點擊 **"我認可這份報告的品質..."** 按鈕。此操作會將報告存入 S3，並同步將其知識存入 **Pinecone**，完成一次完整的學習循環。

## Streamlit community cloud 網址

https://project-contractbot.streamlit.app

