# AI 合約動態比對與知識管理平台 (ContractBot)

這是一個使用 Streamlit、LangChain 和生成式 AI 技術建立的智慧合約分析與知識管理平台。它不僅能比對文件條款差異，更重要的是，它建立了一個**持續學習的循環**：從過往的法務審閱經驗中學習，提供更精準的修改建議，並將高品質的分析結果結構化地保存，反覆用於提升未來 AI 的分析品質。

## 核心功能 ✨

* **動態即時比對 (Session-based Comparison)**: 在 `Review Parameters` 頁面，使用者可以**針對當前任務**，臨時上傳一份「參考文件」與一份「待審文件」。系統會使用 **FAISS** 在記憶體中快速建立向量索引，進行高效的即時比對與分析，無需依賴預先建立的永久知識庫。

* **AI 自我進化學習**:
    * **GCO 實戰經驗**: 在 `Admin Panel`，可上傳包含法務專家 (GCO)「追蹤修訂」和「註解」的 Word 文件。系統會自動提取這些經驗，並存入 **Pinecone** 作為永久知識。
    * **優質報告學習**: 在 `Review Parameters` 頁面產出報告後，使用者可將滿意的分析報告一鍵歸檔。系統會將報告儲存至 **Amazon S3**，並同步將其內容餵給 AI 學習（存入 **Pinecone** 的 `approved-analyses` 命名空間），使其分析能力持續進化。

* **知識庫問答機器人 (Knowledge Base Bot)**:
    `Contract Bot` 提供一個聊天介面，讓使用者能用自然語言對 **Pinecone** 中的永久知識庫（包含所有參考文件、GCO 經驗、已學習的優質報告）進行快速問答。同時，也可以臨時上傳 PDF 檔案，在當前對話中一併查詢。

* **自動化風險分類 (Risk Classification)**:
    在 `Risk Classification` 頁面，使用者可以上傳一份合約 PDF，系統會自動將條款拆解，並利用 AI 模型將每個條款標記為 **高風險 (HIGH)** 或 **中風險 (MEDIUM)**，並提供簡要的理由與標籤。

* **多功能管理介面**:
    * **管理後台 (Admin Panel)**: 統一的後台介面，用於上傳與管理存放在 **Pinecone** 的永久知識庫（GCO 經驗和參考文件），並提供清空索引的危險操作選項。
    * **控制中心 (Control Center)**: 提供模型參數（如溫度、最大字元數）的調整，以及查詢歷史的管理、篩選、釘選與匯出功能。

## 應用架構與運行邏輯 ⚙️

本專案採用了 **RAG (Retrieval-Augmented Generation)** 架構，並透過雙向量資料庫策略，區分了即時分析與長期知識管理。

#### 技術棧 (Technology Stack):

* **前端框架**: Streamlit
* **AI 核心框架**: LangChain, OpenAI (gpt-4o, text-embedding-3-small)
* **資料儲存**:
    * **FAISS (暫存/會話向量庫)**: 用於 `Review_Parameters` 頁面的核心比對功能，在使用者上傳文件後即時建立索引，提供快速、隔離的分析環境。
    * **Pinecone (永久向量庫)**: 作為系統的**長期記憶**。用於儲存「參考文件」、「GCO 審閱經驗」、以及「優質分析範例」。供 `Contract_Bot` 進行跨文件的知識問答。
    * **Amazon S3 (永久檔案庫)**: 用於歸檔儲存由專家認可的 Markdown 格式分析報告，作為永久的企業知識資產。
* **文件處理**: python-docx, PyPDF2, unstructured

## 專案結構 📁

```
/Your_Project_Folder/
├── 📄 Homepage.py               # 應用程式首頁/入口
├── 📄 utils.py                   # 共用函式庫 (Pinecone, DOCX 處理)
├── 📄 storage_utils.py           # S3 儲存相關函式庫
├── 📁 pages/
│   ├── 📄 1_Admin_Panel.py      # 管理後台 (管理 Pinecone)
│   ├── 📄 2_Contract_Bot.py      # 問答機器人 (查詢 Pinecone)
│   ├── 📄 3_Control_Center.py    # 搜尋與參數控制台
│   ├── 📄 4_Review_Parameters.py # 核心分析頁面 (使用 FAISS)
│   └── 📄 5_Risk_Classification.py # 風險分類頁面
├── 📄 .env                        # API 金鑰與環境變數設定檔
├── 📄 requirements.txt            # Python 套件依賴列表
└── 📄 .gitignore                  # Git 忽略清單
```

## 安裝與設定 🛠️

#### 前置需求

* Python 3.9+
* (推薦 for M1/M2/M3 Mac) [Miniforge](https://github.com/conda-forge/miniforge/releases/latest) 以便順利安裝 `faiss-cpu`。

#### 步驟一：建立虛擬環境

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

在本專案的根目錄中，建立一個名為 `.env` 的檔案，並填入您的金鑰。Pinecone 金鑰對於知識庫的長期累積與問答功能仍然是必要的。

**`.env` 檔案:**

```
OPENAI_API_KEY="sk-..."
PINECONE_API_KEY="..."
```

**Streamlit Secrets (`.streamlit/secrets.toml`):**
(若要啟用 S3 歸檔功能)

```toml
[aws]
aws_access_key_id = "YOUR_AWS_ACCESS_KEY_ID"
aws_secret_access_key = "YOUR_AWS_SECRET_ACCESS_KEY"
s3_bucket_name = "your-s3-bucket-name"
s3_region_name = "your-s3-bucket-region"
```

## 如何使用 🚀

#### 1. 啟動應用程式

在終端機中，確認您位於專案根目錄並且虛擬環境已啟動，然後執行：

```bash
streamlit run Homepage.py
```

#### 2. (可選) 建立永久知識庫

* 若要使用 `Contract Bot` 問答功能，請先前往 **Admin Panel** 頁面。
* 上傳「GCO 審閱經驗文件」或「參考文件」，這些文件將被存入 **Pinecone** 作為永久知識。

#### 3. 日常使用（執行比對與學習）

1.  前往 **Review Parameters** 頁面。
2.  **步驟一**: 上傳一份當次分析要使用的「參考文件」。
3.  **步驟二**: 從下拉選單中選擇剛剛上傳的文件作為比對基準。
4.  **步驟三**: 上傳一份您想審查的「待審文件」。
5.  點擊 **"開始 AI 深度審閱"** 按鈕，等待報告產出。
6.  報告生成後，在頁面下方找到 **"分析歸檔與 AI 再學習"** 區塊。
7.  若您認可報告品質，點擊 **"我認可這份報告的品質..."** 按鈕。此舉會將報告存入 S3，並同步將知識存入 **Pinecone**，完成學習循環。

## Streamlit community cloud 網址

https://project-contractbot.streamlit.app

