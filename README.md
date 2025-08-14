# AI 合約動態比對與知識管理平台 (ContractBot)

這是一個使用 Streamlit、LangChain 和生成式 AI 技術建立的智慧合約分析與知識管理平台。它不僅能比對文件條款差異，更重要的是，它建立了一個**持續學習的循環**：從過往的法務審閱經驗中學習，提供更精準的修改建議，並將高品質的分析結果結構化地保存，反覆用於提升未來 AI 的分析品質。

## 核心功能 ✨

* **動態比對基準**: 使用者可以上傳多份合約範本（如公司標準 NDA、採購合約）作為永久的「參考文件」知識庫。在進行比對時，可從知識庫中動態選擇任一份文件作為比對基準。

* **雙重經驗學習 (GCO & Approved Analyses)**:
    * **GCO 實戰經驗**: 可上傳包含法務專家 (GCO)「追蹤修訂」和「註解」的 Word 文件。系統會自動提取這些寶貴的修改經驗，轉化為 AI 的內建知識。
    * **AI 自我進化**: 系統會學習過去被專家認可的「優質分析報告」。在進行新的分析時，AI 會參考這些最佳實踐，產出邏輯更嚴謹、建議更到位的報告，實現持續進化。

* **智慧化條款比對與分析**: 針對使用者自訂的審查項目，系統會運用 `EnsembleRetriever` 技術，同時從「參考文件」、「待審文件」、以及「優質分析範例庫」中提取最相關的條款，交由 AI (GPT-4o) 進行多方比較，最終產出包含「條款摘要」、「差異分析」及「風險提示與建議」的完整報告。

* **多功能應用介面**:
    * **參數與審查 (Review Parameters)**: 核心的合約比對工具，提供審查項目設定、基準選擇與報告產出。
    * **分析歸檔與學習 (Analysis Saving)**: 讓使用者能將高品質的分析報告歸檔至 **Amazon S3**，並同步將其轉化為 AI 的學習材料。
    * **合約問答機器人 (Contract Bot)**: 提供一個聊天介面，讓使用者能用自然語言對所有知識庫（包含參考文件、GCO 經驗、甚至當前上傳的暫存文件）進行快速問答。
    * **管理後台 (Admin Panel)**: 統一的後台介面，用於上傳與管理 GCO 經驗和參考文件，並提供清空索引的危險操作選項。

* **多格式文件支援與匯出**: 支援上傳 `.pdf` 和 `.docx` 等常見文件格式，並可將聊天紀錄匯出為 JSON, TXT, 或 PDF。

## 應用架構與運行邏輯 ⚙️

本專案採用了 **RAG (Retrieval-Augmented Generation)** 架構，並圍繞其建立了一套完整的知識注入、分析、歸檔與再學習的閉環系統。

#### 技術棧 (Technology Stack):

* **前端框架**: Streamlit
* **AI 核心框架**: LangChain, OpenAI (gpt-4o, text-embedding-3-small)
* **資料儲存**:
    * **Pinecone (永久向量庫)**: 用於儲存「參考文件」、「GCO 審閱經驗」、以及「優質分析範例」。每個知識庫都在獨立的 `namespace` 中，便於管理與檢索。
    * **FAISS (暫存向量庫)**: 用於即時處理當前上傳的「待審文件」或聊天時的暫存文件，速度快且資源消耗低。
    * **Amazon S3 (永久檔案庫)**: 用於歸檔儲存由專家認可的 Markdown 格式分析報告，作為永久的企業知識資產。
* **文件處理**: python-docx, PyPDFLoader, unstructured

## 專案結構 📁

一個較完整的專案資料夾結構如下：

```
/Your_Project_Folder/
├── 📄 Homepage.py               # 應用程式首頁/入口
├── 📄 utils.py                   # 共用函式庫 (Pinecone, DOCX 處理)
├── 📄 storage_utils.py           # S3 儲存相關函式庫
├── 📁 pages/
│   ├── 📄 1_Admin_Panel.py      # 管理後台
│   ├── 📄 2_Contract_Bot.py      # 問答機器人
│   ├── 📄 3_Control_Center.py    # 搜尋控制台 (可選)
│   ├── 📄 4_Review_Parameters.py # 核心分析頁面
│   └── 📄 5_Analysis_Saving.py  # 歸檔與學習頁面
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

在本專案的根目錄中，建立一個名為 `.env` 的檔案，並填入您的金鑰。同時，若要使用 S3 功能，請確保 Streamlit 的 `secrets.toml` 設定正確。

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

#### 2. 首次使用（建立知識庫）

* 前往 **Admin Panel** 頁面。
* 使用「上傳 GCO 審閱經驗文件」功能，上傳 Word 文件，為 AI 補充實戰經驗。
* 使用「上傳新的參考文件」功能，上傳您常用的 PDF 合約範本，作為比對基準。

#### 3. 日常使用（執行比對與學習）

1.  前往 **Review Parameters** 頁面。
2.  **步驟二**：從下拉選單中選擇一份「參考文件」作為比對基準。
3.  **步驟三**：上傳一份您想審查的「待審文件」。
4.  點擊 **"開始自動比對與分析"** 按鈕，等待報告產出。
5.  報告生成後，點擊連結前往 **Analysis Saving** 頁面。
6.  **勾選**您認為品質優良的分析項目，並點擊 **"歸檔選定的優質報告至雲端"**。此舉會將報告存入 S3，並同步讓 AI 學習，完成學習循環。
