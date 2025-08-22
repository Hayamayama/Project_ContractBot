# AI 合約審閱與知識進化平台 (ContractBot)

ContractBot 是一個整合 **生成式 AI** 與 **RAG (檢索增強生成)** 架構的智慧平台，旨在徹底改變傳統合約審閱的模式。它的核心理念是建立一個**可持續學習的智慧循環**：不僅僅是分析文件，更是從法務專家的實戰經驗中持續進化，將每一次高品質的審閱成果，轉化為強化未來 AI 分析能力的基石。

此平台透過創新的**雙向量資料庫架構**，巧妙地結合了**即時性分析**的靈活性與**長期性知識管理**的深度，為法務、採購及業務團隊提供前所未有的效率與洞察力。

## ✨ 核心功能

ContractBot 圍繞著一個核心工作流程設計，提供四大主要功能模組：

#### 1. 動態即時比對 (Session-based Comparison)
* **頁面**: `4_Review_Parameters.py`
* **說明**: 此功能專為**單次、臨時性**的合約比對任務設計。使用者可以上傳「我方範本」與「待審文件」，系統會利用 **FAISS** 在記憶體中快速建立向量索引。這確保了比對任務在一個獨立、隔離的環境中高速進行，無需污染永久知識庫，完美兼顧了靈活性與資料安全性。

#### 2. AI 自我進化學習迴圈 (AI Self-Evolution Loop)
* **頁面**: `4_Review_Parameters.py` -> `storage_utils.py` -> `utils.py`
* **說明**: 這是 ContractBot 的核心價值所在。當一份高品質的分析報告在 `Review Parameters` 頁面生成後，使用者可一鍵將其歸檔。
    * **永久歸檔**: 報告的 Markdown 版本會被儲存至 **Amazon S3**，作為可追溯的企業知識資產。
    * **反饋學習**: 報告的核心內容會被萃取，並注入 **Pinecone** 的特定命名空間 (`approved-analyses`)。這使得 AI 能夠從過往的成功案例中學習，持續優化其未來的分析品質。

#### 3. 整合式知識庫問答 (Knowledge Base Q&A)
* **頁面**: `2_Contract_Bot.py`
* **說明**: 一個強大的聊天機器人介面，讓使用者能用自然語言，對儲存在 **Pinecone** 中的**所有永久知識**（包含標準範本、GCO 經驗、已學習的優質報告）進行深度查詢。更支援臨時上傳 PDF，實現**長期知識**與**短期文件**的無縫整合查詢。

#### 4. 自動化風險分類與標註 (Automated Risk Classification)
* **頁面**: `5_Risk_Classification.py`
* **說明**: 使用者可上傳一份合約，系統會自動將其拆解為獨立條款。接著，利用大型語言模型 (LLM) 與預先定義的 `Risk_Knowledge.py` 知識庫進行比對，將每個條款標記為**高風險 (HIGH)** 或 **中風險 (MEDIUM)**，並**精準定位**風險句子、提供分析理由與**完整的修訂建議**，最終可匯出帶有醒目標註的 PDF 文件。

## ⚙️ 應用架構與技術棧

本專案採用先進的 **RAG (Retrieval-Augmented Generation)** 架構，並透過創新的**雙向量資料庫策略**，完美地區分了「即時性分析」與「長期性知識管理」兩種核心需求。



#### 技術棧 (Technology Stack):

* **前端框架**: `Streamlit`
* **AI 核心框架**: `LangChain`, `OpenAI (gpt-4o, text-embedding-3-small)`
* **資料儲存**:
    * **FAISS (暫存/會話向量庫)**: 用於 `Review_Parameters` 頁面的核心比對功能，提供高速、隔離的分析環境。
    * **Pinecone (永久向量庫)**: 作為系統的**長期記憶核心**，儲存標準範本、GCO 審閱經驗及優質分析範例。
    * **Amazon S3 (永久檔案庫)**: 用於歸檔儲存 Markdown 格式的分析報告。
* **文件處理與解析**: `PyMuPDF (fitz)`, `python-docx`, `lxml`, `unstructured`
* **文字切割**: `semantic-text-splitter`, `spaCy`, `RecursiveCharacterTextSplitter`
* **資料驗證**: `pydantic` (用於 `Risk_Classification` 的輸出驗證)

## 📁 專案結構

```
/contractbot/
├── 📄 Homepage.py               # 應用程式首頁/入口
├── 📄 utils.py                   # 共用函式庫 (Pinecone 連線, DOCX 處理)
├── 📄 storage_utils.py           # Amazon S3 儲存相關函式庫
├── 📄 Risk_Knowledge.py          # 風險分類的知識庫與規則定義
├── 📄 text_splitter.py           # 智慧文字切割工具
├── 📁 pages/
│   ├── 📄 1_Admin_Panel.py      # 知識庫管理後台 (管理 Pinecone)
│   ├── 📄 2_Contract_Bot.py      # 整合式知識庫問答機器人
│   ├── 📄 3_Control_Center.py    # 模型參數與查詢歷史控制台
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

#### 步驟三：下載 NLP 模型

`Risk_Classification` 頁面需要 `spaCy` 進行語言偵測與精準文字提取。

```bash
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm
```

#### 步驟四：設定 API 金鑰與雲端服務

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

#### 3. 執行比對與學習 (核心工作流程)

1.  前往 **Review Parameters** 頁面。
2.  **步驟一**: 上傳一份當次分析要使用的「參考文件」（例如：我方標準範本）。
3.  **步驟二**: 從下拉選單中選擇剛剛上傳的文件作為比對基準。
4.  **步驟三**: 依需求調整 AI 分析參數 (溫度、最大 Token 數)。
5.  **步驟四**: 上傳一份您想要審查的「待審文件」（例如：對方草案）。
6.  點擊 **"開始 AI 深度審閱"** 按鈕，等待 AI 生成摘要總覽與逐項分析報告。
7.  報告生成後，在頁面下方找到 **"分析歸檔與 AI 再學習"** 區塊。
8.  若您認可報告品質，點擊 **"我認可這份報告的品質..."** 按鈕。此操作會將報告存入 S3，並同步將其知識存入 **Pinecone**，完成一次完整的學習循環。

#### 4. 進行風險分類

1.  前往 **Risk Classification** 頁面。
2.  上傳合約 PDF，並選擇偏好的文字切割方式與分析模型。
3.  點擊 **"處理並進行 AI 風險分析"**，系統將逐條分析並以卡片形式呈現高/中風險項目。
4.  檢視分析結果，並可將處理完畢的項目標示為 "Resolved"。
5.  點擊 **"匯出含重點標註的 PDF"** 以下載附有 AI 註解的檔案。
