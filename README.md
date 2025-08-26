# AI 合約動態比對與風險分析工具 (ContractBot)

這是一套運用大型語言模型（LLM）技術，專為法務、採購及業務團隊打造的企業級 AI 合約分析平台。本工具旨在將繁瑣、耗時的人工合約審閱流程，轉化為高效、精準且具備持續學習能力的智慧化工作流程。

透過本平台，使用者不僅能秒速比對合約版本間的差異，更能自動識別潛在風險、獲取專業修訂建議，並建立可持續累積的企業內部法務知識庫。

## 核心功能亮點

-   **📝 AI 驅動合約比對分析 (`pages/1_Compare&Analyze.py`)**
    -   **動態比對**：上傳我方標準範本與待審合約草案，AI 將逐條進行深度語意比對。
    -   **風險摘要矩陣**：自動生成清晰的表格，彙總「主要差異」與「核心修改建議」。
    -   **逐項深度報告**：針對每個審查項目（如保密期限、責任上限），產出包含「條款摘要」、「風險分析」及「具體修訂建議」的詳細報告。
    -   **持續學習**：使用者可將高品質的分析報告一鍵歸檔，系統會將其轉化為 AI 的長期記憶，讓未來的分析更貼近企業需求。

-   **🚨 自動化風險分類與標註 (`pages/2_Automated_Risk_Finder.py`)**
    -   **智慧條款切割**：採用多種先進演算法（語意、正則表達式等）將合約精準切割為獨立條款。
    -   **三級風險評鑑**：基於內部專家知識庫 (`Risk_Knowledge.py`)，自動將條款標示為「高風險」、「中風險」或「需注意」。
    -   **PDF 重點標註**：一鍵生成附有高亮標示與註解（包含風險原因與修訂建議）的 PDF 檔案，方便線下溝通。

-   **🤖 智慧合約聊天機器人 (`pages/3_Contract_ChatBot.py`)**
    -   **整合式知識檢索**：可同時對「永久知識庫 (Pinecone)」與「臨時上傳文件 (FAISS)」進行自然語言查詢。
    -   **多文件對話**：輕鬆實現跨文件提問，例如：「請比較 A 合約與 B 合約在賠償責任上的差異？」
    -   **精準溯源**：所有回答皆會附上來源出處，方便使用者追溯與驗證。

-   **📊 歷程記錄與匯出 (`pages/4_History&Export.py`)**
    -   **查詢歷史追蹤**：自動記錄所有分析操作，方便日後審計與複查。
    -   **釘選與標籤**：使用者可將重要的分析結果釘選或加上標籤，快速定位關鍵資訊。
    -   **一鍵匯出**：支援將查詢歷史或釘選項目匯出為 `JSON` 或 `CSV` 檔案。

-   **⚙️ 系統設定與知識庫管理 (`pages/5_System_Settings.py`)**
    -   **經驗萃取與注入**：可上傳包含「追蹤修訂」或「註解」的 Word 文件，系統會自動提取其中的專家審閱經驗，並注入 AI 知識庫。
    -   **索引庫維護**：提供安全的管理介面，供管理員維護向量資料庫的內容。

## 專案結構 (Project Structure)
```
/contractbot/
├── Homepage.py           # 應用程式首頁/入口
├── pages/
│   ├── 1_Compare&Analyze.py      # 核心功能：AI 驅動合約比對分析
│   ├── 2_Automated_Risk_Finder.py # 核心功能：自動化風險識別
│   ├── 3_Contract_ChatBot.py     # 核心功能：互動式合約問答
│   ├── 4_History&Export.py       # 核心功能：歷程記錄與匯出
│   └── 5_System_Settings.py      # 核心功能：系統設定與知識庫管理
├── utils.py              # 共用函式庫 (Pinecone 連線, DOCX 處理)
├── storage_utils.py      # Amazon S3 儲存相關函式庫
├── Risk_Knowledge.py     # 風險分類的知識庫與規則定義
├── logo.png              # 應用程式 Logo
├── text_splitter.py      # 智慧文字切割工具
├── README.md             # 專案說明文件
├── .env                  # API 金鑰與環境變數設定檔
├── requirements.txt      # Python 套件依賴列表
├── NotoSansTC-Var.ttf    # PDF 輸出用標準字體
└── .gitignore            # Git 忽略清單

```

## 應用架構與技術棧 (Application Architecture & Technology Stack)

### RAG (Retrieval-Augmented Generation) 核心架構

本專案採用先進的 **RAG (檢索增強生成)** 架構，這是一種讓大型語言模型（LLM）的回答更具事實性、精準度和相關性的關鍵技術。

**RAG 的核心價值在於：**

1.  **事實接地 (Grounding in Facts)**：傳統 LLM 可能會憑空捏造資訊（稱為「幻覺」）。RAG 架構透過先從指定的知識庫（例如您的合約文件）中檢索相關段落，再將這些段落作為上下文(Context)提供給 LLM，強制模型基於這些事實來生成回答，大幅提升了回答的可靠性。
2.  **運用私有知識 (Leveraging Private Data)**：企業無須耗費巨資重新訓練一個模型。RAG 讓現有的強大 LLM 能夠直接存取並利用您私有的、即時更新的內部文件，使其成為真正懂您業務的專家。
3.  **可追溯性與透明度 (Traceability & Transparency)**：由於答案是基於檢索到的具體文件內容，我們可以輕鬆追溯到答案的來源，方便使用者驗證資訊的正確性。

#### 本專案的雙核 RAG 策略

為了完美地區分「即時性分析」與「長期性知識管理」兩種核心需求，本專案採用了創新的**雙向量資料庫策略**：

-   **FAISS (暫存/會話向量庫)**
    -   **用途**：用於 `4_Review_Parameters.py` 頁面的核心比對功能。
    -   **特性**：當使用者上傳兩份文件進行即時比對時，系統會在記憶體中快速建立一個臨時的 FAISS 索引。這提供了高速、隔離的分析環境，確保每次比對都是獨立的，且會話結束後即銷毀，保障了資料的隱私性。

-   **Pinecone (永久向量庫)**
    -   **用途**：作為系統級的長期記憶核心。
    -   **特性**：用於儲存標準合約範本、GCO 審閱經驗、以及經使用者認可的優質分析範例。`2_Contract_Bot.py` 等功能會從此處檢索資訊，讓 AI 具備跨時間的學習與記憶能力，越用越聰明。

### 技術棧 (Technology Stack)

-   **前端框架**: `Streamlit`
-   **AI 核心框架**: `LangChain`, `OpenAI (gpt-4o, text-embedding-3-small)`
-   **資料儲存**:
    -   `FAISS` (暫存/會話向量庫)
    -   `Pinecone` (永久向量庫)
    -   `Amazon S3` (永久檔案庫，用於歸檔 Markdown 格式的分析報告)
-   **文件處理與解析**: `PyMuPDF (fitz)`, `python-docx`, `lxml`, `unstructured`
-   **文字切割**: `semantic-text-splitter`, `spaCy`, `RecursiveCharacterTextSplitter`
-   **資料驗證**: `pydantic` (用於 `Risk_Classification` 的輸出驗證)

## 安裝與部署指南

### 1. 前置準備

-   確認已安裝 Python 3.9+
-   取得以下服務的 API Keys：
    -   `OPENAI_API_KEY`
    -   `PINECONE_API_KEY`
    -   `AWS_ACCESS_KEY_ID` (用於報告歸檔)
    -   `AWS_SECRET_ACCESS_KEY` (用於報告歸檔)

### 2. 本地端設定

**A. 複製專案**
```bash
git clone [https://github.com/hayamayama/project_contractbot.git](https://github.com/hayamayama/project_contractbot.git)
cd project_contractbot
```

**B. 安裝依賴套件**
```bash
pip install -r requirements.txt
```

**C. 下載 spaCy 語言模型**
本專案使用 spaCy 進行更精準的文字處理，請執行以下指令下載所需模型：
```bash
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm
```

**D. 設定環境變數**
在專案根目錄下建立一個名為 `.env` 的檔案，並填入以下內容：

```env
# .env

# OpenAI API Key
OPENAI_API_KEY="sk-..."

# Pinecone Configuration
PINECONE_API_KEY="..."
PINECONE_INDEX_NAME="contract-assistant" # 您在 Pinecone 建立的索引名稱

# AWS S3 Configuration (用於報告歸檔與 AI 學習功能)
# 若無此需求可留空，但相關功能會受限
AWS_ACCESS_KEY_ID="..."
AWS_SECRET_ACCESS_KEY="..."
S3_BUCKET_NAME="..."
S3_REGION_NAME="..." # e.g., "ap-northeast-1"
```

### 3. 啟動應用程式

在專案根目錄下，執行以下指令：

```bash
streamlit run Homepage.py
```

應用程式將會在您的本地瀏覽器中開啟 (通常是 `http://localhost:8501`)。

## 使用流程概覽

1.  **【首次使用】建立知識庫基準**
    -   前往 **管理後台 (System_Settings)**，上傳公司內部的標準合約範本或過去已審閱過的案例（.docx），建立 AI 檢索的知識基礎。
    -   也可以在 **AI 合約初審 (Compare&Analyze)** 頁面上傳參考文件，作為即時比對的基準。

2.  **【日常審閱】執行合約比對**
    -   前往 **AI 合約初審 (Compare&Analyze)** 頁面。
    -   **步驟一 & 二**：上傳並選擇一份「參考文件」作為比對基準。
    -   **步驟三**：(可選) 調整 AI 分析的溫度 (Temperature) 與最大字元數 (Max Tokens)。
    -   **步驟四**：上傳「待審文件」，點擊「開始 AI 深度審閱」。
    -   稍待片刻，系統將生成完整的風險摘要與逐項分析報告。

3.  **【進階應用】與文件對話**
    -   前往 **合約聊天機器人 (Contract_ChatBot)**。
    -   在側邊欄選擇您想查詢的知識庫（可複選）或臨時上傳 PDF。
    -   在對話框中直接用自然語言提問，例如：「保密義務在合約終止後持續多久？」

4.  **【快速掃描】自動風險分類**
    -   前往 **風險評鑑 (Automated_Risk_Finder)** 頁面。
    -   上傳一份合約 PDF，點擊分析。
    -   系統會自動列出所有高、中風險條款，並提供下載附有重點標註的 PDF 檔案。

## 授權

©2025 Ernst & Young LLP. All Rights Reserved. (詳見頁尾宣告)
