# AI 合約動態比對與知識管理平台 (ContractBot)

這是一個使用 Streamlit、LangChain 和生成式 AI 技術建立的智慧合約分析工具。它不僅可以比對兩份文件的條款差異，還能從過往的法務審閱經驗中學習，提供更貼近公司風格的修改建議，並將這些寶貴的經驗結構化地保存下來。

## 核心功能 ✨

* **動態比對基準**: 使用者可以上傳多份合約範本（如公司標準 NDA、採購合約）作為永久的「參考文件」知識庫。在進行比對時，可從知識庫中動態選擇任一份文件作為比對基準。

* **GCO 經驗學習**: 可上傳包含法務專家 (GCO)「追蹤修訂」和「註解」的 Word 文件。系統會自動提取這些寶貴的修改經驗，轉化為 AI 的內建知識，讓 AI 的建議更貼近真實世界的審閱風格與公司立場。

* **自動化條款比對與分析**: 針對使用者自訂的核心審查項目，系統會自動從「參考文件」與「待審文件」中提取最相關的條款，並交由 AI（GPT-4o）進行成對比較，最終產出包含「條款摘要」、「差異分析」及「風險提示與建議」的完整報告。

* **多格式文件支援**: 支援上傳 `.pdf` 和 `.docx` 等常見文件格式。

* **雙頁面應用介面**:

  * **主頁 (Home)**: 核心的合約比對工具，提供介紹、參數設定與分析報告產出。

  * **管理後台 (Admin Panel)**: 統一的後台介面，用於上傳與管理 GCO 經驗和參考文件，並提供清空索引的危險操作選項。

* **友善的使用者體驗**: 具備專案介紹頁面、可自訂的審查項目、搜尋歷史紀錄，以及將分析結果匯出為 CSV 或 JSON 的功能。

## 技術介紹與程式運行邏輯 ⚙️

本專案採用了 RAG (Retrieval-Augmented Generation) 架構，結合了向量資料庫的快速檢索與大型語言模型的強大生成能力。

#### 技術棧 (Technology Stack):

* **前端框架**: Streamlit

* **語言模型與框架**: LangChain, OpenAI (gpt-4o, text-embedding-3-small)

* **向量資料庫 (Vector Store)**:

  * **Pinecone**: 用於永久儲存「參考文件」和「GCO 審閱經驗」的向量化資料。每個參考文件或 GCO 案例庫會被指定一個獨立的 `namespace` 以作區隔。

  * **FAISS**: 用於暫時儲存當前上傳的「待審文件」的向量化資料，僅在該次會話中使用，比對完成後即釋放。

* **文件處理**: python-docx, PyPDFLoader, lxml

#### 程式運行邏輯:

1. **知識庫建立 (Admin Panel)**:

   * **上傳 GCO 經驗 (.docx)**:

     1. 使用者在管理後台上傳帶有「追蹤修訂」或「註解」的 Word 檔案。

     2. 系統使用 `python-docx` 和 `lxml` 套件解析 Word 文件，精準提取出修訂前後的文字對比，以及註解對應的原文。

     3. 提取出的經驗會被整理成結構化的文字片段，並存入指定的 Pinecone `namespace` (預設為 `gco-case-studies`)。

   * **上傳參考文件 (.pdf)**:

     1. 使用者上傳 PDF 格式的合約範本。

     2. 系統使用 `PyPDFLoader` 讀取文件內容，並透過 `RecursiveCharacterTextSplitter` 將其切割成適當大小的文字區塊 (chunks)。

     3. 這些文字區塊會被轉換為向量，並以該**檔案名稱**作為 `namespace` 存入 Pinecone 的 `contract-assistant` 索引中。

2. **合約比對與分析 (Home)**:

   1. **選擇基準**: 使用者在左側邊欄從 Pinecone 中已有的 `namespace` 列表裡，選擇一份「參考文件」作為比對基準。

   2. **上傳待審文件**: 使用者上傳一份待審的 PDF 合約。這份文件會被即時處理並載入到記憶體中的 FAISS 向量索引，以便快速檢索。

   3. **定義審查重點**: 使用者可以勾選預設的審查項目（如保密期限、管轄法院等），或手動輸入客製化的審查重點。

   4. **智慧檢索 (Retrieval)**:

      * 針對每一個審查重點，系統會分別在 Pinecone（參考文件）和 FAISS（待審文件）中進行向量相似度搜尋。

      * 此步驟會找出與該審查重點最相關的合約條款區塊。

   5. **AI 生成 (Generation)**:

      * 系統將檢索到的「成對條款」（參考條款與待審條款）以及「審查重點」填入一個預設的 Prompt Template 中。

      * 這個完整的 Prompt 被發送到 OpenAI GPT-4o 模型，指令 AI 扮演法務專家，從保護公司利益的角度，產出結構化的分析報告。

   6. **呈現報告**: AI 生成的分析結果會以清晰的卡片形式，分項呈現在 Streamlit 的主畫面上。

## 專案結構 📁

請確保您的專案資料夾包含以下結構：

```
/ContractBot_Project/
├── 📄 Home_final.py             # 主應用程式 (建議更名為 Home.py)
├── 📄 utils.py                 # 共用函式庫
├── 📁 pages/
│   └── 📄 1_Admin_Panel.py    # 管理後台頁面
├── 📄 .env                      # API 金鑰與環境變數設定檔
├── 📄 requirements.txt          # Python 套件依賴列表
└── 📄 .gitignore                # Git 忽略清單
```

## 安裝與設定 🛠️

#### 前置需求

* Python 3.9+

* (推薦 for M1/M2/M3 Mac) [Miniforge](https://github.com/conda-forge/miniforge/releases/latest) 以便順利安裝 `faiss-cpu`。

#### 步驟一：建立虛擬環境

在您的終端機中，進入 `ContractBot_Project` 資料夾，並建立一個虛擬環境。

```bash
# 使用 venv (推薦)
python -m venv venv
source venv/bin/activate

# 或者使用 Conda
conda create -n contract_env python=3.10 -y
conda activate contract_env
```

#### 步驟二：安裝所有依賴套件

執行以下指令來安裝 `requirements.txt` 中列出的所有套件：

```bash
pip install -r requirements.txt
```

#### 步驟三：設定 API 金鑰

在本專案的根目錄 (`ContractBot_Project/`) 中，建立一個名為 `.env` 的檔案。複製以下內容並填入您自己的金鑰：

```
OPENAI_API_KEY="sk-..."
PINECONE_API_KEY="..."
```

## 如何使用 🚀

#### 1. 啟動應用程式

在終端機中，確認您位於 `ContractBot_Project` 資料夾並且虛擬環境已啟動，然後執行：

```bash
# 如果您的主檔案是 Home_final.py，建議先將其改名為 Home.py
# mv Home_final.py Home.py
streamlit run Home.py
```

您的瀏覽器將會自動打開應用程式介面。

#### 2. 首次使用（建立知識庫）

* 在左側導覽列選擇 **Admin Panel** 頁面。

* 使用「上傳 GCO 審閱經驗文件」功能，上傳包含 GCO 註解和修訂的 Word 文件，為 AI 補充實戰經驗。

* 使用「上傳新的參考文件」功能，上傳您常用的合約範本（例如公司標準 NDA），這些將成為未來比對的基準。

#### 3. 日常使用（執行比對）

* 返回主應用程式 **Home** 頁面。

* 在左側邊欄的**步驟二**，從下拉選單中選擇一份您想用來當作比對基準的「參考文件」。

* 在主畫面的**步驟三**，上傳一份您想要審查的「待審文件」。

* 點擊 **"🚀 開始自動比對與分析"** 按鈕，等待系統產出分析報告。


## Streamlit Community Cloud 網址

### https://project-contractbot.streamlit.app