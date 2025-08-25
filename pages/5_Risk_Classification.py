import os
import re
import io
import json
import time
import textwrap
from typing import List, Literal, Optional

import streamlit as st
from pydantic import BaseModel, Field, ValidationError
#from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import fitz

# 假設您的 text_splitter 模組位於專案根目錄
from text_splitter import smart_split

# 為了語言偵測與精準提取，導入 LangChain 與 spaCy
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
import spacy

import Risk_Knowledge

load_dotenv()
# ---------------- UI Configuration ----------------
st.set_page_config(page_title="Contract Risk Classifier", layout="wide")
st.logo("logo.png")

# ---------------- Configuration ----------------
MODEL_NAME = "gpt-4o"
EXTRACTION_MODEL_NAME = "gpt-4o-mini"

# ---------------- Pydantic Schema ----------------
class ClauseRisk(BaseModel):
    """資料結構，包含 risk, reason, 和 suggestion 欄位"""
    clause: str
    risk: Literal["HIGH", "MEDIUM", "NOTICEABLE"]
    risk_sentence: Optional[str] = Field(None)
    reason: str
    suggestion: Optional[str] = Field(None, description="A complete, revised version of the clause text to mitigate risk.")
    tags: List[str] = Field(default_factory=list)

RISK_RUBRIC = Risk_Knowledge.get_risk_rubric_string()

# ---------------- Prompts (核心修改) ----------------
# 用更明確的指令，確保 AI 提供完整的修訂條文
SYSTEM_PROMPT_STAGE1 = f"""
You are a senior legal counsel specializing in contract review. Your sole task is to analyze a **single given contract clause**, classify its risk level, and provide a complete, rewritten version of the clause that is ready for use.

**Analysis Steps (Follow Strictly):**
1.  **Analyze ONLY the Provided Text**: Your entire analysis **MUST** be based **exclusively** on the text of the clause given to you. **It is strictly forbidden to invent, assume, or refer to other clause numbers or topics** (e.g., 'Article 5' or 'indemnification') unless they are explicitly written in the provided text. Your analysis must directly correspond to the content of the input clause.
2.  **Consult Rubric**: Compare the clause against the `Risk Classification Rubric` provided below.
3.  **Determine Risk Level**:
    - Classify as **"HIGH"** or **"MEDIUM"** if it matches a corresponding risk description in the rubric.
    - Classify as **"NOTICEABLE"** only if it does NOT match High/Medium risks but pertains to standard matters (e.g., governing law, confidentiality period).
4.  **Formulate Reason**: In Traditional Chinese (繁體中文), write a concise, clear explanation for your risk classification.
5.  **Provide Full Revision (Crucial Task)**:
    - Provide the revised clause in the **same language** as the original clause.
    - For any "HIGH" or "MEDIUM" risk clause, you **MUST** provide a **complete, standalone, and rewritten version of the clause**. This rewritten clause should mitigate all identified risks and be ready to replace the original text. It is not just a comment, but the full revised text.
    - For "NOTICEABLE" clauses or if the original text is already acceptable, respond with the exact phrase "無需修改".
6.  **Extract Keywords**: Identify 2-4 relevant keywords from the clause.
7.  **Construct JSON**: Assemble your entire analysis into a SINGLE, valid JSON object.

**Risk Classification Rubric:**
---
{RISK_RUBRIC}
---

**Output Format Rules (Strictly Enforced):**
- Your output **MUST** be a single, valid JSON object.
- The JSON keys must be **EXACTLY**: `risk`, `reason`, `suggestion`, `tags`.
- `risk` must be one of "HIGH", "MEDIUM", or "NOTICEABLE".
- `reason` must be in Traditional Chinese.
- `suggestion` **MUST** contain the full, revised clause text in Traditional Chinese, or the exact phrase "無需修改".
- Do **NOT** include the original `clause` in your JSON response.
"""


SYSTEM_PROMPT_STAGE2 = """
You are a legal text analysis assistant. Given a full clause and a reason for its risk, extract the **exact, single sentence (or at most two)** that is the primary source of the risk.
Respond with ONLY the extracted sentence(s). No explanation, no preamble, no quotes.
"""

# ---------------- Helpers (無需修改) ----------------
# pages/5_Risk_Classification.py

def extract_text_from_pdf(file_bytes_io: io.BytesIO) -> str:
    """
    [核心修改] 使用 PyMuPDF (fitz) 進行文字提取。
    這確保了提取文字與後續標註搜尋所用的引擎是同一個，
    從而解決了因函式庫差異導致的中文匹配失敗問題。
    """
    doc = fitz.open(stream=file_bytes_io, filetype="pdf")
    full_text = []
    for page in doc:
        full_text.append(page.get_text() or "")
    doc.close()
    return "\n".join(full_text)

@st.cache_data(show_spinner=False)
def get_language(text_snippet: str) -> str:
    if not text_snippet.strip(): return "en"
    try:
        llm = ChatOpenAI(model_name=EXTRACTION_MODEL_NAME, temperature=0)
        prompt = PromptTemplate.from_template("Detect the primary language (ISO 639-1 code, 'en' or 'zh') of this text: ```{text}```")
        chain = prompt | llm | StrOutputParser()
        lang_code = chain.invoke({"text": text_snippet[:500]}).lower()
        return 'zh' if 'zh' in lang_code else 'en'
    except Exception:
        return "en"

def sanitize_risk_data(risk_data: dict) -> dict:
    if "tags" not in risk_data or not isinstance(risk_data.get("tags"), list):
        risk_data["tags"] = []
    raw_risk = str(risk_data.get("risk", "MEDIUM")).upper()
    if "HIGH" in raw_risk:
        risk_data["risk"] = "HIGH"
    elif "MEDIUM" in raw_risk:
        risk_data["risk"] = "MEDIUM"
    elif "NOTICEABLE" in raw_risk or "LOW" in raw_risk or "STANDARD" in raw_risk:
        risk_data["risk"] = "NOTICEABLE"
    else:
        risk_data["risk"] = "MEDIUM"
        risk_data["tags"].append("risk_parse_failed")
    return risk_data

def classify_clause(client: OpenAI, clause: str) -> ClauseRisk:
    resp_stage1 = client.chat.completions.create(
        model=MODEL_NAME, temperature=0.1,
        messages=[{"role": "system", "content": SYSTEM_PROMPT_STAGE1}, {"role": "user", "content": clause}],
        response_format={"type": "json_object"},
    )
    if not resp_stage1.choices or not resp_stage1.choices[0].message or not resp_stage1.choices[0].message.content:
        raise ValueError("AI model returned an empty or invalid response in Stage 1.")
    content_str = resp_stage1.choices[0].message.content
    try:
        match = re.search(r"\{.*\}", content_str, re.DOTALL)
        if match:
            risk_data = json.loads(match.group(0))
        else:
            risk_data = json.loads(content_str)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse JSON from model response: {content_str}")

    risk_data = sanitize_risk_data(risk_data)
    risk_data["clause"] = clause
    risk_data.setdefault("risk_sentence", None)
    risk_data.setdefault("suggestion", "無法生成建議。")

    if risk_data.get("risk") in ["HIGH", "MEDIUM"]:
        user_prompt_stage2 = f"Full Clause:\n```\n{clause}\n```\n\nReason for Risk:\n{risk_data.get('reason', '')}"
        resp_stage2 = client.chat.completions.create(
            model=EXTRACTION_MODEL_NAME, temperature=0,
            messages=[{"role": "system", "content": SYSTEM_PROMPT_STAGE2}, {"role": "user", "content": user_prompt_stage2}],
        )
        if resp_stage2.choices and resp_stage2.choices[0].message and resp_stage2.choices[0].message.content:
            risk_sentence = resp_stage2.choices[0].message.content.strip()
            if risk_sentence and risk_sentence in clause:
                risk_data["risk_sentence"] = risk_sentence
    return ClauseRisk.model_validate(risk_data)

def classify_batch(clauses: List[str]) -> List[ClauseRisk]:
    client = OpenAI()
    out: List[ClauseRisk] = []
    total = max(len(clauses), 1)
    progress = st.progress(0, text="正在初始化 AI 分析...")
    for i, c in enumerate(clauses, start=1):
        progress.progress(min(i / total, 1.0), text=f"正在分析第 {i}/{total} 條款...")
        try:
            out.append(classify_clause(client, c))
        except (ValueError, ValidationError) as e:
            out.append(ClauseRisk(clause=c, risk="NOTICEABLE", reason=f"模型分析或驗證時發生錯誤: {str(e)[:100]}", suggestion="N/A", tags=["error", "parsing_failed"]))
        except Exception as e:
            out.append(ClauseRisk(clause=c, risk="NOTICEABLE", reason=f"發生未預期的系統錯誤: {str(e)[:100]}", suggestion="N/A", tags=["error", "system_error"]))
        time.sleep(0.05)
    progress.empty()
    return out

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

# pages/5_Risk_Classification.py

def _candidate_snippets(text: str, min_len: int = 15, max_len: int = 100) -> List[str]:
    """
    [核心修改] 全新升級的文字片段生成函式，特別強化對中文的支援。
    1.  **辨識中英文標點**：使用包含「。；！？」的全形標點來進行更自然的斷句。
    2.  **滑動窗口切分**：如果一個長句內沒有標點，則採用重疊的「滑動窗口」方式切分，
        確保整句話都會被完整覆蓋到，而不是只取頭尾。
    3.  **優化長度參數**：調整了最小和最大長度，更適合中文的資訊密度。
    """
    t = _normalize_spaces(text)
    if not t:
        return []
    if len(t) <= max_len:
        return [t]

    # 優先嘗試使用中英文標點符號來切分句子
    # (?<=[...]) 是正規表示式中的 "positive lookbehind"，確保標點符號本身不被切掉
    sentences = re.split(r'(?<=[。？！；!?;\.])\s*', t)
    valid_sentences = [s.strip() for s in sentences if s and s.strip() and len(s.strip()) >= min_len]

    # 如果成功按標點切分出多個句子，就使用這個結果
    if len(valid_sentences) > 1 and any(len(s) < max_len for s in valid_sentences):
         return valid_sentences

    # 如果無法靠標點切分 (例如一個沒有標點的超長條款)
    # 則使用重疊的滑動窗口 (sliding window) 方式，確保覆蓋完整內容
    chunks = []
    # 設定重疊20個字元，讓標註更連貫
    overlap = 20 
    step = max_len - overlap

    for i in range(0, len(t), step):
        chunk = t[i:i + max_len]
        if chunk and len(chunk.strip()) > min_len:
            chunks.append(chunk.strip())

    return chunks if chunks else [t]

def _inset(rect, margin: float) -> "fitz.Rect":
    return fitz.Rect(rect.x0 + margin, rect.y0 + margin, rect.x1 - margin, rect.y1 - margin)

def build_highlighted_pdf(src_pdf_bytes: bytes, items: List[ClauseRisk], include_resolved: bool = False, resolved_idx: set = None) -> bytes:
    """PDF 產生邏輯，確保將完整的修訂條文放入註解"""
    if resolved_idx is None: resolved_idx = set()
    doc = fitz.open(stream=src_pdf_bytes, filetype="pdf")
    not_found = []
    for idx, item in enumerate(items):
        if (not include_resolved and idx in resolved_idx) or item.risk not in ["HIGH", "MEDIUM"]:
            continue
        text_to_search = item.risk_sentence if item.risk_sentence else item.clause
        snippets = _candidate_snippets(text_to_search)

        # 【關鍵修改】新增一個旗標，用來追蹤這個風險項目是否已經被找到並標註了
        item_found_and_annotated = False

        for page in doc:
            # 對於每個頁面，我們檢查所有文字片段
            for snip in snippets:
                if not isinstance(snip, str) or len(snip) < 5: continue
                rects = page.search_for(snip, quads=True)

                # 如果找到了任何一個片段
                if rects:
                    # 就執行標註和新增註解
                    annot = page.add_highlight_annot(rects)
                    annot.set_colors(stroke=(1, 0, 0) if item.risk == "HIGH" else (1, 0.55, 0))

                    info_content = f"【風險原因】\n{item.reason}"
                    if item.suggestion and item.suggestion != "無需修改":
                        info_content += f"\n\n【建議修訂版本】\n{item.suggestion}"

                    annot.set_info(content=f"[{item.risk}] {info_content}")
                    annot.update()

                    # 【關鍵修改】將旗標設為 True，表示這個風險項目已處理完畢
                    item_found_and_annotated = True
                    # 【關鍵修改】立刻跳出最內層的「片段(snippet)」迴圈
                    break
            
            # 【關鍵修改】如果這個風險項目已經處理完畢，也跳出中層的「頁面(page)」迴圈
            if item_found_and_annotated:
                break

        # 如果遍歷完所有頁面後，這個風險項目仍然沒有被找到
        if not item_found_and_annotated:
            not_found.append(item)

    if not_found:
        summary_page = doc.new_page()
        header = "Clauses Not Found For Highlighting"
        text_blocks = [header, ""]
        for miss in not_found:
            text_blocks.append(f"- {miss.risk} · {', '.join(miss.tags) or 'untagged'}")
            text_blocks.append(f"  Reason: {miss.reason}")
            text_blocks.append("  " + textwrap.shorten(_normalize_spaces(miss.clause), width=220))
            text_blocks.append("")
        inner_rect = _inset(summary_page.rect, 36)
        summary_page.insert_textbox(inner_rect, "\n".join(text_blocks), fontsize=10, align=0,fontname="china-tc")
    out = io.BytesIO()
    doc.save(out, deflate=True, garbage=4)
    doc.close()
    return out.getvalue()


# ---------------- App UI (無需修改) ----------------
st.header("合約風險評鑑 Contract Risk Classifier")
st.markdown("上傳合約 PDF，AI 將自動標示**高/中**風險條款，**精準定位風險句子**，並附上分析說明與修訂建議。")
if "results" not in st.session_state: st.session_state.results = []
if "resolved" not in st.session_state: st.session_state.resolved = set()
if "source_pdf_bytes" not in st.session_state: st.session_state.source_pdf_bytes = None
left, right = st.columns([2, 1])
with left:
    uploaded = st.file_uploader("上傳您的合約 (.pdf)", type=["pdf"], key="contract_pdf")
    process_clicked = st.button("處理並進行 AI 風險分析", type="primary", use_container_width=True, disabled=uploaded is None)
with right:
    cap = st.number_input("最大可分析條款數", min_value=5, max_value=50, value=20, step=5)
    split_method = st.selectbox("文字切割方式", options=["semantic", "regex", "recursive"], index=0)
    MODEL_NAME = st.selectbox("分析模型 (進階)", options=["gpt-4o", "gpt-4-turbo"], index=0)

if process_clicked:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY 未設定。")
        st.stop()
    if uploaded is None:
        st.warning("請先上傳 PDF。")
        st.stop()

    try:
        pdf_bytes = uploaded.getvalue()
        st.session_state.source_pdf_bytes = pdf_bytes
        text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
    except Exception as e:
        st.error(f"讀取或解析 PDF 文件時發生錯誤: {e}")
        st.stop()

    lang = get_language(text)
    st.caption(f"偵測到文件主要語言為: **{'中文 (zh)' if lang == 'zh' else 'English (en)'}**")
    clauses = smart_split(text, method=split_method)[:cap]
    st.caption(f"使用 '{split_method}' 方式切割出 {len(clauses)} 個條款區塊。")
    if not clauses:
        st.warning("未能在文件中切割出有效的條款，請檢查文件內容或嘗試其他切割方式。")
        st.stop()

    all_results = classify_batch(clauses)

    high_medium_results = [r for r in all_results if r.risk in ["HIGH", "MEDIUM"]]
    noticeable_count = len(all_results) - len(high_medium_results)

    st.session_state.results = high_medium_results
    st.session_state.resolved = set()

    st.session_state.last_run_message = f"分析完成！已識別出 {len(high_medium_results)} 個高/中風險項目。"
    if noticeable_count > 0:
        st.session_state.last_run_message += f" (已自動過濾 {noticeable_count} 個低風險項目)"

    st.rerun()

if "last_run_message" in st.session_state:
    st.success(st.session_state.last_run_message)
    del st.session_state.last_run_message

if st.session_state.results:
    results: List[ClauseRisk] = st.session_state.results
    resolved: set = st.session_state.resolved
    def mark_resolved(idx: int): st.session_state.resolved.add(idx)
    def undo_resolved(idx: int): st.session_state.resolved.discard(idx)
    active_indices = [i for i in range(len(results)) if i not in resolved]
    active_results = [results[i] for i in active_indices]

    counts = {"HIGH": sum(1 for r in active_results if r.risk == "HIGH"),
              "MEDIUM": sum(1 for r in active_results if r.risk == "MEDIUM")}

    st.divider()
    st.subheader("風險摘要 Summary")
    st.write(f"🔴 **高風險: {counts['HIGH']}** · 🟠 **中風險: {counts['MEDIUM']}** · ✅ **已解決: {len(resolved)}**")

    show_resolved = st.checkbox("顯示已解決項目", value=False)

    badge_map = {"HIGH": "🔴 高風險 HIGH RISK", "MEDIUM": "🟠 中風險 MEDIUM RISK"}

    def render_card(idx: int, item: ClauseRisk, is_resolved: bool):
        with st.container(border=True):
            left_col, right_col = st.columns([8, 2])
            with left_col:
                status = "✅ 已解決 RESOLVED" if is_resolved else badge_map.get(item.risk, "🟠 中風險 MEDIUM RISK")
                st.markdown(f"**{status}** · *Tags: {', '.join(item.tags) or '無'}*")
                if item.risk_sentence and not is_resolved:
                    st.markdown("##### 🔑 風險根源句 (Risk Root-Cause Sentence)")
                    st.markdown(f"> {item.risk_sentence}")

                st.markdown("##### 💬 AI 分析與理由 (AI Analysis & Reason)")
                st.info(f"{item.reason}")

                if item.suggestion and item.suggestion != "無需修改" and not is_resolved:
                    st.markdown("##### ✍️ 建議修訂版本 (Suggested Full Revision)")
                    st.success(f"{item.suggestion}")

                with st.expander("檢視完整條款上下文 (View Full Clause Context)"):
                    st.text_area("Clause Text", value=item.clause, height=150, disabled=True, key=f"clause_text_{idx}")

            with right_col:
                if not is_resolved:
                    st.button("Resolved", key=f"resolve_btn_{idx}", help="將此項目標示為已解決", on_click=mark_resolved, args=(idx,), use_container_width=True)
                else:
                    st.button("Undo", key=f"undo_btn_{idx}", help="將此項目移回待處理清單", on_click=undo_resolved, args=(idx,), use_container_width=True)

    for i in active_indices:
        render_card(i, results[i], is_resolved=False)
    if show_resolved and resolved:
        st.markdown("### Resolved Items")
        for i in sorted(list(resolved)):
            render_card(i, results[i], is_resolved=True)

    st.divider()
    st.subheader("匯出重點標記 Export with Highlights (PDF)")
    if st.session_state.source_pdf_bytes:
        include_resolved_pdf = st.toggle("在 PDF 中包含已解決項目", value=False)
        try:
            pdf_bytes = build_highlighted_pdf(
                st.session_state.source_pdf_bytes,
                items=results,
                include_resolved=include_resolved_pdf,
                resolved_idx=resolved
            )
            st.download_button(
                "匯出含重點標註的 PDF",
                data=pdf_bytes, file_name="contract_highlighted.pdf",
                mime="application/pdf", use_container_width=True,
            )
        except Exception as e:
            st.error(f"產生 PDF 標示時發生錯誤: {e}")
