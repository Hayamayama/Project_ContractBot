import os
import re
import io
import json
import time
import textwrap
from typing import List, Literal, Optional

import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from PyPDF2 import PdfReader
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
    clause: str
    risk: Literal["HIGH", "MEDIUM", "NOTICEABLE"]
    risk_sentence: Optional[str] = Field(None)
    reason: str
    tags: List[str] = Field(default_factory=list)

RISK_RUBRIC = Risk_Knowledge.get_risk_rubric_string()

# ---------------- Prompts ----------------
SYSTEM_PROMPT_STAGE1 = f"""
You are a meticulous contract risk analyst. Your task is to analyze a clause and classify its risk.
**Analysis Steps:**
1.  **Analyze**: Identify core legal and commercial implications.
2.  **Consult Rubric**: Compare against the `Risk Classification Rubric`.
3.  **Determine Risk Level**:
    - If it matches a "High" or "Medium" risk description, classify it as **"HIGH"** or **"MEDIUM"**.
    - If the clause does **NOT** match High/Medium risks but matches a "Noticeable Clause" description (e.g., standard governing law, confidentiality period), classify it as **"NOTICEABLE"**.
4.  **Formulate Reason**: Write a concise explanation in Traditional Chinese (繁體中文) for your classification.
5.  **Extract Keywords**: Identify 2-4 keywords.
6.  **Construct JSON**: Assemble into a SINGLE JSON object.
**Risk Classification Rubric:**
---
{RISK_RUBRIC}
---
**Output Format Rules (Strictly Enforced):**
- Output MUST be a single, valid JSON object.
- Keys must be EXACTLY: `risk`, `reason`, `tags`.
- `risk` must be "HIGH", "MEDIUM", or "NOTICEABLE".
- Do NOT return the original `clause`.
"""

SYSTEM_PROMPT_STAGE2 = """
You are a legal text analysis assistant. Given a full clause and a reason for its risk, extract the **exact, single sentence (or at most two)** that is the primary source of the risk.
Respond with ONLY the extracted sentence(s). No explanation, no preamble, no quotes.
"""

# ---------------- Helpers ----------------
def extract_text_from_pdf(file_bytes_io: io.BytesIO) -> str:
    reader = PdfReader(file_bytes_io)
    return "\n".join((page.extract_text() or "") for page in reader.pages)

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
            out.append(ClauseRisk(clause=c, risk="NOTICEABLE", reason=f"模型分析或驗證時發生錯誤: {str(e)[:100]}", tags=["error", "parsing_failed"]))
        except Exception as e:
            out.append(ClauseRisk(clause=c, risk="NOTICEABLE", reason=f"發生未預期的系統錯誤: {str(e)[:100]}", tags=["error", "system_error"]))
        time.sleep(0.05)
    progress.empty()
    return out

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _candidate_snippets(text: str, min_len: int = 20, max_len: int = 120) -> List[str]:
    t = _normalize_spaces(text)
    if not t: return []
    if len(t) <= max_len: return [t]
    sentences = re.split(r'(?<=[.?!;:])\s+', t)
    valid_sentences = [s.strip() for s in sentences if len(s.strip()) >= min_len]
    if valid_sentences:
        return [s[:max_len] for s in valid_sentences]
    return [t[:max_len], t[-max_len:]]

def _inset(rect, margin: float) -> "fitz.Rect":
    return fitz.Rect(rect.x0 + margin, rect.y0 + margin, rect.x1 - margin, rect.y1 - margin)

def build_highlighted_pdf(src_pdf_bytes: bytes, items: List[ClauseRisk], include_resolved: bool = False, resolved_idx: set = None) -> bytes:
    if resolved_idx is None: resolved_idx = set()
    doc = fitz.open(stream=src_pdf_bytes, filetype="pdf")
    not_found = []
    for idx, item in enumerate(items):
        if (not include_resolved and idx in resolved_idx) or item.risk not in ["HIGH", "MEDIUM"]:
            continue
        text_to_search = item.risk_sentence if item.risk_sentence else item.clause
        snippets = _candidate_snippets(text_to_search)
        found_any = False
        for page in doc:
            for snip in snippets:
                if not isinstance(snip, str) or len(snip) < 5: continue
                rects = page.search_for(snip, quads=True)
                if rects:
                    annot = page.add_highlight_annot(rects)
                    annot.set_colors(stroke=(1, 0, 0) if item.risk == "HIGH" else (1, 0.55, 0))
                    annot.set_info(content=f"[{item.risk}] {item.reason}")
                    annot.update()
                    found_any = True
        if not found_any:
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
        summary_page.insert_textbox(inner_rect, "\n".join(text_blocks), fontsize=10, align=0)
    out = io.BytesIO()
    doc.save(out, deflate=True, garbage=4)
    doc.close()
    return out.getvalue()

# ---------------- App Header & Controls ----------------
st.header("合約風險評鑑 Contract Risk Classifier")
st.markdown("上傳合約 PDF，AI 將自動標示**高/中**風險條款，**精準定位風險句子**，並附上分析說明。")
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
    MODEL_NAME = st.selectbox("分析模型 (進階)", options=["gpt-4o", "gpt-4-turbo", "gpt-5", "gpt-5-pro"], index=0)

# ---------------- Process PDF ----------------
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
    
    # ✅ [核心修改] 在此處過濾掉 NOTICEABLE 的項目
    high_medium_results = [r for r in all_results if r.risk in ["HIGH", "MEDIUM"]]
    noticeable_count = len(all_results) - len(high_medium_results)
    
    st.session_state.results = high_medium_results
    st.session_state.resolved = set()

    # 儲存一個訊息，以便在頁面刷新後顯示
    st.session_state.last_run_message = f"分析完成！已識別出 {len(high_medium_results)} 個高/中風險項目。"
    if noticeable_count > 0:
        st.session_state.last_run_message += f" (已自動過濾 {noticeable_count} 個低風險項目)"

    st.rerun()

# ---------------- Render Results ----------------
# 在頁面頂部顯示上次運行的結果訊息
if "last_run_message" in st.session_state:
    st.success(st.session_state.last_run_message)
    del st.session_state.last_run_message # 顯示一次後就刪除

if st.session_state.results:
    results: List[ClauseRisk] = st.session_state.results
    resolved: set = st.session_state.resolved
    def mark_resolved(idx: int): st.session_state.resolved.add(idx)
    def undo_resolved(idx: int): st.session_state.resolved.discard(idx)
    active_indices = [i for i in range(len(results)) if i not in resolved]
    active_results = [results[i] for i in active_indices]
    
    # ✅ [核心修改] 簡化計數器，不再需要計算 NOTICEABLE
    counts = {"HIGH": sum(1 for r in active_results if r.risk == "HIGH"),
              "MEDIUM": sum(1 for r in active_results if r.risk == "MEDIUM")}
              
    st.divider()
    st.subheader("風險摘要 Summary")
    # ✅ [核心修改] 更新摘要顯示，移除 NOTICEABLE
    st.write(f"🔴 **高風險: {counts['HIGH']}** · 🟠 **中風險: {counts['MEDIUM']}** · ✅ **已解決: {len(resolved)}**")
    
    show_resolved = st.checkbox("顯示已解決項目", value=False)
    
    # ✅ [核心修改] 簡化 badge_map
    badge_map = {"HIGH": "🔴 高風險 HIGH RISK", "MEDIUM": "🟠 中風險 MEDIUM RISK"}

    def render_card(idx: int, item: ClauseRisk, is_resolved: bool):
        with st.container(border=True):
            left_col, right_col = st.columns([8, 2])
            with left_col:
                # 由於已過濾，此處 get 的預設值不再重要，但保留以防萬一
                status = "✅ 已解決 RESOLVED" if is_resolved else badge_map.get(item.risk, "🟠 中風險 MEDIUM RISK")
                st.markdown(f"**{status}** · *Tags: {', '.join(item.tags) or '無'}*")
                if item.risk_sentence and not is_resolved:
                    st.markdown("##### 🔑 風險根源句 (Risk Root-Cause Sentence)")
                    st.markdown(f"> {item.risk_sentence}")
                    st.markdown("---")
                with st.expander("檢視完整條款上下文 (View Full Clause Context)"):
                    st.text_area("Clause Text", value=item.clause, height=150, disabled=True, key=f"clause_text_{idx}")
                st.caption(f"**AI 分析與理由:** {item.reason}")
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
