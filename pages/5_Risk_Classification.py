import os
import re
import io
import json
import time
import textwrap
from typing import List, Literal

import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

import Risk_Knowledge
import fitz  

load_dotenv()
# ---------------- UI Configuration ----------------
st.set_page_config(page_title="Contract Risk Classifier", layout="wide")
st.logo("logo.png")

# ---------------- Configuration ----------------
MODEL_NAME = "gpt-4o"

# ---------------- Pydantic Schema ----------------
class ClauseRisk(BaseModel):
    clause: str
    risk: Literal["HIGH", "MEDIUM"]
    reason: str
    tags: List[str] = Field(default_factory=list)

RISK_RUBRIC = Risk_Knowledge.get_risk_rubric_string()

# ---------------- Prompts ----------------
SYSTEM_PROMPT = """
You are a meticulous and experienced contract risk analyst for a top-tier consulting firm. Your task is to analyze a given contract clause and classify its risk level based on a provided rubric.

**Follow these steps carefully:**
1.  **Analyze the Clause**: Read the user-provided clause and identify its core legal and commercial implications. What obligations, rights, or restrictions does it impose?
2.  **Consult the Rubric**: Compare the clause's characteristics against the `Risk Classification Rubric`. Find the most relevant risk type described in the rubric.
3.  **Determine Risk Level**: Based on the comparison, classify the risk as either "HIGH" or "MEDIUM". Default to "MEDIUM" if you are uncertain but see potential issues.
4.  **Formulate the Reason**: Write a concise, professional explanation (under 150 words) for your classification. Your reason must directly reference the logic from the rubric.
5.  **Extract Keywords**: Identify 2-4 keywords from the clause and your reason that best summarize the topic (e.g., "Indemnification", "Data Usage", "Liability Cap").
6.  **Construct JSON**: Assemble the final analysis into a SINGLE JSON object.

**Risk Classification Rubric:**
---
{RISK_RUBRIC}
---

**Output Format Rules (Strictly Enforced):**
- The output MUST be a single, valid JSON object.
- The JSON object must have these EXACT keys: `clause`, `risk`, `reason`, `tags`.
- `risk` must be either "HIGH" or "MEDIUM" in uppercase.
- Do NOT wrap the JSON in any other keys (e.g., no `{"ClauseRisk": ...}`).
- Do NOT return a list or any text outside of the JSON object.
"""

# ---------------- Helpers ----------------
def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join((page.extract_text() or "") for page in reader.pages)

def split_into_clauses(text: str) -> List[str]:
    parts = re.split(
        r"(?m)(?=^[ \t]*(?:(?:ARTICLE|SECTION)\s+[IVXLC\d]+(?:[:.\-–]\s+|[^\S\r\n]+)|[A-Z][A-Z0-9 ,/&\-–]{3,}[ \t]*$|\d+(?:\.\d+)*(?:\([a-zA-Z0-9]+\))?[ \t]+|\([a-zA-Z0-9]{1,3}\)[ \t]+|[•\-–][ \t]+))",
        text,
        flags=re.MULTILINE,
    )
    clauses = []
    for p in parts:
        p = " ".join(p.split())
        if 80 < len(p) < 1800:
            clauses.append(p)
    if not clauses:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        buf, cur = [], 0
        for s in sentences:
            if cur + len(s) < 800:
                buf.append(s); cur += len(s)
            else:
                chunk = " ".join(buf).strip()
                if len(chunk) > 80:
                    clauses.append(chunk)
                buf, cur = [s], len(s)
        if buf:
            chunk = " ".join(buf).strip()
            if len(chunk) > 80:
                clauses.append(chunk)
    return clauses[:30]

def normalize_model_json(raw: str, original_clause: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            raise
        data = json.loads(m.group(0))

    if isinstance(data, dict) and "ClauseRisk" in data and isinstance(data["ClauseRisk"], dict):
        data = data["ClauseRisk"]

    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]

    synonyms = {
        "riskLevel": "risk",
        "risk_level": "risk",
        "level": "risk",
        "label": "risk",
        "explanation": "reason",
        "rationale": "reason",
        "justification": "reason",
        "text": "clause",
        "content": "clause",
        "clause_text": "clause",
        "summary": "reason",
    }
    if isinstance(data, dict):
        for old, new in list(synonyms.items()):
            if old in data and new not in data:
                data[new] = data[old]
        risk_val = str(data.get("risk", "")).strip().upper()
        if risk_val not in ("HIGH", "MEDIUM"):
            risk_val = "MEDIUM"
        data["risk"] = risk_val
        data.setdefault("tags", [])
        data.setdefault("clause", original_clause)
    return data

def classify_clause(client: OpenAI, clause: str) -> ClauseRisk:
    user_prompt = f"Clause:\n{clause}\n\nReturn the JSON object now."
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    normalized = normalize_model_json(content, original_clause=clause)
    return ClauseRisk.model_validate(normalized)

# ---------------- Running Progress Bar ----------------
def classify_batch(clauses: List[str]) -> List[ClauseRisk]:
    client = OpenAI()
    out: List[ClauseRisk] = []

    total = max(len(clauses), 1)
    progress = st.progress(0, text="Classifying…")

    for i, c in enumerate(clauses, start=1):
        try:
            out.append(classify_clause(client, c))
        except ValidationError as ve:
            out.append(
                ClauseRisk(
                    clause=c,
                    risk="MEDIUM",
                    reason=f"Validation fallback: {ve.errors()[0]['type']}",
                    tags=["fallback"],
                )
            )
        except Exception as e:
            out.append(
                ClauseRisk(
                    clause=c,
                    risk="MEDIUM",
                    reason=f"Model error: {str(e)[:60]}",
                    tags=["error"],
                )
            )
        # update progress each iteration
        progress.progress(min(i / total, 1.0), text=f"Classified {i}/{total}")
        time.sleep(0.05)

    progress.empty()
    return out

# ---------------- Highlighting Helpers ----------------
def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _candidate_snippets(text: str, min_len: int = 40, max_len: int = 140) -> List[str]:
    """
    Produce several overlapping snippets from a clause so that page.search_for()
    stands a better chance of finding matches even when line-breaks / hyphenation differ.
    """
    t = _normalize_spaces(text)
    if len(t) <= max_len:
        return [t] if len(t) >= min_len else [t]

    # sentence-based first
    sentences = re.split(r"(?<=[\.\?!;:])\s+", t)
    picks = []
    for s in sentences:
        s = s.strip()
        if len(s) >= min_len:
            picks.append(s[:max_len])

    # sliding window on words
    words = t.split()
    window = 18  # approx ~90-120 chars for legal text
    step = 8
    for i in range(0, max(1, len(words) - window + 1), step):
        chunk = " ".join(words[i:i+window])
        if len(chunk) >= min_len:
            picks.append(chunk[:max_len])

    # first / middle / last chunk
    picks.extend([
        t[:max_len],
        t[max(0, len(t)//2 - max_len//2): max_len + len(t)//2],
        t[-max_len:]
    ])

    # eliminating duplicates while preserving order
    seen, uniq = set(), []
    for s in picks:
        s2 = s.strip()
        if s2 and s2 not in seen:
            uniq.append(s2)
            seen.add(s2)
    return uniq[:10]

def _risk_color(risk: str):
    if risk == "HIGH":
        return (1, 0, 0)   
    return (1, 0.55, 0)    

def _inset(rect, margin: float) -> "fitz.Rect":
    return fitz.Rect(rect.x0 + margin, rect.y0 + margin, rect.x1 - margin, rect.y1 - margin)

def build_highlighted_pdf(src_pdf_bytes: bytes, items: List[ClauseRisk], include_resolved: bool = False, resolved_idx: set = None) -> bytes:
    """
    Create a copy of the original PDF with highlight annotations for each (unresolved) clause.
    Adds a summary page at end for any clauses not found.
    """
    if resolved_idx is None:
        resolved_idx = set()

    doc = fitz.open(stream=src_pdf_bytes, filetype="pdf")
    not_found = []  # keep tuples of (risk, reason, text)

    for idx, item in enumerate(items):
        if (not include_resolved) and (idx in resolved_idx):   # skip resolved unless include_resolved is True
            continue

        snippets = _candidate_snippets(item.clause)
        found_any = False

        for page in doc:
            page_found_here = False
            for snip in snippets:
                rects = page.search_for(snip, flags=1)      # case-insensitive search compatible across versions
                if rects:
                    annot = page.add_highlight_annot(rects)  # create one combined highlight per snippet occurrence
                    try:
                        annot.set_colors(stroke=_risk_color(item.risk))  # color by risk 
                    except Exception:
                        pass
                    info_text = f"{item.risk} RISK\n{item.reason}".strip() # add popup content with risk + reason
                    try:
                        annot.set_info({"content": info_text})
                    except Exception:
                        pass
                    annot.update()
                    found_any = True
                    page_found_here = True
            if page_found_here:
                continue     # continue scanning next pages to catch multi-page occurrences

        if not found_any:
            not_found.append(item)

    # Add a summary page listing any clauses we could not locate
    if not_found:
        summary = doc.new_page(-1)  # append to end
        header = "Clauses Not Found For Highlighting"
        text_blocks = [header, ""]
        for miss in not_found:
            text_blocks.append(f"- {miss.risk} · {', '.join(miss.tags) or 'untagged'}")
            text_blocks.append(f"  Reason: {miss.reason}")
            text_blocks.append("  " + textwrap.shorten(_normalize_spaces(miss.clause), width=220))  # Show a shortened clause to keep page readable
            text_blocks.append("")  # spacer
        inner_rect = _inset(summary.rect, 36)
        summary.insert_textbox(inner_rect, "\n".join(text_blocks), fontsize=10, align=0)

    # Save to bytes
    out = io.BytesIO()
    doc.save(out, deflate=True, garbage=4)
    doc.close()
    return out.getvalue()

# ---------------- App Header ----------------
st.header("合約風險評鑑 Contract Risk Classifier")
st.markdown("上傳合約 PDF 以標示**高/中**風險條款並附上簡短說明。")

# ---------------- Session State ----------------
if "results" not in st.session_state:
    st.session_state.results = []          # list of ClauseRisk
if "resolved" not in st.session_state:
    st.session_state.resolved = set()      # set of indices (ints)
# keep the original uploaded bytes so we can re-open it for highlighting
if "source_pdf_bytes" not in st.session_state:
    st.session_state.source_pdf_bytes = None

# ---------------- Controls ----------------
left, right = st.columns([2, 1])

with left:
    uploaded = st.file_uploader("Upload a Contract (.pdf)", type=["pdf"], key="contract_pdf")
    process_clicked = st.button("處理上傳 Process Upload", type="primary", use_container_width=True, disabled=uploaded is None)

with right:
    cap = st.number_input("最大可分析子句數 Max Clauses to Analyze", min_value=5, max_value=50, value=20, step=5)
    MODEL_OPTIONS = [
        "gpt-5",
        "gpt-5-pro",
        "gpt-5-mini",
        "gpt-4o",
    ]

    MODEL_NAME = st.selectbox(
        "進階模型 Model",
        options=MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(MODEL_NAME) if MODEL_NAME in MODEL_OPTIONS else 0
    )

# ---------------- Process PDF ----------------
if process_clicked:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not set. Please set it in your environment.")
        st.stop()

    if uploaded is None:
        st.warning("Please choose a PDF first.")
        st.stop()

    # Extract & split
    # Also store the raw bytes for later highlight export
    st.session_state.source_pdf_bytes = uploaded.getvalue()
    text = extract_text_from_pdf(io.BytesIO(st.session_state.source_pdf_bytes))
    clauses = split_into_clauses(text)[:cap]
    st.caption(f"Found {len(clauses)} clause-like chunks.")

    with st.spinner("Classifying clauses…"):
        results = classify_batch(clauses)

    # Persist results and reset resolved flags
    st.session_state.results = results
    st.session_state.resolved = set()

# ---------------- Render Results ----------------
if st.session_state.results:
    results: List[ClauseRisk] = st.session_state.results
    resolved: set = st.session_state.resolved

    # Helpers for buttons (on_click: Streamlit auto-reruns)
    def mark_resolved(idx: int):
        st.session_state.resolved.add(idx)

    def undo_resolved(idx: int):
        st.session_state.resolved.discard(idx)

    def resolve_all():
        st.session_state.resolved = set(range(len(st.session_state.results)))

    def clear_resolved():
        st.session_state.resolved = set()

    # Compute active (unresolved) items and counts
    active_indices = [i for i in range(len(results)) if i not in resolved]
    active_results = [results[i] for i in active_indices]

    counts = {"HIGH": 0, "MEDIUM": 0}
    for r in active_results:
        counts[r.risk] += 1

    st.divider()

    st.subheader("摘要 Summary")
    st.write(
        f"🔴 HIGH: **{counts['HIGH']}**  ·  🟠 MEDIUM: **{counts['MEDIUM']}**  ·  ✅ Resolved: **{len(resolved)}**"
    )

    show_resolved = st.checkbox("Show resolved items", value=False)

    badge_map = {"HIGH": "🔴 HIGH RISK", "MEDIUM": "🟠 MEDIUM RISK"}

    def render_card(idx: int, item: ClauseRisk, is_resolved: bool):
        with st.container(border=True):
            left_col, right_col = st.columns([6, 1])
            with left_col:
                if is_resolved:
                    st.markdown(f"**✅ RESOLVED**  ·  _{', '.join(item.tags) or 'untagged'}_")
                else:
                    st.markdown(f"**{badge_map[item.risk]}**  ·  _{', '.join(item.tags) or 'untagged'}_")
                st.write(textwrap.shorten(item.clause, 550))
                st.caption(item.reason)
            with right_col:
                if not is_resolved:
                    st.button(
                        "Resolved",
                        key=f"resolve_btn_{idx}",
                        help="Mark this clause as resolved (removes from counts and export)",
                        on_click=mark_resolved,
                        args=(idx,),
                        use_container_width=True,
                    )
                else:
                    st.button(
                        "Undo",
                        key=f"undo_btn_{idx}",
                        help="Restore this clause to the active list",
                        on_click=undo_resolved,
                        args=(idx,),
                        use_container_width=True,
                    )

    # Active (unresolved) first
    for i in active_indices:
        render_card(i, results[i], is_resolved=False)

    # Optionally show resolved items
    if show_resolved and len(resolved) > 0:
        st.markdown("### Resolved")
        for i in sorted(resolved):
            render_card(i, results[i], is_resolved=True)

    # ------- Export Highlighted PDF -------
    st.divider()

    st.subheader("匯出重點標記 Export with Highlights (PDF)")
    disabled_pdf = st.session_state.source_pdf_bytes is None
    col_a, col_b = st.columns([1, 1])
    with col_a:
        include_resolved = st.toggle("Include resolved items", value=False, help="Include already-resolved clauses in the highlighted PDF.")

    if disabled_pdf:
        st.warning("Upload and process a PDF first to enable PDF export.")
    else:
        try:
            pdf_bytes = build_highlighted_pdf(
                st.session_state.source_pdf_bytes,
                items=results,
                include_resolved=include_resolved,
                resolved_idx=resolved
            )
            st.download_button(
                "匯出含重點標註 Export PDF with Highlights",
                data=pdf_bytes,
                file_name="contract_highlighted.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Failed to generate highlighted PDF: {e}")
