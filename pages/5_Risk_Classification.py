import os
import re
import json
import time
import textwrap
from typing import List, Literal

import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from PyPDF2 import PdfReader
from openai import OpenAI

import Risk_Knowledge

# ---------------- UI CONFIG (must be the first Streamlit call) ----------------
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
You are a contract risk analyst for a consulting firm.
Return a SINGLE JSON object with EXACT keys:
- clause: the clause text (string)
- risk: one of HIGH, MEDIUM (string, UPPERCASE)
- reason: <= 150 words (string)
- tags: short keywords (array of strings)

**Risk Classification Rubric:**
---
{RISK_RUBRIC}
---
Rules:
- Use the rubric
- Do NOT wrap in any extra key (e.g., no {"ClauseRisk": {...}}).
- Do NOT return a list.
- Do NOT include extra keys.
"""

# ---------------- Helpers ----------------
def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join((page.extract_text() or "") for page in reader.pages)

def split_into_clauses(text: str) -> List[str]:
    """Simple clause splitter: headings, bullets, and long sentences."""
    parts = re.split(
        #r"(?:\n\s*[A-Z][A-Z0-9 /&\-]{3,}\s*\n|\n\s*\d+\.\s+|\n\s*[•\-]\s+)",
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

def classify_batch(clauses: List[str]) -> List[ClauseRisk]:
    client = OpenAI()
    out: List[ClauseRisk] = []
    for c in clauses:
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
        time.sleep(0.05)
    return out

# ---------------- App Header ----------------
st.header("合約風險評鑑 Contract Risk Classifier")
st.markdown("上傳合約 PDF 以標示**高/中**風險條款並附上簡短說明。")

# ---------------- Session State ----------------
if "results" not in st.session_state:
    st.session_state.results = []          # list of ClauseRisk
if "resolved" not in st.session_state:
    st.session_state.resolved = set()      # set of indices (ints)

# ---------------- Controls ----------------
left, right = st.columns([2, 1])

with left:
    uploaded = st.file_uploader("Upload a Contract (.pdf)", type=["pdf"], key="contract_pdf")
    process_clicked = st.button("處理上傳 Process Upload", type="primary", use_container_width=True, disabled=uploaded is None)

with right:
    cap = st.number_input("最大可分析子句數 Max Clauses to Analyze", min_value=5, max_value=80, value=30, step=5)
    model_name = st.text_input("進階模型 Model", value=MODEL_NAME)
    if model_name:
        MODEL_NAME = model_name

# ---------------- Process PDF ----------------
if process_clicked:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not set. Please set it in your environment.")
        st.stop()

    if uploaded is None:
        st.warning("Please choose a PDF first.")
        st.stop()

    # Extract & split
    text = extract_text_from_pdf(uploaded)
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

    st.subheader("Summary")
    st.write(
        f"🔴 HIGH: **{counts['HIGH']}**  ·  🟠 MEDIUM: **{counts['MEDIUM']}**  ·  ✅ Resolved: **{len(resolved)}**"
    )
    st.divider()

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

    # ------------- Exports -------------
    st.markdown("—")

    # Default export: unresolved only (as TXT)
    export_payload = [results[i].model_dump() for i in range(len(results)) if i not in resolved]
    export_text = "\n\n".join(
    f"Clause: {r['clause']}\nRisk: {r['risk']}\nReason: {r['reason']}\nTags: {', '.join(r['tags']) or 'untagged'}"
    for r in export_payload
    )

    st.download_button(
        "Save TXT (Unresolved Only)",
        data=export_text.encode("utf-8"),
        file_name="risk_flags_unresolved.txt",
        mime="text/plain",
        use_container_width=True,
    )
