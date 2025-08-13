# RAG_Chatbot.py — Streamlit chat interface for querying your documents
# Works independently from your Home.py. Uses Pinecone namespaces + ad-hoc PDF uploads (FAISS).
# Env vars needed: OPENAI_API_KEY, PINECONE_API_KEY, (optional) PINECONE_INDEX_NAME

import os
import tempfile
from datetime import datetime
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv

# LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain.schema import HumanMessage, AIMessage, SystemMessage


# -----------------------------
# 1) Page & basic config
# -----------------------------
st.set_page_config(page_title="ContractBot – Chat", layout="wide")
load_dotenv()
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "contract-assistant")


# -----------------------------
# 2) Helpers
# -----------------------------
@st.cache_resource
def _pc_client():
    return Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

@st.cache_data(show_spinner=False)
def list_namespaces(index_name: str) -> List[str]:
    try:
        pc = _pc_client()
        stats = pc.describe_index(index_name).stats
        return list(stats.namespaces.keys()) if stats and stats.namespaces else []
    except Exception:
        return []

@st.cache_resource
def _embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

@st.cache_resource
def _splitter():
    return RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

@st.cache_resource
def _llm(model_name: str, temperature: float, streaming: bool):
    # Keep model_name arg for compatibility with your Home.py style
    return ChatOpenAI(model_name=model_name, temperature=temperature, streaming=streaming)

@st.cache_resource
def pinecone_retriever(namespace: str, k: int):
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=_embeddings(),
        namespace=namespace
    ).as_retriever(search_kwargs={"k": k})


# -----------------------------
<<<<<<< HEAD
# 3) Session-state
=======
# 3) Session-state (no attribute type annotations)
>>>>>>> origin/main
# -----------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("faiss_store", None)
st.session_state.setdefault("namespaces", list_namespaces(INDEX_NAME))
st.session_state.setdefault("chat_model", "gpt-4o")
st.session_state.setdefault("temperature", 0.3)
st.session_state.setdefault("top_k", 4)
st.session_state.setdefault("streaming", True)


# -----------------------------
<<<<<<< HEAD
# 4) Sidebar — Chat Settings & Knowledge Sources
# -----------------------------
with st.sidebar:
    st.header("Chat Settings")
    st.session_state["chat_model"] = st.selectbox("Model", options=["gpt-4o", "gpt-4o-mini"], index=0,)

    with st.expander("Top-k per Retriever"):
        st.caption("Number of top results to keep from each retriever before merging. Lower = faster & more focused; higher = broader & potentially more varied results.")
        st.session_state["top_k"] = st.slider("Top-k per Retriever", 1, 10, st.session_state["top_k"])
    
    st.session_state["streaming"] = st.toggle("Stream responses", value=st.session_state["streaming"])
=======
# 4) Sidebar — sources & settings
# -----------------------------
with st.sidebar:
    st.header("Chat Settings")
    st.session_state["chat_model"] = st.selectbox(
        "Model",
        options=["gpt-4o", "gpt-4o-mini"],
        index=0,
    )
    st.session_state["temperature"] = st.slider(
        "Temperature", 0.0, 1.5, st.session_state["temperature"], 0.1
    )
    st.session_state["top_k"] = st.slider(
        "Top-k per retriever", 1, 10, st.session_state["top_k"]
    )
    st.session_state["streaming"] = st.toggle(
        "Stream responses", value=st.session_state["streaming"]
    )
>>>>>>> origin/main

    st.divider()
    st.subheader("Knowledge Sources")
    st.caption("Mix your permanent Pinecone knowledge with ad-hoc uploads.")

    # Select Pinecone namespaces to include
    if not st.session_state["namespaces"]:
        st.info("No Pinecone namespaces found yet. Add one from your main app.")
    selected_namespaces = st.multiselect(
        "Pinecone namespaces",
        options=st.session_state["namespaces"],
        default=st.session_state["namespaces"][:1] if st.session_state["namespaces"] else [],
        help="Choose one or more reference baselines."
    )

    # Upload PDFs for a temporary (session) FAISS store
    st.markdown("**Add PDFs (temporary this session)**")
    uploaded_pdfs = st.file_uploader(
        "Drag in one or more PDFs", type=["pdf"], accept_multiple_files=True, key="chat_uploader"
    )
    if st.button("Process uploads"):
        if uploaded_pdfs:
            with st.spinner("Indexing PDFs into a temporary vector store…"):
                all_docs = []
                for uf in uploaded_pdfs:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uf.getvalue())
                        path = tmp.name
                    loader = PyPDFLoader(path)
                    pages = loader.load()
                    splitter = _splitter()
                    docs = splitter.split_documents(pages)
                    # Attach lightweight provenance
                    for d in docs:
                        d.metadata = d.metadata or {}
                        d.metadata.update({
                            "source": uf.name,
                            "namespace": "session",
                            "when": datetime.utcnow().isoformat()
                        })
                    all_docs.extend(docs)
                emb = _embeddings()
                if st.session_state["faiss_store"] is None:
                    st.session_state["faiss_store"] = FAISS.from_documents(all_docs, emb)
                else:
                    st.session_state["faiss_store"].add_documents(all_docs)
            st.success("Uploaded documents are now searchable in this chat.")
        else:
            st.warning("No files selected.")

    if st.session_state["faiss_store"] is not None and st.button("Clear session uploads"):
        st.session_state["faiss_store"] = None
        st.success("Cleared temporary FAISS store.")


# -----------------------------
# 5) Build a combined retriever
# -----------------------------
retrievers = []
for ns in (selected_namespaces or []):
    try:
        retrievers.append(pinecone_retriever(ns, st.session_state["top_k"]))
    except Exception:
        st.sidebar.warning(f"Could not connect retriever for namespace: {ns}")

if st.session_state["faiss_store"] is not None:
    retrievers.append(
        st.session_state["faiss_store"].as_retriever(
            search_kwargs={"k": st.session_state["top_k"]}
        )
    )

combo_retriever = None
if retrievers:
    if len(retrievers) == 1:
        combo_retriever = retrievers[0]
    else:
        # Simple equal-weight ensemble
        combo_retriever = EnsembleRetriever(retrievers=retrievers, weights=[1.0] * len(retrievers))


# -----------------------------
# 6) System Prompt & RAG routine
# -----------------------------
SYSTEM_PROMPT = (
    "You are ContractBot, a meticulous legal assistant. Answer using ONLY the provided context "
    "when possible. Quote key passages and include bracketed citations like [1], [2]. If the "
    "answer is not in the context, say you don't have enough information and suggest what to check."
)

ANSWER_PROMPT = (
    "Context:\n{context}\n\n"
    "User question: {question}\n\n"
    "Instructions: Provide a concise, accurate answer. Use a neutral, professional tone. "
    "Cite sources as [#] matching the source list. If there are material differences or risks, call them out."
)

def answer_with_rag(question: str) -> Dict:
    documents = []
    if combo_retriever is not None:
        documents = combo_retriever.get_relevant_documents(question)

    # Build context block and a map of sources
    sources = []
    context_chunks = []
    for i, d in enumerate(documents[:8]):
        src = d.metadata.get("source") or d.metadata.get("path") or d.metadata.get("file_name") or f"doc_{i+1}"
        page = d.metadata.get("page")
        tag = f"{src}{f' p.{page}' if page is not None else ''}"
        sources.append(tag)
        context_chunks.append(f"[{i+1}] {tag}\n{d.page_content.strip()[:2000]}")

    context_text = "\n\n".join(context_chunks) if context_chunks else "(no context retrieved)"

    llm = _llm(
        st.session_state["chat_model"],
        st.session_state["temperature"],
        st.session_state["streaming"]
    )

    # Compose messages
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    # include brief chat history for continuity
    for m in st.session_state["messages"][-6:]:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=ANSWER_PROMPT.format(context=context_text, question=question)))

    # Invoke model
    if st.session_state["streaming"]:
        from langchain.callbacks.base import BaseCallbackHandler

        class StreamHandler(BaseCallbackHandler):
            def __init__(self, container):
                self.container = container
                self.text = ""
            def on_llm_new_token(self, token: str, **kwargs):
                self.text += token
                self.container.markdown(self.text)

        placeholder = st.empty()
        handler = StreamHandler(placeholder)
        _ = llm.invoke(messages, config={"callbacks": [handler]})
        final_text = handler.text
    else:
        resp = llm.invoke(messages)
        final_text = resp.content

    return {
        "answer": final_text,
        "sources": sources,
        "documents": documents,
    }


# -----------------------------
# 7) Chat UI
# -----------------------------
<<<<<<< HEAD
st.title("合約聊天機器人 ContractBot")
st.caption("Ask questions across selected Pinecone namespaces and any PDFs you upload in this session.")

with st.expander("Prompt Suggestions"):
=======
st.title("💬 ContractBot – Document Chat")
st.caption("Ask questions across selected Pinecone namespaces and any PDFs you upload in this session.")

with st.expander("Quick suggestions"):
>>>>>>> origin/main
    st.write("• 這份 NDA 的保密義務何時終止？\n\n• 條款對於機密資訊的例外揭露包含哪些情況？\n\n• 若對方違約，我方可要求的補救措施為何？")

# Replay messages
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Type your question about the documents…")

if user_q:
    st.session_state["messages"].append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        if combo_retriever is None:
            st.warning("No retriever configured. Select a Pinecone namespace and/or upload PDFs.")
            answer_block = {"answer": "", "sources": [], "documents": []}
        else:
            answer_block = answer_with_rag(user_q)

        # Persist assistant message
        st.session_state["messages"].append({"role": "assistant", "content": answer_block.get("answer", "")})

        # Render sources under the answer
        if answer_block.get("sources"):
            st.markdown("\n**Sources**")
            for i, tag in enumerate(answer_block["sources"], start=1):
                with st.expander(f"[{i}] {tag}"):
                    doc = answer_block["documents"][i-1]
                    preview = doc.page_content[:1200]
                    st.write(preview + ("…" if len(doc.page_content) > 1200 else ""))

st.divider()
col1, col2, col3 = st.columns(3)
with col1:
<<<<<<< HEAD
    if st.button("Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()
with col2:
    if st.button("Refresh namespaces", use_container_width=True):
        st.session_state["namespaces"] = list_namespaces(INDEX_NAME)
        st.success("Namespaces updated.")
with col3:
    import io
    import json
    from textwrap import wrap
    st.caption("Export conversation")
    export_fmt = st.selectbox("Format", ["JSON", "TXT", "PDF"], index=0, label_visibility="collapsed")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname_base = f"contractbot_chat_{ts}"

    data = None
    mime = "application/octet-stream"
    fname = f"{fname_base}.bin"

    if export_fmt == "JSON":
        fname = f"{fname_base}.json"
        data = json.dumps(st.session_state["messages"], ensure_ascii=False, indent=2).encode("utf-8")
        mime = "application/json"

    elif export_fmt == "TXT":
        # Simple readable transcript
        lines = []
        for m in st.session_state.get("messages", []):
            role = m.get("role", "").upper()
            content = (m.get("content") or "").strip()
            lines.append(f"[{role}] {content}")
        txt = "\n\n".join(lines)
        data = txt.encode("utf-8")
        fname = f"{fname_base}.txt"
        mime = "text/plain"

    elif export_fmt == "PDF":
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas

            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            width, height = letter

            margin_x = 72  
            margin_y = 72
            y = height - margin_y
            line_height = 14

            c.setTitle(fname_base)
            c.setFont("Helvetica", 11)

            for m in st.session_state.get("messages", []):
                role = m.get("role", "").upper()
                content = (m.get("content") or "").strip()
                line = f"[{role}] {content}"

                # Wrap long lines
                for seg in wrap(line, 95):
                    if y <= margin_y:
                        c.showPage()
                        c.setFont("Helvetica", 11)
                        y = height - margin_y
                    c.drawString(margin_x, y, seg)
                    y -= line_height
                y -= 6  # extra spacing between messages

            c.save()
            data = buffer.getvalue()
            buffer.close()
            fname = f"{fname_base}.pdf"
            mime = "application/pdf"
        except Exception:
            st.error("PDF export requires the 'reportlab' package. Install it with:\n\npip install reportlab")

    # Always render the download button (if data is ready)
    st.download_button(
        f"Download conversation ({export_fmt})",
        data=data if data is not None else b"",
        file_name=fname,
        mime=mime,
        use_container_width=True,
        disabled=(data is None)  # disable only if we couldn't build the chosen format
    )


=======
    if st.button("Clear chat"):
        st.session_state["messages"] = []
        st.rerun()
with col2:
    if st.button("Refresh namespaces"):
        st.session_state["namespaces"] = list_namespaces(INDEX_NAME)
        st.success("Namespaces updated.")
with col3:
    # Export conversation as JSON (always render the download)
    import json
    fname = f"contractbot_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    b = json.dumps(st.session_state["messages"], ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("Download conversation (JSON)", data=b, file_name=fname, mime="application/json")
>>>>>>> origin/main
