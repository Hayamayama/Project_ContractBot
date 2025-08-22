# text_splitter.py
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from semantic_text_splitter import TextSplitter


def regex_split(text: str):
    """條款導向切割"""
    pattern = r"(?=\n\d+\.\s)|(?=\nArticle\s+\d+)|(?=\nSection\s+\d+)"
    chunks = re.split(pattern, text)
    return [c.strip() for c in chunks if c.strip()]


def semantic_split(text: str, max_chunk_size=800):
    """語意導向切割 (semantic-text-splitter)"""
    splitter = TextSplitter(max_chunk_size)
    return splitter.chunks(text)


def recursive_split(text: str, chunk_size=1000, overlap=100):
    """原本的 RecursiveCharacterTextSplitter"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)


def smart_split(text: str, method="semantic"):
    """統一介面，可選擇切割方法"""
    if method == "regex":
        return regex_split(text)
    elif method == "recursive":
        return recursive_split(text)
    elif method == "semantic":
        return semantic_split(text)
    else:
        raise ValueError(f"Unknown split method: {method}")
