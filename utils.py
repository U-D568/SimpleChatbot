import hashlib
import uuid

from pypdf import PdfReader
from openai import OpenAI


def read_pdf(path):
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    text = "\n\n".join(text)

    return text


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def get_embeddings(chunks: str, client: OpenAI):
    emb_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks,
    )
    embeddings = [d.embedding for d in emb_response.data]

    return embeddings


def hash_id(prefix:str, text:str):
    prefix_hash = hashlib.md5(prefix.encode("utf-8")).hexdigest()
    content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"{prefix_hash}-{content_hash}"