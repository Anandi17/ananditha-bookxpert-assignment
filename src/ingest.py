from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from docx import Document
from pypdf import PdfReader
from tqdm import tqdm

try:
    from .config import (
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        COLLECTION_NAME,
        DATA_DIR,
        DB_DIR,
        GEMINI_EMBEDDING_MODEL,
        require_api_key,
    )
except ImportError:
    from config import (
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        COLLECTION_NAME,
        DATA_DIR,
        DB_DIR,
        GEMINI_EMBEDDING_MODEL,
        require_api_key,
    )


DocumentPage = dict[str, Any]
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def clean_text(text: str) -> str:
    """Normalize whitespace while keeping paragraph boundaries useful for chunking."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(file_path: Path) -> list[DocumentPage]:
    extracted_data: list[DocumentPage] = []
    reader = PdfReader(str(file_path))

    for index, page in enumerate(reader.pages):
        text = clean_text(page.extract_text() or "")
        if text:
            extracted_data.append(
                {
                    "text": text,
                    "metadata": {
                        "source": file_path.name,
                        "page": index + 1,
                        "file_type": "pdf",
                    },
                }
            )

    return extracted_data


def extract_txt_pages(file_path: Path) -> list[DocumentPage]:
    text = clean_text(file_path.read_text(encoding="utf-8"))
    if not text:
        return []

    return [
        {
            "text": text,
            "metadata": {
                "source": file_path.name,
                "page": 1,
                "file_type": "txt",
            },
        }
    ]


def extract_docx_pages(file_path: Path) -> list[DocumentPage]:
    document = Document(str(file_path))
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    text = clean_text("\n\n".join(paragraphs))
    if not text:
        return []

    return [
        {
            "text": text,
            "metadata": {
                "source": file_path.name,
                "page": 1,
                "file_type": "docx",
            },
        }
    ]


def extract_document(file_path: Path) -> list[DocumentPage]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_pages(file_path)
    if suffix == ".txt":
        return extract_txt_pages(file_path)
    if suffix == ".docx":
        return extract_docx_pages(file_path)
    raise ValueError(f"Unsupported file type: {file_path}")


def recursive_split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", ""),
) -> list[str]:
    """Split text recursively on natural boundaries, then add overlap."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    def split_segment(segment: str, separator_index: int = 0) -> list[str]:
        segment = segment.strip()
        if len(segment) <= chunk_size:
            return [segment] if segment else []

        separator = separators[separator_index]
        if separator == "":
            return [segment[i : i + chunk_size] for i in range(0, len(segment), chunk_size)]

        pieces = segment.split(separator)
        chunks: list[str] = []
        current = ""

        for piece in pieces:
            candidate = piece if not current else f"{current}{separator}{piece}"
            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.extend(split_segment(current, separator_index + 1))
            current = piece

        if current:
            chunks.extend(split_segment(current, separator_index + 1))

        return chunks

    base_chunks = split_segment(text)
    if not base_chunks or chunk_overlap <= 0:
        return base_chunks

    overlapped: list[str] = []
    previous_tail = ""
    for chunk in base_chunks:
        combined = f"{previous_tail} {chunk}".strip() if previous_tail else chunk
        overlapped.append(combined)
        previous_tail = chunk[-chunk_overlap:]

    return overlapped


def chunk_extracted_pages(
    pages: list[DocumentPage],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[DocumentPage]:
    chunks: list[DocumentPage] = []

    for page in pages:
        page_chunks = recursive_split_text(page["text"], chunk_size, chunk_overlap)
        for index, chunk_text in enumerate(page_chunks):
            chunk_id = stable_chunk_id(page["metadata"]["source"], page["metadata"]["page"], index)
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        **page["metadata"],
                        "chunk_index": index,
                    },
                }
            )

    return chunks


def stable_chunk_id(source: str, page: int, chunk_index: int) -> str:
    raw_id = f"{source}:{page}:{chunk_index}"
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24]


def load_documents(data_dir: Path = DATA_DIR) -> list[DocumentPage]:
    files = sorted(
        path for path in data_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No supported documents found in {data_dir}.")

    extracted: list[DocumentPage] = []
    for file_path in tqdm(files, desc="Extracting documents"):
        extracted.extend(extract_document(file_path))

    return extracted


def save_to_vector_db(chunks: list[DocumentPage], db_path: Path = DB_DIR, reset: bool = True) -> int:
    if not chunks:
        raise ValueError("No chunks were produced. Check that the documents contain extractable text.")

    api_key = require_api_key()
    client = chromadb.PersistentClient(path=str(db_path))
    embedding_fn = GoogleGenerativeAiEmbeddingFunction(
        api_key=api_key,
        model_name=GEMINI_EMBEDDING_MODEL,
    )

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except ValueError:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    return len(chunks)


def ingest(data_dir: Path = DATA_DIR, db_path: Path = DB_DIR, reset: bool = True) -> int:
    pages = load_documents(data_dir)
    chunks = chunk_extracted_pages(pages)
    return save_to_vector_db(chunks, db_path, reset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index documents into the local Chroma vector database.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--db-path", type=Path, default=DB_DIR)
    parser.add_argument("--no-reset", action="store_true", help="Append to the existing collection.")
    args = parser.parse_args()

    count = ingest(args.data_dir, args.db_path, reset=not args.no_reset)
    print(f"Successfully indexed {count} chunks in {args.db_path}.")


if __name__ == "__main__":
    main()
