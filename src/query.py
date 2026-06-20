from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import chromadb
import google.generativeai as genai
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction

try:
    from .config import (
        COLLECTION_NAME,
        DB_DIR,
        GEMINI_EMBEDDING_MODEL,
        GEMINI_GENERATION_MODEL,
        TOP_K,
        require_api_key,
    )
except ImportError:
    from config import (
        COLLECTION_NAME,
        DB_DIR,
        GEMINI_EMBEDDING_MODEL,
        GEMINI_GENERATION_MODEL,
        TOP_K,
        require_api_key,
    )


def get_collection(db_path: Path = DB_DIR):
    api_key = require_api_key()
    embedding_fn = GoogleGenerativeAiEmbeddingFunction(
        api_key=api_key,
        model_name=GEMINI_EMBEDDING_MODEL,
    )
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def retrieve_context(user_query: str, db_path: Path = DB_DIR, k: int = TOP_K) -> dict[str, Any]:
    collection = get_collection(db_path)
    return collection.query(
        query_texts=[user_query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )


def format_context(results: dict[str, Any]) -> tuple[str, list[str]]:
    context_blocks: list[str] = []
    citations: list[str] = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, distance in zip(documents, metadatas, distances):
        source_name = meta["source"]
        page_num = meta["page"]
        citation = f"{source_name}, Page {page_num}"
        context_blocks.append(
            f"[Source: {citation}; Distance: {distance:.4f}]\n{doc}"
        )
        citations.append(citation)

    return "\n\n---\n\n".join(context_blocks), citations


def build_prompt(user_query: str, context_payload: str) -> str:
    system_prompt = (
        "You are a precise document Q&A assistant. Use ONLY the provided context to "
        "answer the user's question. Cite filenames and page numbers inline next to "
        "the facts you use. If the answer is not present in the context, say exactly: "
        "'I cannot find the answer in the provided documents.' Do not use outside knowledge."
    )

    return (
        f"{system_prompt}\n\n"
        f"CONTEXT:\n{context_payload}\n\n"
        f"QUESTION: {user_query}\n\n"
        "ANSWER:"
    )


def query_rag_pipeline(user_query: str, db_path: Path = DB_DIR, k: int = TOP_K) -> dict[str, Any]:
    api_key = require_api_key()
    genai.configure(api_key=api_key)

    results = retrieve_context(user_query, db_path, k)
    context_payload, citations = format_context(results)

    if not context_payload.strip():
        return {
            "answer": "I cannot find the answer in the provided documents.",
            "citations": [],
            "raw_context": [],
        }

    model = genai.GenerativeModel(GEMINI_GENERATION_MODEL)
    response = model.generate_content(build_prompt(user_query, context_payload))

    return {
        "answer": response.text.strip(),
        "citations": citations,
        "raw_context": results["documents"][0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question against the indexed document library.")
    parser.add_argument("question", nargs="?", help="Question to ask. If omitted, starts interactive mode.")
    parser.add_argument("--db-path", type=Path, default=DB_DIR)
    parser.add_argument("--k", type=int, default=TOP_K)
    args = parser.parse_args()

    if args.question:
        result = query_rag_pipeline(args.question, args.db_path, args.k)
        print(result["answer"])
        return

    print("Document Q&A Bot. Type 'exit' to quit.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = query_rag_pipeline(question, args.db_path, args.k)
        print(f"\n{result['answer']}")


if __name__ == "__main__":
    main()
