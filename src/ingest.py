import os
from pypdf import PdfReader

def extract_pdf_pages(file_path: str) -> list[dict]:
    """
    Extracts text page-by-page from a PDF, tracking page numbers and file source.
    """
    extracted_data = []
    file_name = os.path.basename(file_path)

    try:
        reader = PdfReader(file_path)
        for index, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                # Clean up multiple whitespaces and leading/trailing spaces
                clean_text = " ".join(text.split())
                extracted_data.append({
                    "text": clean_text,
                    "metadata": {
                        "source": file_name,
                        "page": index + 1  # 1-indexed for reader readability
                    }
                })
    except Exception as e:
        print(f"Error reading PDF {file_name}: {e}")

    return extracted_data

def chunk_extracted_pages(pages: list[dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[dict]:
    """
    Splits page-level documents into smaller, overlapping chunks.
    Ensures that source metadata is carried over to every individual chunk.
    """
    chunks = []

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]

        start = 0
        text_length = len(text)

        while start < text_length:
            # Determine end point
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end]

            # Store chunk content alongside original page-level metadata
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_range": f"{start}-{end}"
                }
            })

            # Slide window forward by (chunk_size - chunk_overlap)
            start += (chunk_size - chunk_overlap)

    return chunks

import os
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

def save_to_vector_db(chunks: list[dict], db_path: str = "./db"):
    """
    Embeds text chunks and saves them into a persistent disk-based ChromaDB.
    """
    # Create persistent ChromaDB client
    client = chromadb.PersistentClient(path=db_path)

    # Initialize the Gemini embedding function
    embedding_fn = GoogleGenerativeAiEmbeddingFunction(
        api_key=api_key,
        model_name="models/text-embedding-004"
    )

    # Create or fetch collection
    collection = client.get_or_create_collection(
        name="document_knowledge_base",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"} # Use Cosine Distance
    )

    # Prepare batch data
    ids = [f"id_{i}" for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    # Batch upload to ChromaDB
    # ChromaDB automatically handles embedding generation via the custom embedding function
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    print(f"Successfully indexed {len(chunks)} chunks in the vector database.")
