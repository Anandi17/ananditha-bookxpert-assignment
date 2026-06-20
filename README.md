# Document Q&A Bot with RAG

This project is a basic Retrieval-Augmented Generation (RAG) pipeline for asking natural language questions against a local document collection. It ingests PDF, TXT, and DOCX files, chunks the extracted text, stores embeddings in a persistent ChromaDB database, retrieves relevant chunks, and asks Gemini to generate a grounded answer with source citations.

## Features

- Supports `.pdf`, `.txt`, and `.docx` documents.
- Tracks source filename, page number, file type, and chunk index.
- Uses recursive character chunking with overlap for better retrieval quality.
- Persists vectors locally in `db/` with ChromaDB.
- Uses Gemini `text-embedding-004` for embeddings and `gemini-2.5-flash-preview-09-2025` for answer generation.
- Includes an interactive command-line interface and an optional Streamlit UI.

## Project Structure

```text
document-qa-bot/
+-- .env.example
+-- .gitignore
+-- README.md
+-- requirements.txt
+-- app.py
+-- data/
+-- db/
+-- src/
|   +-- __init__.py
|   +-- config.py
|   +-- ingest.py
|   +-- query.py
|   +-- main.py
+-- tests/
    +-- test_chunking.py
```

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Create a `.env` file.

```bash
copy .env.example .env
```

4. Add your Google AI Studio API key to `.env`.

```text
GEMINI_API_KEY=your-key-here
```

## Add Documents

Place 4 to 5 source files in the `data/` directory. The project includes five small TXT sample documents so the pipeline has content immediately. You can replace them with your own PDFs, DOCX files, or TXT files.

## Ingest Documents

Run ingestion once after adding or changing documents.

```bash
python -m src.ingest
```

This extracts text, creates chunks, embeds them, and saves the Chroma database to `db/`.

## Ask Questions from the CLI

Single-question mode:

```bash
python -m src.query "What happens during a SEV1 incident?"
```

Interactive mode:

```bash
python -m src.main
```

## Run the Web UI

```bash
streamlit run app.py
```

## Example Questions

- What are the remote work expectations?
- What does the team plan include in Nimbus Notes?
- What happens during a SEV1 incident?
- How much did first response time improve in the support automation pilot?
- What limitation remains in version 2.4?

## Architecture Decisions

The project separates ingestion from querying. `src/ingest.py` is responsible for reading documents, chunking text, and persisting embeddings. `src/query.py` loads the existing vector database, retrieves relevant chunks, builds a strict grounding prompt, and asks Gemini for a cited answer.

ChromaDB was chosen because it is lightweight, local, persistent, and simple to run without a database server. The chunker uses paragraph, line, sentence, and space boundaries before falling back to fixed-width slicing, which preserves more semantic context than blind slicing.

The prompt explicitly instructs the model to answer only from retrieved context and to say `I cannot find the answer in the provided documents.` when the answer is unavailable. This reduces hallucination risk and keeps answers grounded in the indexed document library.

## Submission Checklist

- Push this project to a public GitHub repository.
- Deploy the Streamlit app using Streamlit Community Cloud or another host.
- Record a 3 to 5 minute walkthrough showing setup, ingestion, question answering, citations, and the code structure.
