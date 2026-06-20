from src.ingest import chunk_extracted_pages, recursive_split_text


def test_recursive_split_text_respects_chunk_size_for_simple_text():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = recursive_split_text(text, chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 141 for chunk in chunks)


def test_chunk_extracted_pages_preserves_metadata():
    pages = [
        {
            "text": "Alpha beta gamma. " * 30,
            "metadata": {"source": "sample.txt", "page": 1, "file_type": "txt"},
        }
    ]

    chunks = chunk_extracted_pages(pages, chunk_size=100, chunk_overlap=20)

    assert chunks
    assert chunks[0]["metadata"]["source"] == "sample.txt"
    assert chunks[0]["metadata"]["page"] == 1
    assert "id" in chunks[0]
