import os
import google.generativeai as genai
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

def query_rag_pipeline(user_query: str, db_path: str = "./db", k: int = 3) -> dict:
    """
    Searches the database, builds a grounded prompt, and queries the LLM.
    """
    client = chromadb.PersistentClient(path=db_path)
    embedding_fn = GoogleGenerativeAiEmbeddingFunction(
        api_key=api_key,
        model_name="models/text-embedding-004"
    )

    collection = client.get_collection(
        name="document_knowledge_base",
        embedding_function=embedding_fn
    )

    # Query collection for top k closest results
    results = collection.query(
        query_texts=[user_query],
        n_results=k
    )

    # Format the retrieved documents as background context
    context_blocks = []
    citations = []

    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        source_name = meta['source']
        page_num = meta['page']
        citation_str = f"Source: {source_name}, Page: {page_num}"

        context_blocks.append(f"[{citation_str}]\nContext: {doc}")
        citations.append(citation_str)

    context_payload = "\n\n---\n\n".join(context_blocks)

    # Set up grounding system prompt
    system_prompt = (
        "You are a professional, accurate document Q&A assistant. "
        "Answer the user's question using ONLY the provided document context below. "
        "Cite the sources (filenames and pages) inline next to facts you cite. "
        "If the answer cannot be found in the context, clearly state: "
        "'I am sorry, but the provided documents do not contain the answer to your question.' "
        "Do not make up facts or use external knowledge sources."
    )

    prompt = (
        f"{system_prompt}\n\n"
        f"CONTEXT INFORMATION:\n{context_payload}\n\n"
        f"USER QUESTION: {user_query}\n\n"
        f"GROUNDED ANSWER:"
    )

    # Call Gemini to generate the answer
    model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
    response = model.generate_content(prompt)

    return {
        "answer": response.text,
        "citations": citations,
        "raw_context": results['documents'][0]
    }
