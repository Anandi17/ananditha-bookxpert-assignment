from __future__ import annotations

try:
    import streamlit as st
except ImportError:
    st = None

try:
    from .config import TOP_K
    from .query import query_rag_pipeline
except ImportError:
    from config import TOP_K
    from query import query_rag_pipeline


def run_cli() -> None:
    print("Document Q&A Bot. Type 'exit' to quit.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = query_rag_pipeline(question, k=TOP_K)
        print(f"\n{result['answer']}")
        if result["citations"]:
            print("\nRetrieved sources:")
            for citation in result["citations"]:
                print(f"- {citation}")


def run_streamlit() -> None:
    if st is None:
        raise RuntimeError("Streamlit is not installed. Install requirements.txt first.")

    st.set_page_config(page_title="Document Q&A Bot", layout="centered")
    st.title("Document Q&A Bot")
    st.caption("Ask questions against the indexed local documents.")

    question = st.text_input("Question")
    k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=TOP_K)

    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving context and generating an answer..."):
            result = query_rag_pipeline(question, k=k)

        st.subheader("Answer")
        st.write(result["answer"])

        if result["citations"]:
            st.subheader("Retrieved sources")
            for citation in result["citations"]:
                st.write(f"- {citation}")


if __name__ == "__main__":
    run_cli()
