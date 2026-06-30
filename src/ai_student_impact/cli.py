import argparse
from .rag import RagChatbot


def main():
    parser = argparse.ArgumentParser(description="Run the Conversational RAG chatbot.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the ai_student_impact dataset file",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of rows to load from CSV/Excel files",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional maximum number of pages to load from PDF files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional Hugging Face model id to use for generation (overrides env vars)",
    )
    args = parser.parse_args()
    model = RagChatbot(dataset_path=args.dataset, max_rows=args.max_rows, max_pages=args.max_pages, model=args.model)
    model.run()


if __name__ == "__main__":
    main()
