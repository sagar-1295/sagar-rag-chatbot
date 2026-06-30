import os
from pathlib import Path
from typing import List

import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.document_loaders import CSVLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency
    fitz = None

try:
    import pdfplumber
except Exception:  # pragma: no cover - optional dependency
    pdfplumber = None
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from tqdm import tqdm

from .config import get_dataset_path


class RagChatbot:
    def __init__(self, dataset_path: str | None = None, max_rows: int | None = None, max_pages: int | None = None, model: str | None = None):
        self.dataset_path = get_dataset_path(dataset_path)
        self.max_rows = max_rows
        self.max_pages = max_pages
        # optional model override (Hugging Face repo id or local path)
        self.model = model
        self.embeddings = self._build_embeddings()
        self.vector_store = self._build_vector_store()
        self.chain = self._build_chain()
        self.history: List[tuple[str, str]] = []

    def _load_documents(self) -> list[Document]:
        if self.dataset_path.is_dir():
            files = sorted(
                [p for p in self.dataset_path.iterdir() if p.suffix.lower() in {".csv", ".xls", ".xlsx", ".pdf"}]
            )
            if not files:
                raise FileNotFoundError(
                    f"No supported dataset files found in directory: {self.dataset_path}. "
                    "Use .csv, .xls, .xlsx, or .pdf files."
                )
            documents: list[Document] = []
            for file_path in files:
                documents.extend(self._load_documents_from_path(file_path))
            if not documents:
                raise ValueError(
                    f"No readable content found in the uploaded dataset directory: {self.dataset_path}. "
                    "Please upload CSV/Excel files or text-based PDFs with extractable text."
                )
            return documents

        if not self.dataset_path.is_file():
            raise FileNotFoundError(
                f"Dataset file not found: {self.dataset_path}. "
                "Please verify the path or set DATASET_PATH to the correct dataset file."
            )
        documents = self._load_documents_from_path(self.dataset_path)
        if not documents:
            raise ValueError(
                f"No readable content found in the uploaded file: {self.dataset_path}. "
                "Please upload CSV/Excel files or text-based PDFs with extractable text."
            )
        return documents

    def _load_documents_from_path(self, file_path: Path) -> list[Document]:
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".csv":
                loader = CSVLoader(file_path=str(file_path))
                documents = loader.load()
            elif suffix in {".xls", ".xlsx"}:
                df = pd.read_excel(file_path)
                documents = []
                for row in df.to_dict(orient="records"):
                    metadata = {k: v for k, v in row.items() if pd.notna(v)}
                    page_content = "\n".join(f"{k}: {v}" for k, v in metadata.items())
                    documents.append(Document(page_content=page_content, metadata=metadata))
            elif suffix == ".pdf":
                try:
                    loader = PyPDFLoader(str(file_path))
                    documents = loader.load()
                except Exception:
                    documents = []

                if not documents and fitz is not None:
                    try:
                        with fitz.open(str(file_path)) as doc:
                            text_chunks = []
                            for page in doc:
                                text = page.get_text()
                                if text and text.strip():
                                    text_chunks.append(text)
                            if text_chunks:
                                documents = [Document(page_content="\n\n".join(text_chunks), metadata={"source": str(file_path)})]
                    except Exception:
                        documents = []

                if not documents and pdfplumber is not None:
                    try:
                        with pdfplumber.open(str(file_path)) as pdf:
                            text_chunks = []
                            for page in pdf.pages:
                                text = page.extract_text() or ""
                                if text and text.strip():
                                    text_chunks.append(text)
                            if text_chunks:
                                documents = [Document(page_content="\n\n".join(text_chunks), metadata={"source": str(file_path)})]
                    except Exception:
                        documents = []
            else:
                raise ValueError(
                    f"Unsupported dataset format: {file_path.suffix}. "
                    "Use .csv, .xls, .xlsx, or .pdf."
                )
        except Exception as exc:
            raise RuntimeError(f"Error loading {file_path}: {exc}") from exc

        documents = [
            doc for doc in documents
            if getattr(doc, "page_content", None) and str(doc.page_content).strip()
        ]

        if suffix == ".pdf":
            if self.max_pages is not None:
                documents = documents[: self.max_pages]
        elif self.max_rows is not None:
            documents = documents[: self.max_rows]

        print(f"Loaded {len(documents)} documents from {file_path}")
        return documents

    def _build_embeddings(self):
        print("Using HuggingFaceEmbeddings")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def _build_vector_store(self):
        documents = self._load_documents()
        chunk_size = int(os.environ.get("RAG_CHUNK_SIZE", 500))
        chunk_overlap = int(os.environ.get("RAG_CHUNK_OVERLAP", 50))
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(documents)
        print(f"Created {len(chunks)} document chunks")

        if not chunks:
            raise ValueError(
                "No readable text chunks could be created from the uploaded dataset. "
                "Please upload files that contain actual text content."
            )

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        print("Embedding documents and building FAISS index...")
        embeddings = self.embeddings.embed_documents(texts)
        text_embedding_pairs = list(tqdm(zip(texts, embeddings), total=len(texts), desc="Embedding"))
        return FAISS.from_embeddings(text_embedding_pairs, self.embeddings, metadatas=metadatas)

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a document Q&A assistant. Answer only from the retrieved context. "
                "If the answer is not found, reply with 'I don't know.'"
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{context}\n\n{question}"),
        ])
        # Prefer OpenAI generation when an API key is present, otherwise use local or Hugging Face Hub.
        llm = None
        hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        use_openai = os.environ.get("OPENAI_API_KEY") is not None
        use_local = os.environ.get("USE_LOCAL_HF") == "1" and not use_openai

        if use_local:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                import torch
                from langchain.llms.base import LLM

                hf_local_model = self.model or os.environ.get("HF_LOCAL_MODEL", "google/flan-t5-large")
                print("Using local Hugging Face model:", hf_local_model)
                tokenizer = AutoTokenizer.from_pretrained(hf_local_model)
                model = AutoModelForSeq2SeqLM.from_pretrained(hf_local_model)
                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                model.to(device)

                from typing import Any
                from pydantic import Field

                class LocalHuggingFace(LLM):
                    model: Any = Field(..., repr=False)
                    tokenizer: Any = Field(..., repr=False)
                    device: Any = Field(..., repr=False)
                    max_length: int = 256

                    def _call(self, prompt: str, stop=None) -> str:
                        max_input_len = getattr(self.tokenizer, "model_max_length", 512)
                        inputs = self.tokenizer(
                            prompt, return_tensors="pt", truncation=True, max_length=max_input_len
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        gen = self.model.generate(**inputs, max_new_tokens=self.max_length)
                        out = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
                        return out

                    @property
                    def _identifying_params(self):
                        return {"max_length": self.max_length}

                    @property
                    def _llm_type(self):
                        return "local_hf"

                llm = LocalHuggingFace(model=model, tokenizer=tokenizer, device=device, max_length=int(os.environ.get("HF_MAX_LENGTH", 256)))
            except Exception as e:
                print("Local Hugging Face model failed to load:", e)
                llm = None

        if use_openai:
            try:
                openai_model = self.model or os.environ.get("OPENAI_MODEL", "gpt-5-mini")
                print("Using OpenAI LLM:", openai_model)
                # Some OpenAI reasoning models require the default temperature setting.
                llm = ChatOpenAI(model_name=openai_model, temperature=1)
            except Exception as e:
                print("OpenAI LLM failed to load:", e)
                llm = None

        if llm is None and hf_token:
            try:
                from langchain.llms import HuggingFaceHub

                print("Using HuggingFaceHub LLM")
                hf_model = self.model or os.environ.get("HF_MODEL", "google/flan-t5-small")
                llm = HuggingFaceHub(repo_id=hf_model, huggingfacehub_api_token=hf_token, model_kwargs={"temperature": 0})
            except Exception:
                llm = None

        if llm is None:
            raise RuntimeError(
                "No LLM configured. Set USE_OPENAI=1 and OPENAI_API_KEY to use OpenAI, "
                "or set HUGGINGFACEHUB_API_TOKEN to use Hugging Face Hub, "
                "or set USE_LOCAL_HF=1 to use a local Hugging Face model."
            )
        retrieval_k = int(os.environ.get("RAG_RETRIEVER_K", 4))
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": retrieval_k}),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt},
        )

    @staticmethod
    def _trim_history(history: list[tuple[str, str]], max_turns: int = 6) -> list[tuple[str, str]]:
        return history[-max_turns:]

    @staticmethod
    def _format_sources(source_documents: list[Document]) -> list[str]:
        sources = []
        seen_ids = set()
        for doc in source_documents:
            student_id = doc.metadata.get("student_id", "unknown")
            if student_id not in seen_ids:
                seen_ids.add(student_id)
                sources.append(f"Student_ID={student_id}")
                if len(sources) >= 5:
                    break
        return sources

    def ask(self, question: str) -> dict:
        trimmed = self._trim_history(self.history)
        chat_messages = []
        for user_q, assistant_a in trimmed:
            chat_messages.append(HumanMessage(content=user_q))
            chat_messages.append(AIMessage(content=assistant_a))

        input_data = {
            "question": question,
            "chat_history": chat_messages,
            "history": chat_messages,
        }

        try:
            result = self.chain(input_data)
            answer = result.get("answer") or result.get("text") or "I don't know."
        except Exception as exc:
            error_message = str(exc)
            print(f"\n--- ERROR: Chain failed: {error_message}")
            if hasattr(exc, '__cause__') and exc.__cause__ is not None:
                print(f"Caused by: {exc.__cause__}")
            print("--- END ERROR ---\n")
            return {"answer": "Error: see console for details.", "sources": []}

        if os.environ.get("RAG_DEBUG"):
            docs = result.get("source_documents", [])
            print("\n--- RAG DEBUG: Retrieved documents ---")
            for idx, doc in enumerate(docs, 1):
                print(f"DOC {idx}: metadata={doc.metadata}")
                print(doc.page_content[:500].replace("\n", " "))
                print()
            print("--- END RAG DEBUG ---\n")

        self.history.append((question, answer))
        return {
            "answer": answer,
            "sources": self._format_sources(result.get("source_documents", [])),
        }

    def run(self):
        print(f"Loading dataset from: {self.dataset_path}")
        print("Conversational RAG agent is ready. Type 'quit' to exit.")
        if os.environ.get("RAG_DEBUG"):
            print("RAG_DEBUG=1 enabled")
            print(f"RAG_CHUNK_SIZE={os.environ.get('RAG_CHUNK_SIZE', 500)}")
            print(f"RAG_CHUNK_OVERLAP={os.environ.get('RAG_CHUNK_OVERLAP', 50)}")
            print(f"RAG_RETRIEVER_K={os.environ.get('RAG_RETRIEVER_K', 4)}")
            print(f"USE_LOCAL_HF={os.environ.get('USE_LOCAL_HF')} HUGGINGFACEHUB_API_TOKEN={'set' if os.environ.get('HUGGINGFACEHUB_API_TOKEN') else 'unset'} USE_OPENAI={os.environ.get('USE_OPENAI')}\n")
        while True:
            question = input("\nUser: ").strip()
            if question.lower() in {"quit", "exit", "q"}:
                print("Goodbye.")
                break
            if not question:
                continue
            result = self.ask(question)
            print("\nAssistant:")
            print(result["answer"])
            if result["sources"]:
                print("\nRetrieved sources:")
                print("\n".join(result["sources"]))
