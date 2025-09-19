import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Sequence
import tiktoken
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )


LOGGER = logging.getLogger(__name__)
DEFAULT_BOOK = Path("pg1497.txt")
DEFAULT_PERSIST_DIR = Path(".chroma")
DEFAULT_COLLECTION_NAME = "book-rag"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_ENCODING = "cl100k_base"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_EMBEDDING_MODEL = "models/embedding-001"


def ensure_api_key(env_var: str = "GEMINI_API_KEY") -> str:
    """Return the Gemini API key or raise an informative error."""

    api_key = os.environ.get(env_var)
    if not api_key:
        raise EnvironmentError(
            "No Gemini API key found. Please set the environment variable"
            f" '{env_var}' with your Google Generative AI key."
        )
    return api_key


def read_book(book_path: Path) -> str:
    """Read the contents of the provided *book_path*."""

    if not book_path.exists():
        raise FileNotFoundError(
            f"Could not locate the book file at '{book_path}'. Make sure the file exists or"
            " specify a different path with --book-path."
        )

    LOGGER.info("Loading book from %s", book_path)
    return book_path.read_text(encoding="utf-8")


def chunk_book(
    *,
    text: str,
    source: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    encoding_name: str = DEFAULT_ENCODING,
) -> List[Document]:
    """Split *text* into :class:`~langchain.docstore.document.Document` objects."""

    encoding = tiktoken.get_encoding(encoding_name)
    token_count = len(encoding.encode(text))
    LOGGER.info("Loaded %s characters (~%s tokens).", len(text), token_count)

    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding_name,
    )
    documents = splitter.create_documents([text])

    for index, document in enumerate(documents):
        document.metadata.setdefault("source", str(source))
        document.metadata.setdefault("chunk", index)
        document.metadata.setdefault("token_count", len(encoding.encode(document.page_content)))

    LOGGER.info("Created %s document chunks (chunk size=%s, overlap=%s).", len(documents), chunk_size, chunk_overlap)
    return documents


def _reset_persist_directory(persist_directory: Path) -> None:
    """Delete *persist_directory* if it exists to ensure a fresh vector store."""

    if persist_directory.exists():
        LOGGER.info("Removing existing Chroma persistence directory at %s", persist_directory)
        shutil.rmtree(persist_directory)


def _vector_count(store: Chroma) -> int:
    """Return the number of documents stored in *store*."""

    collection = getattr(store, "_collection", None)
    if collection is not None:  # pragma: no branch - fast path
        return collection.count()

    result = store.get(ids=None, include=[])
    return len(result.get("ids", []))


def build_vector_store(
    *,
    documents: Sequence[Document],
    persist_directory: Path,
    collection_name: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    google_api_key: str | None = None,
    force_rebuild: bool = False,
) -> Chroma:
    """Create (or reuse) a Chroma vector store for *documents*."""

    if not google_api_key:
        google_api_key = ensure_api_key()

    if force_rebuild:
        _reset_persist_directory(persist_directory)

    persist_directory.mkdir(parents=True, exist_ok=True)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=embedding_model,
        google_api_key=google_api_key,
    )
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

    current_count = _vector_count(vector_store)
    if current_count == 0:
        LOGGER.info("Populating Chroma vector store with %s documents...", len(documents))
        vector_store.add_documents(list(documents))
        vector_store.persist()
        current_count = _vector_count(vector_store)

    LOGGER.info("Vector store contains %s documents.", current_count)
    return vector_store


def build_qa_chain(
    *,
    vector_store: Chroma,
    model: str = DEFAULT_GEMINI_MODEL,
    temperature: float = 0.1,
    google_api_key: str | None = None,
) -> RetrievalQA:
    """Construct a RetrievalQA chain backed by a Gemini chat model."""

    if not google_api_key:
        google_api_key = ensure_api_key()

    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=google_api_key,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return chain


def ask_questions(chain: RetrievalQA, questions: Iterable[str]) -> None:
    """Ask *questions* using *chain* and print the responses."""

    for raw_question in questions:
        question = raw_question.strip()
        if not question:
            continue
        LOGGER.info("Question: %s", question)
        result = chain({"query": question})
        answer = result.get("result")
        sources = result.get("source_documents", [])
        print(f"\nQ: {question}\nA: {answer}\n")
        if sources:
            formatted_sources = sorted(
                {
                    f"{doc.metadata.get('source', 'unknown')} (chunk {doc.metadata.get('chunk', '?')})"
                    for doc in sources
                }
            )
            print(f"Sources: {', '.join(formatted_sources)}\n")


def interactive_session(chain: RetrievalQA) -> None:
    """Start an interactive QA loop with the user."""

    print("Enter a question about the book (type 'exit' or 'quit' to finish).")
    while True:
        try:
            question = input("? ").strip()
        except (EOFError, KeyboardInterrupt):  # pragma: no cover - interactive path
            print()  # friendly newline before exiting
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        result = chain({"query": question})
        answer = result.get("result")
        print(f"\n{answer}\n")
        sources = result.get("source_documents", [])
        if sources:
            formatted_sources = sorted(
                {
                    f"{doc.metadata.get('source', 'unknown')} (chunk {doc.metadata.get('chunk', '?')})"
                    for doc in sources
                }
            )
            print(f"Sources: {', '.join(formatted_sources)}\n")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse CLI arguments from *argv*."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--book-path",
        type=Path,
        default=DEFAULT_BOOK,
        help="Path to the source book text file (default: %(default)s).",
    )
    parser.add_argument(
        "--persist-directory",
        type=Path,
        default=DEFAULT_PERSIST_DIR,
        help="Directory where the Chroma vector store will be persisted.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Name of the Chroma collection (default: %(default)s).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Token chunk size for splitting the book (default: %(default)s).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Token overlap between chunks (default: %(default)s).",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Gemini embeddings model identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini chat model identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for the Gemini chat model (default: %(default)s).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild the Chroma vector store from scratch, even if it already exists.",
    )
    parser.add_argument(
        "--questions",
        nargs="*",
        help="Optional list of questions to ask in non-interactive mode.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by ``python task.py``."""

    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    api_key = ensure_api_key()

    text = read_book(args.book_path)
    documents = chunk_book(
        text=text,
        source=args.book_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    vector_store = build_vector_store(
        documents=documents,
        persist_directory=args.persist_directory,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        google_api_key=api_key,
        force_rebuild=args.force_rebuild,
    )

    qa_chain = build_qa_chain(
        vector_store=vector_store,
        model=args.llm_model,
        temperature=args.temperature,
        google_api_key=api_key,
    )

    if args.questions:
        ask_questions(qa_chain, args.questions)
    else:
        interactive_session(qa_chain)

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())