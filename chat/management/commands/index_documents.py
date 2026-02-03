"""
Django management command for indexing documents into vector store.

Supports both legacy and LangChain-based embedding providers.

Usage:
    python manage.py index_documents --folder <folder_name> [--reindex] [--chunk-size <size>] [--chunk-overlap <overlap>]
    python manage.py index_documents --no-langchain --folder <folder_name> [--reindex]
"""

import logging
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from chat.services import (
    DocumentLoader,
    TextChunker,
    get_embedding_provider,
)
from chat.services.vector_store_service import get_vector_store
from chat.utils.config import get_documents_base_path
from config.settings import djgent_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """
    Management command to index documents into vector store.

    This command loads documents from a specified folder, chunks them,
    generates embeddings, and stores them in a vector store.
    Supports both legacy and LangChain-based embedding providers.
    """

    help = "Index documents from a folder into vector store (supports LangChain mode)"

    def add_arguments(self, parser):
        """Add command line arguments."""
        parser.add_argument(
            "--folder",
            type=str,
            default="",
            help="Name of the folder containing documents (default: default)",
        )
        parser.add_argument(
            "--folder-path",
            type=str,
            default=None,
            help="Absolute path to the documents folder (overrides --folder)",
        )
        parser.add_argument(
            "--reindex",
            action="store_true",
            help="Force re-indexing by deleting existing index",
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=None,
            help="Override default chunk size",
        )
        parser.add_argument(
            "--chunk-overlap",
            type=int,
            default=None,
            help="Override default chunk overlap",
        )
        parser.add_argument(
            "--embedding-provider",
            type=str,
            default=None,
            choices=["openai", "gemini", "huggingface"],
            help="Override default embedding provider",
        )
        parser.add_argument(
            "--chunking-strategy",
            type=str,
            default="langchain",
            choices=[
                "langchain",
                "langchain-sentence",
                "character",
                "sentence",
                "paragraph",
            ],
            help="Chunking strategy to use (default: langchain)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=100,
            help="Number of chunks to process at a time (default: 100)",
        )
        parser.add_argument(
            "--extensions",
            type=str,
            default=None,
            help="Comma-separated list of file extensions to include (e.g., pdf,txt,md)",
        )
        parser.add_argument(
            "--recursive",
            action="store_true",
            default=True,
            help="Search subdirectories recursively (default: True)",
        )
        parser.add_argument(
            "--no-recursive",
            action="store_false",
            dest="recursive",
            help="Do not search subdirectories recursively",
        )
        parser.add_argument(
            "--vector-store-backend",
            type=str,
            default=None,
            choices=["faiss", "chromadb", "pinecone"],
            help="Vector store backend to use (default: from settings)",
        )
        parser.add_argument(
            "--use-langchain",
            action="store_true",
            default=True,
            help="Use LangChain services for indexing (default: True)",
        )
        parser.add_argument(
            "--index-type",
            type=str,
            default="flat",
            choices=["flat", "ivf", "hnsw"],
            help="FAISS index type for LangChain mode (default: flat)",
        )

    def handle(self, *args, **options):
        """Execute the command."""
        # Parse options
        folder_name = options["folder"]
        folder_path = options["folder_path"]
        reindex = options["reindex"]
        chunk_size = options["chunk_size"]
        chunk_overlap = options["chunk_overlap"]
        embedding_provider_name = options["embedding_provider"]
        chunking_strategy = options["chunking_strategy"]
        batch_size = options["batch_size"]
        extensions = options["extensions"]
        recursive = options["recursive"]
        use_langchain = options["use_langchain"]
        index_type = options["index_type"]
        vector_store_backend = options["vector_store_backend"]

        # Apply defaults from settings if still None/empty
        if chunking_strategy == "langchain":
            chunking_strategy = djgent_settings.get(
                "DEFAULT_CHUNKING_STRATEGY", "langchain"
            )
        if batch_size == 100:
            batch_size = djgent_settings.get("DEFAULT_BATCH_SIZE", 100)
        if vector_store_backend is None:
            vector_store_backend = djgent_settings.get("VECTOR_STORE_BACKEND")
        if use_langchain is True:
            use_langchain = djgent_settings.get("USE_LANGCHAIN", True)
        if index_type == "flat":
            index_type = djgent_settings.get("DEFAULT_INDEX_TYPE", "flat")

        # Determine folder path
        if folder_path:
            documents_path = Path(folder_path)
        else:
            documents_base = get_documents_base_path()
            documents_path = documents_base / folder_name

        # Validate folder path
        if not documents_path.exists():
            raise CommandError(
                f"Documents folder does not exist: {documents_path}\n"
                f"Please create the folder or provide a valid path using --folder-path"
            )

        if not documents_path.is_dir():
            raise CommandError(f"Path is not a directory: {documents_path}")

        self.stdout.write(
            self.style.SUCCESS(f"Indexing documents from: {documents_path}")
        )

        # Get configuration from database
        try:
            from chat.models import RAGConfiguration

            config = RAGConfiguration.objects.order_by("-created_at").first()

            if config:
                # Use config values as defaults
                if chunk_size is None:
                    chunk_size = config.chunk_size
                if chunk_overlap is None:
                    chunk_overlap = config.chunk_overlap
                if embedding_provider_name is None:
                    embedding_provider_name = config.embedding_provider
        except Exception as e:
            logger.warning(f"Could not load RAG configuration: {e}")

        # Apply defaults from settings if still None
        if chunk_size is None:
            chunk_size = getattr(settings, "DEFAULT_CHUNK_SIZE", 1000)
        if chunk_overlap is None:
            chunk_overlap = getattr(settings, "DEFAULT_CHUNK_OVERLAP", 200)
        if embedding_provider_name is None:
            embedding_provider_name = getattr(
                settings, "EMBEDDING_PROVIDER", "gemini"
            )

        # Parse extensions
        if extensions:
            extensions_list = [
                ext.strip().lower() for ext in extensions.split(",")
            ]
            # Add leading dot if not present
            extensions_list = [
                ext if ext.startswith(".") else f".{ext}"
                for ext in extensions_list
            ]
        else:
            extensions_list = None

        # Display configuration
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write("Indexing Configuration:")
        self.stdout.write("=" * 60)
        self.stdout.write(f"  Folder: {documents_path}")
        self.stdout.write(f"  Embedding Provider: {embedding_provider_name}")
        self.stdout.write(f"  Chunk Size: {chunk_size}")
        self.stdout.write(f"  Chunk Overlap: {chunk_overlap}")
        self.stdout.write(f"  Chunking Strategy: {chunking_strategy}")
        self.stdout.write(f"  Batch Size: {batch_size}")
        self.stdout.write(f"  Recursive: {recursive}")
        if extensions_list:
            self.stdout.write(f"  Extensions: {', '.join(extensions_list)}")
        self.stdout.write(f"  Use LangChain: {use_langchain}")
        if use_langchain:
            self.stdout.write(
                f"  Vector Store Backend: {vector_store_backend or 'default'}"
            )
            self.stdout.write(f"  Index Type: {index_type}")
        self.stdout.write(f"  Reindex: {reindex}")
        self.stdout.write("=" * 60 + "\n")

        try:
            # Initialize vector store based on mode
            if use_langchain:
                # Use LangChain-based embeddings and vector store
                self.stdout.write(
                    "Initializing LangChain embedding provider..."
                )
                from chat.services.langchain_embeddings import (
                    get_langchain_embeddings,
                )

                embeddings = get_langchain_embeddings(embedding_provider_name)
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✓ Using LangChain embeddings with provider: {embedding_provider_name}"
                    )
                )

                # Initialize LangChain FAISS vector store
                self.stdout.write(
                    "Initializing LangChain FAISS vector store..."
                )
                from chat.services.langchain_vectorstore import LangChainFAISS
                from chat.utils.config import get_faiss_index_base_path

                index_base_path = get_faiss_index_base_path()
                index_path = index_base_path / folder_name

                vector_store = LangChainFAISS(
                    embedding=embeddings,
                    index_path=str(index_path),
                    index_type=index_type,
                )

                # Check if index exists
                if vector_store._load_index():
                    if reindex:
                        self.stdout.write(
                            "  Deleting existing index (reindex mode)..."
                        )
                        vector_store.delete_collection()
                    else:
                        # Get vector count from loaded index
                        vector_count = (
                            vector_store._index.ntotal
                            if vector_store._index
                            else 0
                        )
                        self.stdout.write(
                            self.style.WARNING(
                                f"  Index already exists with {vector_count} vectors. "
                                "Use --reindex to overwrite."
                            )
                        )
                        response = input(
                            "  Continue and add to existing index? (y/N): "
                        )
                        if response.lower() != "y":
                            self.stdout.write("Indexing cancelled.")
                            return
                else:
                    self.stdout.write("  Creating new index...")
            else:
                # Use legacy embedding provider
                self.stdout.write("Initializing embedding provider...")
                embedding_provider = get_embedding_provider(
                    embedding_provider_name
                )
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✓ Using model: {embedding_provider.model_name}"
                    )
                )

                # Initialize vector store
                self.stdout.write("Initializing vector store...")
                vector_store = get_vector_store(
                    folder_name=folder_name,
                    embedding_provider=embedding_provider,
                    backend=vector_store_backend,
                )

                # Check if index exists
                if vector_store.is_loaded():
                    if reindex:
                        self.stdout.write(
                            "  Deleting existing index (reindex mode)..."
                        )
                        vector_store.delete_collection()
                    else:
                        self.stdout.write(
                            self.style.WARNING(
                                f"  Index already exists with {vector_store.get_vector_count()} vectors. "
                                "Use --reindex to overwrite."
                            )
                        )
                        response = input(
                            "  Continue and add to existing index? (y/N): "
                        )
                        if response.lower() != "y":
                            self.stdout.write("Indexing cancelled.")
                            return

            # Load documents
            self.stdout.write("Loading documents...")
            documents = DocumentLoader.load_documents_from_folder(
                folder_path=str(documents_path),
                recursive=recursive,
                extensions=extensions_list,
            )

            if not documents:
                self.stdout.write(
                    self.style.WARNING("  No documents found to index.")
                )
                return

            self.stdout.write(
                self.style.SUCCESS(f"  ✓ Loaded {len(documents)} documents")
            )

            # Display document summary
            total_chars = sum(len(text) for text in documents.values())
            self.stdout.write(f"  Total characters: {total_chars:,}")

            # Chunk documents
            self.stdout.write("Chunking documents...")
            chunker = TextChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strategy=chunking_strategy,
            )

            chunks = chunker.chunk_documents(
                documents, base_path=str(documents_path)
            )

            if not chunks:
                self.stdout.write(self.style.WARNING("  No chunks generated."))
                return

            self.stdout.write(
                self.style.SUCCESS(f"  ✓ Generated {len(chunks)} chunks")
            )

            # Add chunks to index (different methods for LangChain vs legacy)
            if use_langchain:
                # LangChain mode: use add_texts
                self.stdout.write(
                    "Generating embeddings and adding to index..."
                )
                texts = [chunk["text"] for chunk in chunks]
                metadatas = [
                    {k: v for k, v in chunk.items() if k != "text"}
                    for chunk in chunks
                ]
                vector_store.add_texts(texts, metadatas)
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✓ Added {len(chunks)} chunks to index"
                    )
                )

                # Display final statistics
                self.stdout.write("\n" + "=" * 60)
                self.stdout.write("Indexing Complete!")
                self.stdout.write("=" * 60)
                stats = vector_store.get_stats()
                self.stdout.write(f"  Total vectors: {stats['vector_count']:,}")
                self.stdout.write(f"  Dimension: {stats['dimension']}")
                self.stdout.write(f"  Index type: {stats['index_type']}")
                self.stdout.write(f"  Index path: {stats['index_path']}")
                self.stdout.write("=" * 60)
            else:
                # Legacy mode: use add_texts (auto-creates index if needed)
                self.stdout.write(
                    "Generating embeddings and adding to index..."
                )

                # Convert chunks to texts and metadatas
                texts = [chunk["text"] for chunk in chunks]
                metadatas = [
                    {k: v for k, v in chunk.items() if k != "text"}
                    for chunk in chunks
                ]

                vector_store.add_texts(texts, metadatas)
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✓ Added {len(chunks)} chunks to index"
                    )
                )

                # Display final statistics
                self.stdout.write("\n" + "=" * 60)
                self.stdout.write("Indexing Complete!")
                self.stdout.write("=" * 60)
                stats = vector_store.get_stats()
                self.stdout.write(f"  Total vectors: {stats['vector_count']:,}")
                self.stdout.write(f"  Dimension: {stats['dimension']}")
                self.stdout.write(f"  Index type: {stats['index_type']}")
                self.stdout.write(f"  Index path: {stats['index_path']}")
                self.stdout.write("=" * 60)

            self.stdout.write(
                self.style.SUCCESS("\n✓ Documents indexed successfully!")
            )

        except Exception as e:
            logger.exception("Error during indexing")
            raise CommandError(f"Error during indexing: {e}")
