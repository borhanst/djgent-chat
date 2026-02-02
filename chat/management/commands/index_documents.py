"""
Django management command for indexing documents into FAISS.

Usage:
    python manage.py index_documents --folder <folder_name> [--reindex] [--chunk-size <size>] [--chunk-overlap <overlap>]
"""

import os
import sys
import logging
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

from chat.services import (
    get_embedding_provider,
    get_vector_store,
    DocumentLoader,
    TextChunker,
)
from chat.utils import get_documents_base_path, ensure_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """
    Management command to index documents into FAISS vector store.
    
    This command loads documents from a specified folder, chunks them,
    generates embeddings, and stores them in a FAISS index.
    """
    
    help = 'Index documents from a folder into FAISS vector store'
    
    def add_arguments(self, parser):
        """Add command line arguments."""
        parser.add_argument(
            '--folder',
            type=str,
            default='default',
            help='Name of the folder containing documents (default: default)'
        )
        parser.add_argument(
            '--folder-path',
            type=str,
            default=None,
            help='Absolute path to the documents folder (overrides --folder)'
        )
        parser.add_argument(
            '--reindex',
            action='store_true',
            help='Force re-indexing by deleting existing index'
        )
        parser.add_argument(
            '--chunk-size',
            type=int,
            default=None,
            help='Override default chunk size'
        )
        parser.add_argument(
            '--chunk-overlap',
            type=int,
            default=None,
            help='Override default chunk overlap'
        )
        parser.add_argument(
            '--embedding-provider',
            type=str,
            default=None,
            choices=['openai', 'gemini', 'huggingface'],
            help='Override default embedding provider'
        )
        parser.add_argument(
            '--chunking-strategy',
            type=str,
            default='character',
            choices=['character', 'sentence', 'paragraph'],
            help='Chunking strategy to use (default: character)'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help='Number of chunks to process at a time (default: 100)'
        )
        parser.add_argument(
            '--extensions',
            type=str,
            default=None,
            help='Comma-separated list of file extensions to include (e.g., pdf,txt,md)'
        )
        parser.add_argument(
            '--recursive',
            action='store_true',
            default=True,
            help='Search subdirectories recursively (default: True)'
        )
        parser.add_argument(
            '--no-recursive',
            action='store_false',
            dest='recursive',
            help='Do not search subdirectories recursively'
        )
    
    def handle(self, *args, **options):
        """Execute the command."""
        # Parse options
        folder_name = options['folder']
        folder_path = options['folder_path']
        reindex = options['reindex']
        chunk_size = options['chunk_size']
        chunk_overlap = options['chunk_overlap']
        embedding_provider_name = options['embedding_provider']
        chunking_strategy = options['chunking_strategy']
        batch_size = options['batch_size']
        extensions = options['extensions']
        recursive = options['recursive']
        
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
            raise CommandError(
                f"Path is not a directory: {documents_path}"
            )
        
        self.stdout.write(self.style.SUCCESS(f"Indexing documents from: {documents_path}"))
        
        # Get configuration
        try:
            from chat.models import RAGConfiguration
            config = RAGConfiguration.objects.order_by('-created_at').first()
            
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
            chunk_size = getattr(settings, 'DEFAULT_CHUNK_SIZE', 1000)
        if chunk_overlap is None:
            chunk_overlap = getattr(settings, 'DEFAULT_CHUNK_OVERLAP', 200)
        if embedding_provider_name is None:
            embedding_provider_name = getattr(settings, 'EMBEDDING_PROVIDER', 'openai')
        
        # Parse extensions
        if extensions:
            extensions_list = [ext.strip().lower() for ext in extensions.split(',')]
            # Add leading dot if not present
            extensions_list = [
                ext if ext.startswith('.') else f'.{ext}'
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
        self.stdout.write(f"  Reindex: {reindex}")
        self.stdout.write("=" * 60 + "\n")
        
        try:
            # Initialize embedding provider
            self.stdout.write("Initializing embedding provider...")
            embedding_provider = get_embedding_provider(embedding_provider_name)
            self.stdout.write(self.style.SUCCESS(f"  ✓ Using model: {embedding_provider.model_name}"))
            
            # Initialize vector store
            self.stdout.write("Initializing vector store...")
            vector_store = get_vector_store(
                folder_name=folder_name,
                embedding_provider=embedding_provider
            )
            
            # Check if index exists
            if vector_store.is_loaded():
                if reindex:
                    self.stdout.write("  Deleting existing index (reindex mode)...")
                    vector_store.delete_index()
                else:
                    self.stdout.write(
                        self.style.WARNING(
                            f"  Index already exists with {vector_store.get_vector_count()} vectors. "
                            "Use --reindex to overwrite."
                        )
                    )
                    response = input("  Continue and add to existing index? (y/N): ")
                    if response.lower() != 'y':
                        self.stdout.write("Indexing cancelled.")
                        return
            
            # Load documents
            self.stdout.write("Loading documents...")
            documents = DocumentLoader.load_documents_from_folder(
                folder_path=str(documents_path),
                recursive=recursive,
                extensions=extensions_list
            )
            
            if not documents:
                self.stdout.write(self.style.WARNING("  No documents found to index."))
                return
            
            self.stdout.write(self.style.SUCCESS(f"  ✓ Loaded {len(documents)} documents"))
            
            # Display document summary
            total_chars = sum(len(text) for text in documents.values())
            self.stdout.write(f"  Total characters: {total_chars:,}")
            
            # Chunk documents
            self.stdout.write("Chunking documents...")
            chunker = TextChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strategy=chunking_strategy
            )
            
            chunks = chunker.chunk_documents(documents, base_path=str(documents_path))
            
            if not chunks:
                self.stdout.write(self.style.WARNING("  No chunks generated."))
                return
            
            self.stdout.write(self.style.SUCCESS(f"  ✓ Generated {len(chunks)} chunks"))
            
            # Create or load index
            if not vector_store.is_loaded():
                self.stdout.write("Creating new FAISS index...")
                dimension = embedding_provider.get_dimension()
                vector_store.create_index(dimension)
                self.stdout.write(self.style.SUCCESS(f"  ✓ Created index with dimension {dimension}"))
            
            # Add chunks to index
            self.stdout.write("Generating embeddings and adding to index...")
            vector_store.add_text_chunks(chunks, batch_size=batch_size)
            self.stdout.write(self.style.SUCCESS(f"  ✓ Added {len(chunks)} chunks to index"))
            
            # Save index
            self.stdout.write("Saving index to disk...")
            vector_store.save_index()
            self.stdout.write(self.style.SUCCESS(f"  ✓ Index saved to: {vector_store.index_path}"))
            
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
            
            self.stdout.write(self.style.SUCCESS("\n✓ Documents indexed successfully!"))
            
        except Exception as e:
            logger.exception("Error during indexing")
            raise CommandError(f"Error during indexing: {e}")
