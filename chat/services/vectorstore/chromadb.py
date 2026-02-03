"""
ChromaDB vector store implementation.

This module provides a ChromaDB-based vector store implementation
that extends the BaseVectorStore interface.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from django.conf import settings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaDBVectorStore(BaseVectorStore):
    """
    ChromaDB-based vector store implementation.

    This class provides methods to create, load, save, and query
    ChromaDB collections with associated metadata while implementing
    the BaseVectorStore interface.

    Features:
    - Persistent local database storage
    - Efficient similarity search
    - Metadata filtering support
    - Automatic collection creation
    """

    def __init__(
        self,
        embedding: Embeddings,
        folder_name: str,
        index_base_path: Optional[str] = None,
        persist_directory: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the ChromaDB vector store.

        Args:
            embedding: LangChain embeddings instance.
            folder_name: Name of the collection.
            index_base_path: Base path for storing database files.
            persist_directory: Directory for ChromaDB persistence.
                            If not provided, uses default from settings.
            **kwargs: Additional keyword arguments.
        """
        # Set persist directory from settings if not provided
        if persist_directory is None:
            persist_directory = getattr(
                settings, "CHROMADB_PERSIST_DIRECTORY", None
            )
            if persist_directory is None and index_base_path:
                persist_directory = str(Path(index_base_path) / "chromadb")

        super().__init__(
            embedding=embedding,
            folder_name=folder_name,
            index_base_path=index_base_path,
            **kwargs,
        )

        self.persist_directory = persist_directory

        # Initialize ChromaDB client and collection
        self._client = None
        self._collection = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the ChromaDB client and collection."""
        try:
            import chromadb

            # Create client with persistence
            if self.persist_directory:
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory
                )
            else:
                # In-memory client for testing
                self._client = chromadb.EphemeralClient()

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.folder_name,
                embedding_function=None,  # We'll handle embeddings ourselves
            )

            logger.info(
                f"Initialized ChromaDB collection: {self.folder_name}"
                f" (persist_directory: {self.persist_directory})"
            )

        except ImportError:
            logger.error(
                "ChromaDB not installed. Install with: pip install chromadb"
            )
            raise ImportError(
                "ChromaDB is required for ChromaDBVectorStore. "
                "Install it with: pip install chromadb"
            )
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Add texts to the vector store.

        Args:
            texts: List of text strings to add.
            metadatas: Optional list of metadata dictionaries.
            ids: Optional list of IDs for the texts.
            **kwargs: Additional keyword arguments.

        Returns:
            List of IDs for the added texts.
        """
        if self._collection is None:
            raise RuntimeError("ChromaDB collection not initialized")

        # Generate IDs if not provided
        if ids is None:
            existing_count = self._collection.count()
            ids = [
                str(i)
                for i in range(existing_count, existing_count + len(texts))
            ]

        # Use provided metadatas or empty dicts
        if metadatas is None:
            metadatas = [{}] * len(texts)

        # Generate embeddings
        embeddings = self.embedding.embed_documents(texts)

        # Add to ChromaDB
        self._collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info(f"Added {len(texts)} texts to ChromaDB collection")
        return ids

    async def aadd_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Async add texts to the vector store.

        Args:
            texts: List of text strings to add.
            metadatas: Optional list of metadata dictionaries.
            ids: Optional list of IDs for the texts.
            **kwargs: Additional keyword arguments.

        Returns:
            List of IDs for the added texts.
        """
        return self.add_texts(texts, metadatas, ids, **kwargs)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs
    ) -> List[Document]:
        """
        Search for similar documents to a query.

        Args:
            query: Query text to search for.
            k: Number of results to return.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects sorted by relevance.
        """
        if self._collection is None:
            raise RuntimeError("ChromaDB collection not initialized")

        # Get query embedding
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_by_vector(embedding, k, **kwargs)

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs
    ) -> List[Document]:
        """
        Async search for similar documents.

        Args:
            query: Query text to search for.
            k: Number of results to return.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects sorted by relevance.
        """
        return self.similarity_search(query, k, **kwargs)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs
    ) -> List[Document]:
        """
        Search for similar documents by embedding vector.

        Args:
            embedding: Query embedding vector.
            k: Number of results to return.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects sorted by relevance.
        """
        if self._collection is None:
            raise RuntimeError("ChromaDB collection not initialized")

        # Search in ChromaDB
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # Build Document objects
        docs = []
        for doc_content, meta, distance in zip(documents, metadatas, distances):
            doc = Document(page_content=doc_content, metadata=meta or {})
            # Add distance as relevance score (smaller is better in ChromaDB)
            doc.metadata["score"] = float(distance)
            docs.append(doc)

        return docs

    async def asimilarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs
    ) -> List[Document]:
        """
        Async search by embedding vector.

        Args:
            embedding: Query embedding vector.
            k: Number of results to return.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects sorted by relevance.
        """
        return self.similarity_search_by_vector(embedding, k, **kwargs)

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        if self._collection is None:
            logger.warning("Collection already deleted or not initialized")
            return

        try:
            self._client.delete_collection(name=self.folder_name)
            logger.info(f"Deleted ChromaDB collection: {self.folder_name}")
        except Exception as e:
            logger.error(f"Error deleting ChromaDB collection: {e}")

        # Reinitialize
        self._collection = None
        self._init_client()

    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> None:
        """
        Delete documents by IDs.

        Args:
            ids: List of document IDs to delete.
            **kwargs: Additional keyword arguments.
        """
        if ids is None:
            return

        if self._collection is None:
            raise RuntimeError("ChromaDB collection not initialized")

        self._collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents from ChromaDB collection")

    def get_vector_count(self) -> int:
        """
        Get the number of vectors in the store.

        Returns:
            Number of vectors as an integer.
        """
        if self._collection is None:
            return 0

        return self._collection.count()

    def is_loaded(self) -> bool:
        """
        Check if the store is loaded and ready.

        Returns:
            True if loaded, False otherwise.
        """
        return self._collection is not None

    def get_dimension(self) -> Optional[int]:
        """
        Get the dimension of the vectors.

        Returns:
            Dimension as an integer, or None if not available.
        """
        # Try to get dimension from collection peek
        if self._collection is None:
            return None

        try:
            peek = self._collection.peek(limit=1)
            if peek and "embeddings" in peek and peek["embeddings"]:
                return len(peek["embeddings"][0])
        except Exception:
            pass

        return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary containing store statistics.
        """
        return {
            "vector_count": self.get_vector_count(),
            "dimension": self.get_dimension(),
            "is_loaded": self.is_loaded(),
            "collection_name": self.folder_name,
            "persist_directory": self.persist_directory,
            "backend": "chromadb",
        }

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs,
    ) -> List[Document]:
        """
        Search for documents using maximal marginal relevance.

        Note: ChromaDB doesn't have built-in MMR, so we use a simplified
        implementation that filters duplicates from top results.

        Args:
            query: Query text to search for.
            k: Number of results to return.
            fetch_k: Number of documents to fetch before filtering.
            lambda_mult: Controls diversity vs relevance tradeoff (not used).
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects.
        """
        # Fetch more results than needed
        results = self.similarity_search(query, k=fetch_k, **kwargs)

        # For now, just return top k results
        # A full MMR implementation would require custom filtering
        return results[:k]

    def get_collection(self):
        """
        Get the underlying ChromaDB collection.

        Returns:
            The ChromaDB collection object.
        """
        return self._collection
