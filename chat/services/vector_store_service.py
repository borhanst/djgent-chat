"""
Vector store service with multi-backend support.

This module provides a unified interface for vector storage operations
with support for multiple backends (FAISS, ChromaDB, Pinecone).

FAISS is the default backend for local storage.
"""

import logging
from typing import Any, Dict, List, Optional

from django.conf import settings

from .vectorstore.base import BaseVectorStore
from .vectorstore.factory import VectorStoreFactory

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Unified vector store service with multi-backend support.

    This class provides a consistent interface for vector storage operations
    regardless of the underlying backend. It uses VectorStoreFactory to
    create and manage the appropriate backend instance.
    """

    def __init__(
        self,
        folder_name: str = "default",
        embedding_provider=None,
        backend: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the vector store service.

        Args:
            folder_name: Name of the collection/index.
            embedding_provider: An embedding provider instance.
            backend: Backend to use ('faiss', 'chromadb', 'pinecone').
                    Defaults to VECTOR_STORE_BACKEND from settings.
            **kwargs: Additional backend-specific arguments.
        """
        self.folder_name = folder_name
        self._embedding_provider = embedding_provider
        self._backend = backend
        self._kwargs = kwargs
        self._vector_store: Optional[BaseVectorStore] = None

    @property
    def embedding_provider(self):
        """Get the embedding provider."""
        return self._embedding_provider

    @property
    def backend(self) -> str:
        """Get the backend name."""
        if self._backend:
            return self._backend
        return getattr(settings, "VECTOR_STORE_BACKEND", "faiss")

    def _get_vector_store(self) -> BaseVectorStore:
        """Get or create the vector store instance."""
        if self._vector_store is None:
            self._vector_store = VectorStoreFactory.get_vector_store(
                backend=self._backend,
                folder_name=self.folder_name,
                embedding=self._embedding_provider,
                **self._kwargs,
            )
        return self._vector_store

    def create_index(self, dimension: int) -> None:
        """
        Create a new index with the given dimension.

        Note: Some backends (like FAISS) create the index automatically
        when adding the first document.

        Args:
            dimension: The dimension of the vectors to be stored.
        """
        vs = self._get_vector_store()
        logger.info(
            f"Index creation for {self.backend} backend with dimension "
            f"{dimension} handled automatically"
        )

    def load_index(self) -> bool:
        """
        Load an existing index from storage.

        Returns:
            True if the index was loaded successfully, False otherwise.
        """
        vs = self._get_vector_store()
        if hasattr(vs, '_load_index'):
            return vs._load_index()
        return vs.is_loaded()

    def save_index(self) -> None:
        """Save the current index to storage."""
        vs = self._get_vector_store()
        if hasattr(vs, '_save_index'):
            vs._save_index()

    def add_vectors(
        self, vectors: List[List[float]], metadata: List[Dict]
    ) -> None:
        """
        Add vectors to the index with associated metadata.

        Note: This method maintains compatibility but uses text-based
        interface internally.

        Args:
            vectors: A list of vectors to add.
            metadata: A list of metadata dictionaries, one per vector.
        """
        vs = self._get_vector_store()

        # Convert to text format for compatibility
        texts = []
        metadatas = []
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            text = meta.get("text", f"chunk_{i}")
            texts.append(text)
            metadatas.append(meta)

        vs.add_texts(texts, metadatas)

    def search(self, query_vector: List[float], k: int = 5) -> List:
        """
        Search for the k most similar vectors to the query.

        Args:
            query_vector: The query vector.
            k: The number of results to return.

        Returns:
            A list of search results.
        """
        vs = self._get_vector_store()
        docs = vs.similarity_search_by_vector(query_vector, k=k)

        # Convert to standard format
        from .base import SearchResult

        results = []
        for doc in docs:
            results.append(
                SearchResult(
                    text=doc.page_content,
                    score=doc.metadata.get("score", 0.0),
                    metadata=doc.metadata,
                )
            )

        return results

    def delete_index(self) -> None:
        """Delete the current index and remove from storage."""
        vs = self._get_vector_store()
        vs.delete_collection()

    def get_vector_count(self) -> int:
        """
        Get the number of vectors in the index.

        Returns:
            The number of vectors as an integer.
        """
        vs = self._get_vector_store()
        return vs.get_vector_count()

    def get_dimension(self) -> Optional[int]:
        """
        Get the dimension of the vectors in the index.

        Returns:
            The dimension as an integer, or None if no index is loaded.
        """
        vs = self._get_vector_store()
        return vs.get_dimension()

    def is_loaded(self) -> bool:
        """
        Check if an index is currently loaded.

        Returns:
            True if an index is loaded, False otherwise.
        """
        vs = self._get_vector_store()
        return vs.is_loaded()

    def add_text_chunks(
        self, chunks: List[Dict[str, Any]], batch_size: int = 100
    ) -> None:
        """
        Add text chunks to the index by generating embeddings.

        Args:
            chunks: A list of chunk dictionaries, each containing 'text' and metadata.
            batch_size: Number of chunks to process at a time.
        """
        vs = self._get_vector_store()

        texts = [chunk["text"] for chunk in chunks]
        metadatas = [
            {k: v for k, v in chunk.items() if k != "text"} for chunk in chunks
        ]

        vs.add_texts(texts, metadatas)
        logger.info(f"Added {len(chunks)} text chunks to {self.backend} index")

    def search_by_text(self, query: str, k: int = 5) -> List:
        """
        Search for similar chunks by text query.

        Args:
            query: The text query.
            k: The number of results to return.

        Returns:
            A list of search results.
        """
        vs = self._get_vector_store()
        docs = vs.similarity_search(query, k=k)

        # Convert to standard format
        from .base import SearchResult

        results = []
        for doc in docs:
            results.append(
                SearchResult(
                    text=doc.page_content,
                    score=doc.metadata.get("score", 0.0),
                    metadata=doc.metadata,
                )
            )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.

        Returns:
            A dictionary containing index statistics.
        """
        vs = self._get_vector_store()
        stats = vs.get_stats()
        stats["backend"] = self.backend
        return stats


def get_vector_store(
    folder_name: str,
    embedding_provider=None,
    index_base_path: Optional[str] = None,
    backend: Optional[str] = None,
) -> BaseVectorStore:
    """
    Factory function to get a vector store instance.

    This is a convenience function that creates a VectorStoreService
    and returns its underlying vector store.

    Args:
        folder_name: Name of the folder (used to create index path).
        embedding_provider: An embedding provider instance.
        index_base_path: Base path for storing indexes.
        backend: Backend to use ('faiss', 'chromadb', 'pinecone').

    Returns:
        A BaseVectorStore instance.
    """
    return VectorStoreFactory.get_vector_store(
        backend=backend,
        folder_name=folder_name,
        embedding=embedding_provider,
        index_base_path=index_base_path,
    )


# Keep FAISSVectorStore for backward compatibility
from .vectorstore.faiss import FAISSVectorStore

__all__ = [
    "VectorStoreService",
    "get_vector_store",
    "FAISSVectorStore",
]
