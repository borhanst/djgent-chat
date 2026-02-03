"""
Abstract base class for vector store backends.

This module defines the interface that all vector store implementations
must follow to ensure consistent behavior across different backends.
"""

import logging
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """
    Abstract base class for vector store backends.

    This class defines the interface that all vector store implementations
    must follow. It extends LangChain's VectorStore interface while
    providing additional methods specific to this application.

    Subclasses must implement all abstract methods. They can optionally
    extend existing functionality to support backend-specific features.
    """

    def __init__(
        self,
        embedding: Embeddings,
        folder_name: str,
        index_base_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the vector store.

        Args:
            embedding: LangChain embeddings instance.
            folder_name: Name of the collection/index.
            index_base_path: Base path for storing index files (if applicable).
            **kwargs: Additional backend-specific arguments.
        """
        self.embedding = embedding
        self.folder_name = folder_name
        self.index_base_path = index_base_path
        self.kwargs = kwargs

    @property
    def embedding(self) -> Embeddings:
        """Get the embedding function."""
        return self._embedding

    @embedding.setter
    def embedding(self, value: Embeddings) -> None:
        """Set the embedding function."""
        self._embedding = value

    @property
    def folder_name(self) -> str:
        """Get the folder/collection name."""
        return self._folder_name

    @folder_name.setter
    def folder_name(self, value: str) -> None:
        """Set the folder/collection name."""
        self._folder_name = value

    @property
    def index_base_path(self) -> Optional[str]:
        """Get the base path for index storage."""
        return self._index_base_path

    @index_base_path.setter
    def index_base_path(self, value: Optional[str]) -> None:
        """Set the base path for index storage."""
        self._index_base_path = value

    @property
    def collection_name(self) -> str:
        """
        Get the collection/index name.

        Returns:
            Collection name as a string.
        """
        return self.folder_name

    @property
    def index_path(self) -> Optional[Path]:
        """
        Get the index path (if applicable).

        Returns:
            Path object or None.
        """
        if self.index_base_path:
            safe_folder_name = self.folder_name.replace("/", "_").replace(
                "\\", "_"
            )
            return Path(self.index_base_path) / safe_folder_name
        return None

    # Abstract methods that must be implemented by subclasses

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
        raise NotImplementedError("Subclasses must implement add_texts")

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
        raise NotImplementedError("Subclasses must implement similarity_search")

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
        raise NotImplementedError(
            "Subclasses must implement similarity_search_by_vector"
        )

    def delete_collection(self) -> None:
        """
        Delete the entire collection/index.
        """
        raise NotImplementedError("Subclasses must implement delete_collection")

    def get_vector_count(self) -> int:
        """
        Get the number of vectors in the store.

        Returns:
            Number of vectors as an integer.
        """
        raise NotImplementedError("Subclasses must implement get_vector_count")

    def is_loaded(self) -> bool:
        """
        Check if the store is loaded and ready.

        Returns:
            True if loaded, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement is_loaded")

    def get_dimension(self) -> Optional[int]:
        """
        Get the dimension of the vectors.

        Returns:
            Dimension as an integer, or None if not available.
        """
        raise NotImplementedError("Subclasses must implement get_dimension")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary containing store statistics.
        """
        raise NotImplementedError("Subclasses must implement get_stats")

    # Optional async methods with default implementations

    async def aadd_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Async add texts to the vector store.

        Default implementation calls the sync version.
        Subclasses can override for better async support.

        Args:
            texts: List of text strings to add.
            metadatas: Optional list of metadata dictionaries.
            ids: Optional list of IDs for the texts.
            **kwargs: Additional keyword arguments.

        Returns:
            List of IDs for the added texts.
        """
        return self.add_texts(texts, metadatas, ids, **kwargs)

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs
    ) -> List[Document]:
        """
        Async search for similar documents.

        Default implementation calls the sync version.
        Subclasses can override for better async support.

        Args:
            query: Query text to search for.
            k: Number of results to return.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects sorted by relevance.
        """
        return self.similarity_search(query, k, **kwargs)

    async def asimilarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs
    ) -> List[Document]:
        """
        Async search by embedding vector.

        Default implementation calls the sync version.
        Subclasses can override for better async support.

        Args:
            embedding: Query embedding vector.
            k: Number of results to return.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects sorted by relevance.
        """
        return self.similarity_search_by_vector(embedding, k, **kwargs)

    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> None:
        """
        Delete documents by IDs.

        Args:
            ids: List of document IDs to delete.
            **kwargs: Additional keyword arguments.
        """
        # Default implementation - subclasses can override
        if ids is None:
            return
        logger.warning(
            "delete() not fully implemented for this backend. "
            "Consider using delete_collection() for a full reset."
        )

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

        Default implementation uses similarity_search.
        Subclasses can override for better performance.

        Args:
            query: Query text to search for.
            k: Number of results to return.
            fetch_k: Number of documents to fetch before filtering.
            lambda_mult: Controls diversity vs relevance tradeoff.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects.
        """
        # Fallback to regular similarity search
        return self.similarity_search(query, k, **kwargs)
