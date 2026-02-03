"""
Vector store factory for dynamic backend selection.

This module provides a factory class that creates instances of different
vector store backends based on configuration settings.
"""

import logging
from typing import Dict, Optional

from django.conf import settings

from .base import BaseVectorStore
from .faiss import FAISSVectorStore

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """
    Factory class for creating vector store instances.

    This factory dynamically selects and creates the appropriate vector store
    backend based on configuration settings or runtime parameters.

    Supported backends:
    - 'faiss': Local FAISS index (default)
    - 'chromadb': ChromaDB local database
    - 'pinecone': Pinecone cloud vector database
    """

    # Mapping of backend names to their classes
    _backends = {
        "faiss": FAISSVectorStore,
        # These will be populated when the modules are imported
        "chromadb": None,
        "pinecone": None,
    }

    @classmethod
    def get_vector_store(
        cls,
        backend: Optional[str] = None,
        folder_name: str = "default",
        embedding=None,
        index_base_path: Optional[str] = None,
        **kwargs,
    ) -> BaseVectorStore:
        """
        Create a vector store instance based on the specified backend.

        Args:
            backend: Backend name ('faiss', 'chromadb', 'pinecone').
                    If None, uses VECTOR_STORE_BACKEND from settings.
            folder_name: Name of the collection/index.
            embedding: LangChain embeddings instance.
            index_base_path: Base path for storing indexes (if applicable).
            **kwargs: Additional backend-specific arguments.

        Returns:
            A BaseVectorStore instance.

        Raises:
            ValueError: If an unsupported backend is specified.
            ImportError: If a backend's dependencies are not installed.
        """
        # Get backend from settings if not specified
        if backend is None:
            backend = settings.djgent_settings.get(
                "VECTOR_STORE_BACKEND", "faiss"
            )

        backend = backend.lower()

        # Get the appropriate backend class
        if backend == "faiss":
            return cls._create_faiss_store(
                embedding=embedding,
                folder_name=folder_name,
                index_base_path=index_base_path,
                **kwargs,
            )
        elif backend == "chromadb":
            return cls._create_chromadb_store(
                embedding=embedding,
                folder_name=folder_name,
                index_base_path=index_base_path,
                **kwargs,
            )
        elif backend == "pinecone":
            return cls._create_pinecone_store(
                embedding=embedding,
                folder_name=folder_name,
                index_base_path=index_base_path,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported vector store backend: {backend}. "
                f"Supported backends: {list(cls._backends.keys())}"
            )

    @classmethod
    def _create_faiss_store(
        cls,
        embedding=None,
        folder_name: str = "default",
        index_base_path: Optional[str] = None,
        **kwargs,
    ) -> FAISSVectorStore:
        """
        Create a FAISS vector store instance.

        Args:
            embedding: LangChain embeddings instance.
            folder_name: Name of the collection/index.
            index_base_path: Base path for storing indexes.
            **kwargs: Additional FAISS-specific arguments.

        Returns:
            A FAISSVectorStore instance.
        """
        return FAISSVectorStore(
            embedding=embedding,
            folder_name=folder_name,
            index_base_path=index_base_path,
            **kwargs,
        )

    @classmethod
    def _create_chromadb_store(
        cls,
        embedding=None,
        folder_name: str = "default",
        index_base_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Create a ChromaDB vector store instance.

        Args:
            embedding: LangChain embeddings instance.
            folder_name: Name of the collection.
            index_base_path: Base path for storing database files.
            **kwargs: Additional ChromaDB-specific arguments.

        Returns:
            A ChromaDBVectorStore instance.

        Raises:
            ImportError: If ChromaDB is not installed.
        """
        try:
            from .chromadb import ChromaDBVectorStore

            return ChromaDBVectorStore(
                embedding=embedding,
                folder_name=folder_name,
                index_base_path=index_base_path,
                **kwargs,
            )
        except ImportError:
            logger.error(
                "ChromaDB backend requested but chromadb is not installed"
            )
            raise ImportError(
                "ChromaDB is required for chromadb backend. "
                "Install it with: pip install chromadb"
            )

    @classmethod
    def _create_pinecone_store(
        cls,
        embedding=None,
        folder_name: str = "default",
        index_base_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Create a Pinecone vector store instance.

        Args:
            embedding: LangChain embeddings instance.
            folder_name: Name of the index.
            index_base_path: Ignored for Pinecone (cloud-based).
            **kwargs: Additional Pinecone-specific arguments.

        Returns:
            A PineconeVectorStore instance.

        Raises:
            ImportError: If Pinecone client is not installed.
        """
        try:
            from .pinecone import PineconeVectorStore

            return PineconeVectorStore(
                embedding=embedding,
                folder_name=folder_name,
                index_base_path=index_base_path,
                **kwargs,
            )
        except ImportError:
            logger.error(
                "Pinecone backend requested but pinecone-client is not installed"
            )
            raise ImportError(
                "Pinecone client is required for pinecone backend. "
                "Install it with: pip install pinecone-client"
            )

    @classmethod
    def get_supported_backends(cls) -> Dict[str, str]:
        """
        Get a dictionary of supported backends and their status.

        Returns:
            Dictionary mapping backend names to availability status.
        """
        backends = {
            "faiss": "available",
            "chromadb": "unavailable",
            "pinecone": "unavailable",
        }

        # Check if optional backends are available
        try:
            import chromadb  # noqa: F401

            backends["chromadb"] = "available"
        except ImportError:
            pass

        try:
            import pinecone  # noqa: F401

            backends["pinecone"] = "available"
        except ImportError:
            pass

        return backends

    @classmethod
    def get_default_backend(cls) -> str:
        """
        Get the default backend name.

        Returns:
            Backend name as a string.
        """
        return getattr(settings, "VECTOR_STORE_BACKEND", "faiss")

    @classmethod
    def set_default_backend(cls, backend: str) -> None:
        """
        Set the default backend for the application.

        Note: This only affects the current process.

        Args:
            backend: Backend name to set as default.
        """
        backend = backend.lower()
        if backend not in cls._backends:
            raise ValueError(
                f"Unsupported backend: {backend}. "
                f"Supported: {list(cls._backends.keys())}"
            )
        # Update settings dynamically
        cls._default_backend = backend

    @classmethod
    def is_backend_available(cls, backend: str) -> bool:
        """
        Check if a backend is available (dependencies installed).

        Args:
            backend: Backend name to check.

        Returns:
            True if available, False otherwise.
        """
        backend = backend.lower()

        if backend == "faiss":
            try:
                import faiss  # noqa: F401

                return True
            except ImportError:
                return False
        elif backend == "chromadb":
            try:
                import chromadb  # noqa: F401

                return True
            except ImportError:
                return False
        elif backend == "pinecone":
            try:
                import pinecone  # noqa: F401

                return True
            except ImportError:
                return False

        return False


def get_vector_store(
    folder_name: str,
    embedding_provider=None,
    index_base_path: Optional[str] = None,
    backend: Optional[str] = None,
    **kwargs,
) -> BaseVectorStore:
    """
    Factory function to get a vector store instance.

    This is a convenience function that delegates to VectorStoreFactory.

    Args:
        folder_name: Name of the collection/index.
        embedding_provider: An embedding provider instance.
        index_base_path: Base path for storing indexes.
        backend: Backend name to use (optional).
        **kwargs: Additional arguments.

    Returns:
        A BaseVectorStore instance.
    """
    return VectorStoreFactory.get_vector_store(
        backend=backend,
        folder_name=folder_name,
        embedding=embedding_provider,
        index_base_path=index_base_path,
        **kwargs,
    )
