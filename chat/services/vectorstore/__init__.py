"""
Vector store package for multi-backend vector storage support.

This package provides a unified interface for different vector storage backends:
- FAISS: Local file-based vector storage (default)
- ChromaDB: Local persistent database
- Pinecone: Cloud-based vector storage
"""

from .base import BaseVectorStore
from .factory import VectorStoreFactory, get_vector_store
from .faiss import FAISSVectorStore

__all__ = [
    "BaseVectorStore",
    "VectorStoreFactory",
    "get_vector_store",
    "FAISSVectorStore",
]
