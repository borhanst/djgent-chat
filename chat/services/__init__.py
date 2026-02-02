"""
Services module for the chat application.

This module contains all business logic services including:
- Embedding providers (OpenAI, Gemini, HuggingFace)
- Vector store service (FAISS)
- Document loading service
- Text chunking service
- RAG orchestration service
"""

from .embedding_service import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    GeminiEmbeddingProvider,
    HuggingFaceEmbeddingProvider,
    get_embedding_provider,
)
from .vector_store_service import FAISSVectorStore
from .document_service import DocumentLoader
from .chunking_service import TextChunker
from .rag_service import RAGService

__all__ = [
    # Embedding providers
    'EmbeddingProvider',
    'OpenAIEmbeddingProvider',
    'GeminiEmbeddingProvider',
    'HuggingFaceEmbeddingProvider',
    'get_embedding_provider',
    # Vector store
    'FAISSVectorStore',
    # Document handling
    'DocumentLoader',
    'TextChunker',
    # RAG service
    'RAGService',
]
