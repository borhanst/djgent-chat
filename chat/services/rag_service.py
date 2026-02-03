"""
RAG (Retrieval-Augmented Generation) service.

This module provides the main RAG orchestration service that combines
vector retrieval with LLM generation to answer questions based on
indexed documents.

This module now uses LangChain/LangGraph for the RAG pipeline.
"""

import logging
from typing import Any, Dict, Optional

from .langchain_rag_service import LangChainRAGService

logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG service for retrieval-augmented generation.

    This service wraps LangChainRAGService for backward compatibility.
    It orchestrates the entire RAG pipeline:
    1. Retrieve relevant chunks from the vector store
    2. Generate a response using the LLM with retrieved context
    3. Save conversation history
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RAG service.

        Args:
            config: Configuration dictionary with keys:
                - rag_folder_name: Name of the documents folder
                - embedding_provider: Embedding provider name
                - llm_provider: LLM provider name
                - chunk_size: Chunk size for text splitting
                - chunk_overlap: Chunk overlap
                - top_k: Number of chunks to retrieve
        """
        self.config = config or {}
        self._service = LangChainRAGService(self.config)

    def query(
        self, question: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system with a question.

        Args:
            question: The user's question.
            session_id: Optional session ID for conversation tracking.

        Returns:
            A dictionary containing:
                - answer: The generated answer
                - context: List of retrieved chunks
                - sources: List of source documents
                - session_id: The session ID
                - error: Error message if any
        """
        return self._service.query(question, session_id)

    def is_index_ready(self) -> bool:
        """
        Check if the index is ready for querying.

        Returns:
            True if the index is loaded and has vectors, False otherwise.
        """
        return self._service.is_index_ready()

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.

        Returns:
            A dictionary containing index statistics.
        """
        return self._service.get_index_stats()

    def add_documents(
        self,
        texts: list,
        metadatas: Optional[list] = None,
        ids: Optional[list] = None,
    ) -> list:
        """
        Add documents to the vector store.

        Args:
            texts: List of text strings to add.
            metadatas: Optional list of metadata dictionaries.
            ids: Optional list of IDs for the documents.

        Returns:
            List of IDs for the added documents.
        """
        return self._service.add_documents(texts, metadatas, ids)

    def delete_documents(self, ids: list) -> None:
        """
        Delete documents from the vector store.

        Args:
            ids: List of document IDs to delete.
        """
        self._service.delete_documents(ids)

    def clear_index(self) -> None:
        """Clear all documents from the vector store."""
        self._service.clear_index()


def get_rag_service(config: Optional[Dict[str, Any]] = None) -> RAGService:
    """
    Factory function to get a RAG service instance.

    Args:
        config: Optional configuration dictionary.

    Returns:
        A RAGService instance.
    """
    return RAGService(config)
