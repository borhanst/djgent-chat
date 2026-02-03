"""
Async RAG Service with LangChain/LangGraph support.

This module provides the high-level RAG service that wraps the LangGraph
workflow and provides both sync and async operations.
"""

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from django.conf import settings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from chat.langgraph.workflow import create_rag_workflow

from .langchain_embeddings import get_langchain_embeddings
from .vectorstore.factory import VectorStoreFactory

logger = logging.getLogger(__name__)


class LangChainRAGService:
    """
    RAG service using LangChain and LangGraph.

    This service provides a high-level interface for the RAG pipeline,
    wrapping the LangGraph workflow with both sync and async operations.

    Features:
    - LangChain embeddings and vector store integration
    - LangGraph workflow orchestration
    - Sync and async query methods
    - Streaming response support
    - Multiple LLM provider support (OpenAI, Gemini, Anthropic)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RAG service.

        Args:
            config: Configuration dictionary with keys:
                - rag_folder_name: Documents folder name (default: 'default')
                - embedding_provider: Embedding provider (default: from settings)
                - llm_provider: LLM provider (default: from settings)
                - top_k: Number of chunks to retrieve (default: 5)
                - chunk_size: Chunk size for indexing
                - chunk_overlap: Chunk overlap for indexing
        """
        self.config = config or {}

        # Get configuration values
        self.rag_folder_name = self.config.get("rag_folder_name", "default")
        self.embedding_provider_name = self.config.get(
            "embedding_provider",
            settings.djgent_settings.get("EMBEDDING_PROVIDER", "openai"),
        )
        self.llm_provider_name = self.config.get(
            "llm_provider",
            settings.djgent_settings.get("LLM_PROVIDER", "openai"),
        )
        self.top_k = self.config.get(
            "top_k", settings.djgent_settings.get("DEFAULT_TOP_K", 5)
        )

        # Initialize components
        self._init_components()

    def _init_components(self) -> None:
        """Initialize LangChain components (embeddings, vector store, LLM, workflow)."""
        # Get embeddings
        try:
            self.embeddings = get_langchain_embeddings(
                self.embedding_provider_name
            )
            logger.info(
                f"Initialized embeddings with provider: {self.embedding_provider_name}"
            )
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            raise

        # Get vector store
        try:
            # Get backend from config or djgent_settings
            backend = self.config.get(
                "vector_store_backend",
                settings.djgent_settings.get("VECTOR_STORE_BACKEND", "faiss"),
            )
            self.vector_store = VectorStoreFactory.get_vector_store(
                backend=backend,
                folder_name=self.rag_folder_name,
                embedding=self.embeddings,
            )
            logger.info(
                f"Initialized {backend} vector store for folder: {self.rag_folder_name}"
            )
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise

        # Create LLM
        try:
            self.llm = self._create_llm()
            logger.info(
                f"Initialized LLM with provider: {self.llm_provider_name}"
            )
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise

        # Create workflow
        try:
            self.workflow = create_rag_workflow(
                vector_store=self.vector_store,
                llm=self.llm,
                config={
                    "top_k": self.top_k,
                    "max_iterations": 1,  # Single pass for basic workflow
                },
            )
            logger.info("Initialized RAG workflow")
        except Exception as e:
            logger.error(f"Error initializing workflow: {e}")
            raise

    def _create_llm(self):
        """Create LLM based on provider configuration from djgent_settings."""
        if self.llm_provider_name == "openai":
            api_key = settings.djgent_settings.get("LLM_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key is required. Set LLM_PROVIDER=openai and LLM_API_KEY in settings or .env file."
                )
            return ChatOpenAI(
                model_name=settings.djgent_settings.get(
                    "LLM_MODEL", "gpt-4o-mini"
                ),
                openai_api_key=api_key,
                temperature=0.7,
                max_tokens=1000,
            )

        elif self.llm_provider_name == "gemini":
            api_key = settings.djgent_settings.get("LLM_API_KEY")
            if not api_key:
                raise ValueError(
                    "Gemini API key is required. Set LLM_PROVIDER=gemini and LLM_API_KEY in settings or .env file."
                )
            return ChatGoogleGenerativeAI(
                model=settings.djgent_settings.get(
                    "LLM_MODEL", "gemini-1.5-flash"
                ),
                google_api_key=api_key,
                temperature=0.7,
            )

        elif self.llm_provider_name == "anthropic":
            api_key = settings.djgent_settings.get("LLM_API_KEY")
            if not api_key:
                raise ValueError(
                    "Anthropic API key is required. Set LLM_PROVIDER=anthropic and LLM_API_KEY in settings or .env file."
                )
            return ChatAnthropic(
                model_name=settings.djgent_settings.get(
                    "LLM_MODEL", "claude-3-haiku-20240307"
                ),
                anthropic_api_key=api_key,
                max_tokens=1000,
            )

        else:
            raise ValueError(
                f"Unsupported LLM provider: {self.llm_provider_name}"
            )

    def query(
        self, question: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system synchronously.

        Args:
            question: The user's question.
            session_id: Optional session ID for conversation tracking.

        Returns:
            Dictionary containing:
                - answer: The generated answer
                - context: List of retrieved context chunks
                - sources: List of source document information
                - session_id: The session ID
                - error: Error message if any
        """
        try:
            # Invoke the workflow
            state = self.workflow.invoke(
                question=question, session_id=session_id
            )

            # Convert documents to response format
            context = []
            for doc in state.documents:
                context.append(
                    {"text": doc.page_content, "metadata": doc.metadata}
                )

            return {
                "answer": state.answer,
                "context": context,
                "sources": state.sources,
                "session_id": session_id,
                "error": state.error,
            }

        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return {
                "answer": "",
                "context": [],
                "sources": [],
                "session_id": session_id,
                "error": str(e),
            }

    async def aquery(
        self, question: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system asynchronously.

        Args:
            question: The user's question.
            session_id: Optional session ID for conversation tracking.

        Returns:
            Dictionary containing answer, sources, context, and error info.
        """
        try:
            # Invoke workflow asynchronously
            state = await self.workflow.ainvoke(
                question=question, session_id=session_id
            )

            # Convert documents to response format
            context = []
            for doc in state.documents:
                context.append(
                    {"text": doc.page_content, "metadata": doc.metadata}
                )

            return {
                "answer": state.answer,
                "context": context,
                "sources": state.sources,
                "session_id": session_id,
                "error": state.error,
            }

        except Exception as e:
            logger.error(f"Error in async RAG query: {e}")
            return {
                "answer": "",
                "context": [],
                "sources": [],
                "session_id": session_id,
                "error": str(e),
            }

    async def astream(
        self, question: str, session_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream the RAG response asynchronously.

        This method yields chunks of the generated response as they
        are produced by the LLM.

        Args:
            question: The user's question.
            session_id: Optional session ID for conversation tracking.

        Yields:
            Chunks of the generated response
        """
        try:
            # Get the generate node for streaming
            from chat.langgraph.nodes.generate import create_generate_node

            generate_node = create_generate_node(
                llm=self.llm, include_conversation_history=True
            )

            # Create initial state
            from chat.langgraph.state import RAGState

            state = RAGState(current_question=question, session_id=session_id)
            state.add_user_message(question)

            # Get context from vector store
            docs = self.vector_store.similarity_search(question, k=self.top_k)
            state.documents = docs
            state.context_str = state.format_context_for_prompt()

            # Stream the generation
            async for chunk in generate_node.astream(state):
                yield chunk

        except Exception as e:
            logger.error(f"Error in streaming RAG: {e}")
            yield f"Error: {str(e)}"

    def is_index_ready(self) -> bool:
        """
        Check if the vector index is ready for querying.

        Returns:
            True if index is loaded or can be loaded, False otherwise
        """
        # Check if the vector store is loaded
        if hasattr(self.vector_store, "is_loaded"):
            return self.vector_store.is_loaded()
        return False

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.

        Returns:
            Dictionary containing index statistics
        """
        return self.vector_store.get_stats()

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs for the documents

        Returns:
            List of IDs for the added documents
        """
        return self.vector_store.add_texts(
            texts=texts, metadatas=metadatas, ids=ids
        )

    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the vector store.

        Args:
            ids: List of document IDs to delete
        """
        self.vector_store.delete(ids=ids)

    def clear_index(self) -> None:
        """Clear all documents from the vector store."""
        self.vector_store.delete_collection()


def get_rag_service(
    config: Optional[Dict[str, Any]] = None,
) -> LangChainRAGService:
    """
    Factory function to get a RAG service instance.

    Args:
        config: Optional configuration dictionary

    Returns:
        A LangChainRAGService instance
    """
    return LangChainRAGService(config)
