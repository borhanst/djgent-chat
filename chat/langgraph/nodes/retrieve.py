"""
Document retrieval node for LangGraph workflow.

This module provides the node implementation for retrieving relevant
documents from the vector store based on the user's question.
"""

import logging
from typing import Optional

from langchain_core.vectorstores import VectorStore

from ..state import RAGState

logger = logging.getLogger(__name__)


class RetrieveNode:
    """
    Node for retrieving relevant documents from the vector store.

    This node takes a question from the state, searches the vector store
    for relevant documents, and adds them to the state.

    Features:
    - Configurable number of results (k)
    - Support for both sync and async operations
    - Automatic context formatting
    - Source metadata extraction
    """

    def __init__(
        self,
        vector_store: VectorStore,
        k: int = 5,
        search_type: str = "similarity",
    ):
        """
        Initialize the retrieval node.

        Args:
            vector_store: LangChain vector store instance
            k: Number of documents to retrieve
            search_type: Type of search ('similarity' or 'mmr')
        """
        self.vector_store = vector_store
        self.k = k
        self.search_type = search_type.lower()

    def __call__(self, state: RAGState) -> RAGState:
        """
        Execute retrieval synchronously.

        Args:
            state: Current RAG state

        Returns:
            Updated state with retrieved documents
        """
        question = state.current_question

        if not question:
            logger.warning("No question provided for retrieval")
            state.set_error("No question provided for retrieval")
            return state

        try:
            # Perform the search
            if self.search_type == "mmr":
                documents = self.vector_store.max_marginal_relevance_search(
                    question, k=self.k, fetch_k=self.k * 2
                )
            else:
                # Default to similarity search
                documents = self.vector_store.similarity_search(
                    question, k=self.k
                )

            # Update state
            state.documents = documents
            state.context_str = state.format_context_for_prompt()
            state.sources = state.extract_sources()
            state.increment_iterations()

            logger.info(
                f"Retrieved {len(documents)} documents for question: {question[:50]}..."
            )

            # Check if documents were found
            if not documents:
                logger.warning("No documents found matching the question")

        except Exception as e:
            logger.error(f"Error during document retrieval: {e}")
            state.set_error(f"Retrieval error: {str(e)}")

        return state

    async def ainvoke(self, state: RAGState) -> RAGState:
        """
        Execute retrieval asynchronously.

        Args:
            state: Current RAG state

        Returns:
            Updated state with retrieved documents
        """
        # For now, use sync implementation
        # Vector store async support varies by implementation
        return self(state)

    def invoke_with_filter(
        self, state: RAGState, filter_dict: Optional[dict[str, str]] = None
    ) -> RAGState:
        """
        Execute retrieval with metadata filtering.

        Args:
            state: Current RAG state
            filter_dict: Metadata filter conditions

        Returns:
            Updated state with retrieved documents
        """
        question = state.current_question

        if not question:
            state.set_error("No question provided for retrieval")
            return state

        try:
            # Search with filter if provided
            if filter_dict:
                documents = self.vector_store.similarity_search(
                    question, k=self.k, filter=filter_dict
                )
            else:
                documents = self.vector_store.similarity_search(
                    question, k=self.k
                )

            state.documents = documents
            state.context_str = state.format_context_for_prompt()
            state.sources = state.extract_sources()
            state.increment_iterations()

        except Exception as e:
            logger.error(f"Error during filtered retrieval: {e}")
            state.set_error(f"Filtered retrieval error: {str(e)}")

        return state


def create_retrieve_node(
    vector_store: VectorStore, k: int = 5, search_type: str = "similarity"
) -> RetrieveNode:
    """
    Factory function to create a retrieval node.

    Args:
        vector_store: LangChain vector store instance
        k: Number of documents to retrieve
        search_type: Type of search ('similarity' or 'mmr')

    Returns:
        A RetrieveNode instance
    """
    return RetrieveNode(vector_store=vector_store, k=k, search_type=search_type)
