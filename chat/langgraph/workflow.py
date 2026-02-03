"""
LangGraph RAG workflow assembly.

This module provides the workflow assembly that combines the retrieval
and generation nodes into a complete RAG pipeline using LangGraph.
"""

import logging
from typing import Any, AsyncGenerator, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph


from .nodes.generate import create_generate_node
from .nodes.retrieve import create_retrieve_node
from .state import RAGState

logger = logging.getLogger(__name__)


class RAGWorkflow:
    """
    LangGraph-based RAG workflow.

    This class assembles the complete RAG pipeline using LangGraph,
    combining document retrieval and response generation into a
    single workflow with state management.

    Features:
    - Configurable retrieval and generation nodes
    - Checkpoint memory for conversation state persistence
    - Both sync and async invocation support
    - Streaming support for responses
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm: BaseChatModel,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the RAG workflow.

        Args:
            vector_store: LangChain vector store instance
            llm: LangChain LLM instance
            config: Optional configuration dictionary with keys:
                - top_k: Number of documents to retrieve (default: 5)
                - search_type: Type of search (default: 'similarity')
                - max_iterations: Maximum workflow iterations (default: 2)
                - include_history: Include conversation history (default: True)
                - system_prompt: Custom system prompt
        """
        self.vector_store = vector_store
        self.llm = llm
        self.config = config or {}

        # Configuration with defaults
        self.k = self.config.get("top_k", 5)
        self.search_type = self.config.get("search_type", "similarity")
        self.max_iterations = self.config.get("max_iterations", 2)
        self.include_history = self.config.get("include_history", True)
        self.system_prompt = self.config.get("system_prompt", None)

        # Build the graph
        self.graph: StateGraph = self._build_graph()

        logger.info(
            f"RAGWorkflow initialized with k={self.k}, search_type={self.search_type}, "
            f"max_iterations={self.max_iterations}"
        )

    def _build_graph(self) -> StateGraph:
        """
        Build the workflow graph.

        Returns:
            Compiled StateGraph with retrieval and generation nodes
        """
        # Create nodes
        retrieve_node = create_retrieve_node(
            vector_store=self.vector_store,
            k=self.k,
            search_type=self.search_type,
        )

        generate_node = create_generate_node(
            llm=self.llm,
            system_prompt=self.system_prompt,
            include_conversation_history=self.include_history,
        )

        # Create the state graph
        workflow = StateGraph(RAGState)

        # Add nodes
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("generate", generate_node)

        # Set entry point
        workflow.set_entry_point("retrieve")

        # Add edges
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        # Compile with memory checkpointing
        memory = MemorySaver()
        compiled_graph = workflow.compile(checkpointer=memory)

        return compiled_graph

    def invoke(
        self,
        question: str,
        session_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> RAGState:
        """
        Invoke the workflow synchronously.

        Args:
            question: User's question
            session_id: Optional session ID for checkpointing
            config: Optional runtime configuration

        Returns:
            Final RAGState with answer and sources
        """
        # Create initial state
        initial_state = RAGState(
            current_question=question,
            session_id=session_id,
            max_iterations=self.max_iterations,
        )
        initial_state.add_user_message(question)

        # Prepare graph config
        graph_config = config or {}
        graph_config["configurable"] = {"thread_id": session_id or "default"}

        try:
            # Invoke the graph
            final_state = self.graph.invoke(initial_state, config=graph_config)

            logger.info(
                f"Workflow completed for question: {question[:50]}... "
                f"(iterations: {final_state.iterations})"
            )

            return final_state

        except Exception as e:
            logger.error(f"Error invoking workflow: {e}")
            # Return state with error
            initial_state.set_error(str(e))
            return initial_state

    async def ainvoke(
        self,
        question: str,
        session_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> RAGState:
        """
        Invoke the workflow asynchronously.

        Args:
            question: User's question
            session_id: Optional session ID for checkpointing
            config: Optional runtime configuration

        Returns:
            Final RAGState with answer and sources
        """
        # Create initial state
        initial_state = RAGState(
            current_question=question,
            session_id=session_id,
            max_iterations=self.max_iterations,
        )
        initial_state.add_user_message(question)

        # Prepare graph config
        graph_config = config or {}
        graph_config["configurable"] = {"thread_id": session_id or "default"}

        try:
            # Invoke the graph asynchronously
            final_state = await self.graph.ainvoke(
                initial_state, config=graph_config
            )

            logger.info(
                f"Async workflow completed for question: {question[:50]}... "
                f"(iterations: {final_state.iterations})"
            )

            return final_state

        except Exception as e:
            logger.error(f"Error invoking async workflow: {e}")
            initial_state.set_error(str(e))
            return initial_state

    def stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream the workflow output.

        This is useful for showing progress as documents are retrieved
        and the response is generated.

        Args:
            question: User's question
            session_id: Optional session ID for checkpointing
            config: Optional runtime configuration

        Yields:
            State updates from each node
        """

        # Create initial state
        initial_state = RAGState(
            current_question=question, session_id=session_id
        )
        initial_state.add_user_message(question)

        # Prepare graph config
        graph_config = config or {}
        graph_config["configurable"] = {"thread_id": session_id or "default"}

        # Stream the graph output
        try:
            for chunk in self.graph.stream(initial_state, config=graph_config):
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming workflow: {e}")
            yield {"error": str(e)}

    def get_checkpoints(self, session_id: str = "default") -> list:
        """
        Get stored checkpoints for a session.

        Args:
            session_id: Session ID to get checkpoints for

        Returns:
            List of checkpoint states
        """
        # This requires access to the checkpointer's internal storage
        # For now, return empty list as a placeholder
        logger.warning("Checkpoint retrieval not yet implemented")
        return []

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update workflow configuration.

        Note: This requires rebuilding the graph for most changes.

        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)

        # Update individual settings
        if "top_k" in new_config:
            self.k = new_config["top_k"]
        if "search_type" in new_config:
            self.search_type = new_config["search_type"]
        if "max_iterations" in new_config:
            self.max_iterations = new_config["max_iterations"]
        if "include_history" in new_config:
            self.include_history = new_config["include_history"]
        if "system_prompt" in new_config:
            self.system_prompt = new_config["system_prompt"]

        # Rebuild the graph
        self.graph = self._build_graph()

        logger.info("Workflow configuration updated, graph rebuilt")


def create_rag_workflow(
    vector_store: VectorStore,
    llm: BaseChatModel,
    config: Optional[Dict[str, Any]] = None,
) -> RAGWorkflow:
    """
    Factory function to create a RAG workflow.

    Args:
        vector_store: LangChain vector store instance
        llm: LangChain LLM instance
        config: Optional configuration dictionary

    Returns:
        A RAGWorkflow instance
    """
    return RAGWorkflow(vector_store=vector_store, llm=llm, config=config)
