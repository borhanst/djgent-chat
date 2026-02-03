"""
Response generation node for LangGraph workflow.

This module provides the node implementation for generating responses
using the LLM with retrieved context.
"""

import logging
from typing import Optional, AsyncGenerator, Dict, Any
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel

from ..state import RAGState

logger = logging.getLogger(__name__)


class GenerateNode:
    """
    Node for generating responses using the LLM.
    
    This node takes the retrieved context and user's question,
    generates a response using the LLM, and updates the state.
    
    Features:
    - Configurable system prompt
    - Conversation history context
    - Streaming support for token-by-token output
    - Source citation extraction
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided context.

When answering:
1. Use the context provided to answer the question
2. If the context doesn't contain enough information, say so
3. Be concise and helpful
4. Reference sources when appropriate"""

    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: Optional[str] = None,
        include_conversation_history: bool = True
    ):
        """
        Initialize the generation node.
        
        Args:
            llm: LangChain LLM instance
            system_prompt: Custom system prompt (uses default if not provided)
            include_conversation_history: Whether to include history in prompt
        """
        self.llm = llm
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.include_conversation_history = include_conversation_history
        
        # Create prompt template
        self._build_prompt_template()
    
    def _build_prompt_template(self) -> None:
        """Build the prompt template based on configuration."""
        if self.include_conversation_history:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", """Context from documents:
{context}

Conversation history:
{conversation}

Question: {question}

Please answer the question based on the provided context.""")
            ])
        else:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", """Context from documents:
{context}

Question: {question}

Please answer the question based on the provided context.""")
            ])
        
        # Create the chain
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def __call__(self, state: RAGState) -> RAGState:
        """
        Generate response synchronously.
        
        Args:
            state: Current RAG state with context and question
            
        Returns:
            Updated state with generated answer
        """
        try:
            # Prepare prompt inputs
            inputs = self._prepare_inputs(state)
            
            # Invoke the chain
            answer = self.chain.invoke(inputs)
            
            # Update state
            state.answer = answer
            state.add_assistant_message(answer)
            state.sources = state.extract_sources()
            
            logger.info(
                f"Generated answer ({len(answer)} chars) for question: "
                f"{state.current_question[:50]}..."
            )
            
        except Exception as e:
            logger.error(f"Error during response generation: {e}")
            state.set_error(f"Generation error: {str(e)}")
        
        return state
    
    async def ainvoke(self, state: RAGState) -> RAGState:
        """
        Generate response asynchronously.
        
        Args:
            state: Current RAG state with context and question
            
        Returns:
            Updated state with generated answer
        """
        try:
            inputs = self._prepare_inputs(state)
            
            # Use async invoke if available
            if hasattr(self.chain, 'ainvoke'):
                answer = await self.chain.ainvoke(inputs)
            else:
                # Fall back to sync
                answer = self.chain.invoke(inputs)
            
            state.answer = answer
            state.add_assistant_message(answer)
            state.sources = state.extract_sources()
            
        except Exception as e:
            logger.error(f"Error during async response generation: {e}")
            state.set_error(f"Async generation error: {str(e)}")
        
        return state
    
    async def astream(self, state: RAGState) -> AsyncGenerator[str, None]:
        """
        Stream response token by token.
        
        Args:
            state: Current RAG state with context and question
            
        Yields:
            Chunks of the generated response
        """
        try:
            inputs = self._prepare_inputs(state)
            
            # Use async stream if available
            if hasattr(self.chain, 'astream'):
                async for chunk in self.chain.astream(inputs):
                    yield chunk
            else:
                # Fall back to sync stream
                for chunk in self.chain.stream(inputs):
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Error during streaming generation: {e}")
            yield f"Error: {str(e)}"
    
    def _prepare_inputs(self, state: RAGState) -> Dict[str, str]:
        """
        Prepare prompt inputs from state.
        
        Args:
            state: Current RAG state
            
        Returns:
            Dictionary of prompt inputs
        """
        inputs = {
            "context": state.context_str or "No context available",
            "question": state.current_question,
        }
        
        if self.include_conversation_history:
            inputs["conversation"] = state.get_conversation_context()
        
        return inputs
    
    def update_system_prompt(self, new_prompt: str) -> None:
        """
        Update the system prompt.
        
        Args:
            new_prompt: New system prompt string
        """
        self.system_prompt = new_prompt
        self._build_prompt_template()
        logger.info("System prompt updated")


def create_generate_node(
    llm: BaseChatModel,
    system_prompt: Optional[str] = None,
    include_conversation_history: bool = True
) -> GenerateNode:
    """
    Factory function to create a generation node.
    
    Args:
        llm: LangChain LLM instance
        system_prompt: Custom system prompt
        include_conversation_history: Whether to include history in prompt
        
    Returns:
        A GenerateNode instance
    """
    return GenerateNode(
        llm=llm,
        system_prompt=system_prompt,
        include_conversation_history=include_conversation_history
    )
