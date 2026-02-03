"""
RAG state schema for LangGraph workflow.

This module defines the state that flows through the LangGraph RAG workflow,
including conversation messages, retrieved documents, and generation context.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.documents import Document


class RAGState(BaseModel):
    """
    State for the RAG workflow.
    
    This defines all the data that flows through the graph during the
    retrieval-augmented generation process.
    
    Attributes:
        messages: Conversation messages (human and AI)
        current_question: The current user question being processed
        documents: Retrieved documents from the vector store
        question_embedding: Embedded question vector (for caching)
        context_str: Formatted context string for LLM prompt
        answer: Generated answer from the LLM
        sources: Source document information for citations
        iterations: Number of retrieval iterations performed
        max_iterations: Maximum retrieval iterations allowed
        error: Error message if any step failed
        session_id: Conversation session identifier
    """
    
    # Conversation context
    messages: List[BaseMessage] = Field(
        default_factory=list,
        description="Conversation messages"
    )
    current_question: str = Field(
        default="",
        description="Current user question"
    )
    
    # Retrieval context
    documents: List[Document] = Field(
        default_factory=list,
        description="Retrieved documents"
    )
    question_embedding: Optional[List[float]] = Field(
        default=None,
        description="Embedded question vector for caching"
    )
    
    # Generation context
    context_str: str = Field(
        default="",
        description="Formatted context string for LLM prompt"
    )
    answer: str = Field(
        default="",
        description="Generated answer from the LLM"
    )
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Source document information for citations"
    )
    
    # Workflow state
    iterations: int = Field(
        default=0,
        description="Number of retrieval iterations performed"
    )
    max_iterations: int = Field(
        default=2,
        description="Maximum retrieval iterations allowed"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if any step failed"
    )
    
    # Session info
    session_id: Optional[str] = Field(
        default=None,
        description="Conversation session identifier"
    )
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_user_message(self, question: str) -> None:
        """Add a user message to the conversation."""
        self.current_question = question
        self.messages.append(HumanMessage(content=question))
    
    def add_assistant_message(self, answer: str) -> None:
        """Add an assistant message to the conversation."""
        self.messages.append(AIMessage(content=answer))
        self.answer = answer
    
    def get_conversation_context(self, max_messages: int = 5) -> str:
        """
        Get formatted conversation history for context.
        
        Args:
            max_messages: Maximum number of recent messages to include
            
        Returns:
            Formatted conversation history string
        """
        context_parts = []
        # Get recent messages
        recent_messages = self.messages[-max_messages:] if max_messages > 0 else self.messages
        
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                context_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_parts.append(f"Assistant: {msg.content}")
            else:
                # Handle other message types
                role = msg.type if hasattr(msg, 'type') else 'unknown'
                content = msg.content if hasattr(msg, 'content') else str(msg)
                context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    def format_context_for_prompt(self, include_sources: bool = True) -> str:
        """
        Format the retrieved documents into a context string for the LLM.
        
        Args:
            include_sources: Whether to include source citations
            
        Returns:
            Formatted context string
        """
        if not self.documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(self.documents, 1):
            source = doc.metadata.get('file_name', f'Document {i}')
            chunk_id = doc.metadata.get('chunk_id', i)
            
            if include_sources:
                context_parts.append(
                    f"[Source: {source}, Chunk {chunk_id}]\n{doc.page_content}"
                )
            else:
                context_parts.append(doc.page_content)
        
        return "\n\n".join(context_parts)
    
    def extract_sources(self) -> List[Dict[str, Any]]:
        """
        Extract unique sources from retrieved documents.
        
        Returns:
            List of source dictionaries with file_name, source_path, chunk_count
        """
        sources = {}
        
        for doc in self.documents:
            file_name = doc.metadata.get('file_name', 'Unknown')
            source_path = doc.metadata.get('source_path', '')
            
            if file_name not in sources:
                sources[file_name] = {
                    'file_name': file_name,
                    'source_path': source_path,
                    'chunk_count': 0,
                }
            
            sources[file_name]['chunk_count'] += 1
        
        return list(sources.values())
    
    def should_continue(self) -> bool:
        """
        Check if the workflow should continue (more iterations allowed).
        
        Returns:
            True if more iterations are allowed, False otherwise
        """
        return self.iterations < self.max_iterations
    
    def increment_iterations(self) -> None:
        """Increment the iteration counter."""
        self.iterations += 1
    
    def set_error(self, error: str) -> None:
        """Set an error message."""
        self.error = error
    
    def clear_error(self) -> None:
        """Clear the error message."""
        self.error = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a dictionary (for serialization)."""
        return {
            'messages': [
                {'type': msg.type, 'content': msg.content}
                for msg in self.messages
            ],
            'current_question': self.current_question,
            'context_str': self.context_str,
            'answer': self.answer,
            'sources': self.sources,
            'iterations': self.iterations,
            'max_iterations': self.max_iterations,
            'error': self.error,
            'session_id': self.session_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGState':
        """Create state from a dictionary."""
        messages = []
        for msg_data in data.get('messages', []):
            msg_type = msg_data.get('type', 'human')
            content = msg_data.get('content', '')
            
            if msg_type == 'human':
                messages.append(HumanMessage(content=content))
            elif msg_type == 'ai':
                messages.append(AIMessage(content=content))
            else:
                messages.append(BaseMessage(content=content, type=msg_type))
        
        return cls(
            messages=messages,
            current_question=data.get('current_question', ''),
            context_str=data.get('context_str', ''),
            answer=data.get('answer', ''),
            sources=data.get('sources', []),
            iterations=data.get('iterations', 0),
            max_iterations=data.get('max_iterations', 2),
            error=data.get('error'),
            session_id=data.get('session_id'),
        )
