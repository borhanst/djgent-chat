"""
RAG (Retrieval-Augmented Generation) service.

This module provides the main RAG orchestration service that combines
vector retrieval with LLM generation to answer questions based on
indexed documents.
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

from django.conf import settings

from .embedding_service import get_embedding_provider
from .vector_store_service import FAISSVectorStore, get_vector_store
from .document_service import DocumentLoader
from .chunking_service import TextChunker

logger = logging.getLogger(__name__)


class LLMProvider:
    """
    LLM provider for generating responses.
    
    Supports multiple LLM providers: OpenAI, Gemini, Anthropic.
    """
    
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the LLM provider.
        
        Args:
            provider: The LLM provider name ('openai', 'gemini', 'anthropic').
            model: The model name to use.
        """
        self.provider = provider or getattr(settings, 'LLM_PROVIDER', 'openai')
        self.model = model or self._get_default_model()
        self._client = None
    
    def _get_default_model(self) -> str:
        """Get the default model for the configured provider."""
        models = {
            'openai': 'gpt-4o-mini',
            'gemini': 'gemini-1.5-flash',
            'anthropic': 'claude-3-haiku-20240307',
        }
        return models.get(self.provider, 'gpt-4o-mini')
    
    def _get_client(self):
        """Lazy load the LLM client."""
        if self._client is not None:
            return self._client
        
        if self.provider == 'openai':
            api_key = getattr(settings, 'OPENAI_API_KEY', None)
            if not api_key:
                raise ValueError("OpenAI API key is required")
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
        
        elif self.provider == 'gemini':
            api_key = getattr(settings, 'GEMINI_API_KEY', None)
            if not api_key:
                raise ValueError("Gemini API key is required")
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.model)
        
        elif self.provider == 'anthropic':
            api_key = getattr(settings, 'ANTHROPIC_API_KEY', None)
            if not api_key:
                raise ValueError("Anthropic API key is required")
            from anthropic import Anthropic
            self._client = Anthropic(api_key=api_key)
        
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        return self._client
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: The user's question or prompt.
            context: Optional context from retrieved documents.
            
        Returns:
            The generated response.
        """
        # Build the full prompt with context
        full_prompt = self._build_prompt(prompt, context)
        
        try:
            if self.provider == 'openai':
                return self._generate_openai(full_prompt)
            elif self.provider == 'gemini':
                return self._generate_gemini(full_prompt)
            elif self.provider == 'anthropic':
                return self._generate_anthropic(full_prompt)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
        except Exception as e:
            logger.error(f"Error generating response with {self.provider}: {e}")
            raise
    
    def _build_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        """Build the full prompt with context."""
        if context:
            return f"""Context from documents:
{context}

Question: {prompt}

Please answer the question based on the provided context. If the context doesn't contain enough information to answer the question, please say so."""
        else:
            return f"Question: {prompt}"
    
    def _generate_openai(self, prompt: str) -> str:
        """Generate response using OpenAI."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    
    def _generate_gemini(self, prompt: str) -> str:
        """Generate response using Gemini."""
        client = self._get_client()
        response = client.generate_content(prompt)
        return response.text
    
    def _generate_anthropic(self, prompt: str) -> str:
        """Generate response using Anthropic."""
        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text


class RAGService:
    """
    RAG service for retrieval-augmented generation.
    
    This service orchestrates the entire RAG pipeline:
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
        
        # Get configuration values
        self.rag_folder_name = self.config.get('rag_folder_name', 'default')
        self.embedding_provider_name = self.config.get(
            'embedding_provider',
            getattr(settings, 'EMBEDDING_PROVIDER', 'openai')
        )
        self.llm_provider_name = self.config.get(
            'llm_provider',
            getattr(settings, 'LLM_PROVIDER', 'openai')
        )
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.chunk_overlap = self.config.get('chunk_overlap', 200)
        self.top_k = self.config.get('top_k', 5)
        
        # Initialize services
        self.embedding_provider = get_embedding_provider(self.embedding_provider_name)
        self.vector_store = get_vector_store(
            self.rag_folder_name,
            self.embedding_provider
        )
        self.llm_provider = LLMProvider(self.llm_provider_name)
        
        # Load the index if it exists
        self._ensure_index_loaded()
    
    def _ensure_index_loaded(self):
        """Ensure the vector index is loaded."""
        if not self.vector_store.is_loaded():
            loaded = self.vector_store.load_index()
            if not loaded:
                logger.warning(
                    f"No index found for folder '{self.rag_folder_name}'. "
                    "Please run the index_documents command first."
                )
    
    def query(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
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
        """
        # Retrieve relevant context
        context_chunks = self.get_context(question)
        
        # Build context string
        context_text = self._build_context_text(context_chunks)
        
        # Generate response
        answer = self.generate_response(question, context_text)
        
        # Extract sources
        sources = self._extract_sources(context_chunks)
        
        result = {
            'answer': answer,
            'context': context_chunks,
            'sources': sources,
            'session_id': session_id,
        }
        
        logger.info(f"Generated answer for question: {question[:50]}...")
        return result
    
    def get_context(self, question: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context chunks for a question.
        
        Args:
            question: The user's question.
            
        Returns:
            A list of relevant chunks with metadata.
        """
        if not self.vector_store.is_loaded():
            logger.warning("Vector index is not loaded. No context available.")
            return []
        
        # Search for relevant chunks
        search_results = self.vector_store.search_by_text(question, k=self.top_k)
        
        # Convert to context format
        context = []
        for result in search_results:
            context.append({
                'text': result.text,
                'score': result.score,
                'metadata': result.metadata,
            })
        
        logger.debug(f"Retrieved {len(context)} context chunks")
        return context
    
    def generate_response(self, question: str, context: Optional[str] = None) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            question: The user's question.
            context: Optional context from retrieved documents.
            
        Returns:
            The generated response.
        """
        return self.llm_provider.generate_response(question, context)
    
    def _build_context_text(self, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Build a context string from retrieved chunks.
        
        Args:
            context_chunks: List of retrieved chunks.
            
        Returns:
            A formatted context string.
        """
        if not context_chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            metadata = chunk.get('metadata', {})
            source = metadata.get('file_name', 'Unknown')
            chunk_id = metadata.get('chunk_id', i)
            
            context_parts.append(
                f"[Source: {source}, Chunk {chunk_id}]\n{chunk['text']}"
            )
        
        return "\n\n".join(context_parts)
    
    def _extract_sources(self, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract unique sources from context chunks.
        
        Args:
            context_chunks: List of retrieved chunks.
            
        Returns:
            A list of unique sources.
        """
        sources = {}
        
        for chunk in context_chunks:
            metadata = chunk.get('metadata', {})
            file_name = metadata.get('file_name', 'Unknown')
            source_path = metadata.get('source_path', '')
            
            if file_name not in sources:
                sources[file_name] = {
                    'file_name': file_name,
                    'source_path': source_path,
                    'chunk_count': 0,
                }
            
            sources[file_name]['chunk_count'] += 1
        
        return list(sources.values())
    
    def save_conversation(
        self,
        session_id: str,
        role: str,
        content: str,
        context: Optional[List[Dict]] = None
    ) -> None:
        """
        Save a conversation message to the database.
        
        Args:
            session_id: The session ID.
            role: The role ('user' or 'assistant').
            content: The message content.
            context: Optional context chunks used for the response.
        """
        from chat.models import ChatMessage
        
        ChatMessage.objects.create(
            session_id=session_id,
            role=role,
            content=content,
            context_used=context or []
        )
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.
        
        Returns:
            A dictionary containing index statistics.
        """
        return self.vector_store.get_stats()
    
    def is_index_ready(self) -> bool:
        """
        Check if the index is ready for querying.
        
        Returns:
            True if the index is loaded and has vectors, False otherwise.
        """
        return self.vector_store.is_loaded() and self.vector_store.get_vector_count() > 0


def get_rag_service(config: Optional[Dict[str, Any]] = None) -> RAGService:
    """
    Factory function to get a RAG service instance.
    
    Args:
        config: Optional configuration dictionary.
    
    Returns:
        A RAGService instance.
    """
    return RAGService(config)
