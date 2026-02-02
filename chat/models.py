"""
Django models for the chat application.

This module defines the database models for:
- RAGConfiguration: User-configurable RAG settings
- ChatMessage: Chat history and conversation tracking
"""

import uuid
import json
from django.db import models
from django.core.exceptions import ValidationError


class RAGConfiguration(models.Model):
    """
    Model for storing RAG (Retrieval-Augmented Generation) configuration.
    
    Users can configure various settings for the RAG system including
    document folder, embedding provider, LLM provider, and chunking parameters.
    """
    
    # Embedding provider choices
    EMBEDDING_PROVIDER_CHOICES = [
        ('openai', 'OpenAI'),
        ('gemini', 'Google Gemini'),
        ('huggingface', 'HuggingFace (Local)'),
    ]
    
    # LLM provider choices
    LLM_PROVIDER_CHOICES = [
        ('openai', 'OpenAI'),
        ('gemini', 'Google Gemini'),
        ('anthropic', 'Anthropic'),
    ]
    
    # Configuration fields
    rag_folder_name = models.CharField(
        max_length=255,
        default='default',
        help_text='Name of the folder containing documents to index'
    )
    
    embedding_provider = models.CharField(
        max_length=20,
        choices=EMBEDDING_PROVIDER_CHOICES,
        default='openai',
        help_text='Provider for generating embeddings'
    )
    
    llm_provider = models.CharField(
        max_length=20,
        choices=LLM_PROVIDER_CHOICES,
        default='openai',
        help_text='Provider for LLM generation'
    )
    
    chunk_size = models.PositiveIntegerField(
        default=1000,
        help_text='Target size of each chunk in characters'
    )
    
    chunk_overlap = models.PositiveIntegerField(
        default=200,
        help_text='Number of characters to overlap between chunks'
    )
    
    top_k = models.PositiveIntegerField(
        default=5,
        help_text='Number of relevant chunks to retrieve'
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'RAG Configuration'
        verbose_name_plural = 'RAG Configurations'
    
    def __str__(self):
        return f"RAG Config: {self.rag_folder_name} ({self.embedding_provider})"
    
    def clean(self):
        """Validate the configuration."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValidationError(
                'chunk_overlap must be less than chunk_size'
            )
        
        if self.top_k < 1:
            raise ValidationError('top_k must be at least 1')
        
        if self.chunk_size < 100:
            raise ValidationError('chunk_size must be at least 100')
    
    def save(self, *args, **kwargs):
        """Override save to run validation."""
        self.full_clean()
        super().save(*args, **kwargs)
    
    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            'rag_folder_name': self.rag_folder_name,
            'embedding_provider': self.embedding_provider,
            'llm_provider': self.llm_provider,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'top_k': self.top_k,
        }


class ChatMessage(models.Model):
    """
    Model for storing chat messages and conversation history.
    
    Messages are grouped by session_id to track conversations.
    Each message stores the role (user/assistant), content, and
    optionally the context chunks used for generating the response.
    """
    
    # Role choices
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]
    
    # Primary key - use UUID for better security and distribution
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Session tracking
    session_id = models.UUIDField(
        db_index=True,
        help_text='UUID for grouping messages into conversations'
    )
    
    # Message content
    role = models.CharField(
        max_length=10,
        choices=ROLE_CHOICES,
        help_text='Role of the message sender'
    )
    
    content = models.TextField(
        help_text='The message content'
    )
    
    # Context used for assistant responses
    context_used = models.JSONField(
        default=list,
        blank=True,
        help_text='List of context chunks used for generating the response'
    )
    
    # Sources referenced
    sources = models.JSONField(
        default=list,
        blank=True,
        help_text='List of source documents referenced'
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    
    class Meta:
        ordering = ['created_at']
        verbose_name = 'Chat Message'
        verbose_name_plural = 'Chat Messages'
        indexes = [
            models.Index(fields=['session_id', 'created_at']),
        ]
    
    def __str__(self):
        content_preview = self.content[:50] + '...' if len(self.content) > 50 else self.content
        return f"{self.role}: {content_preview}"
    
    @classmethod
    def get_conversation(cls, session_id: uuid.UUID) -> models.QuerySet:
        """
        Get all messages for a conversation session.
        
        Args:
            session_id: The session UUID.
            
        Returns:
            QuerySet of messages ordered by creation time.
        """
        return cls.objects.filter(session_id=session_id).order_by('created_at')
    
    @classmethod
    def create_user_message(cls, session_id: uuid.UUID, content: str) -> 'ChatMessage':
        """
        Create a user message.
        
        Args:
            session_id: The session UUID.
            content: The message content.
            
        Returns:
            The created ChatMessage instance.
        """
        return cls.objects.create(
            session_id=session_id,
            role='user',
            content=content
        )
    
    @classmethod
    def create_assistant_message(
        cls,
        session_id: uuid.UUID,
        content: str,
        context: list = None,
        sources: list = None
    ) -> 'ChatMessage':
        """
        Create an assistant message.
        
        Args:
            session_id: The session UUID.
            content: The message content.
            context: Optional list of context chunks.
            sources: Optional list of source documents.
            
        Returns:
            The created ChatMessage instance.
        """
        return cls.objects.create(
            session_id=session_id,
            role='assistant',
            content=content,
            context_used=context or [],
            sources=sources or []
        )
    
    def get_context_summary(self) -> str:
        """
        Get a summary of the context used for this message.
        
        Returns:
            A string summarizing the context.
        """
        if not self.context_used:
            return "No context used"
        
        sources = set()
        for chunk in self.context_used:
            metadata = chunk.get('metadata', {})
            source = metadata.get('file_name', 'Unknown')
            sources.add(source)
        
        return f"Used {len(self.context_used)} chunks from {len(sources)} source(s): {', '.join(sorted(sources))}"
