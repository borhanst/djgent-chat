"""
Embedding service for generating text embeddings.

This module provides implementations for multiple embedding providers:
- OpenAI Embeddings
- Google Gemini Embeddings
- HuggingFace Sentence Transformers (local)
"""

import os
import logging
from typing import List, Optional
from abc import ABC

from django.conf import settings

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider using the OpenAI API.
    
    Supports models like:
    - text-embedding-3-small (1536 dimensions)
    - text-embedding-3-large (3072 dimensions)
    - text-embedding-ada-002 (1536 dimensions)
    """
    
    # Model dimensions mapping
    MODEL_DIMENSIONS = {
        'text-embedding-3-small': 1536,
        'text-embedding-3-large': 3072,
        'text-embedding-ada-002': 1536,
    }
    
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the OpenAI embedding provider.
        
        Args:
            model_name: The model name to use. Defaults to text-embedding-3-small.
            api_key: The OpenAI API key. If not provided, reads from settings.
        """
        self.api_key = api_key or getattr(settings, 'OPENAI_API_KEY', None)
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in settings or .env file.")
        
        super().__init__(model_name)
        self._client = None
    
    def get_default_model(self) -> str:
        """Get the default OpenAI embedding model."""
        return 'text-embedding-3-small'
    
    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.MODEL_DIMENSIONS.get(self.model_name, 1536)
    
    def _get_client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI package is not installed. "
                    "Install it with: pip install openai"
                )
        return self._client
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text using OpenAI API.
        
        Args:
            text: The text to embed.
            
        Returns:
            The embedding vector as a list of floats.
        """
        try:
            client = self._get_client()
            response = client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using OpenAI API.
        
        Args:
            texts: A list of texts to embed.
            
        Returns:
            A list of embedding vectors.
        """
        try:
            client = self._get_client()
            response = client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating OpenAI batch embeddings: {e}")
            raise


class GeminiEmbeddingProvider(EmbeddingProvider):
    """
    Google Gemini embedding provider using the Google Generative AI API.
    
    Supports models like:
    - models/embedding-001 (768 dimensions)
    - models/text-embedding-004 (768 dimensions)
    """
    
    # Model dimensions mapping
    MODEL_DIMENSIONS = {
        'models/embedding-001': 768,
        'models/text-embedding-004': 768,
        'models/multimodalembedding': 768,
    }
    
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the Gemini embedding provider.
        
        Args:
            model_name: The model name to use. Defaults to models/text-embedding-004.
            api_key: The Google API key. If not provided, reads from settings.
        """
        self.api_key = api_key or getattr(settings, 'GEMINI_API_KEY', None)
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY in settings or .env file.")
        
        super().__init__(model_name)
        self._client = None
    
    def get_default_model(self) -> str:
        """Get the default Gemini embedding model."""
        return 'models/text-embedding-004'
    
    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.MODEL_DIMENSIONS.get(self.model_name, 768)
    
    def _get_client(self):
        """Lazy load the Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai
            except ImportError:
                raise ImportError(
                    "Google Generative AI package is not installed. "
                    "Install it with: pip install google-generativeai"
                )
        return self._client
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text using Gemini API.
        
        Args:
            text: The text to embed.
            
        Returns:
            The embedding vector as a list of floats.
        """
        try:
            client = self._get_client()
            result = client.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating Gemini embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using Gemini API.
        
        Note: Gemini API doesn't support batch embedding directly,
        so we make individual calls.
        
        Args:
            texts: A list of texts to embed.
            
        Returns:
            A list of embedding vectors.
        """
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text))
        return embeddings


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """
    HuggingFace embedding provider using sentence-transformers.
    
    This runs locally and doesn't require an API key.
    Supports any model from the sentence-transformers library.
    
    Popular models:
    - all-MiniLM-L6-v2 (384 dimensions, fast)
    - all-mpnet-base-v2 (768 dimensions, better quality)
    - paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions, multilingual)
    """
    
    # Model dimensions mapping for common models
    MODEL_DIMENSIONS = {
        'all-MiniLM-L6-v2': 384,
        'all-mpnet-base-v2': 768,
        'paraphrase-multilingual-MiniLM-L12-v2': 384,
        'all-distilroberta-v1': 768,
    }
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the HuggingFace embedding provider.
        
        Args:
            model_name: The model name to use. Defaults to all-MiniLM-L6-v2.
            device: The device to use ('cpu' or 'cuda'). Defaults to CPU.
        """
        self.device = device or 'cpu'
        super().__init__(model_name)
        self._model = None
    
    def get_default_model(self) -> str:
        """Get the default HuggingFace embedding model."""
        return getattr(settings, 'HUGGINGFACE_MODEL_NAME', 'all-MiniLM-L6-v2')
    
    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        # If we have the model loaded, get dimension from it
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        
        # Otherwise, use the mapping or default to 384
        return self.MODEL_DIMENSIONS.get(self.model_name, 384)
    
    def _get_model(self):
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                raise ImportError(
                    "sentence-transformers package is not installed. "
                    "Install it with: pip install sentence-transformers"
                )
        return self._model
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text using sentence-transformers.
        
        Args:
            text: The text to embed.
            
        Returns:
            The embedding vector as a list of floats.
        """
        try:
            model = self._get_model()
            embedding = model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating HuggingFace embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using sentence-transformers.
        
        Args:
            texts: A list of texts to embed.
            
        Returns:
            A list of embedding vectors.
        """
        try:
            model = self._get_model()
            embeddings = model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating HuggingFace batch embeddings: {e}")
            raise


def get_embedding_provider(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> EmbeddingProvider:
    """
    Factory function to get an embedding provider instance.
    
    Args:
        provider: The provider name ('openai', 'gemini', 'huggingface').
                 If not provided, uses the default from settings.
        model_name: Optional model name to use.
        **kwargs: Additional keyword arguments to pass to the provider.
    
    Returns:
        An instance of the requested embedding provider.
    
    Raises:
        ValueError: If the provider is not supported.
    """
    if provider is None:
        provider = getattr(settings, 'EMBEDDING_PROVIDER', 'openai')
    
    provider = provider.lower()
    
    providers = {
        'openai': OpenAIEmbeddingProvider,
        'gemini': GeminiEmbeddingProvider,
        'huggingface': HuggingFaceEmbeddingProvider,
    }
    
    provider_class = providers.get(provider)
    if provider_class is None:
        raise ValueError(
            f"Unsupported embedding provider: {provider}. "
            f"Supported providers: {', '.join(providers.keys())}"
        )
    
    return provider_class(model_name=model_name, **kwargs)
