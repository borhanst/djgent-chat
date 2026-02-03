"""
LangChain-compatible embedding providers with sync/async support.

This module provides wrappers that make the existing embedding providers
compatible with the LangChain embeddings interface.
"""

import logging
from typing import List, Optional

# from sentence_transformers import SentenceTransformer
from config import settings
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class LegacyEmbeddingWrapper(Embeddings):
    """
    Wrapper for legacy/custom embedding providers to make them LangChain-compatible.

    This allows using the existing embedding providers (OpenAI, Gemini, HuggingFace)
    with LangChain's interface while maintaining backward compatibility.
    """

    def __init__(self, provider_name: str, model_name: Optional[str] = None):
        """
        Initialize the wrapper.

        Args:
            provider_name: The name of the provider ('openai', 'gemini', 'huggingface')
            model_name: Optional model name to use
        """
        self.provider_name = provider_name
        self.model_name = model_name
        self._instance = None

    def _get_instance(self):
        """Get the underlying embedding provider instance."""
        if self._instance is None:
            from .embedding_service import get_embedding_provider

            self._instance = get_embedding_provider(
                self.provider_name, self.model_name
            )
        return self._instance

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts (documents).

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        try:
            instance = self._get_instance()
            return instance.embed_batch(texts)
        except Exception as e:
            logger.error(
                f"Error embedding documents with {self.provider_name}: {e}"
            )
            raise

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Async embed a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        # For now, use sync implementation
        # Can be optimized with concurrent execution for API-based providers
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector
        """
        try:
            instance = self._get_instance()
            return instance.embed_text(text)
        except Exception as e:
            logger.error(
                f"Error embedding query with {self.provider_name}: {e}"
            )
            raise

    async def aembed_query(self, text: str) -> List[float]:
        """
        Async embed a single query text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector
        """
        return self.embed_query(text)


class HuggingFaceLangChainEmbeddings(Embeddings):
    """
    LangChain-compatible HuggingFace embeddings using sentence-transformers.

    This runs locally and doesn't require an API key.
    Supports any model from the sentence-transformers library.

    Popular models:
    - all-MiniLM-L6-v2 (384 dimensions, fast)
    - all-mpnet-base-v2 (768 dimensions, better quality)
    - paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions, multilingual)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ):
        """
        Initialize the HuggingFace embeddings.

        Args:
            model_name: The model name to use
            device: The device to use ('cpu' or 'cuda')
            normalize_embeddings: Whether to normalize embeddings
        """
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model = None

    def _get_model(self):
        """Lazy load the sentence-transformers model."""
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            try:
                self._model = SentenceTransformer(
                    self.model_name, device=self.device
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers package is not installed. "
                    "Install it with: pip install sentence-transformers"
                )
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using sentence-transformers.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        try:
            model = self._get_model()
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating HuggingFace embeddings: {e}")
            raise

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Async embed a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        # Sentence-transformers doesn't have native async, so we use sync
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector
        """
        try:
            model = self._get_model()
            embedding = model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating HuggingFace embedding: {e}")
            raise

    async def aembed_query(self, text: str) -> List[float]:
        """
        Async embed a single query text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector
        """
        return self.embed_query(text)

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        model = self._get_model()
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dimension": model.get_sentence_embedding_dimension(),
            "max_seq_length": model.max_seq_length,
        }


def get_langchain_embeddings(provider: Optional[str] = None) -> Embeddings:
    """
    Factory function to get LangChain-compatible embeddings.

    This function returns an embeddings instance that implements the
    LangChain Embeddings interface, allowing it to be used with
    LangChain vector stores and other components.

    Args:
        provider: The provider name ('openai', 'gemini', 'huggingface').
                 If not provided, uses the default from settings.

    Returns:
        An Embeddings instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    if provider is None:
        provider = settings.djgent_settings.get("EMBEDDING_PROVIDER", "openai")

    provider = provider.lower()

    if provider == "openai":
        api_key = settings.djgent_settings.get("EMBEDDING_API_KEY")
        if not api_key:
            raise ValueError(
                "Embedding API key is required. Set EMBEDDING_PROVIDER=openai and EMBEDDING_API_KEY in settings or .env file."
            )
        return OpenAIEmbeddings(
            model=settings.djgent_settings.get(
                "EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            openai_api_key=api_key,
        )

    elif provider == "gemini":
        api_key = settings.djgent_settings.get("EMBEDDING_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key is required. Set EMBEDDING_PROVIDER=gemini and EMBEDDING_API_KEY in settings or .env file."
            )
        return GoogleGenerativeAIEmbeddings(
            model=settings.djgent_settings.get(
                "EMBEDDING_MODEL", "models/text-embedding-004"
            ),
            google_api_key=api_key,
        )

    elif provider == "huggingface":
        model_name = settings.djgent_settings.get(
            "HUGGINGFACE_MODEL_NAME", "all-MiniLM-L6-v2"
        )
        device = settings.djgent_settings.get("HUGGINGFACE_DEVICE", "cpu")
        return HuggingFaceLangChainEmbeddings(
            model_name=model_name, device=device
        )

    else:
        # Use legacy wrapper for backward compatibility
        # This allows using custom or unknown providers
        logger.warning(
            f"Unknown provider '{provider}', using legacy wrapper. "
            f"Supported providers: openai, gemini, huggingface"
        )
        return LegacyEmbeddingWrapper(provider)


def get_async_embeddings(provider: Optional[str] = None) -> Embeddings:
    """
    Get embeddings with async support optimization.

    For API-based providers (OpenAI, Gemini), this returns an optimized
    version that can handle concurrent requests better.

    Args:
        provider: The provider name

    Returns:
        An Embeddings instance with async support
    """
    embeddings = get_langchain_embeddings(provider)

    # Add async methods if not present (for providers that don't implement them)
    if not hasattr(embeddings, "aembed_documents"):

        async def aembed_documents(texts):
            return embeddings.embed_documents(texts)

        embeddings.aembed_documents = aembed_documents

    if not hasattr(embeddings, "aembed_query"):

        async def aembed_query(text):
            return embeddings.embed_query(text)

        embeddings.aembed_query = aembed_query

    return embeddings
