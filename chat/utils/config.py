"""
Configuration utilities for the chat application.

This module provides functions to access RAG configuration from
the database, environment variables, and djgent_settings.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from django.conf import settings

logger = logging.getLogger(__name__)


def get_djgent_settings() -> Dict[str, Any]:
    """
    Get the djgent_settings dictionary.

    Returns:
        A dictionary containing all application settings.
    """
    return getattr(settings, "djgent_settings", {})


def get_rag_settings() -> Dict[str, Any]:
    """
    Get RAG-related settings from djgent_settings.

    Returns:
        A dictionary containing RAG configuration values.
    """
    djgent = get_djgent_settings()
    return {
        "rag_faiss_index_base_path": djgent.get("RAG_FAISS_INDEX_BASE_PATH"),
        "rag_documents_base_path": djgent.get("RAG_DOCUMENTS_BASE_PATH"),
    }


def get_vector_store_settings() -> Dict[str, Any]:
    """
    Get vector store settings from djgent_settings.

    Returns:
        A dictionary containing vector store configuration.
    """
    djgent = get_djgent_settings()
    return {
        "backend": djgent.get("VECTOR_STORE_BACKEND", "faiss"),
        "chromadb_persist_directory": djgent.get("CHROMADB_PERSIST_DIRECTORY"),
        "pinecone_environment": djgent.get("PINECONE_ENVIRONMENT"),
        "pinecone_api_key": djgent.get("PINECONE_API_KEY"),
    }


def get_llm_settings() -> Dict[str, Any]:
    """
    Get LLM settings from djgent_settings.

    Returns:
        A dictionary containing LLM configuration.
    """
    djgent = get_djgent_settings()
    return {
        "provider": djgent.get("LLM_PROVIDER", "openai"),
        "api_key": djgent.get("LLM_API_KEY"),
        "default_model": _get_default_llm_model(
            djgent.get("LLM_PROVIDER", "openai")
        ),
    }


def get_embedding_settings() -> Dict[str, Any]:
    """
    Get embedding settings from djgent_settings.

    Returns:
        A dictionary containing embedding configuration.
    """
    djgent = get_djgent_settings()
    return {
        "provider": djgent.get("EMBEDDING_PROVIDER", "openai"),
        "api_key": djgent.get("EMBEDDING_API_KEY"),
        "huggingface_model_name": djgent.get(
            "HUGGINGFACE_MODEL_NAME", "all-MiniLM-L6-v2"
        ),
        "default_model": _get_default_embedding_model(
            djgent.get("EMBEDDING_PROVIDER", "openai")
        ),
    }


def get_chunking_settings() -> Dict[str, Any]:
    """
    Get chunking settings from djgent_settings.

    Returns:
        A dictionary containing chunking configuration.
    """
    djgent = get_djgent_settings()
    return {
        "chunk_size": djgent.get("DEFAULT_CHUNK_SIZE", 1000),
        "chunk_overlap": djgent.get("DEFAULT_CHUNK_OVERLAP", 200),
        "top_k": djgent.get("DEFAULT_TOP_K", 5),
    }


def _get_default_llm_model(provider: str) -> str:
    """Get the default model for a given LLM provider."""
    defaults = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "anthropic": "claude-3-haiku-20240307",
    }
    return defaults.get(provider, "gpt-4o-mini")


def _get_default_embedding_model(provider: str) -> str:
    """Get the default model for a given embedding provider."""
    defaults = {
        "openai": "text-embedding-3-small",
        "gemini": "models/text-embedding-004",
        "huggingface": "all-MiniLM-L6-v2",
    }
    return defaults.get(provider, "text-embedding-3-small")


def get_api_key(provider: str) -> Optional[str]:
    """
    Get the API key for a given provider from djgent_settings.

    Args:
        provider: The provider name ('openai', 'gemini', 'anthropic').

    Returns:
        The API key, or None if not found.
    """
    djgent = get_djgent_settings()
    provider = provider.lower()

    # Try to get from LLM_API_KEY first if it matches the provider
    llm_provider = djgent.get("LLM_PROVIDER", "openai")
    if provider == llm_provider:
        api_key = djgent.get("LLM_API_KEY")
        if api_key:
            return api_key

    # Fall back to environment-specific keys
    env_keys = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    return os.getenv(env_keys.get(provider, ""))


def get_vector_store_backend() -> str:
    """
    Get the configured vector store backend.

    Returns:
        The backend name ('faiss', 'chromadb', 'pinecone').
    """
    return get_djgent_settings().get("VECTOR_STORE_BACKEND", "faiss")


def get_llm_provider() -> str:
    """
    Get the configured LLM provider.

    Returns:
        The provider name ('openai', 'gemini', 'anthropic').
    """
    return get_djgent_settings().get("LLM_PROVIDER", "openai")


def get_embedding_provider() -> str:
    """
    Get the configured embedding provider.

    Returns:
        The provider name ('openai', 'gemini', 'huggingface').
    """
    return get_djgent_settings().get("EMBEDDING_PROVIDER", "openai")


def get_env_config() -> Dict[str, Any]:
    """
    Get configuration from environment variables.

    Returns:
        A dictionary containing configuration values.
    """
    djgent = get_djgent_settings()
    return {
        # LLM Configuration
        "llm_provider": "gemini",
        "llm_api_key": djgent.get("LLM_API_KEY", os.getenv("LLM_API_KEY")),
        # Embedding Configuration
        "embedding_provider": djgent.get(
            "EMBEDDING_PROVIDER", os.getenv("EMBEDDING_PROVIDER", "gemini")
        ),
        "embedding_api_key": djgent.get(
            "EMBEDDING_API_KEY", os.getenv("EMBEDDING_API_KEY")
        ),
        "huggingface_model_name": djgent.get(
            "HUGGINGFACE_MODEL_NAME",
            os.getenv("HUGGINGFACE_MODEL_NAME", "all-MiniLM-L6-v2"),
        ),
        # Vector Store Configuration
        "vector_store_backend": djgent.get(
            "VECTOR_STORE_BACKEND", os.getenv("VECTOR_STORE_BACKEND", "faiss")
        ),
        "chromadb_persist_directory": djgent.get(
            "CHROMADB_PERSIST_DIRECTORY",
            os.getenv("CHROMADB_PERSIST_DIRECTORY"),
        ),
        "pinecone_api_key": djgent.get(
            "PINECONE_API_KEY", os.getenv("PINECONE_API_KEY")
        ),
        "pinecone_environment": djgent.get(
            "PINECONE_ENVIRONMENT",
            os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),
        ),
        # RAG Configuration
        "rag_faiss_index_base_path": djgent.get("RAG_FAISS_INDEX_BASE_PATH"),
        "rag_documents_base_path": djgent.get("RAG_DOCUMENTS_BASE_PATH"),
        # Chunking Configuration
        "default_chunk_size": djgent.get(
            "DEFAULT_CHUNK_SIZE", int(os.getenv("DEFAULT_CHUNK_SIZE", 1000))
        ),
        "default_chunk_overlap": djgent.get(
            "DEFAULT_CHUNK_OVERLAP",
            int(os.getenv("DEFAULT_CHUNK_OVERLAP", 200)),
        ),
        "default_top_k": djgent.get(
            "DEFAULT_TOP_K", int(os.getenv("DEFAULT_TOP_K", 5))
        ),
    }


def get_rag_config() -> Dict[str, Any]:
    """
    Get RAG configuration from the database.

    Returns:
        A dictionary containing RAG configuration values.
        If no configuration exists in the database, returns defaults from djgent_settings.
    """
    try:
        from chat.models import RAGConfiguration

        # Get the active configuration (most recent)
        config = RAGConfiguration.objects.order_by("-created_at").first()

        if config:
            return {
                "rag_folder_name": config.rag_folder_name,
                "embedding_provider": config.embedding_provider,
                "llm_provider": config.llm_provider,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "top_k": config.top_k,
            }
    except Exception as e:
        logger.error(f"Error getting RAG configuration from database: {e}")

    # Return default configuration from djgent_settings
    env_config = get_env_config()
    print("env config", env_config)
    chunking = get_chunking_settings()
    return {
        "rag_folder_name": "default",
        "embedding_provider": env_config["embedding_provider"],
        "llm_provider": env_config["llm_provider"],
        "chunk_size": chunking["chunk_size"],
        "chunk_overlap": chunking["chunk_overlap"],
        "top_k": chunking["top_k"],
    }


def get_faiss_index_base_path() -> Path:
    """
    Get the base path for storing FAISS indexes.

    Returns:
        A Path object pointing to the FAISS index base directory.
    """
    djgent = get_djgent_settings()
    path_str = djgent.get("RAG_FAISS_INDEX_BASE_PATH")
    if path_str:
        return Path(path_str)
    return settings.BASE_DIR / "faiss_indexes"


def get_documents_base_path() -> Path:
    """
    Get the base path for storing documents.

    Returns:
        A Path object pointing to the documents base directory.
    """
    djgent = get_djgent_settings()
    path_str = djgent.get("RAG_DOCUMENTS_BASE_PATH")
    if path_str:
        return Path(path_str)
    return settings.BASE_DIR / "data" / "documents"


def ensure_directories_exist() -> None:
    """
    Ensure that all required directories exist.
    Creates them if they don't exist.
    """
    directories = [
        get_faiss_index_base_path(),
        get_documents_base_path(),
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")


def validate_api_keys() -> Dict[str, bool]:
    """
    Validate that required API keys are configured.

    Returns:
        A dictionary mapping provider names to validation status.
    """
    env_config = get_env_config()

    return {
        "openai": bool(
            env_config.get("llm_provider") == "openai"
            and env_config.get("llm_api_key")
        ),
        "gemini": bool(
            env_config.get("llm_provider") == "gemini"
            and env_config.get("llm_api_key")
        ),
        "anthropic": bool(
            env_config.get("llm_provider") == "anthropic"
            and env_config.get("llm_api_key")
        ),
    }
