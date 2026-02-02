"""
Configuration utilities for the chat application.

This module provides functions to access RAG configuration from
the database and environment variables.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)


def get_env_config() -> Dict[str, Any]:
    """
    Get configuration from environment variables.
    
    Returns:
        A dictionary containing configuration values.
    """
    return {
        # LLM Configuration
        'llm_provider': os.getenv('LLM_PROVIDER', 'openai'),
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'gemini_api_key': os.getenv('GEMINI_API_KEY'),
        'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
        
        # Embedding Configuration
        'embedding_provider': os.getenv('EMBEDDING_PROVIDER', 'openai'),
        'huggingface_model_name': os.getenv('HUGGINGFACE_MODEL_NAME', 'all-MiniLM-L6-v2'),
        
        # RAG Configuration
        'rag_faiss_index_base_path': os.getenv(
            'RAG_FAISS_INDEX_BASE_PATH',
            str(settings.BASE_DIR / 'faiss_indexes')
        ),
        'rag_documents_base_path': os.getenv(
            'RAG_DOCUMENTS_BASE_PATH',
            str(settings.BASE_DIR / 'data' / 'documents')
        ),
        
        # Chunking Configuration
        'default_chunk_size': int(os.getenv('DEFAULT_CHUNK_SIZE', 1000)),
        'default_chunk_overlap': int(os.getenv('DEFAULT_CHUNK_OVERLAP', 200)),
        'default_top_k': int(os.getenv('DEFAULT_TOP_K', 5)),
    }


def get_rag_config() -> Dict[str, Any]:
    """
    Get RAG configuration from the database.
    
    Returns:
        A dictionary containing RAG configuration values.
        If no configuration exists in the database, returns defaults.
    """
    try:
        from chat.models import RAGConfiguration
        
        # Get the active configuration (most recent)
        config = RAGConfiguration.objects.order_by('-created_at').first()
        
        if config:
            return {
                'rag_folder_name': config.rag_folder_name,
                'embedding_provider': config.embedding_provider,
                'llm_provider': config.llm_provider,
                'chunk_size': config.chunk_size,
                'chunk_overlap': config.chunk_overlap,
                'top_k': config.top_k,
            }
        else:
            # Return default configuration
            env_config = get_env_config()
            return {
                'rag_folder_name': 'default',
                'embedding_provider': env_config['embedding_provider'],
                'llm_provider': env_config['llm_provider'],
                'chunk_size': env_config['default_chunk_size'],
                'chunk_overlap': env_config['default_chunk_overlap'],
                'top_k': env_config['default_top_k'],
            }
    except Exception as e:
        logger.error(f"Error getting RAG configuration: {e}")
        # Return default configuration
        env_config = get_env_config()
        return {
            'rag_folder_name': 'default',
            'embedding_provider': env_config['embedding_provider'],
            'llm_provider': env_config['llm_provider'],
            'chunk_size': env_config['default_chunk_size'],
            'chunk_overlap': env_config['default_chunk_overlap'],
            'top_k': env_config['default_top_k'],
        }


def get_faiss_index_base_path() -> Path:
    """
    Get the base path for storing FAISS indexes.
    
    Returns:
        A Path object pointing to the FAISS index base directory.
    """
    path_str = getattr(
        settings,
        'RAG_FAISS_INDEX_BASE_PATH',
        str(settings.BASE_DIR / 'faiss_indexes')
    )
    return Path(path_str)


def get_documents_base_path() -> Path:
    """
    Get the base path for storing documents.
    
    Returns:
        A Path object pointing to the documents base directory.
    """
    path_str = getattr(
        settings,
        'RAG_DOCUMENTS_BASE_PATH',
        str(settings.BASE_DIR / 'data' / 'documents')
    )
    return Path(path_str)


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


def get_api_key(provider: str) -> Optional[str]:
    """
    Get the API key for a given provider.
    
    Args:
        provider: The provider name ('openai', 'gemini', 'anthropic').
        
    Returns:
        The API key, or None if not found.
    """
    provider = provider.lower()
    
    if provider == 'openai':
        return getattr(settings, 'OPENAI_API_KEY', None) or os.getenv('OPENAI_API_KEY')
    elif provider == 'gemini':
        return getattr(settings, 'GEMINI_API_KEY', None) or os.getenv('GEMINI_API_KEY')
    elif provider == 'anthropic':
        return getattr(settings, 'ANTHROPIC_API_KEY', None) or os.getenv('ANTHROPIC_API_KEY')
    else:
        return None


def validate_api_keys() -> Dict[str, bool]:
    """
    Validate that required API keys are configured.
    
    Returns:
        A dictionary mapping provider names to validation status.
    """
    env_config = get_env_config()
    
    return {
        'openai': bool(env_config['openai_api_key']),
        'gemini': bool(env_config['gemini_api_key']),
        'anthropic': bool(env_config['anthropic_api_key']),
    }
