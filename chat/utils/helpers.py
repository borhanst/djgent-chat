"""
Helper functions for the chat application.

This module provides utility functions for common operations
like filename sanitization, path validation, and text processing.
"""

import os
import re
import uuid
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing or replacing unsafe characters.
    
    Args:
        filename: The filename to sanitize.
        
    Returns:
        The sanitized filename.
    """
    # Replace path separators with underscores
    filename = filename.replace('/', '_').replace('\\', '_')
    
    # Remove or replace other unsafe characters
    filename = re.sub(r'[<>:"|?*\x00-\x1f]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255 - len(ext)] + ext
    
    return filename


def validate_path(path: str, base_path: Optional[str] = None) -> bool:
    """
    Validate that a path is safe to access.
    
    This prevents directory traversal attacks by ensuring the path
    is within the allowed base directory.
    
    Args:
        path: The path to validate.
        base_path: The base path to restrict access to.
                  If None, uses the documents base path from settings.
        
    Returns:
        True if the path is safe, False otherwise.
    """
    try:
        from django.conf import settings
        
        if base_path is None:
            base_path = str(settings.BASE_DIR / 'data' / 'documents')
        
        # Resolve absolute paths
        abs_path = Path(path).resolve()
        abs_base = Path(base_path).resolve()
        
        # Check if the path is within the base path
        try:
            abs_path.relative_to(abs_base)
            return True
        except ValueError:
            return False
            
    except Exception as e:
        logger.error(f"Error validating path {path}: {e}")
        return False


def generate_session_id() -> str:
    """
    Generate a unique session ID.
    
    Returns:
        A UUID string.
    """
    return str(uuid.uuid4())


def truncate_text(text: str, max_length: int = 100, suffix: str = '...') -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: The text to truncate.
        max_length: The maximum length.
        suffix: The suffix to add if truncated.
        
    Returns:
        The truncated text.
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_file_size(size_bytes: int) -> str:
    """
    Format a file size in human-readable format.
    
    Args:
        size_bytes: The size in bytes.
        
    Returns:
        A formatted string (e.g., "1.5 MB").
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def get_file_extension(file_path: str) -> str:
    """
    Get the file extension from a file path.
    
    Args:
        file_path: The file path.
        
    Returns:
        The file extension (including the dot), or empty string if no extension.
    """
    return Path(file_path).suffix.lower()


def is_supported_document(file_path: str) -> bool:
    """
    Check if a file is a supported document type.
    
    Args:
        file_path: The file path.
        
    Returns:
        True if the file is supported, False otherwise.
    """
    from chat.services import DocumentLoader
    
    return DocumentLoader.get_loader_for_file(file_path) is not None


def safe_join(base_path: str, *paths: str) -> str:
    """
    Safely join paths, preventing directory traversal.
    
    Args:
        base_path: The base path.
        *paths: Additional path components.
        
    Returns:
        The joined path if safe, or the base path if unsafe.
    """
    try:
        result = Path(base_path)
        for path in paths:
            result = result / path
        
        # Resolve to get absolute path
        result = result.resolve()
        
        # Check if result is within base_path
        base = Path(base_path).resolve()
        try:
            result.relative_to(base)
            return str(result)
        except ValueError:
            return str(base)
            
    except Exception as e:
        logger.error(f"Error joining paths: {e}")
        return str(base_path)


def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: The directory path.
        
    Returns:
        A Path object for the directory.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_relative_path(file_path: str, base_path: str) -> Optional[str]:
    """
    Get the relative path of a file from a base path.
    
    Args:
        file_path: The file path.
        base_path: The base path.
        
    Returns:
        The relative path, or None if the file is not within the base path.
    """
    try:
        file_abs = Path(file_path).resolve()
        base_abs = Path(base_path).resolve()
        return str(file_abs.relative_to(base_abs))
    except ValueError:
        return None


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.
    
    Args:
        text: The text to clean.
        
    Returns:
        The cleaned text.
    """
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_keywords(text: str, max_keywords: int = 10) -> list:
    """
    Extract keywords from text.
    
    This is a simple implementation that extracts words
    that are longer than 3 characters and appear frequently.
    
    Args:
        text: The text to extract keywords from.
        max_keywords: Maximum number of keywords to return.
        
    Returns:
        A list of keywords.
    """
    # Simple word extraction
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    
    # Count word frequency
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:max_keywords]]
