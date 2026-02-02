"""
Document service for loading and processing various document types.

This module provides document loaders for:
- PDF files (using PyPDF2)
- TXT files (plain text)
- MD files (Markdown)
- DOCX files (using python-docx)
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

from django.conf import settings

from .base import DocumentLoader

logger = logging.getLogger(__name__)


class PDFLoader(DocumentLoader):
    """
    Document loader for PDF files using PyPDF2.
    """
    
    def __init__(self):
        """Initialize the PDF loader."""
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are installed."""
        try:
            import PyPDF2
        except ImportError:
            raise ImportError(
                "PyPDF2 package is not installed. "
                "Install it with: pip install PyPDF2"
            )
    
    def get_supported_extensions(self) -> List[str]:
        """Get the list of supported file extensions."""
        return ['.pdf']
    
    def can_load(self, file_path: str) -> bool:
        """Check if this loader can load the given file."""
        return Path(file_path).suffix.lower() in self.get_supported_extensions()
    
    def load_document(self, file_path: str) -> str:
        """
        Load a PDF document and extract text.
        
        Args:
            file_path: The path to the PDF file.
            
        Returns:
            The extracted text content.
        """
        import PyPDF2
        
        text_parts = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(
                            f"Error extracting text from page {page_num} "
                            f"in {file_path}: {e}"
                        )
            
            # Join all pages with newlines
            full_text = '\n\n'.join(text_parts)
            
            if not full_text.strip():
                logger.warning(f"No text extracted from PDF: {file_path}")
            
            return full_text
            
        except Exception as e:
            logger.error(f"Error loading PDF document {file_path}: {e}")
            raise


class TextLoader(DocumentLoader):
    """
    Document loader for plain text files.
    """
    
    def get_supported_extensions(self) -> List[str]:
        """Get the list of supported file extensions."""
        return ['.txt']
    
    def can_load(self, file_path: str) -> bool:
        """Check if this loader can load the given file."""
        return Path(file_path).suffix.lower() in self.get_supported_extensions()
    
    def load_document(self, file_path: str) -> str:
        """
        Load a plain text document.
        
        Args:
            file_path: The path to the text file.
            
        Returns:
            The text content.
        """
        try:
            # Try UTF-8 first, then fallback to other encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, raise error
            raise ValueError(f"Could not decode file {file_path} with any supported encoding")
            
        except Exception as e:
            logger.error(f"Error loading text document {file_path}: {e}")
            raise


class MarkdownLoader(TextLoader):
    """
    Document loader for Markdown files.
    
    Inherits from TextLoader as Markdown is just plain text.
    """
    
    def get_supported_extensions(self) -> List[str]:
        """Get the list of supported file extensions."""
        return ['.md', '.markdown']


class DocxLoader(DocumentLoader):
    """
    Document loader for DOCX files using python-docx.
    """
    
    def __init__(self):
        """Initialize the DOCX loader."""
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are installed."""
        try:
            import docx
        except ImportError:
            raise ImportError(
                "python-docx package is not installed. "
                "Install it with: pip install python-docx"
            )
    
    def get_supported_extensions(self) -> List[str]:
        """Get the list of supported file extensions."""
        return ['.docx']
    
    def can_load(self, file_path: str) -> bool:
        """Check if this loader can load the given file."""
        return Path(file_path).suffix.lower() in self.get_supported_extensions()
    
    def load_document(self, file_path: str) -> str:
        """
        Load a DOCX document and extract text.
        
        Args:
            file_path: The path to the DOCX file.
            
        Returns:
            The extracted text content.
        """
        import docx
        
        try:
            doc = docx.Document(file_path)
            
            # Extract text from all paragraphs
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            
            # Join paragraphs with newlines
            full_text = '\n\n'.join(text_parts)
            
            if not full_text.strip():
                logger.warning(f"No text extracted from DOCX: {file_path}")
            
            return full_text
            
        except Exception as e:
            logger.error(f"Error loading DOCX document {file_path}: {e}")
            raise


class DocumentLoader:
    """
    Unified document loader that delegates to specific loaders.
    
    This class provides a single interface for loading documents
    of various types by delegating to the appropriate loader.
    """
    
    # Available loaders
    LOADERS = [
        PDFLoader(),
        TextLoader(),
        MarkdownLoader(),
        DocxLoader(),
    ]
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """
        Get all supported file extensions.
        
        Returns:
            A list of all supported file extensions.
        """
        extensions = []
        for loader in cls.LOADERS:
            extensions.extend(loader.get_supported_extensions())
        return list(set(extensions))
    
    @classmethod
    def get_loader_for_file(cls, file_path: str) -> Optional[DocumentLoader]:
        """
        Get the appropriate loader for a given file.
        
        Args:
            file_path: The path to the file.
            
        Returns:
            The appropriate loader, or None if no loader supports the file.
        """
        for loader in cls.LOADERS:
            if loader.can_load(file_path):
                return loader
        return None
    
    @classmethod
    def load_document(cls, file_path: str) -> str:
        """
        Load a document from the given path.
        
        Args:
            file_path: The path to the document file.
            
        Returns:
            The text content of the document.
            
        Raises:
            ValueError: If the file type is not supported.
        """
        loader = cls.get_loader_for_file(file_path)
        
        if loader is None:
            raise ValueError(
                f"Unsupported file type: {Path(file_path).suffix}. "
                f"Supported types: {', '.join(cls.get_supported_extensions())}"
            )
        
        return loader.load_document(file_path)
    
    @classmethod
    def load_documents_from_folder(
        cls,
        folder_path: str,
        recursive: bool = True,
        extensions: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Load all supported documents from a folder.
        
        Args:
            folder_path: The path to the folder.
            recursive: Whether to search subdirectories recursively.
            extensions: Optional list of extensions to filter by.
                       If None, uses all supported extensions.
            
        Returns:
            A dictionary mapping file paths to their text content.
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")
        
        if not folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")
        
        # Determine which extensions to use
        if extensions is None:
            extensions = cls.get_supported_extensions()
        else:
            # Normalize extensions to lowercase with leading dot
            extensions = [
                ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                for ext in extensions
            ]
        
        documents = {}
        
        # Find all matching files
        if recursive:
            pattern = '**/*'
        else:
            pattern = '*'
        
        for file_path in folder_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    text = cls.load_document(str(file_path))
                    documents[str(file_path)] = text
                    logger.info(f"Loaded document: {file_path}")
                except Exception as e:
                    logger.error(f"Error loading document {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {folder_path}")
        return documents
    
    @classmethod
    def get_file_info(cls, file_path: str) -> Dict[str, any]:
        """
        Get information about a file.
        
        Args:
            file_path: The path to the file.
            
        Returns:
            A dictionary containing file information.
        """
        file_path = Path(file_path)
        
        return {
            'name': file_path.name,
            'path': str(file_path),
            'extension': file_path.suffix,
            'size': file_path.stat().st_size if file_path.exists() else 0,
            'is_supported': cls.get_loader_for_file(str(file_path)) is not None,
        }
    
    @classmethod
    def validate_path(cls, path: str, base_path: Optional[str] = None) -> bool:
        """
        Validate that a path is safe to access.
        
        Args:
            path: The path to validate.
            base_path: The base path to restrict access to.
                      If None, uses the documents base path from settings.
            
        Returns:
            True if the path is safe, False otherwise.
        """
        if base_path is None:
            base_path = getattr(settings, 'RAG_DOCUMENTS_BASE_PATH', 'data/documents')
        
        try:
            # Resolve absolute paths
            abs_path = Path(path).resolve()
            abs_base = Path(base_path).resolve()
            
            # Check if the path is within the base path
            try:
                abs_path.relative_to(abs_base)
                return True
            except ValueError:
                return False
                
        except Exception:
            return False
