"""
Text chunking service for splitting documents into manageable pieces.

This module provides various chunking strategies:
- Character-based chunking
- Sentence-based chunking
- Paragraph-based chunking
"""

import re
import logging
from typing import List, Dict, Optional, Callable

from .base import TextChunker

logger = logging.getLogger(__name__)


class CharacterChunker(TextChunker):
    """
    Text chunker that splits text by character count.
    
    This is the simplest chunking strategy, splitting text into
    chunks of approximately equal character count with overlap.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the character chunker.
        
        Args:
            chunk_size: The target size of each chunk in characters.
            chunk_overlap: The number of characters to overlap between chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split text into chunks by character count.
        
        Args:
            text: The text to chunk.
            metadata: Metadata to include with each chunk.
            
        Returns:
            A list of chunk dictionaries.
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)
            
            # Extract chunk text
            chunk_text = text[start:end]
            
            # Create chunk with metadata
            chunk = {
                'text': chunk_text,
                'chunk_id': chunk_id,
                **metadata
            }
            chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            chunk_id += 1
            
            # Prevent infinite loop if overlap is too large
            if start <= 0:
                start = end
        
        logger.debug(f"Chunked text into {len(chunks)} chunks (character-based)")
        return chunks
    
    def get_chunk_count(self, text: str) -> int:
        """
        Get the number of chunks that would be created from the text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            The number of chunks as an integer.
        """
        if not text or not text.strip():
            return 0
        
        text_length = len(text)
        effective_size = self.chunk_size - self.chunk_overlap
        
        if effective_size <= 0:
            return 1
        
        return max(1, (text_length + effective_size - 1) // effective_size)


class SentenceChunker(TextChunker):
    """
    Text chunker that splits text by sentences.
    
    This strategy tries to keep sentences together, which is
    better for semantic understanding than character-based chunking.
    """
    
    # Sentence boundary patterns
    SENTENCE_PATTERN = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])|'  # Sentence ending followed by space and capital
        r'(?<=[.!?])\s*\n|'          # Sentence ending followed by newline
        r'(?<=[.!?])\s+(?=[a-z])'    # Sentence ending followed by lowercase (for abbreviations)
    )
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the sentence chunker.
        
        Args:
            chunk_size: The target size of each chunk in characters.
            chunk_overlap: The number of characters to overlap between chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: The text to split.
            
        Returns:
            A list of sentences.
        """
        # First, split by the pattern
        sentences = self.SENTENCE_PATTERN.split(text)
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split text into chunks by sentences.
        
        Args:
            text: The text to chunk.
            metadata: Metadata to include with each chunk.
            
        Returns:
            A list of chunk dictionaries.
        """
        if not text or not text.strip():
            return []
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunk = {
                    'text': chunk_text,
                    'chunk_id': chunk_id,
                    **metadata
                }
                chunks.append(chunk)
                
                # Start new chunk with overlap
                # Keep some sentences from the end of the previous chunk
                overlap_sentences = []
                overlap_length = 0
                
                for sent in reversed(current_chunk):
                    if overlap_length + len(sent) <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += len(sent)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
                chunk_id += 1
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the last chunk if it has content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = {
                'text': chunk_text,
                'chunk_id': chunk_id,
                **metadata
            }
            chunks.append(chunk)
        
        logger.debug(f"Chunked text into {len(chunks)} chunks (sentence-based)")
        return chunks
    
    def get_chunk_count(self, text: str) -> int:
        """
        Get the number of chunks that would be created from the text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            The number of chunks as an integer.
        """
        chunks = self.chunk_text(text, {})
        return len(chunks)


class ParagraphChunker(TextChunker):
    """
    Text chunker that splits text by paragraphs.
    
    This strategy keeps paragraphs together, which is useful for
    documents where paragraph boundaries are meaningful.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the paragraph chunker.
        
        Args:
            chunk_size: The target size of each chunk in characters.
            chunk_overlap: The number of characters to overlap between chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Args:
            text: The text to split.
            
        Returns:
            A list of paragraphs.
        """
        # Split by double newlines (paragraph separator)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split text into chunks by paragraphs.
        
        Args:
            text: The text to chunk.
            metadata: Metadata to include with each chunk.
            
        Returns:
            A list of chunk dictionaries.
        """
        if not text or not text.strip():
            return []
        
        # Split into paragraphs
        paragraphs = self._split_paragraphs(text)
        
        if not paragraphs:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            
            # If adding this paragraph would exceed chunk size
            if current_length + paragraph_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunk = {
                    'text': chunk_text,
                    'chunk_id': chunk_id,
                    **metadata
                }
                chunks.append(chunk)
                
                # Start new chunk with overlap
                # Keep some paragraphs from the end of the previous chunk
                overlap_paragraphs = []
                overlap_length = 0
                
                for para in reversed(current_chunk):
                    if overlap_length + len(para) <= self.chunk_overlap:
                        overlap_paragraphs.insert(0, para)
                        overlap_length += len(para)
                    else:
                        break
                
                current_chunk = overlap_paragraphs
                current_length = overlap_length
                chunk_id += 1
            
            # Add paragraph to current chunk
            current_chunk.append(paragraph)
            current_length += paragraph_length
        
        # Add the last chunk if it has content
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk = {
                'text': chunk_text,
                'chunk_id': chunk_id,
                **metadata
            }
            chunks.append(chunk)
        
        logger.debug(f"Chunked text into {len(chunks)} chunks (paragraph-based)")
        return chunks
    
    def get_chunk_count(self, text: str) -> int:
        """
        Get the number of chunks that would be created from the text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            The number of chunks as an integer.
        """
        chunks = self.chunk_text(text, {})
        return len(chunks)


class TextChunker:
    """
    Unified text chunker that supports multiple chunking strategies.
    
    This class provides a single interface for chunking text using
    different strategies (character, sentence, paragraph).
    """
    
    # Available chunking strategies
    STRATEGIES = {
        'character': CharacterChunker,
        'sentence': SentenceChunker,
        'paragraph': ParagraphChunker,
    }
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: str = 'character'
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: The target size of each chunk in characters.
            chunk_overlap: The number of characters to overlap between chunks.
            strategy: The chunking strategy to use ('character', 'sentence', 'paragraph').
        """
        strategy = strategy.lower()
        
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown chunking strategy: {strategy}. "
                f"Available strategies: {', '.join(self.STRATEGIES.keys())}"
            )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        
        # Create the appropriate chunker
        chunker_class = self.STRATEGIES[strategy]
        self._chunker = chunker_class(chunk_size, chunk_overlap)
    
    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split text into chunks using the configured strategy.
        
        Args:
            text: The text to chunk.
            metadata: Metadata to include with each chunk.
            
        Returns:
            A list of chunk dictionaries.
        """
        return self._chunker.chunk_text(text, metadata)
    
    def get_chunk_count(self, text: str) -> int:
        """
        Get the number of chunks that would be created from the text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            The number of chunks as an integer.
        """
        return self._chunker.get_chunk_count(text)
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """
        Get the list of available chunking strategies.
        
        Returns:
            A list of strategy names.
        """
        return list(cls.STRATEGIES.keys())
    
    def chunk_documents(
        self,
        documents: Dict[str, str],
        base_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Chunk multiple documents.
        
        Args:
            documents: A dictionary mapping file paths to text content.
            base_path: Optional base path for relative paths in metadata.
            
        Returns:
            A list of all chunks from all documents.
        """
        all_chunks = []
        
        for file_path, text in documents.items():
            # Create metadata for this document
            from pathlib import Path
            file_path_obj = Path(file_path)
            
            metadata = {
                'file_name': file_path_obj.name,
                'source_path': file_path,
            }
            
            # Add relative path if base_path is provided
            if base_path:
                try:
                    base_path_obj = Path(base_path).resolve()
                    file_path_abs = file_path_obj.resolve()
                    relative_path = file_path_abs.relative_to(base_path_obj)
                    metadata['relative_path'] = str(relative_path)
                except ValueError:
                    pass
            
            # Chunk the document
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks
