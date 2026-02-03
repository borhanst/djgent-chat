"""
Text chunking service for splitting documents into manageable pieces.

This module provides various chunking strategies including:
- LangChain's RecursiveCharacterTextSplitter (recommended)
- Character-based chunking
- Sentence-based chunking
- Paragraph-based chunking
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import TextChunker

logger = logging.getLogger(__name__)


class LangchainChunker(TextChunker):
    """
    Text chunker that uses LangChain's RecursiveCharacterTextSplitter.

    This is the recommended chunking strategy as it recursively splits text
    using multiple separators to keep semantically related content together.
    It tries to split by paragraphs first, then sentences, then characters.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = False,
    ):
        """
        Initialize the LangChain-based chunker.

        Args:
            chunk_size: The target size of each chunk in characters.
            chunk_overlap: The number of characters to overlap between chunks.
            separators: Custom separators to use for splitting. If None, uses
                       default separators (["\n\n", "\n", " ", ""]).
            keep_separator: Whether to keep the separator in the chunk text.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Create LangChain's RecursiveCharacterTextSplitter
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=keep_separator,
            length_function=len,
        )

    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split text into chunks using LangChain's recursive splitter.

        Args:
            text: The text to chunk.
            metadata: Metadata to include with each chunk.

        Returns:
            A list of chunk dictionaries.
        """
        if not text or not text.strip():
            return []

        # Use LangChain's split_text method
        chunk_texts = self._splitter.split_text(text)

        chunks = []
        for chunk_id, chunk_text in enumerate(chunk_texts):
            chunk = {"text": chunk_text, "chunk_id": chunk_id, **metadata}
            chunks.append(chunk)

        logger.debug(
            f"Chunked text into {len(chunks)} chunks (langchain-recursive)"
        )
        return chunks

    def chunk_documents(
        self, documents: Dict[str, str], base_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Chunk multiple documents using LangChain's splitter with Document objects.

        Args:
            documents: A dictionary mapping file paths to text content.
            base_path: Optional base path for relative paths in metadata.

        Returns:
            A list of all chunks from all documents.
        """
        from langchain_core.documents import Document

        all_chunks = []

        for file_path, text in documents.items():
            # Create metadata for this document
            file_path_obj = Path(file_path)

            metadata = {
                "file_name": file_path_obj.name,
                "source_path": file_path,
            }

            # Add relative path if base_path is provided
            if base_path:
                try:
                    base_path_obj = Path(base_path).resolve()
                    file_path_abs = file_path_obj.resolve()
                    relative_path = file_path_abs.relative_to(base_path_obj)
                    metadata["relative_path"] = str(relative_path)
                except ValueError:
                    pass

            # Create a LangChain Document object
            doc = Document(page_content=text, metadata=metadata)

            # Split using LangChain's split_documents method
            split_docs = self._splitter.split_documents([doc])

            # Convert to our chunk format
            for chunk_id, split_doc in enumerate(split_docs):
                chunk = {
                    "text": split_doc.page_content,
                    "chunk_id": chunk_id,
                    **split_doc.metadata,
                }
                all_chunks.append(chunk)

        logger.info(
            f"Chunked {len(documents)} documents into {len(all_chunks)} chunks (langchain)"
        )
        return all_chunks

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

        chunk_texts = self._splitter.split_text(text)
        return len(chunk_texts)


class SentenceAwareChunker(TextChunker):
    """
    Text chunker that uses LangChain's RecursiveCharacterTextSplitter
    with sentence-aware separators.

    This is optimized for documents where sentence boundaries are important.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the sentence-aware chunker.

        Args:
            chunk_size: The target size of each chunk in characters.
            chunk_overlap: The number of characters to overlap between chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Use separators that respect sentence boundaries
        separators = [
            "\n\n",  # Paragraphs
            "\n",  # Lines
            ". ",  # Sentences (period + space)
            "! ",  # Exclamation
            "? ",  # Question
            "; ",  # Semicolon
            ", ",  # Comma
            " ",  # Words
            "",  # Characters
        ]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=False,
            length_function=len,
        )

    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split text into sentence-aware chunks.

        Args:
            text: The text to chunk.
            metadata: Metadata to include with each chunk.

        Returns:
            A list of chunk dictionaries.
        """
        if not text or not text.strip():
            return []

        chunk_texts = self._splitter.split_text(text)

        chunks = []
        for chunk_id, chunk_text in enumerate(chunk_texts):
            chunk = {"text": chunk_text, "chunk_id": chunk_id, **metadata}
            chunks.append(chunk)

        logger.debug(
            f"Chunked text into {len(chunks)} chunks (langchain-sentence-aware)"
        )
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

        chunk_texts = self._splitter.split_text(text)
        return len(chunk_texts)


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
            chunk = {"text": chunk_text, "chunk_id": chunk_id, **metadata}
            chunks.append(chunk)

            # Move start position with overlap
            start = end - self.chunk_overlap
            chunk_id += 1

            # Prevent infinite loop if overlap is too large
            if start <= 0:
                start = end

        logger.debug(
            f"Chunked text into {len(chunks)} chunks (character-based)"
        )
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
        r"(?<=[.!?])\s+(?=[A-Z])|"  # Sentence ending followed by space and capital
        r"(?<=[.!?])\s*\n|"  # Sentence ending followed by newline
        r"(?<=[.!?])\s+(?=[a-z])"  # Sentence ending followed by lowercase (for abbreviations)
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
            if (
                current_length + sentence_length > self.chunk_size
                and current_chunk
            ):
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunk = {"text": chunk_text, "chunk_id": chunk_id, **metadata}
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
            chunk_text = " ".join(current_chunk)
            chunk = {"text": chunk_text, "chunk_id": chunk_id, **metadata}
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
        paragraphs = re.split(r"\n\s*\n", text)

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
            if (
                current_length + paragraph_length > self.chunk_size
                and current_chunk
            ):
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunk = {"text": chunk_text, "chunk_id": chunk_id, **metadata}
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
            chunk_text = "\n\n".join(current_chunk)
            chunk = {"text": chunk_text, "chunk_id": chunk_id, **metadata}
            chunks.append(chunk)

        logger.debug(
            f"Chunked text into {len(chunks)} chunks (paragraph-based)"
        )
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
    different strategies (langchain, langchain-sentence, character, sentence, paragraph).
    """

    # Available chunking strategies
    STRATEGIES = {
        "langchain": LangchainChunker,
        "langchain-sentence": SentenceAwareChunker,
        "character": CharacterChunker,
        "sentence": SentenceChunker,
        "paragraph": ParagraphChunker,
    }

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: str = "langchain",
    ):
        """
        Initialize the text chunker.

        Args:
            chunk_size: The target size of each chunk in characters.
            chunk_overlap: The number of characters to overlap between chunks.
            strategy: The chunking strategy to use ('langchain', 'langchain-sentence',
                     'character', 'sentence', 'paragraph').
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

    def chunk_documents(
        self, documents: Dict[str, str], base_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Chunk multiple documents.

        Args:
            documents: A dictionary mapping file paths to text content.
            base_path: Optional base path for relative paths in metadata.

        Returns:
            A list of all chunks from all documents.
        """
        # Use LangChain chunker's optimized chunk_documents method if available
        if isinstance(self._chunker, (LangchainChunker, SentenceAwareChunker)):
            return self._chunker.chunk_documents(documents, base_path)

        # Fallback to basic implementation
        all_chunks = []

        for file_path, text in documents.items():
            # Create metadata for this document
            file_path_obj = Path(file_path)

            metadata = {
                "file_name": file_path_obj.name,
                "source_path": file_path,
            }

            # Add relative path if base_path is provided
            if base_path:
                try:
                    base_path_obj = Path(base_path).resolve()
                    file_path_abs = file_path_obj.resolve()
                    relative_path = file_path_abs.relative_to(base_path_obj)
                    metadata["relative_path"] = str(relative_path)
                except ValueError:
                    pass

            # Chunk the document
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)

        logger.info(
            f"Chunked {len(documents)} documents into {len(all_chunks)} chunks"
        )
        return all_chunks

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """
        Get the list of available chunking strategies.

        Returns:
            A list of strategy names.
        """
        return list(cls.STRATEGIES.keys())
