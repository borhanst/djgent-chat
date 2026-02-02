"""
Base classes and interfaces for the chat application services.

This module defines abstract base classes that all service implementations
must follow, ensuring consistent interfaces across different providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """
    Result of an embedding operation.
    
    Attributes:
        vector: The embedding vector as a list of floats
        dimension: The dimension of the embedding vector
        model: The model name used for embedding
    """
    vector: List[float]
    dimension: int
    model: str


@dataclass
class SearchResult:
    """
    Result of a vector search operation.
    
    Attributes:
        text: The text content of the chunk
        score: The similarity score (lower is better for FAISS)
        metadata: Additional metadata about the chunk
    """
    text: str
    score: float
    metadata: Dict[str, Any]


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    
    All embedding providers (OpenAI, Gemini, HuggingFace) must implement
    these methods to ensure a consistent interface.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding provider.
        
        Args:
            model_name: Optional model name to use. If not provided,
                       uses the default model for the provider.
        """
        self.model_name = model_name or self.get_default_model()
    
    @abstractmethod
    def get_default_model(self) -> str:
        """
        Get the default model name for this provider.
        
        Returns:
            The default model name as a string.
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            The dimension as an integer.
        """
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.
        
        Args:
            text: The text to embed.
            
        Returns:
            The embedding vector as a list of floats.
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: A list of texts to embed.
            
        Returns:
            A list of embedding vectors.
        """
        pass
    
    def embed_result(self, text: str) -> EmbeddingResult:
        """
        Generate an embedding result for a single text.
        
        Args:
            text: The text to embed.
            
        Returns:
            An EmbeddingResult object containing the vector and metadata.
        """
        vector = self.embed_text(text)
        return EmbeddingResult(
            vector=vector,
            dimension=len(vector),
            model=self.model_name
        )
    
    def embed_batch_results(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Generate embedding results for multiple texts.
        
        Args:
            texts: A list of texts to embed.
            
        Returns:
            A list of EmbeddingResult objects.
        """
        vectors = self.embed_batch(texts)
        return [
            EmbeddingResult(
                vector=vector,
                dimension=len(vector),
                model=self.model_name
            )
            for vector in vectors
        ]


class VectorStore(ABC):
    """
    Abstract base class for vector stores.
    
    All vector store implementations must implement these methods
    to ensure a consistent interface.
    """
    
    @abstractmethod
    def create_index(self, dimension: int) -> None:
        """
        Create a new index with the given dimension.
        
        Args:
            dimension: The dimension of the vectors to be stored.
        """
        pass
    
    @abstractmethod
    def load_index(self) -> bool:
        """
        Load an existing index from disk.
        
        Returns:
            True if the index was loaded successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def save_index(self) -> None:
        """
        Save the current index to disk.
        """
        pass
    
    @abstractmethod
    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict]) -> None:
        """
        Add vectors to the index with associated metadata.
        
        Args:
            vectors: A list of vectors to add.
            metadata: A list of metadata dictionaries, one per vector.
        """
        pass
    
    @abstractmethod
    def search(self, query_vector: List[float], k: int) -> List[SearchResult]:
        """
        Search for the k most similar vectors to the query.
        
        Args:
            query_vector: The query vector.
            k: The number of results to return.
            
        Returns:
            A list of SearchResult objects.
        """
        pass
    
    @abstractmethod
    def delete_index(self) -> None:
        """
        Delete the current index.
        """
        pass
    
    @abstractmethod
    def get_vector_count(self) -> int:
        """
        Get the number of vectors in the index.
        
        Returns:
            The number of vectors as an integer.
        """
        pass


class DocumentLoader(ABC):
    """
    Abstract base class for document loaders.
    
    All document loaders must implement these methods to ensure
    a consistent interface.
    """
    
    @abstractmethod
    def load_document(self, file_path: str) -> str:
        """
        Load a document from the given path.
        
        Args:
            file_path: The path to the document file.
            
        Returns:
            The text content of the document.
        """
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """
        Get the list of file extensions supported by this loader.
        
        Returns:
            A list of file extensions (e.g., ['.pdf', '.txt']).
        """
        pass
    
    @abstractmethod
    def can_load(self, file_path: str) -> bool:
        """
        Check if this loader can load the given file.
        
        Args:
            file_path: The path to the file.
            
        Returns:
            True if the file can be loaded, False otherwise.
        """
        pass


class TextChunker(ABC):
    """
    Abstract base class for text chunkers.
    
    All text chunkers must implement these methods to ensure
    a consistent interface.
    """
    
    @abstractmethod
    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split text into chunks.
        
        Args:
            text: The text to chunk.
            metadata: Metadata to include with each chunk.
            
        Returns:
            A list of chunk dictionaries, each containing 'text' and metadata.
        """
        pass
    
    @abstractmethod
    def get_chunk_count(self, text: str) -> int:
        """
        Get the number of chunks that would be created from the text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            The number of chunks as an integer.
        """
        pass
