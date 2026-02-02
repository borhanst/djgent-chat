"""
Vector store service using FAISS for efficient similarity search.

This module provides a FAISS-based vector store implementation with
support for saving/loading indexes to disk and metadata management.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from django.conf import settings

from .base import VectorStore, SearchResult

logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """
    FAISS-based vector store implementation.
    
    This class provides methods to create, load, save, and query
    FAISS indexes with associated metadata.
    """
    
    def __init__(
        self,
        index_path: str,
        embedding_provider,
        index_type: str = 'flat'
    ):
        """
        Initialize the FAISS vector store.
        
        Args:
            index_path: Path to the FAISS index file (without extension).
            embedding_provider: An embedding provider instance.
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw').
                       'flat' is exact search, others are approximate.
        """
        self.index_path = Path(index_path)
        self.embedding_provider = embedding_provider
        self.index_type = index_type.lower()
        
        # FAISS index file paths
        self.index_file = self.index_path / 'index.faiss'
        self.metadata_file = self.index_path / 'metadata.pkl'
        
        # Internal state
        self._index = None
        self._metadata: List[Dict[str, Any]] = []
        self._dimension = None
        self._is_loaded = False
    
    def create_index(self, dimension: int) -> None:
        """
        Create a new FAISS index with the given dimension.
        
        Args:
            dimension: The dimension of the vectors to be stored.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS package is not installed. "
                "Install it with: pip install faiss-cpu"
            )
        
        self._dimension = dimension
        self._metadata = []
        
        # Create index based on type
        if self.index_type == 'flat':
            # Flat L2 index - exact search
            self._index = faiss.IndexFlatL2(dimension)
        elif self.index_type == 'ivf':
            # IVF index - approximate search, faster for large datasets
            quantizer = faiss.IndexFlatL2(dimension)
            nlist = min(100, dimension)  # Number of clusters
            self._index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        elif self.index_type == 'hnsw':
            # HNSW index - approximate search, good balance of speed/accuracy
            self._index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        self._is_loaded = True
        logger.info(f"Created new FAISS index with dimension {dimension}")
    
    def load_index(self) -> bool:
        """
        Load an existing FAISS index from disk.
        
        Returns:
            True if the index was loaded successfully, False otherwise.
        """
        if not self.index_file.exists() or not self.metadata_file.exists():
            logger.warning(f"Index files not found at {self.index_path}")
            return False
        
        try:
            import faiss
            
            # Load FAISS index
            self._index = faiss.read_index(str(self.index_file))
            
            # Load metadata
            with open(self.metadata_file, 'rb') as f:
                self._metadata = pickle.load(f)
            
            # Get dimension from index
            self._dimension = self._index.d
            self._is_loaded = True
            
            logger.info(
                f"Loaded FAISS index from {self.index_path} "
                f"with {self._index.ntotal} vectors"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return False
    
    def save_index(self) -> None:
        """
        Save the current FAISS index to disk.
        """
        if self._index is None:
            raise RuntimeError("No index to save. Create or load an index first.")
        
        try:
            import faiss
            
            # Create directory if it doesn't exist
            self.index_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self._index, str(self.index_file))
            
            # Save metadata
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self._metadata, f)
            
            logger.info(
                f"Saved FAISS index to {self.index_path} "
                f"with {self._index.ntotal} vectors"
            )
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            raise
    
    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict]) -> None:
        """
        Add vectors to the index with associated metadata.
        
        Args:
            vectors: A list of vectors to add.
            metadata: A list of metadata dictionaries, one per vector.
        """
        if self._index is None:
            raise RuntimeError("No index available. Create or load an index first.")
        
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match number of metadata entries")
        
        if not vectors:
            logger.warning("No vectors to add")
            return
        
        try:
            import numpy as np
            
            # Convert vectors to numpy array
            vectors_array = np.array(vectors, dtype='float32')
            
            # Train IVF index if needed (first time adding vectors)
            if self.index_type == 'ivf' and not self._index.is_trained:
                self._index.train(vectors_array)
            
            # Add vectors to index
            self._index.add(vectors_array)
            
            # Add metadata
            self._metadata.extend(metadata)
            
            logger.info(f"Added {len(vectors)} vectors to index")
            
        except Exception as e:
            logger.error(f"Error adding vectors to index: {e}")
            raise
    
    def search(self, query_vector: List[float], k: int) -> List[SearchResult]:
        """
        Search for the k most similar vectors to the query.
        
        Args:
            query_vector: The query vector.
            k: The number of results to return.
            
        Returns:
            A list of SearchResult objects.
        """
        if self._index is None:
            raise RuntimeError("No index available. Create or load an index first.")
        
        if self._index.ntotal == 0:
            logger.warning("Index is empty, no results to return")
            return []
        
        try:
            import numpy as np
            
            # Convert query to numpy array
            query_array = np.array([query_vector], dtype='float32')
            
            # Adjust k if necessary
            k = min(k, self._index.ntotal)
            
            # Search
            distances, indices = self._index.search(query_array, k)
            
            # Build results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                metadata = self._metadata[idx]
                results.append(SearchResult(
                    text=metadata.get('text', ''),
                    score=float(distances[0][i]),
                    metadata=metadata
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching index: {e}")
            raise
    
    def delete_index(self) -> None:
        """
        Delete the current index and remove files from disk.
        """
        self._index = None
        self._metadata = []
        self._dimension = None
        self._is_loaded = False
        
        # Remove files if they exist
        if self.index_file.exists():
            self.index_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        
        logger.info(f"Deleted index at {self.index_path}")
    
    def get_vector_count(self) -> int:
        """
        Get the number of vectors in the index.
        
        Returns:
            The number of vectors as an integer.
        """
        if self._index is None:
            return 0
        return self._index.ntotal
    
    def get_dimension(self) -> Optional[int]:
        """
        Get the dimension of the vectors in the index.
        
        Returns:
            The dimension as an integer, or None if no index is loaded.
        """
        return self._dimension
    
    def is_loaded(self) -> bool:
        """
        Check if an index is currently loaded.
        
        Returns:
            True if an index is loaded, False otherwise.
        """
        return self._is_loaded
    
    def add_text_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> None:
        """
        Add text chunks to the index by generating embeddings.
        
        Args:
            chunks: A list of chunk dictionaries, each containing 'text' and metadata.
            batch_size: Number of chunks to process at a time.
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_provider.embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        # Add vectors with metadata
        self.add_vectors(all_embeddings, chunks)
        
        logger.info(f"Added {len(chunks)} text chunks to index")
    
    def search_by_text(
        self,
        query: str,
        k: int = 5
    ) -> List[SearchResult]:
        """
        Search for similar chunks by text query.
        
        Args:
            query: The text query.
            k: The number of results to return.
            
        Returns:
            A list of SearchResult objects.
        """
        # Generate embedding for query
        query_embedding = self.embedding_provider.embed_text(query)
        
        # Search
        return self.search(query_embedding, k)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            A dictionary containing index statistics.
        """
        return {
            'vector_count': self.get_vector_count(),
            'dimension': self.get_dimension(),
            'index_type': self.index_type,
            'is_loaded': self.is_loaded(),
            'index_path': str(self.index_path),
        }


def get_vector_store(
    folder_name: str,
    embedding_provider,
    index_base_path: Optional[str] = None
) -> FAISSVectorStore:
    """
    Factory function to get a FAISS vector store instance.
    
    Args:
        folder_name: Name of the folder (used to create index path).
        embedding_provider: An embedding provider instance.
        index_base_path: Base path for storing indexes. If not provided,
                         uses the default from settings.
    
    Returns:
        A FAISSVectorStore instance.
    """
    if index_base_path is None:
        index_base_path = getattr(settings, 'RAG_FAISS_INDEX_BASE_PATH', 'faiss_indexes')
    
    # Create index path from folder name (sanitize)
    safe_folder_name = folder_name.replace('/', '_').replace('\\', '_')
    index_path = Path(index_base_path) / safe_folder_name
    
    return FAISSVectorStore(
        index_path=index_path,
        embedding_provider=embedding_provider
    )
