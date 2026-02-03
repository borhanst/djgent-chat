"""
FAISS vector store implementation.

This module provides a FAISS-based vector store implementation that
extends the BaseVectorStore interface with FAISS-specific functionality.
"""

import logging
import pickle
from typing import Any, Dict, List, Optional

from django.conf import settings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based vector store implementation.

    This class provides methods to create, load, save, and query
    FAISS indexes with associated metadata while implementing the
    BaseVectorStore interface.

    Features:
    - Local file-based storage
    - Support for different index types (flat, ivf, hnsw)
    - Metadata storage and retrieval
    - Configurable vector normalization
    """

    def __init__(
        self,
        embedding: Embeddings,
        folder_name: str,
        index_base_path: Optional[str] = None,
        index_type: str = "flat",
        normalize_l2: bool = False,
        **kwargs,
    ):
        """
        Initialize the FAISS vector store.

        Args:
            embedding: LangChain embeddings instance.
            folder_name: Name of the folder (used to create index path).
            index_base_path: Base path for storing indexes. If not provided,
                           uses the default from settings.
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw').
                       'flat' is exact search, others are approximate.
            normalize_l2: Whether to normalize vectors to unit length.
            **kwargs: Additional keyword arguments.
        """
        # Set index base path from settings if not provided
        if index_base_path is None:
            index_base_path = getattr(
                settings, "RAG_FAISS_INDEX_BASE_PATH", "faiss_indexes"
            )

        super().__init__(
            embedding=embedding,
            folder_name=folder_name,
            index_base_path=index_base_path,
            **kwargs,
        )

        self.index_type = index_type.lower()
        self.normalize_l2 = normalize_l2

        # Set up file paths
        self._setup_file_paths()

        # Internal state
        self._index = None
        self._docstore: Dict[str, Document] = {}
        self._dimension = None

    def _setup_file_paths(self) -> None:
        """Set up the index and metadata file paths."""
        if self.index_path:
            self.index_file = self.index_path / "index.faiss"
            self.metadata_file = self.index_path / "metadata.pkl"
        else:
            self.index_file = None
            self.metadata_file = None

    def _load_index(self) -> bool:
        """
        Load an existing FAISS index from disk.

        Returns:
            True if the index was loaded successfully, False otherwise.
        """
        if self.index_file is None or not self.index_file.exists():
            logger.warning(f"Index files not found at {self.index_path}")
            return False

        if self.metadata_file is None or not self.metadata_file.exists():
            logger.warning(f"Metadata file not found at {self.index_path}")
            return False

        try:
            import faiss

            # Load FAISS index
            self._index = faiss.read_index(str(self.index_file))

            # Load metadata (document store)
            with open(self.metadata_file, "rb") as f:
                self._docstore = pickle.load(f)

            # Get dimension from index
            self._dimension = self._index.d

            logger.info(
                f"Loaded FAISS index from {self.index_path} "
                f"with {self._index.ntotal} vectors"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return False

    def _save_index(self) -> None:
        """Save the current FAISS index and metadata to disk."""
        if self._index is None:
            raise RuntimeError(
                "No index to save. Create or load an index first."
            )

        if self.index_path is None:
            raise RuntimeError("No index path configured")

        try:
            import faiss

            # Create directory if it doesn't exist
            self.index_path.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            faiss.write_index(self._index, str(self.index_file))

            # Save metadata (document store)
            with open(self.metadata_file, "wb") as f:
                pickle.dump(self._docstore, f)

            logger.info(
                f"Saved FAISS index to {self.index_path} "
                f"with {self._index.ntotal} vectors"
            )

        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            raise

    def _create_index(self, dimension: int) -> None:
        """
        Create a new FAISS index with the given dimension.

        Args:
            dimension: The dimension of the vectors to be stored.
        """
        import faiss

        self._dimension = dimension

        if self.index_type == "flat":
            # Flat L2 index - exact search
            if self.normalize_l2:
                self._index = faiss.IndexFlatIP(dimension)
            else:
                self._index = faiss.IndexFlatL2(dimension)

        elif self.index_type == "ivf":
            # IVF index - approximate search, faster for large datasets
            quantizer = faiss.IndexFlatL2(dimension)
            nlist = min(100, dimension)  # Number of clusters
            self._index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

        elif self.index_type == "hnsw":
            # HNSW index - approximate search, good balance of speed/accuracy
            self._index = faiss.IndexHNSWFlat(dimension, 32)

        else:
            # Default to flat
            if self.normalize_l2:
                self._index = faiss.IndexFlatIP(dimension)
            else:
                self._index = faiss.IndexFlatL2(dimension)

        logger.info(f"Created new FAISS index with dimension {dimension}")

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Add texts to the vector store.

        Args:
            texts: List of text strings to add.
            metadatas: Optional list of metadata dictionaries.
            ids: Optional list of IDs for the texts.
            **kwargs: Additional keyword arguments.

        Returns:
            List of IDs for the added texts.
        """
        import numpy as np

        # Load or create index
        if self._index is None:
            if not self._load_index():
                # Create new index with first embedding
                test_embeddings = self.embedding.embed_documents(texts[:1])
                self._create_index(len(test_embeddings[0]))

        # Generate embeddings
        embeddings = self.embedding.embed_documents(texts)
        embeddings_array = np.array(embeddings, dtype="float32")

        # Normalize if needed
        if self.normalize_l2:
            faiss.normalize_L2(embeddings_array)

        # Train IVF index if needed (first time adding vectors)
        if self.index_type == "ivf" and not self._index.is_trained:
            self._index.train(embeddings_array)

        # Add to index
        start_id = self._index.ntotal
        self._index.add(embeddings_array)

        # Generate IDs if not provided
        if ids is None:
            ids = [str(i) for i in range(start_id, start_id + len(texts))]

        # Use provided metadatas or empty dicts
        if metadatas is None:
            metadatas = [{}] * len(texts)

        # Store documents
        for text, metadata, id_ in zip(texts, metadatas, ids):
            self._docstore[id_] = Document(page_content=text, metadata=metadata)

        # Save index
        self._save_index()

        logger.info(f"Added {len(texts)} texts to index")
        return ids

    async def aadd_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Async add texts to the vector store.

        Args:
            texts: List of text strings to add.
            metadatas: Optional list of metadata dictionaries.
            ids: Optional list of IDs for the texts.
            **kwargs: Additional keyword arguments.

        Returns:
            List of IDs for the added texts.
        """
        # For now, use sync implementation
        return self.add_texts(texts, metadatas, ids, **kwargs)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs
    ) -> List[Document]:
        """
        Search for similar documents to a query.

        Args:
            query: Query text to search for.
            k: Number of results to return.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects sorted by relevance.
        """
        # Get query embedding
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_by_vector(embedding, k, **kwargs)

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs
    ) -> List[Document]:
        """
        Async search for similar documents.

        Args:
            query: Query text to search for.
            k: Number of results to return.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects sorted by relevance.
        """
        return self.similarity_search(query, k, **kwargs)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs
    ) -> List[Document]:
        """
        Search for similar documents by embedding vector.

        Args:
            embedding: Query embedding vector.
            k: Number of results to return.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects sorted by relevance.
        """
        import numpy as np

        # Load index if needed
        if self._index is None:
            self._load_index()

        if self._index is None:
            logger.warning("No index available for search")
            return []

        if self._index.ntotal == 0:
            logger.warning("Index is empty, no results to return")
            return []

        # Convert query to numpy array
        query_array = np.array([embedding], dtype="float32")

        # Normalize if needed
        if self.normalize_l2:
            import faiss

            faiss.normalize_L2(query_array)

        # Adjust k if necessary
        k = min(k, self._index.ntotal)

        # Search
        distances, indices = self._index.search(query_array, k)

        # Build results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue

            doc_id = str(idx)
            if doc_id in self._docstore:
                doc = self._docstore[doc_id]
                # Add score to metadata
                doc.metadata["score"] = float(distances[0][i])
                results.append(doc)

        return results

    async def asimilarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs
    ) -> List[Document]:
        """
        Async search by embedding vector.

        Args:
            embedding: Query embedding vector.
            k: Number of results to return.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects sorted by relevance.
        """
        return self.similarity_search_by_vector(embedding, k, **kwargs)

    def delete_collection(self) -> None:
        """Delete the entire collection (index and documents)."""
        self._index = None
        self._docstore = {}
        self._dimension = None

        # Remove files
        if self.index_file and self.index_file.exists():
            self.index_file.unlink()
        if self.metadata_file and self.metadata_file.exists():
            self.metadata_file.unlink()

        logger.info(f"Deleted collection at {self.index_path}")

    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> None:
        """
        Delete documents from the vector store.

        Args:
            ids: List of document IDs to delete.
            **kwargs: Additional keyword arguments.
        """
        if ids is None:
            return

        for id_ in ids:
            if id_ in self._docstore:
                del self._docstore[id_]

        # Note: FAISS doesn't support efficient deletion
        # The index would need to be rebuilt for true deletion
        logger.warning(
            "Documents removed from docstore but FAISS index not rebuilt. "
            "Consider recreating the index for true deletion."
        )

    def get_vector_count(self) -> int:
        """
        Get the number of vectors in the store.

        Returns:
            Number of vectors as an integer.
        """
        if self._index is None:
            self._load_index()

        if self._index is None:
            return 0

        return self._index.ntotal

    def is_loaded(self) -> bool:
        """
        Check if the store is loaded and ready.

        Returns:
            True if loaded, False otherwise.
        """
        return self._index is not None

    def get_dimension(self) -> Optional[int]:
        """
        Get the dimension of the vectors.

        Returns:
            Dimension as an integer, or None if not available.
        """
        if self._dimension is None and self._index is not None:
            self._dimension = self._index.d

        return self._dimension

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary containing store statistics.
        """
        return {
            "vector_count": self.get_vector_count(),
            "dimension": self.get_dimension(),
            "index_type": self.index_type,
            "is_loaded": self.is_loaded(),
            "index_path": str(self.index_path) if self.index_path else None,
            "normalize_l2": self.normalize_l2,
            "backend": "faiss",
        }

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs,
    ) -> List[Document]:
        """
        Search for documents using maximal marginal relevance.

        This method aims to diversify results by selecting documents
        that are both relevant to the query and different from each other.

        Args:
            query: Query text to search for.
            k: Number of results to return.
            fetch_k: Number of documents to fetch before filtering.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects.
        """
        import numpy as np

        # Load index if needed
        if self._index is None:
            self._load_index()

        if self._index is None or self._index.ntotal == 0:
            return []

        # Get query embedding
        embedding = self.embedding.embed_query(query)
        query_array = np.array([embedding], dtype="float32")

        # Search for more documents than needed
        fetch_k = min(fetch_k, self._index.ntotal)
        distances, indices = self._index.search(query_array, fetch_k)

        # Get embeddings of fetched documents
        fetched_embeddings = []
        for idx in indices[0]:
            if idx == -1:
                continue
            doc_id = str(idx)
            if doc_id in self._docstore:
                doc_embedding = self.embedding.embed_query(
                    self._docstore[doc_id].page_content
                )
                fetched_embeddings.append(doc_embedding)

        if not fetched_embeddings:
            return []

        fetched_embeddings = np.array(fetched_embeddings, dtype="float32")

        # Select diverse results using MMR
        selected_indices = []
        selected_embeddings = []

        for i in range(k):
            if not selected_indices:
                # First selection: most relevant
                best_idx = 0
            else:
                # Subsequent selections: balance relevance and diversity
                best_score = -1
                best_idx = -1

                for j, emb in enumerate(fetched_embeddings):
                    if j in selected_indices:
                        continue

                    # Relevance score (negative distance)
                    relevance = -distances[0][indices[0][j]]

                    # Diversity penalty
                    diversity = 0
                    if selected_embeddings:
                        query_emb = np.array([embedding], dtype="float32")
                        selected_arr = np.array(
                            selected_embeddings, dtype="float32"
                        )
                        similarity = np.dot(emb, selected_arr.T).max()
                        diversity = 1 - similarity

                    # MMR score (lambda controls diversity emphasis)
                    lambda_param = 0.5
                    mmr_score = (
                        lambda_param * relevance
                        + (1 - lambda_param) * diversity
                    )

                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = j

            if best_idx >= 0:
                selected_indices.append(best_idx)
                selected_embeddings.append(fetched_embeddings[best_idx])

        # Build results
        results = []
        for idx in selected_indices:
            doc_idx = indices[0][idx]
            if doc_idx == -1:
                continue
            doc_id = str(doc_idx)
            if doc_id in self._docstore:
                doc = self._docstore[doc_id]
                doc.metadata["score"] = float(distances[0][doc_idx])
                results.append(doc)

        return results
