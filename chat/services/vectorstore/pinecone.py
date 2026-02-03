"""
Pinecone vector store implementation.

This module provides a Pinecone-based vector store implementation
that extends the BaseVectorStore interface for cloud-based vector storage.
"""

import logging
from typing import Any, Dict, List, Optional

from django.conf import settings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone-based vector store implementation.

    This class provides methods to create, load, save, and query
    Pinecone indexes with associated metadata while implementing
    the BaseVectorStore interface.

    Features:
    - Cloud-based scalable vector storage
    - Efficient similarity search
    - Metadata filtering support
    - Automatic index management
    """

    def __init__(
        self,
        embedding: Embeddings,
        folder_name: str,
        index_base_path: Optional[str] = None,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Pinecone vector store.

        Args:
            embedding: LangChain embeddings instance.
            folder_name: Name of the Pinecone index.
            index_base_path: Ignored for Pinecone (cloud-based).
            api_key: Pinecone API key. Uses PINECONE_API_KEY from settings if not provided.
            environment: Pinecone environment. Uses PINECONE_ENVIRONMENT from settings if not provided.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            embedding=embedding,
            folder_name=folder_name,
            index_base_path=index_base_path,
            **kwargs,
        )

        # Get API credentials from settings if not provided
        self.api_key = api_key or getattr(settings, "PINECONE_API_KEY", None)
        self.environment = environment or getattr(
            settings, "PINECONE_ENVIRONMENT", "us-east-1"
        )

        # Initialize Pinecone client and index
        self._index = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Pinecone client and index."""
        try:
            import pinecone

            # Initialize Pinecone
            if self.api_key:
                pinecone.init(
                    api_key=self.api_key, environment=self.environment
                )
            else:
                raise ValueError(
                    "Pinecone API key is required. Set PINECONE_API_KEY in settings or pass api_key parameter."
                )

            # Get or create index
            try:
                self._index = pinecone.Index(self.folder_name)
                logger.info(f"Connected to Pinecone index: {self.folder_name}")
            except Exception as e:
                logger.warning(f"Index not found, attempting to create: {e}")
                # Note: Index creation typically requires Pinecone console access
                # This is a simplified approach
                raise ValueError(
                    f"Pinecone index '{self.folder_name}' not found. "
                    "Please create it in the Pinecone console first."
                )

        except ImportError:
            logger.error(
                "Pinecone client not installed. Install with: pip install pinecone-client"
            )
            raise ImportError(
                "Pinecone client is required for PineconeVectorStore. "
                "Install it with: pip install pinecone-client"
            )
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise

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
        if self._index is None:
            raise RuntimeError("Pinecone index not initialized")

        # Generate embeddings
        embeddings = self.embedding.embed_documents(texts)

        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]

        # Use provided metadatas or empty dicts
        if metadatas is None:
            metadatas = [{}] * len(texts)

        # Prepare vectors for upsert
        vectors = []
        for id_, embedding, text, metadata in zip(
            ids, embeddings, texts, metadatas
        ):
            # Add text content to metadata for retrieval
            metadata_with_text = dict(metadata)
            metadata_with_text["text"] = text
            vectors.append((id_, embedding, metadata_with_text))

        # Upsert to Pinecone
        self._index.upsert(vectors=vectors)

        logger.info(f"Added {len(texts)} texts to Pinecone index")
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
        if self._index is None:
            raise RuntimeError("Pinecone index not initialized")

        # Search in Pinecone
        results = self._index.query(
            vector=embedding,
            top_k=k,
            include_metadata=True,
            include_values=False,
        )

        # Build Document objects
        docs = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            # Extract text from metadata
            text = metadata.pop("text", "")
            score = match.get("score", 0.0)

            doc = Document(page_content=text, metadata=metadata)
            doc.metadata["score"] = float(score)
            docs.append(doc)

        return docs

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
        """Delete all vectors from the index."""
        if self._index is None:
            logger.warning("Index already deleted or not initialized")
            return

        try:
            # Delete all vectors in the namespace
            self._index.delete(delete_all=True)
            logger.info(
                f"Deleted all vectors from Pinecone index: {self.folder_name}"
            )
        except Exception as e:
            logger.error(f"Error deleting Pinecone vectors: {e}")

    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> None:
        """
        Delete documents by IDs.

        Args:
            ids: List of document IDs to delete.
            **kwargs: Additional keyword arguments.
        """
        if ids is None:
            return

        if self._index is None:
            raise RuntimeError("Pinecone index not initialized")

        self._index.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents from Pinecone index")

    def get_vector_count(self) -> int:
        """
        Get the number of vectors in the store.

        Returns:
            Number of vectors as an integer.
        """
        if self._index is None:
            return 0

        try:
            stats = self._index.describe_index_stats()
            return stats.get("total_vector_count", 0)
        except Exception:
            return 0

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
        if self._index is None:
            return None

        try:
            stats = self._index.describe_index_stats()
            return stats.get("dimension")
        except Exception:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary containing store statistics.
        """
        return {
            "vector_count": self.get_vector_count(),
            "dimension": self.get_dimension(),
            "is_loaded": self.is_loaded(),
            "index_name": self.folder_name,
            "environment": self.environment,
            "backend": "pinecone",
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

        Note: Pinecone doesn't have built-in MMR. We use a simplified
        implementation that fetches more results and returns top k.

        Args:
            query: Query text to search for.
            k: Number of results to return.
            fetch_k: Number of documents to fetch before filtering.
            lambda_mult: Controls diversity vs relevance tradeoff (not used).
            **kwargs: Additional keyword arguments.

        Returns:
            List of Document objects.
        """
        # Fetch more results than needed
        results = self.similarity_search(query, k=fetch_k, **kwargs)

        # For now, just return top k results
        return results[:k]

    def query_with_metadata_filter(
        self,
        embedding: List[float],
        k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Search with metadata filtering.

        Args:
            embedding: Query embedding vector.
            k: Number of results to return.
            filter_dict: Metadata filter dictionary.

        Returns:
            List of Document objects.
        """
        if self._index is None:
            raise RuntimeError("Pinecone index not initialized")

        # Search in Pinecone with filter
        results = self._index.query(
            vector=embedding,
            top_k=k,
            filter=filter_dict,
            include_metadata=True,
            include_values=False,
        )

        # Build Document objects
        docs = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            text = metadata.pop("text", "")
            score = match.get("score", 0.0)

            doc = Document(page_content=text, metadata=metadata)
            doc.metadata["score"] = float(score)
            docs.append(doc)

        return docs

    def get_index(self):
        """
        Get the underlying Pinecone index.

        Returns:
            The Pinecone index object.
        """
        return self._index
