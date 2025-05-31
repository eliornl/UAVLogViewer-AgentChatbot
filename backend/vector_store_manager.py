import asyncio
import structlog
import time
import hashlib
from typing import List, Optional, TypeAlias, Dict, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import openai
import requests
from backend.telemetry_schema import TELEMETRY_SCHEMA

logger = structlog.get_logger(__name__)

# Type alias for clarity
DocumentList: TypeAlias = List[Document]
CacheKey: TypeAlias = str
CacheEntry: TypeAlias = Tuple[DocumentList, float]  # (results, timestamp)

# Constants for vector store operations
VECTOR_INIT_TIMEOUT_SECONDS: float = 30.0  # Timeout for vector store initialization
VECTOR_SEARCH_TIMEOUT_SECONDS: float = (
    30.0  # Timeout for vector store similarity search
)
DEFAULT_SEARCH_RESULTS: int = (
    3  # Default number of results to return from similarity search
)
CACHE_TTL_SECONDS: float = 600.0  # Cache TTL: 10 minutes
MAX_CACHE_SIZE: int = 50  # Maximum number of cache entries (LRU eviction)

# Error messages for better consistency
ERROR_NOT_INITIALIZED: str = "Vector store not initialized"
ERROR_EMPTY_SCHEMA: str = "TELEMETRY_SCHEMA is empty; cannot initialize vector store"
ERROR_INVALID_API_KEY: str = "Failed to initialize vector store: Invalid OpenAI API key"
ERROR_NETWORK: str = (
    "Failed to initialize vector store: Network error during embeddings API call"
)


class VectorStoreManager:
    """Manages the FAISS vector store for telemetry schema embeddings.

    Handles asynchronous initialization and similarity search for the FAISS vector store,
    which stores embeddings of telemetry schema tables for relevant table identification.
    Includes time-based caching to avoid redundant embedding computations.

    Attributes:
        embeddings: Pre-initialized OpenAIEmbeddings instance for vector store.
        vector_store: FAISS vector store instance (initialized lazily).
        _search_cache: Cache for similarity search results with timestamps.
        _cache_ttl: Time-to-live for cache entries in seconds.
    """

    def __init__(self, embeddings: OpenAIEmbeddings, cache_ttl_seconds: float = CACHE_TTL_SECONDS) -> None:
        """Initialize VectorStoreManager with embeddings and caching.

        Args:
            embeddings: Pre-initialized OpenAIEmbeddings instance.
            cache_ttl_seconds: Time-to-live for cache entries in seconds (default: 10 minutes).

        Note:
            Call initialize() to set up the FAISS vector store before performing searches.
        """
        self.embeddings: OpenAIEmbeddings = embeddings
        self.vector_store: Optional[FAISS] = None
        self._search_cache: Dict[CacheKey, CacheEntry] = {}
        self._cache_access_order: List[CacheKey] = []  # Track access order for LRU
        self._cache_ttl: float = cache_ttl_seconds
        self.logger = logger.bind(class_name=self.__class__.__name__)

    def _generate_cache_key(self, query: str, k: int) -> CacheKey:
        """Generate a cache key for the given query and k value.

        Args:
            query: Search query string.
            k: Number of results requested.

        Returns:
            CacheKey: Hash-based cache key for consistent lookup.
        """
        # Create a consistent cache key by hashing query and k
        cache_input = f"{query.strip().lower()}:{k}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if a cache entry is still valid based on TTL.

        Args:
            timestamp: Timestamp when the cache entry was created.

        Returns:
            bool: True if cache entry is still valid, False otherwise.
        """
        return (time.time() - timestamp) < self._cache_ttl

    def _cleanup_expired_cache_entries(self) -> None:
        """Remove expired cache entries to prevent memory bloat."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._search_cache.items()
            if (current_time - timestamp) >= self._cache_ttl
        ]

        for key in expired_keys:
            del self._search_cache[key]
            if key in self._cache_access_order:
                self._cache_access_order.remove(key)

        if expired_keys:
            self.logger.debug(
                "Cleaned up expired cache entries",
                expired_count=len(expired_keys),
                remaining_count=len(self._search_cache)
            )

    def clear_cache(self) -> None:
        """Clear all cached search results."""
        cache_size = len(self._search_cache)
        self._search_cache.clear()
        self._cache_access_order.clear()
        self.logger.info("Cleared similarity search cache", cleared_entries=cache_size)

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for monitoring purposes.

        Returns:
            Dict with cache size and valid entry count.
        """
        current_time = time.time()
        valid_entries = sum(
            1 for _, timestamp in self._search_cache.values()
            if (current_time - timestamp) < self._cache_ttl
        )

        return {
            "total_entries": len(self._search_cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self._search_cache) - valid_entries
        }

    async def initialize(
        self, timeout_seconds: float = VECTOR_INIT_TIMEOUT_SECONDS
    ) -> None:
        """Initialize the FAISS vector store asynchronously.

        Creates a FAISS vector store from TELEMETRY_SCHEMA embeddings, including table descriptions,
        column names, and anomaly hints. Clears any existing cache upon reinitialization.

        Args:
            timeout_seconds: Timeout for vector store initialization in seconds (default: VECTOR_INIT_TIMEOUT_SECONDS).

        Raises:
            ValueError: If initialization fails due to authentication, network, or other errors.
            asyncio.TimeoutError: If initialization exceeds timeout_seconds.
        """
        documents: List[Document] = [
            Document(
                page_content=f"{meta['table']}: {meta['description']} Columns: {', '.join(col['name'] for col in meta['columns'])} Anomaly hint: {meta['anomaly_hint']}",
                metadata={"table": meta["table"], "anomaly_hint": meta["anomaly_hint"]},
            )
            for meta in TELEMETRY_SCHEMA
        ]
        if not documents:
            self.logger.error("TELEMETRY_SCHEMA is empty")
            raise ValueError(ERROR_EMPTY_SCHEMA)

        try:

            def init_vector_store() -> FAISS:
                return FAISS.from_documents(documents, self.embeddings)

            self.vector_store = await asyncio.wait_for(
                asyncio.to_thread(init_vector_store), timeout=timeout_seconds
            )

            # Clear cache when reinitializing vector store
            self.clear_cache()

            self.logger.info(
                "Initialized FAISS vector store", document_count=len(documents)
            )
        except openai.AuthenticationError as e:
            self.logger.error("Invalid OpenAI API key for embeddings", error=str(e))
            raise ValueError(ERROR_INVALID_API_KEY) from e
        except requests.RequestException as e:
            self.logger.error("Network error during embeddings API call", error=str(e))
            raise ValueError(ERROR_NETWORK) from e
        except asyncio.TimeoutError:
            self.logger.error(
                "Vector store initialization timed out", timeout=timeout_seconds
            )
            raise ValueError(
                f"Vector store initialization timed out after {timeout_seconds} seconds"
            )
        except Exception as e:
            self.logger.error(
                "Unexpected error during vector store initialization",
                error=str(e),
                exc_info=True,
            )
            raise ValueError(f"Failed to initialize vector store: {str(e)}") from e

    async def async_similarity_search(
        self, query: str, k: int = DEFAULT_SEARCH_RESULTS
    ) -> DocumentList:
        """Perform an asynchronous similarity search on the vector store with caching.

        Searches the vector store for documents similar to the query and returns
        the top k results with their metadata. Results are cached for the configured
        TTL to avoid redundant embedding computations for similar queries.

        Args:
            query: Query string to search for (natural language description)
            k: Number of results to return (default: DEFAULT_SEARCH_RESULTS)

        Returns:
            DocumentList: List of Document objects with table and anomaly_hint metadata

        Raises:
            ValueError: If search fails due to vector store not being initialized,
                       network errors, or other issues
            asyncio.TimeoutError: If search exceeds VECTOR_SEARCH_TIMEOUT_SECONDS
        """
        if self.vector_store is None:
            self.logger.error(ERROR_NOT_INITIALIZED)
            raise ValueError(ERROR_NOT_INITIALIZED)

        # Generate cache key and check for cached results
        cache_key = self._generate_cache_key(query, k)

        # Check cache first
        if cache_key in self._search_cache:
            cached_results, timestamp = self._search_cache[cache_key]
            if self._is_cache_valid(timestamp):
                # Update LRU order
                self._cache_access_order.remove(cache_key)
                self._cache_access_order.append(cache_key)
                self.logger.debug(
                    "Returning cached similarity search results",
                    query=query,
                    result_count=len(cached_results),
                    cache_age_seconds=time.time() - timestamp
                )
                return cached_results
            else:
                # Remove expired entry
                del self._search_cache[cache_key]
                if cache_key in self._cache_access_order:
                    self._cache_access_order.remove(cache_key)

        # Periodically clean up expired entries (every 10th search to avoid overhead)
        if len(self._search_cache) > 0 and len(self._search_cache) % 10 == 0:
            self._cleanup_expired_cache_entries()

        try:

            def run_search() -> DocumentList:
                results = self.vector_store.similarity_search(query, k=k)
                return [
                    Document(
                        page_content=result.page_content,
                        metadata={
                            "table": result.metadata.get("table", ""),
                            "anomaly_hint": result.metadata.get("anomaly_hint", ""),
                        },
                    )
                    for result in results
                ]

            results = await asyncio.wait_for(
                asyncio.to_thread(run_search), timeout=VECTOR_SEARCH_TIMEOUT_SECONDS
            )

            # Cache the results with LRU eviction
            self._search_cache[cache_key] = (results, time.time())
            self._cache_access_order.append(cache_key)
            
            # Enforce cache size limit with LRU eviction
            while len(self._search_cache) > MAX_CACHE_SIZE:
                oldest_key = self._cache_access_order.pop(0)
                if oldest_key in self._search_cache:
                    del self._search_cache[oldest_key]
                    self.logger.debug("Evicted cache entry due to size limit", evicted_key=oldest_key)

            self.logger.debug(
                "Performed vector store similarity search and cached results",
                query=query,
                result_count=len(results),
                cache_size=len(self._search_cache),
                max_cache_size=MAX_CACHE_SIZE
            )
            return results
        except asyncio.TimeoutError:
            self.logger.error(
                "Vector store search timed out",
                query=query,
                timeout=VECTOR_SEARCH_TIMEOUT_SECONDS,
            )
            raise ValueError(
                f"Vector store search timed out after {VECTOR_SEARCH_TIMEOUT_SECONDS} seconds"
            )
        except Exception as e:
            self.logger.error(
                "Vector store search failed", query=query, error=str(e), exc_info=True
            )
            raise ValueError(f"Vector store search failed: {str(e)}") from e
