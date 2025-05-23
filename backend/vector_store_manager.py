import asyncio
import structlog
from typing import List, Optional, TypeAlias
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import openai
import requests
from backend.telemetry_schema import TELEMETRY_SCHEMA

logger = structlog.get_logger(__name__)

# Type alias for clarity
DocumentList: TypeAlias = List[Document]

class VectorStoreManager:
    """Manages the FAISS vector store for telemetry schema embeddings.

    Handles asynchronous initialization, caching, and similarity search for the FAISS vector store,
    which stores embeddings of telemetry schema tables for relevant table identification.

    Attributes:
        openai_api_key: OpenAI API key for embeddings.
        vector_store: Cached FAISS vector store instance (initialized lazily).
        _lock: Async lock for thread-safe initialization.
    """
    _vector_store_cache: Optional[FAISS] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self, openai_api_key: str) -> None:
        """Initialize VectorStoreManager with OpenAI API key.

        Args:
            openai_api_key: OpenAI API key for embeddings.

        Raises:
            ValueError: If openai_api_key is empty or invalid.
        """
        if not openai_api_key:
            logger.error("OpenAI API key is empty")
            raise ValueError("OpenAI API key must be provided")
        self.openai_api_key: str = openai_api_key
        self.vector_store: Optional[FAISS] = None
        self.logger = logger.bind(class_name=self.__class__.__name__)

    async def initialize(self, timeout_seconds: float = 30.0) -> None:
        """Initialize the FAISS vector store asynchronously.

        Creates a FAISS vector store from TELEMETRY_SCHEMA embeddings if not already cached.
        Thread-safe using an async lock.

        Args:
            timeout_seconds: Timeout for vector store initialization in seconds.

        Raises:
            ValueError: If initialization fails due to authentication, network, or other errors.
            asyncio.TimeoutError: If initialization exceeds timeout_seconds.
        """
        async with self._lock:
            if VectorStoreManager._vector_store_cache is not None:
                self.vector_store = VectorStoreManager._vector_store_cache
                self.logger.debug("Using cached FAISS vector store")
                return

            documents: List[str] = [
                f"{meta['table']}: {meta['description']} Columns: {', '.join(col['name'] for col in meta['columns'])}"
                for meta in TELEMETRY_SCHEMA
            ]
            if not documents:
                self.logger.error("TELEMETRY_SCHEMA is empty")
                raise ValueError("TELEMETRY_SCHEMA is empty; cannot initialize vector store")

            try:
                async def init_vector_store() -> FAISS:
                    embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
                    return FAISS.from_texts(documents, embeddings)

                self.vector_store = await asyncio.wait_for(
                    asyncio.to_thread(init_vector_store),
                    timeout=timeout_seconds
                )
                VectorStoreManager._vector_store_cache = self.vector_store
                self.logger.info("Initialized FAISS vector store", document_count=len(documents))
            except openai.AuthenticationError as e:
                self.logger.error("Invalid OpenAI API key for embeddings", error=str(e))
                raise ValueError("Failed to initialize vector store: Invalid OpenAI API key") from e
            except requests.RequestException as e:
                self.logger.error("Network error during embeddings API call", error=str(e))
                raise ValueError("Failed to initialize vector store: Network error during embeddings API call") from e
            except asyncio.TimeoutError:
                self.logger.error("Vector store initialization timed out", timeout=timeout_seconds)
                raise ValueError(f"Vector store initialization timed out after {timeout_seconds} seconds")
            except Exception as e:
                self.logger.error("Unexpected error during vector store initialization", error=str(e), exc_info=True)
                raise ValueError(f"Failed to initialize vector store: {str(e)}") from e

    async def asimilarity_search(self, query: str, k: int = 4) -> DocumentList:
        """Perform an asynchronous similarity search on the vector store.

        Args:
            query: Query string to search for.
            k: Number of results to return (default: 4).

        Returns:
            DocumentList: List of FAISS Document objects matching the query.

        Raises:
            ValueError: If vector store is not initialized or search fails.
            asyncio.TimeoutError: If search exceeds timeout (30 seconds).
        """
        if not self.vector_store:
            self.logger.error("Vector store not initialized")
            raise ValueError("Vector store not initialized; call initialize() first")

        try:
            async def run_search() -> DocumentList:
                return self.vector_store.similarity_search(query, k=k)

            results = await asyncio.wait_for(
                asyncio.to_thread(run_search),
                timeout=30.0
            )
            self.logger.debug("Performed vector store similarity search", query=query, result_count=len(results))
            return results
        except asyncio.TimeoutError:
            self.logger.error("Vector store search timed out", query=query, timeout=30.0)
            raise ValueError("Vector store search timed out after 30 seconds")
        except Exception as e:
            self.logger.error("Vector store search failed", query=query, error=str(e), exc_info=True)
            raise ValueError(f"Vector store search failed: {str(e)}") from e