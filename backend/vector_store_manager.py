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

# Constants
VECTOR_INIT_TIMEOUT_SECONDS: float = 30.0  # Timeout for vector store initialization
VECTOR_SEARCH_TIMEOUT_SECONDS: float = 30.0  # Timeout for vector store similarity search

class VectorStoreManager:
    """Manages the FAISS vector store for telemetry schema embeddings.

    Handles asynchronous initialization and similarity search for the FAISS vector store,
    which stores embeddings of telemetry schema tables for relevant table identification.

    Attributes:
        embeddings: Pre-initialized OpenAIEmbeddings instance for vector store.
        vector_store: FAISS vector store instance (initialized lazily).
    """

    def __init__(self, embeddings: OpenAIEmbeddings) -> None:
        """Initialize VectorStoreManager with embeddings.

        Args:
            embeddings: Pre-initialized OpenAIEmbeddings instance.

        Note:
            Call initialize() to set up the FAISS vector store before performing searches.
        """
        self.embeddings: OpenAIEmbeddings = embeddings
        self.vector_store: Optional[FAISS] = None
        self.logger = logger.bind(class_name=self.__class__.__name__)

    async def initialize(self, timeout_seconds: float = VECTOR_INIT_TIMEOUT_SECONDS) -> None:
        """Initialize the FAISS vector store asynchronously.

        Creates a FAISS vector store from TELEMETRY_SCHEMA embeddings.

        Args:
            timeout_seconds: Timeout for vector store initialization in seconds (default: VECTOR_INIT_TIMEOUT_SECONDS).

        Raises:
            ValueError: If initialization fails due to authentication, network, or other errors.
            asyncio.TimeoutError: If initialization exceeds timeout_seconds.
        """
        documents: List[str] = [
            f"{meta['table']}: {meta['description']} Columns: {', '.join(col['name'] for col in meta['columns'])}"
            for meta in TELEMETRY_SCHEMA
        ]
        if not documents:
            self.logger.error("TELEMETRY_SCHEMA is empty")
            raise ValueError("TELEMETRY_SCHEMA is empty; cannot initialize vector store")

        try:
            async def init_vector_store() -> FAISS:
                return FAISS.from_texts(documents, self.embeddings)

            self.vector_store = await asyncio.wait_for(
                asyncio.to_thread(init_vector_store),
                timeout=timeout_seconds
            )
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
            ValueError: If search fails due to network or other errors.
            asyncio.TimeoutError: If search exceeds VECTOR_SEARCH_TIMEOUT_SECONDS.
        """
        try:
            async def run_search() -> DocumentList:
                return self.vector_store.similarity_search(query, k=k)

            results = await asyncio.wait_for(
                asyncio.to_thread(run_search),
                timeout=VECTOR_SEARCH_TIMEOUT_SECONDS
            )
            self.logger.debug("Performed vector store similarity search", query=query, result_count=len(results))
            return results
        except asyncio.TimeoutError:
            self.logger.error("Vector store search timed out", query=query, timeout=VECTOR_SEARCH_TIMEOUT_SECONDS)
            raise ValueError(f"Vector store search timed out after {VECTOR_SEARCH_TIMEOUT_SECONDS} seconds")
        except Exception as e:
            self.logger.error("Vector store search failed", query=query, error=str(e), exc_info=True)
            raise ValueError(f"Vector store search failed: {str(e)}") from e