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

# Constants for vector store operations
VECTOR_INIT_TIMEOUT_SECONDS: float = 30.0  # Timeout for vector store initialization
VECTOR_SEARCH_TIMEOUT_SECONDS: float = (
    30.0  # Timeout for vector store similarity search
)
DEFAULT_SEARCH_RESULTS: int = (
    4  # Default number of results to return from similarity search
)

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

    async def initialize(
        self, timeout_seconds: float = VECTOR_INIT_TIMEOUT_SECONDS
    ) -> None:
        """Initialize the FAISS vector store asynchronously.

        Creates a FAISS vector store from TELEMETRY_SCHEMA embeddings, including table descriptions,
        column names, and anomaly hints.

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
        """Perform an asynchronous similarity search on the vector store.

        Searches the vector store for documents similar to the query and returns
        the top k results with their metadata. This is useful for finding relevant
        telemetry tables based on natural language descriptions.

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
            self.logger.debug(
                "Performed vector store similarity search",
                query=query,
                result_count=len(results),
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
