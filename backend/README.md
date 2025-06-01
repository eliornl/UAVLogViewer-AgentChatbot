

https://github.com/user-attachments/assets/5ff08fce-19fa-49fc-b9e4-a03a49996144

<div align="center">

# 🚁 UAV Log Viewer Agentic Chatbot

#### Advanced AI-powered telemetry analysis with natural language interface for UAV logs

#### [Watch how the Agentic AI Chatbot works](#agentic-ai-chatbot-demo)

[![UAV Log Viewer](https://img.shields.io/badge/UAV%20Log%20Viewer-Agentic%20Chatbot-blue?style=for-the-badge)](https://github.com/yourusername/UAVLogViewer-AgentChatbot)
[![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0+-blue?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0+-orange?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![DuckDB](https://img.shields.io/badge/DuckDB-0.8.1+-blue?style=for-the-badge&logo=duckdb&logoColor=white)](https://duckdb.org)
[![FAISS](https://img.shields.io/badge/FAISS-1.7.4+-blueviolet?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.0+-red?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![OpenAI API](https://img.shields.io/badge/OpenAI-API-teal?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![httpx](https://img.shields.io/badge/httpx-0.24.0+-lightgrey?style=for-the-badge)](https://www.python-httpx.org)
[![Cursor](https://img.shields.io/badge/Built%20with-Cursor-purple?style=for-the-badge)](https://cursor.sh)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

> **Harness the power of AI to decode complex UAV telemetry data through natural language**

## 📍 Table of Contents

- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [AI Agent Capabilities](#ai-agent-capabilities)
- [WebSocket Integration](#websocket-integration)
- [Telemetry Processing Pipeline](#telemetry-processing-pipeline)
- [Telemetry Schema Intelligence](#telemetry-schema-intelligence)
- [Vector Search Technology](#vector-search-technology)
- [Security](#security-considerations)
- [License](#license)
- [Contributing](#contributing)
- [AI Chatbot Demo](#agentic-ai-chatbot-demo)

---

## 🚀 System Overview

The **UAV Log Viewer** is an AI-powered platform that revolutionizes drone telemetry data analysis. Instead of manually parsing complex flight logs, users can interact with telemetry data through a powerful agentic chatbot—asking questions, uncovering insights, and diagnosing issues using natural language. This dramatically accelerates analysis workflows and enhances decision-making across development, testing, and operations.

### ✨ Key Features

- **🔍 Conversational Flight Analysis**: Transform complex telemetry data into clear, actionable insights using natural language. Ask questions about system performance, sensor readings, or operational anomalies—no technical queries required.

- **⚡ Real-Time Response Streaming**: Experience instant feedback with WebSocket-powered token streaming for a truly responsive analytical experience.

- **🧠 Intelligent Context Management**: Maintain coherent conversations with advanced memory architecture that blends conversation history and semantic recall via vector embeddings.

- **📊 Proactive Anomaly Detection**: Identify hidden issues across flight systems automatically using machine learning algorithms that detect statistical anomalies in real time.

- **💾 High-Performance Data Engine**: Process gigabytes of telemetry data in seconds with DuckDB's efficient columnar storage and optimized query execution.

- **🔄 Flexible Data Export**: Export analytical sessions in TXT, JSON, or CSV formats with consistent timestamps and formatting for seamless downstream analysis.

- **🛡️ Robust Error Handling**: Handle unexpected inputs gracefully with intelligent response routing, context-aware fallbacks, and helpful user guidance.

---

## 🏗️ Architecture

The **UAV Log Viewer** backend implements a clean, modular architecture optimized for performance, scalability, and maintainability. Built on modern microservice design principles, each component functions independently while maintaining seamless system-wide integration.

### 🧩 Core Components

- **🌐 FastAPI Server**: High-performance asynchronous API framework with automatic OpenAPI documentation, strong type validation, dependency injection, and native WebSocket support.

- **🤖 Telemetry Agent**: LangChain-powered reactive agent implementing step-by-step reasoning with specialized analytical tools and contextual memory for multi-turn conversation analysis.

- **💾 DuckDB Database**: In-memory analytical database with columnar storage optimized for complex time-series queries over large telemetry datasets.

- **📂 Telemetry Processor**: Fault-tolerant pipeline for ingesting, parsing, and normalizing UAV log data with optimized transformation capabilities and efficient data organization.

- **📃 Vector Store**: FAISS-powered semantic search engine that efficiently indexes and retrieves contextual information using dense vector embeddings.

- **🚨 Anomaly Detector**: Isolation Forest implementation with model caching and performance optimizations for real-time statistical anomaly detection across telemetry parameters.

- **💭 Memory Manager**: Multi-tier memory system that combines token-aware buffering with a three-level strategy to maintain context across conversation turns.

- **📝 AgentScratchpad**: Tracks the agent’s reasoning steps across multi-turn conversations by storing actions and observations in a structured format, helping maintain context and avoid repetition.

- **📱 WebSocket Handler**: Bidirectional communication layer for real-time token streaming with proper connection lifecycle management and HTTP fallback mechanisms.

### Data Flow

The system orchestrates data through a five-stage pipeline optimized for performance and reliability:

1. **Ingestion**: UAV log files (.tlog/.bin) → Telemetry Processor → DuckDB tables with optimized schemas
2. **Query**: User natural language input → FastAPI router → Telemetry Agent → Tool selection
3. **Analysis**: Agent tools → DuckDB queries/ML models → Structured analytical results
4. **Response**: Result synthesis → Memory updates → Token-by-token streaming via WebSockets
5. **Context Management**: Historical interactions → Vector indexing → Semantic retrieval for conversation continuity

---

## 🔧️ Technology Stack

The UAV Log Viewer leverages a carefully selected stack of technologies, each chosen to address specific technical challenges in telemetry processing, AI interaction, and real-time communication.

### Core Technologies

- **Python 3.10+**: Powers the backend with modern language features including async/await syntax, structural pattern matching, and improved type annotations for safer, more maintainable code.

- **FastAPI**: Modern, high-throughput web framework delivering performance up to 200% faster than traditional alternatives, with automatic OpenAPI generation, request validation, and native async support.

- **LangChain**: Orchestration framework implementing the ReAct agent pattern, managing tool routing, prompt engineering, and structured memory systems for LLM-powered applications.

- **DuckDB**: Analytical database engine providing vectorized query execution with columnar storage, resulting in 10-100x performance improvement over row-based alternatives for time-series telemetry queries.

- **FAISS**: High-performance similarity search library from Facebook AI Research that scales to billions of vectors with sub-millisecond query time using optimized nearest-neighbor algorithms and SIMD acceleration.

- **scikit-learn**: Statistical learning library providing the Isolation Forest implementation used for unsupervised anomaly detection across high-dimensional telemetry parameter spaces.

- **OpenAI API**: Provides access to GPT-4o and embedding models, handled with proper error retry logic, token management, and context optimization for cost-effective AI capabilities.

- **WebSockets**: Implements full-duplex communication with token streaming, heartbeat mechanisms, and automatic reconnection strategies for reliable real-time interactions.

### Performance Optimizations

- **Async I/O**: Non-blocking operations throughout the stack for high concurrency.
- **Batched Database Operations**: Grouped inserts/updates for 10-20x throughput improvement.
- **Token Management**: Precise tracking and optimization of LLM context windows.
- **Caching Strategy**: Multi-level caching for queries, embeddings, and model results.
- **Resource Pooling**: Connection and thread pools for efficient resource utilization.

---

## 🚀 Quick Start

### System Requirements

- **💻 Hardware**: 8GB+ RAM and 4+ CPU cores for optimal performance
- **🔧 Software**: Python 3.10+, Node.js 18+, and Git
- **🔑 API Keys**: OpenAI API key with GPT-4/GPT-4o access

### Installation

```bash
# Clone repository and navigate to project directory
git clone https://github.com/yourusername/UAVLogViewer-AgentChatbot.git
cd UAVLogViewer-AgentChatbot

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
cd backend
pip install -U pip setuptools wheel
pip install -r requirements.txt

### Configuration

Create a `.env` file in the backend directory with the following variables (Only OPENAI_API_KEY is required, other settings have defaults):

```ini
STORAGE_DIR=path_to_storage_directory
LOG_LEVEL=logging_verbosity_level
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=openai_model_to_use
MAX_MODEL_TOKENS=maximum_tokens_per_request
```

### Running the Application

#### Development Mode

```bash
# Start backend with auto-reload for development
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Production Mode

```bash
# Option 1: Using Gunicorn with multiple workers (recommended for production)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app --bind 0.0.0.0:8000

# Option 2: Using Docker
docker build -t uav-log-viewer .
docker run -d --name uav-log-viewer -p 8000:8000 \
  -v $(pwd)/storage:/app/storage --env-file .env uav-log-viewer
```

### Verification

Once running, verify your installation by:

1. Opening the API documentation at `http://localhost:8000/docs`
2. Accessing the frontend at `http://localhost:8080`
3. Uploading a sample UAV log file or open sample to test the telemetry processing

---

## 📞 API Reference

The UAV Log Viewer exposes a comprehensive API for integrating telemetry analysis capabilities into your applications.

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | `POST` | Upload UAV log files (.tlog, .bin) |
| `/chat` | `POST` | Send message to AI agent (HTTP) |
| `/ws/{session_id}` | `WebSocket` | Real-time chat with token streaming |
| `/session/{session_id}/delete` | `POST` | Delete session and associated data |
| `/export_chat/{session_id}` | `GET` | Export chat history (TXT/JSON/CSV) |
| `/get_messages/{session_id}` | `GET` | Retrieve all messages for a session |

\* *Authentication should be implemented for production deployments*

For complete API documentation, access the interactive Swagger UI at `/docs` when the server is running.

---

## 🤖 AI Agent Capabilities

The UAV Log Viewer implements a sophisticated AI agent based on the ReAct (Reasoning + Acting) paradigm, combining LLM reasoning with specialized analytical tools.

### 💬 Enhanced User Interaction

The agent features an advanced interaction system optimized for telemetry analysis workflows:

- **Intent Classification**: Accurately categorizes user queries into analytical types (anomaly detection, parameter analysis, flight phase investigation) to select appropriate processing paths.

- **Greeting & Off-Topic Management**: Efficiently handles non-analytical queries without triggering expensive data operations, providing contextual guidance while preserving system resources.

- **Adaptive Assistance**: Dynamically adjusts response detail and suggestion specificity based on query clarity and user expertise level.

- **Resource Lifecycle Management**: Automatically cleans up sessions and releases database connections when users navigate away or close their browser.

### 💭 Dynamic Memory Architecture

The agent implements an adaptive memory system that automatically selects the appropriate strategy based on conversation length and complexity:

**Memory Strategy Selection**

The Conversation Memory Manager dynamically transitions between memory strategies as the conversation evolves:

- **Short-Term Strategy** (< 1,000 tokens): Uses a token-aware buffer window that preserves recent messages while respecting token limits.

- **Medium-Term Strategy** (1,000-3,000 tokens): Combines buffer memory with a summarization mechanism that condenses older conversation turns into compact summaries.

- **Advanced Strategy** (> 3,000 tokens): Combines buffer memory, summarization, and vector-based retrieval using FAISS with time-weighted decay to maintain relevance.

- **Fallback Strategy**: Implements aggressive summarization when token limits are exceeded, ensuring conversations can continue indefinitely.

This progressive approach ensures optimal context preservation while managing token usage efficiently, allowing the system to maintain coherent conversations over extended sessions without performance degradation.

### 🔍 Specialized Analytical Tools

The agent leverages purpose-built tools for telemetry analysis:

- **SQL Query Generation**: Converts natural language to optimized DuckDB queries with schema awareness.

- **Anomaly Detection**: Implements Isolation Forest with configurable sensitivity parameters for identifying statistical outliers across telemetry parameters.

- **Flight Phase Analysis**: Segments flight data into takeoff, cruise, and landing phases for targeted investigation.

- **Parameter Correlation**: Identifies relationships between telemetry parameters for root cause analysis.

### 🔧 System Integration

The agent is fully integrated with the core system architecture:

- **Strongly Typed Interfaces**: All agent inputs and outputs use Pydantic models for validation.

- **Concurrent Processing**: Leverages asyncio for non-blocking operation during complex analyses.

- **Error Handling**: Implements graceful degradation with informative user feedback.

- **Session Isolation**: Maintains independent agent state per user session with proper resource management.

---

## 🔄 WebSocket Integration

The UAV Log Viewer implements a bidirectional WebSocket communication system for real-time token streaming with automatic HTTP fallback for maximum reliability.

### Implementation Details

- **Connection Pool Management**: Session-isolated WebSocket connections with proper lifecycle tracking.
- **Binary Frame Optimization**: Uses binary WebSocket frames for 15-20% lower latency compared to text frames.
- **Heartbeat Mechanism**: 30-second ping/pong cycle to detect and manage stale connections.
- **Backpressure Handling**: Token buffer with configurable high-water mark to prevent client overflow.
- **Error Recovery**: Implements exponential backoff for reconnection attempts with jitter.

---

## 📊 Telemetry Processing Pipeline

The UAV Log Viewer features a high-throughput processing pipeline that transforms raw MAVLink binary logs into optimized analytical data structures through a multi-stage ETL process.

---

## 📝 Telemetry Schema Intelligence

At the heart of the UAV Log Viewer's natural language understanding lies the Telemetry Schema Intelligence - a powerful semantic layer connecting raw telemetry data to human-friendly insights.

### ✨ Schema Intelligence Features

- **🗺️ Knowledge Graph**: Comprehensive mapping of telemetry parameters including relationships and rich metadata.

- **🔍 Semantic Search**: Vector-based similarity search powered by OpenAI embeddings and FAISS for precise context retrieval.

- **📝 Parameter Context**: Detailed descriptions, units, expected ranges, and anomaly indicators for each telemetry parameter.

- **⚙️ SQL Translation**: Converts natural language queries into optimized, schema-aware SQL statements.

- **📊 Result Interpretation**: Applies domain expertise to provide meaningful, contextual analysis of query results.

### 💡 System Integration

The Telemetry Schema Intelligence system is deeply integrated throughout the UAV Log Viewer application to enhance various components:

- **🧠 AI Reasoning**: Supplies domain-specific knowledge that guides the agent's analysis strategies and response generation.

- **💾 Database Layer**: Informs query optimization by aiding table selection and join strategies for efficient data retrieval.

- **📈 Visualization**: Recommends suitable visualization types based on parameter characteristics and their relationships.

- **💎 Anomaly Detection**: Establishes baseline expectations to detect unusual patterns and potential issues in flight telemetry data.

---

## 🔎 Vector Search Technology

The UAV Log Viewer employs advanced neural embedding and approximate nearest neighbor search technologies to enable semantic retrieval of conversation context and telemetry knowledge.

### ✨ Vector Store Implementation

- **📚 FAISS Integration**: Uses Facebook AI Similarity Search for efficient vector operations with optimized nearest-neighbor algorithms.

- **🧐 OpenAI Embeddings**: Provides high-quality semantic representations (text-embedding-ada-002) with 1,536 dimensions for precise similarity matching.

- **📦 Batched Processing**: Implements cost-efficient embedding generation with proper rate limiting and error handling.

- **⏱️ Time-Weighted Retrieval**: Enhances search relevance by applying decay functions to boost recent information in conversation context.

### 🔥 Key Innovations

**Time-weighted Retrieval** enhances search relevance by boosting similarity scores for recent conversation turns. An exponential decay function maintains contextual freshness, controlled via a configurable half-life parameter. This approach balances semantic meaning with recency to ensure results remain timely and pertinent.

**Multi-tier Memory Architecture** structures memory into three layers: a sliding window buffer for immediate context (~300 tokens), a medium-term memory summarized by the language model (~1,500 tokens), and an unlimited, vector-indexed long-term memory. This design dynamically selects relevant context based on query demands and token limits.

**Hybrid Search** combines vector similarity with exact keyword matching, using a blended ranking algorithm that balances both methods. Metadata-aware filtering refines results further, optimizing retrieval for specialized telemetry domain terminology.

---

## 🔒 Security Considerations

UAV Log Viewer is built with strong security principles to ensure the integrity of telemetry data and the safety of user interactions.

Session isolation ensures that each user operates in a sandboxed environment with dedicated locking mechanisms. All incoming inputs are rigorously validated through Pydantic models, enforcing strict data integrity. Request rate limiting protects the API from abuse, while structured error handling prevents accidental information disclosure. Sensitive configurations, such as API keys and credentials, are securely managed through environment variables.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

Copyright (c) 2025 Elior Nataf Lackritz

## 🤝 Contributing

Contributions are welcome! Here's how you can contribute:

1. **Fork** the repository on GitHub
2. **Clone** your fork and create a new feature branch
3. **Implement** your changes with appropriate tests
4. **Submit** a pull request with a clear description

Please follow the project's coding style and include appropriate tests for new functionality.

---

## 🎥 Agentic AI Chatbot Demo

Watch how the Agentic AI Chatbot works, seamlessly integrated with the UAV Log Viewer Project.

https://github.com/eliornl/UAVLogViewer-AgentChatbot/blob/1d42516699ac8a881cfde98a3e2dffbf75f151de/agentic_ai_chatbot_demo.mp4
