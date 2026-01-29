# Document Summarizer API

A production-ready Python backend for document summarization using a multi-agent architecture with FastAPI, PostgreSQL, Redis, and OpenAI GPT.

## 🏗️ Architecture Overview

### Multi-Agent Design

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  ChunkingAgent  │ -> │SummarizationAgent│ -> │ ValidatorAgent  │
│                 │    │                 │    │                 │
│ • Text splitting│    │ • LLM summaries │    │ • Quality checks│
│ • Token counting│    │ • Async parallel│    │ • Validation    │
│ • Context       │    │ • Confidence    │    │ • Suggestions   │
│   preservation  │    │   scoring       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### System Components

- **FastAPI**: Async REST API with automatic OpenAPI documentation
- **PostgreSQL**: Document storage, metadata, and summaries
- **Redis**: Summary caching and session management
- **OpenAI GPT**: LLM-powered text summarization
- **Docker**: Containerized deployment with multi-service orchestration

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd document_summarizer

# Copy environment configuration
cp .env.example .env

# Edit .env with your OpenAI API key and other settings
vim .env
```

### 2. Docker Deployment (Recommended)

```bash
# Start all services
docker-compose up -d

# Run database migrations
docker-compose run --rm migration

# Check service health
curl http://localhost:8000/health
```

### 3. Manual Setup (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL and Redis (locally or via Docker)
docker run -d --name postgres -p 5432:5432 -e POSTGRES_DB=doc_summarizer -e POSTGRES_USER=docuser -e POSTGRES_PASSWORD=docpass postgres:15
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Run migrations
export DATABASE_URL="postgresql://docuser:docpass@localhost:5432/doc_summarizer"
alembic upgrade head

# Start API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```