# RAG Agent Testing Framework

This repository contains a comprehensive testing framework for the RAG (Retrieval-Augmented Generation) agent system, designed for the Physical AI & Humanoid Robotics textbook project.

## Architecture Overview

The system implements a RAG pipeline with the following components:

1. **Query Processing**: FastAPI endpoints for accepting user queries
2. **Embedding Generation**: Cohere API for creating vector embeddings
3. **Vector Storage**: Qdrant for similarity search and content retrieval
4. **Response Generation**: Google Gemini for generating contextual answers
5. **Validation Layer**: Comprehensive testing for retrieval accuracy, content matching, and metadata integrity

## Components

### Core Modules

- `main.py`: FastAPI application with query endpoints
- `agent.py`: Google Gemini integration and response generation
- `retrieval_tool.py`: Qdrant similarity search and validation functions
- `config.py`: Configuration management and environment loading
- `models.py`: Pydantic models for request/response schemas

### Testing Modules

- `test_retrieval_accuracy.py`: Top-K retrieval accuracy testing
- `test_content_matching.py`: Content matching verification
- `test_metadata_integrity.py`: Metadata integrity validation
- `pipeline_test.py`: End-to-end pipeline testing
- `integration_test.py`: Integrated user story testing
- `test_pytest_cases.py`: Pytest test cases for all user stories

## User Stories Implemented

### User Story 1: Top-K Retrieval Accuracy
- Query embedding generation using Cohere (matching ingestion pipeline)
- Qdrant similarity search with configurable top-k results
- Results ordering by descending similarity score
- Configurable top-k parameter handling
- Retrieval accuracy testing with sample queries

### User Story 2: Content Matching Verification
- Content matching validation function comparing retrieved vs original content
- Text similarity calculation using Levenshtein distance/ratio
- 95%+ similarity threshold validation
- Handling for exact matches and near matches
- Content matching tests with original embedded text samples

### User Story 3: Metadata Integrity Testing
- Metadata validation function for all fields
- URL field validation with proper format checking
- Chunk ID field validation with format requirements
- Comprehensive metadata validation with detailed error reporting
- Metadata retrieval tests with original samples

### User Story 4: End-to-End Pipeline Testing
- Clean JSON output formatting with consistent schema
- Complete flow execution testing
- JSON schema validation for consistent output format
- Logging for retrieval correctness and errors
- Multiple query type testing with accuracy metrics

## Setup and Installation

### Prerequisites

- Python 3.8+
- Google Gemini API key
- Cohere API key
- Qdrant instance (cloud or local)

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements-agent.txt
   ```
3. Set up environment variables in `.env`:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   COHERE_API_KEY=your_cohere_api_key
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   QDRANT_COLLECTION_NAME=rag_embedding
   ```

### Running the Application

Start the FastAPI server:
```bash
cd backend
uvicorn rag_agent.main:app --host 0.0.0.0 --port 8000
```

## Testing

### Running Tests

Execute all test suites:
```bash
# Unit tests
python -m pytest backend/rag_agent/test_pytest_cases.py

# Integration tests
python -m backend.rag_agent.integration_test

# Individual user story tests
python -m backend.rag_agent.test_retrieval_accuracy
python -m backend.rag_agent.test_content_matching
python -m backend.rag_agent.test_metadata_integrity
python -m backend.rag_agent.pipeline_test
```

### API Endpoints

- `POST /query`: Submit queries to the RAG agent
- `GET /health`: Health check endpoint

### Query Request Format

```json
{
  "query": "Your question here",
  "top_k": 5,
  "temperature": 0.7
}
```

### Response Format

```json
{
  "success": true,
  "data": {
    "answer": "Generated response",
    "sources": [...],
    "confidence": 0.9,
    "processing_time": 1.23,
    "tokens_used": {...}
  },
  "error": null,
  "request_id": "unique-id"
}
```

## Configuration

### Environment Variables

- `GEMINI_API_KEY`: Google Gemini API key
- `COHERE_API_KEY`: Cohere API key
- `QDRANT_URL`: Qdrant cloud URL (optional for local)
- `QDRANT_API_KEY`: Qdrant API key (optional for local)
- `QDRANT_COLLECTION_NAME`: Vector collection name (default: rag_embedding)
- `COHERE_MODEL`: Embedding model name (default: embed-multilingual-v3.0)
- `DEFAULT_TOP_K`: Default number of results (default: 5)
- `DEFAULT_TEMPERATURE`: Default temperature for generation (default: 0.7)
- `SIMILARITY_THRESHOLD`: Minimum similarity threshold (default: 0.3)

## Validation Features

### Retrieval Validation
- Top-K accuracy verification
- Similarity score validation
- Result ordering validation

### Content Validation
- 95%+ content similarity threshold
- Exact and near match detection
- Text similarity calculation

### Metadata Validation
- URL format validation
- Chunk ID format validation
- 99% metadata integrity threshold
- Field-specific validation

## Performance Monitoring

The system includes performance monitoring for key operations:
- Query processing time
- Embedding generation time
- Vector search time
- Response generation time

## Error Handling

Comprehensive error handling for:
- Invalid queries
- API rate limits
- Network timeouts
- Configuration issues
- Vector database errors

## Security

- API keys stored in environment variables
- Input validation for all endpoints
- Rate limiting considerations
- Secure configuration management

## Development

### Adding New Tests

To add new tests, create test files following the existing patterns in the `backend/rag_agent/` directory.

### Extending Functionality

The modular design allows for easy extension of:
- New embedding models
- Additional vector databases
- Alternative LLM providers
- Custom validation rules

## Troubleshooting

### Common Issues

1. **API Quotas**: Ensure API keys have sufficient quotas
2. **Network Issues**: Verify connectivity to external services
3. **Configuration**: Check all required environment variables are set
4. **Rate Limits**: Implement appropriate delays for API calls

### Logging

Detailed logging is available for debugging:
- Set `LOG_LEVEL=DEBUG` for verbose output
- Check log files in the application directory
- Monitor API request/response patterns

## License

This project is part of the Physical AI & Humanoid Robotics textbook and is provided for educational purposes.