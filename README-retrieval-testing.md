# Retrieval Pipeline Testing

This module provides comprehensive testing for the RAG retrieval pipeline to verify that stored vectors in Qdrant can be retrieved accurately.

## Features

- Query embedding generation using the same model as ingestion
- Qdrant similarity search with configurable top-k results
- Content matching verification with 95%+ similarity threshold
- Metadata integrity testing for URL and chunk_id fields
- End-to-end pipeline testing with structured JSON output
- Performance monitoring and error handling

## Requirements

- Python 3.11+
- Existing embedded content in Qdrant collection named "rag_embedding"
- Cohere API key
- Qdrant API key and cluster URL (or local Qdrant instance)

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-test.txt
   ```

2. **Configure Environment**:
   Ensure your `.env` file contains the necessary credentials:
   ```bash
   COHERE_API_KEY=your_cohere_api_key_here
   QDRANT_URL=your_qdrant_cluster_url_here  # or leave empty if using local instance
   QDRANT_API_KEY=your_qdrant_api_key_here  # optional for local instance
   ```

## Usage

### Running a Single Test

```python
from retrieval_tester import run_retrieval_test

# Run a retrieval test
result = run_retrieval_test(query_text="What is ROS2?", top_k=3)
print(f"Test accuracy: {result['overall_accuracy']}")
print(f"Retrieved {len(result['retrieved_results'])} results")
```

### Running Multiple Tests

```python
from retrieval_tester import run_comprehensive_test

queries = ["ROS2 concepts", "simulation environments", "AI planning"]
results = run_comprehensive_test(queries, top_k=5)
```

### Testing Edge Cases

```python
from retrieval_tester import test_edge_cases

edge_case_results = test_edge_cases()
for result in edge_case_results:
    print(f"Test case: {result['test_case']}, Status: {result['status']}")
```

### Configuration Validation

```python
from retrieval_tester import validate_configuration

config_status = validate_configuration()
print(f"All checks passed: {config_status['all_checks_passed']}")
```

## Functions

### `run_retrieval_test(query_text: str, top_k: int = 5) -> Dict`
Main function that executes the complete end-to-end test flow:
- Generates embedding for the query using the same model as ingestion
- Performs similarity search in Qdrant
- Validates retrieved results
- Returns structured test results

### `generate_query_embedding(query_text: str) -> List[float]`
Generates an embedding vector for a query text using the same Cohere model as the ingestion pipeline.

### `search_qdrant(query_embedding: List[float], top_k: int = 5) -> List[Dict]`
Performs similarity search in Qdrant and returns top-k results with metadata.

### `validate_content_match(retrieved_content: str, original_content: str) -> Dict`
Validates that retrieved content matches original content with 95%+ similarity.

### `validate_metadata(retrieved_metadata: Dict, expected_metadata: Dict) -> Dict`
Verifies that metadata fields (URL, chunk_id) are correctly preserved.

### `format_json_output(test_results: Dict) -> str`
Formats test results into clean, structured JSON output.

## Testing

Run the pytest test suite:

```bash
python -m pytest test_retrieval.py -v
```

## Output Format

The test results follow this JSON schema:

```json
{
  "query_request": {
    "query_text": "input query text",
    "top_k": 5,
    "test_timestamp": 1234567890
  },
  "retrieved_results": [
    {
      "chunk_id": "unique identifier",
      "content": "retrieved text content",
      "similarity_score": 0.85,
      "metadata": {"url": "...", "title": "..."},
      "position_in_results": 0
    }
  ],
  "validation_results": [
    {
      "chunk_id": "...",
      "content_validation": {...},
      "metadata_validation": {...}
    }
  ],
  "overall_accuracy": 0.95,
  "test_timestamp": "2023-12-14 10:30:00",
  "test_status": "COMPLETED",
  "execution_time": 1.23,
  "summary": {
    "total_results": 5,
    "accuracy_percentage": "95.00%"
  }
}
```

## Success Criteria

- Achieves 95% accuracy in top-k retrieval where returned results match expected similar content
- Retrieved content chunks match original text with 95%+ semantic similarity
- All metadata (URL, chunk_id) is returned correctly for 99% of retrieval requests
- End-to-end pipeline produces clean JSON output for 100% of test queries with proper error handling