"""
Pytest module for retrieval pipeline testing
"""
import pytest
import json
from retrieval_tester import run_retrieval_test, generate_query_embedding, search_qdrant, validate_content_match, validate_metadata, format_json_output


def test_query_embedding_generation():
    """Test that query embedding is generated correctly"""
    query_text = "test query for retrieval"
    embedding = generate_query_embedding(query_text)

    assert isinstance(embedding, list), "Embedding should be a list"
    assert len(embedding) == 1024, "Embedding should have 1024 dimensions (same as ingestion model)"
    assert all(isinstance(val, float) for val in embedding), "All embedding values should be floats"


def test_basic_retrieval():
    """Test basic retrieval functionality"""
    result = run_retrieval_test("ROS2 concepts", top_k=3)

    assert 'query_request' in result
    assert 'retrieved_results' in result
    assert 'validation_results' in result
    assert 'overall_accuracy' in result
    assert len(result['retrieved_results']) <= 3


def test_content_validation():
    """Test content validation functionality"""
    retrieved_content = "This is a test content string"
    original_content = "This is a test content string"

    validation_result = validate_content_match(retrieved_content, original_content)

    assert validation_result['is_match'] == True
    assert 0.0 <= validation_result['similarity_ratio'] <= 1.0


def test_metadata_validation():
    """Test metadata validation functionality"""
    retrieved_metadata = {
        'url': 'https://example.com/test',
        'chunk_id': 'test-chunk-123',
        'title': 'Test Document'
    }
    expected_metadata = {
        'url': 'https://example.com/test',
        'chunk_id': 'test-chunk-123',
        'title': 'Test Document'
    }

    validation_result = validate_metadata(retrieved_metadata, expected_metadata)

    assert validation_result['is_valid'] == True
    assert len(validation_result['validation_errors']) == 0


def test_json_output_format():
    """Test JSON output formatting"""
    test_results = {
        'query_request': {'query_text': 'test', 'top_k': 5},
        'retrieved_results': [],
        'validation_results': []
    }

    json_output = format_json_output(test_results)
    parsed_output = json.loads(json_output)

    assert isinstance(json_output, str), "Output should be a JSON string"
    assert 'query_request' in parsed_output
    assert 'retrieved_results' in parsed_output
    assert 'validation_results' in parsed_output


if __name__ == "__main__":
    pytest.main([__file__])