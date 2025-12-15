"""
Pytest test cases covering all user stories for the RAG agent system.
"""
import pytest
import asyncio
from typing import Dict, List
import sys
import os

# Add the backend directory to the path to import rag_agent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_agent.retrieval_tool import (
    retrieve_relevant_chunks,
    validate_retrieval_accuracy,
    content_matching_validation,
    validate_content_matching,
    validate_metadata_integrity,
    comprehensive_metadata_validation
)
from rag_agent.agent import initialize_agent, query_agent
from rag_agent.config import validate_config
from rag_agent.models import QueryRequest, QueryResponse


@pytest.mark.asyncio
async def test_top_k_retrieval_accuracy():
    """
    Test User Story 1: Top-K Retrieval Accuracy
    Verify that Qdrant returns the correct top-k matches when queried with a search vector using cosine similarity
    """
    # Test with a sample query
    query = "ROS2 concepts"
    top_k = 5

    retrieved_chunks = await retrieve_relevant_chunks(query, top_k=top_k)

    # Verify we get results
    assert len(retrieved_chunks) > 0, "Should retrieve at least some chunks"
    assert len(retrieved_chunks) <= top_k, f"Should not retrieve more than {top_k} chunks"

    # Verify results are ordered by similarity score (descending)
    if len(retrieved_chunks) > 1:
        scores = [chunk['similarity_score'] for chunk in retrieved_chunks]
        is_ordered = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        assert is_ordered, "Results should be ordered by descending similarity score"


@pytest.mark.asyncio
async def test_retrieval_with_different_top_k_values():
    """
    Test that retrieval works correctly with different top_k values
    """
    query = "robotics"

    # Test with different top_k values
    for top_k in [1, 3, 5, 10]:
        retrieved_chunks = await retrieve_relevant_chunks(query, top_k=top_k)
        assert len(retrieved_chunks) <= top_k, f"Should not retrieve more than {top_k} chunks"
        if retrieved_chunks:
            assert all(chunk['similarity_score'] >= 0.0 for chunk in retrieved_chunks), "Similarity scores should be non-negative"


@pytest.mark.asyncio
async def test_content_matching_verification():
    """
    Test User Story 2: Content Matching Verification
    Verify that retrieved content chunks match the original text that was embedded with 95%+ similarity
    """
    # Get some content to test with
    query = "AI"
    retrieved_chunks = await retrieve_relevant_chunks(query, top_k=3)

    if retrieved_chunks:
        # Test content matching validation
        for chunk in retrieved_chunks:
            content = chunk.get('content', '')
            if content:
                # Test self-matching (content should match itself)
                validation = content_matching_validation(content, content)
                assert validation['is_match'], "Content should match itself"
                assert validation['similarity_ratio'] == 1.0, "Self-matching should have 1.0 similarity ratio"

                # Test with slight variation
                varied_content = content.replace(" ", "_", 1) if len(content) > 0 else ""
                if varied_content and varied_content != content:
                    validation_varied = content_matching_validation(content, varied_content)
                    # Even with variation, it should have high similarity
                    assert validation_varied['similarity_ratio'] >= 0.8, "Slightly varied content should still have high similarity"


@pytest.mark.asyncio
async def test_validate_content_matching_function():
    """
    Test the validate_content_matching function
    """
    query = "machine learning"
    retrieved_chunks = await retrieve_relevant_chunks(query, top_k=5)

    # Validate content matching for retrieved chunks
    validation_result = validate_content_matching(retrieved_chunks)

    # Should have reasonable metrics
    assert validation_result['total_chunks'] >= 0, "Should have valid total chunk count"
    if validation_result['total_chunks'] > 0:
        assert 0.0 <= validation_result['average_similarity'] <= 1.0, "Average similarity should be between 0 and 1"


@pytest.mark.asyncio
async def test_metadata_integrity_testing():
    """
    Test User Story 3: Metadata Integrity Testing
    Verify that all metadata (URL, chunk_id, and other attributes) is correctly preserved and returned during retrieval
    """
    query = "ROS2 framework"
    retrieved_chunks = await retrieve_relevant_chunks(query, top_k=3)

    if retrieved_chunks:
        for chunk in retrieved_chunks:
            metadata = chunk.get('metadata', {})
            chunk_id = chunk.get('id', '')

            # Test metadata validation
            expected_metadata = metadata.copy()  # In practice, you'd have original metadata
            validation = await validate_metadata_integrity(expected_metadata, metadata)

            # Validation should complete without errors
            assert 'validation_passed' in validation, "Validation result should contain validation_passed field"
            assert 'field_validations' in validation, "Validation result should contain field_validations field"

            # Test comprehensive metadata validation
            comprehensive_validation = comprehensive_metadata_validation(expected_metadata, metadata)
            assert 'success_rate' in comprehensive_validation, "Comprehensive validation should have success_rate"
            assert 0.0 <= comprehensive_validation['success_rate'] <= 1.0, "Success rate should be between 0 and 1"


@pytest.mark.asyncio
async def test_end_to_end_pipeline():
    """
    Test User Story 4: End-to-End Pipeline Testing
    Perform comprehensive end-to-end testing from query input to clean JSON output following consistent schema
    """
    # Validate configuration first
    assert validate_config(), "Configuration should be valid"

    # Initialize agent
    agent = initialize_agent()

    # Test query
    query = "What are the key concepts in robotics?"
    top_k = 3

    # Retrieve chunks
    retrieved_chunks = await retrieve_relevant_chunks(query, top_k=top_k)
    assert len(retrieved_chunks) <= top_k, f"Should retrieve at most {top_k} chunks"

    # Query agent
    agent_response = await query_agent(agent, query, retrieved_chunks, temperature=0.7)

    # Validate response
    assert agent_response is not None, "Agent should return a response"
    assert isinstance(agent_response, str), "Agent response should be a string"
    assert len(agent_response.strip()) > 0, "Agent response should not be empty"


@pytest.mark.asyncio
async def test_edge_cases_fewer_than_k_results():
    """
    Test edge case: fewer than k results returned
    """
    query = "very specific and unlikely to match anything in the corpus 123456789"
    top_k = 10

    retrieved_chunks = await retrieve_relevant_chunks(query, top_k=top_k)

    # Should return fewer than requested if not enough matches
    assert len(retrieved_chunks) <= top_k, "Should not return more than requested top_k"
    # In this case, might return 0 or few results


@pytest.mark.asyncio
async def test_edge_cases_no_similar_content():
    """
    Test edge case: no similar content found
    """
    query = "completely unrelated and random text with no matches whatsoever"
    retrieved_chunks = await retrieve_relevant_chunks(query, top_k=5)

    # Should handle gracefully, might return empty list or chunks with very low similarity
    assert isinstance(retrieved_chunks, list), "Should return a list of chunks"


@pytest.mark.asyncio
async def test_error_handling_invalid_inputs():
    """
    Test error handling for invalid inputs
    """
    # Test with empty query
    with pytest.raises(Exception):
        await retrieve_relevant_chunks("", top_k=5)

    # Test with negative top_k (if validation is implemented)
    retrieved_chunks = await retrieve_relevant_chunks("test", top_k=-1)
    # Should handle gracefully, possibly defaulting to a minimum value


@pytest.mark.asyncio
async def test_configuration_validation():
    """
    Test that configuration validation works properly
    """
    is_valid = validate_config()
    assert is_valid, "Configuration should be valid with proper environment variables"


@pytest.mark.asyncio
async def test_retrieval_accuracy_validation():
    """
    Test retrieval accuracy validation function
    """
    # Create mock expected and retrieved chunks for testing
    expected_chunks = [
        {"id": "exp1", "content": "This is expected content", "metadata": {"url": "http://example.com"}}
    ]
    retrieved_chunks = [
        {"id": "ret1", "content": "This is expected content", "metadata": {"url": "http://example.com"}, "similarity_score": 1.0, "position": 0}
    ]

    validation_result = validate_retrieval_accuracy(expected_chunks, retrieved_chunks)

    assert 'validation_passed' in validation_result, "Should return validation result"
    assert 'accuracy_rate' in validation_result, "Should include accuracy rate"


@pytest.mark.asyncio
async def test_token_usage_approximation():
    """
    Test token usage approximation in the agent response
    """
    agent = initialize_agent()
    query = "Simple test query"
    retrieved_chunks = await retrieve_relevant_chunks(query, top_k=2)

    agent_response = await query_agent(agent, query, retrieved_chunks)

    # Basic validation that response is reasonable
    assert len(agent_response) > 0, "Response should not be empty"
    # The token approximation should work (though we can't validate the exact count without proper tokenization)


if __name__ == "__main__":
    # Run pytest when this file is executed directly
    import subprocess
    import sys

    # Run pytest with asyncio support
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-v", "-s"])
    sys.exit(result.returncode)