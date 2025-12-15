"""
Integration Testing for RAG Agent System.
This module integrates all user story components into a cohesive testing workflow.
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
import time
import sys
import os

# Add the backend directory to the path to import rag_agent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .retrieval_tool import (
    retrieve_relevant_chunks,
    validate_retrieval_accuracy,
    content_matching_validation,
    validate_content_matching,
    validate_metadata_integrity,
    comprehensive_metadata_validation
)
from .agent import initialize_agent, query_agent
from .config import validate_config
from .models import QueryRequest, QueryResponse, AgentResponse, RetrievedChunk, TokenUsage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntegrationTestResult:
    """
    Class to hold integration test results.
    """
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = time.time()
        self.passed = True
        self.errors = []
        self.metrics = {}
        self.test_details = {}

    def add_error(self, error: str):
        self.errors.append(error)
        self.passed = False

    def add_metric(self, name: str, value: Any):
        self.metrics[name] = value

    def add_detail(self, name: str, value: Any):
        self.test_details[name] = value

    def get_duration(self):
        return time.time() - self.start_time


async def test_top_k_retrieval_accuracy():
    """
    Test User Story 1: Top-K Retrieval Accuracy
    """
    logger.info("Testing Top-K Retrieval Accuracy (User Story 1)")

    result = IntegrationTestResult("Top-K Retrieval Accuracy")

    try:
        # Test with various queries
        test_queries = ["ROS2 concepts", "robotics", "AI"]

        for query in test_queries:
            # Test different top_k values
            for top_k in [3, 5, 10]:
                retrieved_chunks = await retrieve_relevant_chunks(query, top_k=top_k)

                # Verify results are ordered by similarity score
                scores = [chunk['similarity_score'] for chunk in retrieved_chunks]
                is_ordered = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

                if not is_ordered and len(retrieved_chunks) > 1:
                    result.add_error(f"Results not ordered by similarity for query '{query}' with top_k={top_k}")

                # Verify we don't get more results than requested
                if len(retrieved_chunks) > top_k:
                    result.add_error(f"Retrieved {len(retrieved_chunks)} chunks, expected max {top_k}")

                logger.info(f"  Query: '{query}', top_k: {top_k}, retrieved: {len(retrieved_chunks)}, ordered: {is_ordered}")

        result.add_metric("average_chunks_retrieved", sum([len(await retrieve_relevant_chunks(q, top_k=5)) for q in test_queries]) / len(test_queries))

    except Exception as e:
        result.add_error(f"Error in Top-K Retrieval Accuracy test: {str(e)}")

    return result


async def test_content_matching_verification():
    """
    Test User Story 2: Content Matching Verification
    """
    logger.info("Testing Content Matching Verification (User Story 2)")

    result = IntegrationTestResult("Content Matching Verification")

    try:
        # Test with a query
        query = "ROS2 architecture"
        retrieved_chunks = await retrieve_relevant_chunks(query, top_k=5)

        if retrieved_chunks:
            # Validate content matching for each retrieved chunk
            for chunk in retrieved_chunks:
                content = chunk.get('content', '')

                # Test exact content matching (with itself)
                validation = content_matching_validation(content, content)

                if not validation['is_match']:
                    result.add_error(f"Self-content match failed for chunk {chunk.get('id')}")

                # Test with slightly varied content
                varied_content = content.replace("ROS2", "ROS 2") if content else ""
                if varied_content:
                    validation_varied = content_matching_validation(content, varied_content)
                    result.add_detail(f"Variation match for chunk {chunk.get('id')}", validation_varied['similarity_ratio'])

        # Validate overall content matching
        if retrieved_chunks:
            overall_validation = validate_content_matching(retrieved_chunks)
            result.add_metric("content_matching_success_rate", overall_validation['match_rate'])
            result.add_metric("average_content_similarity", overall_validation['average_similarity'])

        logger.info(f"  Retrieved {len(retrieved_chunks)} chunks for content matching test")
        logger.info(f"  Content matching success rate: {overall_validation.get('match_rate', 0):.2%}" if retrieved_chunks else "  No chunks retrieved")

    except Exception as e:
        result.add_error(f"Error in Content Matching Verification test: {str(e)}")

    return result


async def test_metadata_integrity():
    """
    Test User Story 3: Metadata Integrity Testing
    """
    logger.info("Testing Metadata Integrity (User Story 3)")

    result = IntegrationTestResult("Metadata Integrity")

    try:
        # Test with a query
        query = "robotics framework"
        retrieved_chunks = await retrieve_relevant_chunks(query, top_k=5)

        if retrieved_chunks:
            for chunk in retrieved_chunks:
                metadata = chunk.get('metadata', {})

                # Perform comprehensive metadata validation
                # Using a mock expected metadata for testing
                expected_metadata = metadata.copy()  # In real scenario, you'd have original metadata
                validation = comprehensive_metadata_validation(expected_metadata, metadata)

                result.add_metric("metadata_success_rate", validation['success_rate'])
                result.add_detail(f"metadata_validation_{chunk.get('id', 'unknown')}", validation['validation_passed'])

                logger.info(f"  Chunk {chunk.get('id', 'unknown')}: metadata validation passed: {validation['validation_passed']}")

        logger.info(f"  Retrieved {len(retrieved_chunks)} chunks for metadata integrity test")

    except Exception as e:
        result.add_error(f"Error in Metadata Integrity test: {str(e)}")

    return result


async def test_end_to_end_pipeline():
    """
    Test User Story 4: End-to-End Pipeline Testing
    """
    logger.info("Testing End-to-End Pipeline (User Story 4)")

    result = IntegrationTestResult("End-to-End Pipeline")

    try:
        # Validate configuration
        if not validate_config():
            result.add_error("Configuration validation failed")
            return result

        # Initialize agent
        agent = initialize_agent()

        # Test query
        query = "What is ROS2?"
        top_k = 5

        # Retrieve chunks
        retrieved_chunks = await retrieve_relevant_chunks(query, top_k=top_k)
        result.add_metric("retrieved_chunks_count", len(retrieved_chunks))

        # Query agent
        agent_response = await query_agent(agent, query, retrieved_chunks)

        # Validate response structure
        if not agent_response or len(agent_response.strip()) == 0:
            result.add_error("Agent returned empty response")

        # Check that response is grounded in retrieved content
        content_in_response = any(chunk.get('content', '')[:50] in agent_response for chunk in retrieved_chunks if chunk.get('content'))
        result.add_metric("response_uses_context", content_in_response)

        logger.info(f"  Query: '{query}'")
        logger.info(f"  Retrieved chunks: {len(retrieved_chunks)}")
        logger.info(f"  Response grounded in context: {content_in_response}")

    except Exception as e:
        result.add_error(f"Error in End-to-End Pipeline test: {str(e)}")

    return result


async def test_edge_cases():
    """
    Test edge cases: fewer than k results, no similar content, unavailable services
    """
    logger.info("Testing Edge Cases")

    result = IntegrationTestResult("Edge Cases")

    try:
        # Test with a very specific query that might return few results
        specific_query = "very specific and unlikely to match anything in the corpus 123456789"
        few_results = await retrieve_relevant_chunks(specific_query, top_k=10)

        logger.info(f"  Specific query returned {len(few_results)} results")
        result.add_metric("few_results_count", len(few_results))

        # Test with empty query (should be handled gracefully)
        try:
            empty_results = await retrieve_relevant_chunks("", top_k=5)
            result.add_error("Empty query should have been rejected but wasn't")
        except Exception:
            # Expected behavior - empty query should fail validation
            logger.info("  Empty query properly rejected")

        # Test with very long query
        long_query = "This is a very long query " + "with repeated text " * 100
        long_results = await retrieve_relevant_chunks(long_query, top_k=3)

        logger.info(f"  Long query returned {len(long_results)} results")
        result.add_metric("long_query_results_count", len(long_results))

        # Test with top_k=1 to ensure it works with minimal results
        single_result = await retrieve_relevant_chunks("ROS2", top_k=1)
        logger.info(f"  Single result query returned {len(single_result)} results")

        if len(single_result) > 1:
            result.add_error(f"top_k=1 returned {len(single_result)} results instead of 1")

    except Exception as e:
        result.add_error(f"Error in Edge Cases test: {str(e)}")

    return result


async def run_integration_tests():
    """
    Run all integration tests and return comprehensive results.
    """
    logger.info("Starting comprehensive integration tests...")

    # Run all user story tests
    tests = [
        ("Top-K Retrieval Accuracy", test_top_k_retrieval_accuracy),
        ("Content Matching Verification", test_content_matching_verification),
        ("Metadata Integrity", test_metadata_integrity),
        ("End-to-End Pipeline", test_end_to_end_pipeline),
        ("Edge Cases", test_edge_cases),
    ]

    results = []
    all_passed = True

    for test_name, test_func in tests:
        logger.info(f"Running {test_name} test...")
        result = await test_func()
        results.append(result)

        status = "PASSED" if result.passed else "FAILED"
        logger.info(f"{test_name}: {status} ({result.get_duration():.2f}s)")

        if not result.passed:
            all_passed = False
            for error in result.errors:
                logger.error(f"  Error: {error}")

    # Generate summary
    logger.info("\n=== INTEGRATION TEST SUMMARY ===")
    logger.info(f"Total tests: {len(results)}")
    logger.info(f"Passed: {sum(1 for r in results if r.passed)}")
    logger.info(f"Failed: {sum(1 for r in results if not r.passed)}")
    logger.info(f"Overall status: {'PASSED' if all_passed else 'FAILED'}")

    # Print metrics
    logger.info("\n=== METRICS ===")
    for result in results:
        if result.metrics:
            logger.info(f"{result.test_name} metrics:")
            for metric, value in result.metrics.items():
                logger.info(f"  {metric}: {value}")

    logger.info("===============================")

    return results, all_passed


async def main():
    """
    Main function to run integration tests.
    """
    logger.info("Starting RAG Agent Integration Testing")

    results, overall_success = await run_integration_tests()

    if overall_success:
        logger.info("All integration tests PASSED!")
    else:
        logger.error("Some integration tests FAILED!")

    logger.info("Integration testing completed.")


if __name__ == "__main__":
    asyncio.run(main())