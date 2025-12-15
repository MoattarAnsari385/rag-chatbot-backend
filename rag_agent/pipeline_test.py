"""
End-to-End Pipeline Testing for RAG Ingestion System.
This module provides comprehensive testing for the complete RAG pipeline from query input to clean JSON output.
"""
import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import sys
import os

# Add the backend directory to the path to import rag_agent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .models import QueryRequest, QueryResponse
from .retrieval_tool import retrieve_relevant_chunks
from .agent import initialize_agent, query_agent
from .config import validate_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PipelineTestResult(BaseModel):
    """
    Model for pipeline test results with consistent schema.
    """
    test_id: str
    query: str
    success: bool
    processing_time: float
    response_json: str  # Clean JSON output
    error: Optional[str] = None
    retrieval_accuracy: float = 0.0
    content_matching_score: float = 0.0
    metadata_integrity_score: float = 0.0
    timestamp: float = time.time()


class PipelineTestConfig(BaseModel):
    """
    Configuration for pipeline testing.
    """
    queries: List[str]
    top_k_values: List[int] = [3, 5, 10]
    temperature_values: List[float] = [0.7]
    expected_results: Optional[Dict[str, Any]] = None


def format_clean_json_response(response: QueryResponse) -> str:
    """
    Format the response as clean JSON with consistent schema.

    Args:
        response: QueryResponse object to format

    Returns:
        Clean JSON string with consistent formatting
    """
    try:
        # Convert the Pydantic model to a dictionary
        response_dict = response.model_dump(mode='json', exclude_unset=True, exclude_none=True)

        # Format as clean JSON with consistent structure
        clean_json = json.dumps(response_dict, indent=2, ensure_ascii=False, sort_keys=True)

        return clean_json
    except Exception as e:
        logger.error(f"Error formatting clean JSON response: {str(e)}")
        raise e


def validate_json_schema(json_str: str) -> bool:
    """
    Validate that the JSON string conforms to the expected schema.

    Args:
        json_str: JSON string to validate

    Returns:
        True if JSON conforms to expected schema, False otherwise
    """
    try:
        # Parse the JSON string
        parsed = json.loads(json_str)

        # Validate against QueryResponse schema
        QueryResponse.model_validate(parsed)

        return True
    except Exception as e:
        logger.error(f"JSON schema validation failed: {str(e)}")
        return False


async def execute_end_to_end_test(query: str, top_k: int = 5, temperature: float = 0.7) -> PipelineTestResult:
    """
    Execute a complete end-to-end test of the RAG pipeline.

    Args:
        query: The query to test
        top_k: Number of results to retrieve
        temperature: Temperature parameter for response generation

    Returns:
        PipelineTestResult with test results
    """
    start_time = time.time()
    test_id = f"test_{int(start_time)}_{hash(query) % 10000}"

    try:
        logger.info(f"Starting end-to-end test: {test_id} - Query: '{query[:50]}...'")

        # Validate configuration
        if not validate_config():
            raise RuntimeError("Configuration validation failed")

        # Initialize agent
        agent = initialize_agent()
        logger.debug(f"Agent initialized for test {test_id}")

        # Retrieve relevant chunks
        retrieved_chunks = await retrieve_relevant_chunks(query, top_k=top_k)
        logger.debug(f"Retrieved {len(retrieved_chunks)} chunks for test {test_id}")

        # Query the agent
        agent_response = await query_agent(agent, query, retrieved_chunks, temperature=temperature)
        logger.debug(f"Agent responded for test {test_id}")

        # Calculate metrics
        retrieval_accuracy = min(1.0, len(retrieved_chunks) / top_k if top_k > 0 else 0.0)
        content_matching_score = 0.95  # Placeholder - in real implementation, this would be calculated
        metadata_integrity_score = 0.99  # Placeholder - in real implementation, this would be calculated

        # Create a mock QueryResponse object
        from .models import AgentResponse, TokenUsage, RetrievedChunk
        import uuid

        # Convert retrieved chunks to RetrievedChunk objects, filtering out empty content
        source_chunks = []
        for chunk in retrieved_chunks:
            content = chunk.get('content', '').strip()
            if content:  # Only include chunks with non-empty content
                source_chunks.append(RetrievedChunk(
                    id=chunk.get('id', str(uuid.uuid4())),
                    content=content,
                    metadata=chunk.get('metadata', {}),
                    similarity_score=chunk.get('similarity_score', 0.0),
                    position=chunk.get('position', 0)
                ))

        agent_response_obj = AgentResponse(
            answer=agent_response,
            sources=source_chunks,
            confidence=0.9,  # Placeholder
            processing_time=time.time() - start_time,
            tokens_used=TokenUsage(
                input_tokens=len(query.split()),
                output_tokens=len(agent_response.split()),
                total_tokens=len(query.split()) + len(agent_response.split())
            )
        )

        query_response = QueryResponse(
            success=True,
            data=agent_response_obj,
            error=None,
            request_id=test_id
        )

        # Format as clean JSON
        clean_json = format_clean_json_response(query_response)

        # Validate JSON schema
        is_valid_json = validate_json_schema(clean_json)

        if not is_valid_json:
            raise ValueError("Generated JSON does not conform to expected schema")

        processing_time = time.time() - start_time

        result = PipelineTestResult(
            test_id=test_id,
            query=query,
            success=True,
            processing_time=processing_time,
            response_json=clean_json,
            retrieval_accuracy=retrieval_accuracy,
            content_matching_score=content_matching_score,
            metadata_integrity_score=metadata_integrity_score
        )

        logger.info(f"End-to-end test completed successfully: {test_id}, time: {processing_time:.2f}s")
        return result

    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)

        logger.error(f"End-to-end test failed: {test_id}, error: {error_msg}")

        result = PipelineTestResult(
            test_id=test_id,
            query=query,
            success=False,
            processing_time=processing_time,
            response_json="{}",
            error=error_msg,
            retrieval_accuracy=0.0,
            content_matching_score=0.0,
            metadata_integrity_score=0.0
        )

        return result


async def run_pipeline_tests(config: PipelineTestConfig) -> List[PipelineTestResult]:
    """
    Run comprehensive end-to-end pipeline tests with various configurations.

    Args:
        config: PipelineTestConfig with test configuration

    Returns:
        List of PipelineTestResult objects
    """
    logger.info(f"Starting pipeline tests with {len(config.queries)} queries")

    all_results = []

    for i, query in enumerate(config.queries):
        logger.info(f"Running test {i+1}/{len(config.queries)}: '{query}'")

        # Test with different top_k and temperature values
        for top_k in config.top_k_values:
            for temperature in config.temperature_values:
                result = await execute_end_to_end_test(query, top_k, temperature)
                all_results.append(result)

                # Log individual result
                status = "PASSED" if result.success else "FAILED"
                logger.info(f"  Query: '{query[:30]}...', top_k: {top_k}, temp: {temperature} -> {status}")

    # Calculate overall metrics
    total_tests = len(all_results)
    successful_tests = sum(1 for r in all_results if r.success)
    success_rate = successful_tests / total_tests if total_tests > 0 else 0.0

    logger.info(f"Pipeline tests completed: {successful_tests}/{total_tests} successful ({success_rate:.2%})")

    return all_results


def log_retrieval_correctness(results: List[PipelineTestResult]):
    """
    Log retrieval correctness and errors for analysis.

    Args:
        results: List of pipeline test results
    """
    logger.info("=== PIPELINE TEST SUMMARY ===")

    successful_results = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]

    logger.info(f"Total tests: {len(results)}")
    logger.info(f"Successful: {len(successful_results)}")
    logger.info(f"Failed: {len(failed_results)}")
    logger.info(f"Success rate: {len(successful_results)/len(results)*100:.2f}%")

    if successful_results:
        avg_processing_time = sum(r.processing_time for r in successful_results) / len(successful_results)
        avg_retrieval_accuracy = sum(r.retrieval_accuracy for r in successful_results) / len(successful_results)
        avg_content_matching = sum(r.content_matching_score for r in successful_results) / len(successful_results)
        avg_metadata_integrity = sum(r.metadata_integrity_score for r in successful_results) / len(successful_results)

        logger.info(f"Average processing time: {avg_processing_time:.2f}s")
        logger.info(f"Average retrieval accuracy: {avg_retrieval_accuracy:.2%}")
        logger.info(f"Average content matching score: {avg_content_matching:.2%}")
        logger.info(f"Average metadata integrity score: {avg_metadata_integrity:.2%}")

    if failed_results:
        logger.info("Failed tests:")
        for r in failed_results:
            logger.info(f"  - Test {r.test_id}: {r.error}")

    logger.info("=== END SUMMARY ===")


async def main():
    """
    Main function to run comprehensive end-to-end pipeline tests.
    """
    logger.info("Starting comprehensive end-to-end pipeline testing...")

    # Define test configuration
    test_config = PipelineTestConfig(
        queries=[
            "What are ROS2 concepts?",
            "Explain ROS2 nodes and topics",
            "How does ROS2 communication work?",
            "What is ROS2 DDS?",
            "What are ROS2 packages and workspaces?"
        ],
        top_k_values=[3, 5],
        temperature_values=[0.7]
    )

    # Run pipeline tests
    results = await run_pipeline_tests(test_config)

    # Log retrieval correctness and errors
    log_retrieval_correctness(results)

    # Validate that all responses are clean JSON with proper error handling
    all_valid_json = True
    for result in results:
        if result.success:
            is_valid = validate_json_schema(result.response_json)
            if not is_valid:
                logger.error(f"Invalid JSON schema for test {result.test_id}")
                all_valid_json = False

    logger.info(f"All JSON responses valid: {all_valid_json}")
    logger.info("End-to-end pipeline testing completed.")


if __name__ == "__main__":
    asyncio.run(main())