"""
Test script for Metadata Integrity Testing in the RAG agent system.
This script tests the metadata validation functionality to ensure all metadata
fields (URL, chunk_id, and other attributes) are correctly preserved and returned.
"""
import asyncio
import logging
from typing import List, Dict

# Add the backend directory to the path to import rag_agent modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_agent.retrieval_tool import (
    retrieve_relevant_chunks,
    validate_metadata_integrity,
    validate_url_field,
    validate_chunk_id_field,
    comprehensive_metadata_validation
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_metadata_integrity():
    """
    Test metadata retrieval with known original metadata samples.
    """
    logger.info("Starting metadata integrity tests...")

    # Sample original metadata that would have been stored with embedded content
    original_metadata_samples = [
        {
            "url": "https://robotics.stackexchange.com/questions/about-ros2",
            "chunk_id": "ros2_intro_001",
            "source_type": "documentation",
            "section": "introduction",
            "language": "en"
        },
        {
            "url": "https://docs.ros.org/en/rolling/Concepts.html",
            "chunk_id": "ros2_concepts_002",
            "source_type": "official_documentation",
            "section": "concepts",
            "language": "en"
        },
        {
            "url": "https://github.com/ros2/tutorials",
            "chunk_id": "ros2_tutorials_003",
            "source_type": "tutorial",
            "section": "tutorials",
            "language": "en"
        }
    ]

    # Test each metadata sample
    for i, original_metadata in enumerate(original_metadata_samples):
        logger.info(f"Testing metadata integrity for sample {i+1}")

        # Validate individual fields
        url_validation = validate_url_field(original_metadata.get('url'))
        chunk_id_validation = validate_chunk_id_field(original_metadata.get('chunk_id'))

        logger.info(f"  URL validation: {url_validation['is_valid_url']}, value: {url_validation['value']}")
        logger.info(f"  Chunk ID validation: {chunk_id_validation['is_valid_chunk_id']}, value: {chunk_id_validation['value']}")

        # Test retrieval and metadata validation
        try:
            # Use a query related to the content to retrieve chunks with metadata
            query = "What is ROS2?"
            retrieved_chunks = await retrieve_relevant_chunks(query, top_k=2)

            if retrieved_chunks:
                logger.info(f"  Retrieved {len(retrieved_chunks)} chunks for query '{query}'")

                # Validate metadata for each retrieved chunk
                for j, chunk in enumerate(retrieved_chunks):
                    retrieved_metadata = chunk.get('metadata', {})

                    logger.info(f"  Validating metadata for chunk {j+1}")

                    # Perform comprehensive metadata validation
                    validation_result = comprehensive_metadata_validation(original_metadata, retrieved_metadata)

                    logger.info(f"    Metadata validation passed: {validation_result['validation_passed']}")
                    logger.info(f"    Success rate: {validation_result['success_rate']:.2%}")
                    logger.info(f"    Total fields: {validation_result['total_fields']}")
                    logger.info(f"    Matching fields: {validation_result['matching_fields']}")

                    # Show detailed validation for important fields
                    if 'url' in validation_result['field_validations']:
                        url_validation = validation_result['field_validations']['url']
                        logger.info(f"    URL match: {url_validation['values_match']}")

                    if 'chunk_id' in validation_result['field_validations']:
                        chunk_id_validation = validation_result['field_validations']['chunk_id']
                        logger.info(f"    Chunk ID match: {chunk_id_validation['values_match']}")

                    # Show any errors
                    if validation_result['detailed_error_report']['missing_fields_in_retrieved']:
                        logger.warning(f"    Missing fields: {validation_result['detailed_error_report']['missing_fields_in_retrieved']}")

                    if validation_result['detailed_error_report']['mismatched_fields']:
                        logger.warning(f"    Mismatched fields: {validation_result['detailed_error_report']['mismatched_fields']}")
            else:
                logger.warning(f"  No chunks retrieved for query '{query}'")

        except Exception as e:
            logger.error(f"  Error during metadata validation test: {str(e)}")
            continue

    logger.info("Metadata integrity tests completed.")


def test_field_validations():
    """
    Test individual field validation functions.
    """
    logger.info("Testing individual field validations...")

    # Test URL validation
    test_urls = [
        ("https://example.com", True),
        ("http://example.com", True),
        ("ftp://example.com", False),  # Invalid scheme
        ("", False),  # Empty
        ("not-a-url", False),  # Invalid format
        ("https://docs.ros.org/en/rolling/", True)
    ]

    for url, expected_valid in test_urls:
        result = validate_url_field(url)
        logger.info(f"  URL '{url[:30]}...': valid={result['is_valid_url']}, expected={expected_valid}, match={result['is_valid_url'] == expected_valid}")

    # Test chunk_id validation
    test_chunk_ids = [
        ("chunk_001", True),
        ("ros2_intro_001", True),
        ("a", False),  # Too short
        ("a" * 101, False),  # Too long
        ("chunk with spaces", False),  # Invalid characters
        ("valid_chunk-id_123", True)
    ]

    for chunk_id, expected_valid in test_chunk_ids:
        result = validate_chunk_id_field(chunk_id)
        logger.info(f"  Chunk ID '{chunk_id[:20]}...': valid={result['is_valid_chunk_id']}, expected={expected_valid}, match={result['is_valid_chunk_id'] == expected_valid}")


async def main():
    """
    Main function to run all metadata integrity tests.
    """
    logger.info("Starting comprehensive metadata integrity tests...")

    # Test individual field validations
    test_field_validations()

    # Test metadata integrity
    await test_metadata_integrity()

    logger.info("All metadata integrity tests completed.")


if __name__ == "__main__":
    asyncio.run(main())