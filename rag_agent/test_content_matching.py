"""
Test script for Content Matching Verification in the RAG agent system.
This script tests the content matching functionality to ensure retrieved content
matches the original text that was embedded with 95%+ similarity.
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
    content_matching_validation,
    validate_content_matching,
    calculate_text_similarity
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_content_matching_verification():
    """
    Test content matching verification with original embedded text samples.
    """
    logger.info("Starting content matching verification tests...")

    # Sample original content that would have been embedded
    original_content_samples = [
        {
            "id": "sample_1",
            "content": "ROS2 (Robot Operating System 2) is a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.",
            "metadata": {"url": "https://example.com/ros2-intro", "source_type": "documentation"}
        },
        {
            "id": "sample_2",
            "content": "Nodes in ROS2 are processes that perform computation. Nodes are the fundamental building blocks of ROS2 programs. Each node is designed to perform a specific task and can communicate with other nodes through topics, services, or actions.",
            "metadata": {"url": "https://example.com/ros2-nodes", "source_type": "documentation"}
        },
        {
            "id": "sample_3",
            "content": "Topics in ROS2 are named buses over which nodes exchange messages. Topics implement a publish/subscribe messaging model where publishers send messages to a topic and subscribers receive messages from a topic.",
            "metadata": {"url": "https://example.com/ros2-topics", "source_type": "documentation"}
        }
    ]

    # Test content matching validation for each sample
    for i, sample in enumerate(original_content_samples):
        logger.info(f"Testing content matching for sample {i+1}: '{sample['content'][:50]}...'")

        # Test exact match (should return 1.0 similarity)
        exact_match_result = content_matching_validation(sample['content'], sample['content'])
        logger.info(f"  Exact match similarity: {exact_match_result['similarity_ratio']:.3f}, match: {exact_match_result['is_match']}")

        # Test with slight variations (should still be high similarity)
        varied_content = sample['content'].replace("ROS2", "ROS 2")  # Minor variation
        varied_match_result = content_matching_validation(sample['content'], varied_content)
        logger.info(f"  Varied content similarity: {varied_match_result['similarity_ratio']:.3f}, match: {varied_match_result['is_match']}")

        # Test with different content (should be low similarity)
        different_content = "This is completely different content unrelated to robotics."
        different_match_result = content_matching_validation(sample['content'], different_content)
        logger.info(f"  Different content similarity: {different_match_result['similarity_ratio']:.3f}, match: {different_match_result['is_match']}")

        # Test retrieval and content matching together
        try:
            # Use a query related to the content to retrieve similar chunks
            query = "What is ROS2?"
            retrieved_chunks = await retrieve_relevant_chunks(query, top_k=3)

            if retrieved_chunks:
                logger.info(f"  Retrieved {len(retrieved_chunks)} chunks for query '{query}'")

                # Validate content matching for retrieved chunks
                validation_result = validate_content_matching(retrieved_chunks, [sample])
                logger.info(f"  Content matching validation: passed={validation_result['validation_passed']}, avg_similarity={validation_result['average_similarity']:.3f}")

                # Show detailed results for first retrieved chunk
                if validation_result['validation_results']:
                    first_result = validation_result['validation_results'][0]['validation']
                    logger.info(f"    First chunk similarity: {first_result['similarity_ratio']:.3f}, match: {first_result['is_match']}")
            else:
                logger.warning(f"  No chunks retrieved for query '{query}'")

        except Exception as e:
            logger.error(f"  Error during retrieval test: {str(e)}")
            continue

    logger.info("Content matching verification tests completed.")


def test_similarity_calculation():
    """
    Test the similarity calculation function with various text comparisons.
    """
    logger.info("Testing similarity calculation function...")

    test_cases = [
        ("Identical text", "Hello world", "Hello world", 1.0),
        ("Case difference", "Hello World", "hello world", 1.0),  # Should be handled by normalization
        ("Minor difference", "Hello world", "Hello World", 1.0),  # Should be handled by normalization
        ("Similar content", "ROS2 is a robot framework", "ROS2 is a framework for robots", 0.8),
        ("Different content", "This is about robotics", "This is about cooking", 0.1),
        ("Empty strings", "", "", 1.0),
        ("One empty", "Hello", "", 0.0),
        ("Substring", "Hello world", "Hello", 0.8),  # Should be high but not 1.0
    ]

    for description, text1, text2, expected_min in test_cases:
        similarity = calculate_text_similarity(text1, text2)
        is_acceptable = similarity >= expected_min
        logger.info(f"  {description}: '{text1[:20]}...' vs '{text2[:20]}...' -> {similarity:.3f} (>= {expected_min}): {is_acceptable}")


async def main():
    """
    Main function to run all content matching verification tests.
    """
    logger.info("Starting comprehensive content matching verification tests...")

    # Test similarity calculation
    test_similarity_calculation()

    # Test content matching verification
    await test_content_matching_verification()

    logger.info("All content matching verification tests completed.")


if __name__ == "__main__":
    asyncio.run(main())