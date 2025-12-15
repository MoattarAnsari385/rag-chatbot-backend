"""
Test script for Top-K Retrieval Accuracy in the RAG agent system.
This script tests the retrieval accuracy with sample queries about ROS2 concepts.
"""
import asyncio
import logging
import sys
import os
from typing import List, Dict

# Add the backend directory to the path to import rag_agent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_agent.retrieval_tool import retrieve_relevant_chunks, calculate_text_similarity
from rag_agent.config import QDRANT_COLLECTION_NAME

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_ros2_retrieval_accuracy():
    """
    Test retrieval accuracy with sample query about ROS2 concepts.
    """
    logger.info("Starting ROS2 retrieval accuracy test...")

    # Sample queries about ROS2 concepts
    test_queries = [
        "What are ROS2 concepts?",
        "Explain ROS2 nodes and topics",
        "What is ROS2 DDS?",
        "How does ROS2 communication work?",
        "What are ROS2 packages and workspaces?"
    ]

    # Test each query
    for i, query in enumerate(test_queries):
        logger.info(f"Testing query {i+1}: '{query}'")

        try:
            # Retrieve relevant chunks with different top_k values
            for top_k in [3, 5, 10]:
                logger.info(f"  Testing with top_k={top_k}")

                retrieved_chunks = await retrieve_relevant_chunks(query, top_k=top_k)

                logger.info(f"  Retrieved {len(retrieved_chunks)} chunks (top_k={top_k})")

                # Validate that we have results
                if len(retrieved_chunks) > 0:
                    # Check that results are ordered by descending similarity score
                    scores = [chunk['similarity_score'] for chunk in retrieved_chunks]
                    is_ordered = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

                    logger.info(f"  Results ordered by similarity: {is_ordered}")
                    logger.info(f"  Top similarity scores: {[f'{score:.3f}' for score in scores[:5]]}")

                    # Show first few results
                    for j, chunk in enumerate(retrieved_chunks[:3]):
                        content_preview = chunk['content'][:100] + "..." if len(chunk['content']) > 100 else chunk['content']
                        logger.info(f"    Result {j+1}: Score={chunk['similarity_score']:.3f}, Content='{content_preview}'")
                else:
                    logger.warning(f"  No results retrieved for query: {query}")

        except Exception as e:
            logger.error(f"  Error testing query '{query}': {str(e)}")
            continue

    logger.info("ROS2 retrieval accuracy test completed.")


def validate_similarity_ordering(retrieved_chunks: List[Dict]) -> bool:
    """
    Validate that results are returned with descending similarity scores.

    Args:
        retrieved_chunks: List of retrieved chunks with similarity scores

    Returns:
        True if chunks are ordered by descending similarity score, False otherwise
    """
    if len(retrieved_chunks) <= 1:
        return True

    scores = [chunk['similarity_score'] for chunk in retrieved_chunks]
    is_descending = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    return is_descending


async def test_similarity_score_validation():
    """
    Test that results are returned with descending similarity scores.
    """
    logger.info("Starting similarity score validation test...")

    # Test query
    test_query = "ROS2 architecture and design patterns"

    try:
        # Retrieve chunks
        retrieved_chunks = await retrieve_relevant_chunks(test_query, top_k=5)

        # Validate ordering
        is_ordered = validate_similarity_ordering(retrieved_chunks)

        logger.info(f"Query: '{test_query}'")
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        logger.info(f"Results ordered by descending similarity: {is_ordered}")

        if is_ordered and len(retrieved_chunks) > 0:
            scores = [chunk['similarity_score'] for chunk in retrieved_chunks]
            logger.info(f"Similarity scores: {[f'{score:.3f}' for score in scores]}")
        else:
            logger.warning("Results are not properly ordered by similarity")

        return is_ordered

    except Exception as e:
        logger.error(f"Error in similarity score validation: {str(e)}")
        return False


async def main():
    """
    Main function to run all retrieval accuracy tests.
    """
    logger.info("Starting comprehensive retrieval accuracy tests...")

    # Test 1: ROS2 retrieval accuracy
    await test_ros2_retrieval_accuracy()

    # Test 2: Similarity score validation
    is_ordered = await test_similarity_score_validation()

    logger.info("All retrieval accuracy tests completed.")
    logger.info(f"Similarity ordering validation: {'PASSED' if is_ordered else 'FAILED'}")


if __name__ == "__main__":
    asyncio.run(main())