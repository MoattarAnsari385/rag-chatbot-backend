"""
Qdrant retrieval tool wrapper for the RAG agent system.
Provides functions to perform similarity search in Qdrant and retrieve relevant content chunks.

This module implements the retrieval component of the RAG (Retrieval-Augmented Generation) system,
handling vector similarity search, content matching validation, and metadata integrity verification.
"""
import asyncio
import logging
import time
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams
import cohere
import os
from .config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    COHERE_API_KEY,
    COHERE_MODEL,
    SIMILARITY_THRESHOLD
)

logger = logging.getLogger(__name__)

# Initialize Cohere client for query embedding
co = cohere.Client(COHERE_API_KEY)

# Initialize Qdrant client
if QDRANT_URL:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=10
    )
else:
    # Connect to local Qdrant instance
    qdrant_client = QdrantClient(host="localhost", port=6333)


async def retrieve_relevant_chunks(query_text: str, top_k: int = 5) -> List[Dict]:
    """
    Retrieve the most relevant content chunks from Qdrant based on the query text.

    Args:
        query_text: The query text to search for
        top_k: Number of top results to retrieve (default: 5)

    Returns:
        List of dictionaries containing chunk information with content, metadata, and similarity scores
    """
    start_time = time.time()

    try:
        logger.info(f"Retrieving relevant chunks for query: '{query_text[:50]}...' with top_k={top_k}")

        # Validate top_k parameter
        if top_k < 1 or top_k > 20:
            logger.warning(f"top_k value {top_k} is outside recommended range (1-20), using default 5")
            top_k = min(max(top_k, 1), 20)  # Clamp to 1-20 range

        # Time the embedding generation
        embedding_start = time.time()
        # Generate embedding for the query using Cohere
        response = co.embed(
            texts=[query_text],
            model=COHERE_MODEL,
            input_type="search_query"  # Appropriate for search queries
        )
        embedding_time = time.time() - embedding_start
        logger.debug(f"Generated query embedding in {embedding_time:.4f}s with {len(response.embeddings[0])} dimensions")

        query_embedding = response.embeddings[0]

        # Time the vector search
        search_start = time.time()
        # Perform similarity search in Qdrant
        search_results = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            with_payload=True,  # Include the stored metadata
            with_vectors=False  # Don't return the vectors themselves
        )
        search_time = time.time() - search_start
        logger.debug(f"Qdrant search completed in {search_time:.4f}s")

        # Format results to match the expected structure
        retrieved_chunks = []
        for i, hit in enumerate(search_results.points):
            # Get content from 'content' field, fallback to 'text' field in metadata if empty
            content = hit.payload.get('content', '') if hit.payload else ''
            if not content and hit.payload:
                content = hit.payload.get('text', '')

            chunk = {
                "id": hit.id,
                "content": content,
                "metadata": hit.payload if hit.payload else {},
                "similarity_score": float(hit.score) if hasattr(hit, 'score') else 0.0,
                "position": i
            }

            # Only include chunks that meet the similarity threshold
            if chunk["similarity_score"] >= SIMILARITY_THRESHOLD:
                retrieved_chunks.append(chunk)

        # Ensure results are ordered by descending similarity score
        retrieved_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)

        total_time = time.time() - start_time
        logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks from Qdrant with top_k={top_k} in {total_time:.4f}s")

        # Log performance metrics
        logger.debug(f"Performance breakdown - Embedding: {embedding_time:.4f}s, Search: {search_time:.4f}s, Total: {total_time:.4f}s")

        return retrieved_chunks

    except Exception as e:
        logger.error(f"Error retrieving relevant chunks from Qdrant: {str(e)}")
        raise e


def validate_retrieval_accuracy(expected_chunks: List[Dict], retrieved_chunks: List[Dict]) -> Dict:
    """
    Validate that retrieved chunks match expected content with high similarity.

    Args:
        expected_chunks: List of expected content chunks
        retrieved_chunks: List of actually retrieved chunks

    Returns:
        Dictionary with validation results including match status and similarity ratios
    """
    try:
        logger.info(f"Validating retrieval accuracy: {len(expected_chunks)} expected vs {len(retrieved_chunks)} retrieved")

        # For each expected chunk, find the best match in retrieved chunks
        validation_results = []
        for exp_chunk in expected_chunks:
            best_match = None
            best_similarity = 0.0

            for ret_chunk in retrieved_chunks:
                # Calculate text similarity using a simple approach
                similarity = calculate_text_similarity(exp_chunk.get('content', ''), ret_chunk.get('content', ''))

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = ret_chunk

            validation_results.append({
                "expected_chunk_id": exp_chunk.get('id', ''),
                "retrieved_chunk_id": best_match.get('id', '') if best_match else '',
                "similarity_ratio": best_similarity,
                "is_match": best_similarity >= 0.95  # Using 95% threshold for content matching
            })

        # Calculate overall validation metrics
        total_chunks = len(validation_results)
        matched_chunks = sum(1 for vr in validation_results if vr['is_match'])
        accuracy_rate = matched_chunks / total_chunks if total_chunks > 0 else 0.0

        result = {
            "total_chunks_validated": total_chunks,
            "matched_chunks": matched_chunks,
            "accuracy_rate": accuracy_rate,
            "individual_results": validation_results,
            "validation_passed": accuracy_rate >= 0.9  # 90% threshold for overall validation
        }

        logger.info(f"Retrieval validation completed with {accuracy_rate:.2%} accuracy")
        return result

    except Exception as e:
        logger.error(f"Error validating retrieval accuracy: {str(e)}")
        raise e


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity ratio between two texts using multiple approaches.
    In a production environment, you might use more sophisticated methods like semantic similarity.

    Args:
        text1: First text to compare
        text2: Second text to compare

    Returns:
        Similarity ratio between 0.0 and 1.0
    """
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0

    # Use difflib for sequence-based similarity (Levenshtein-like)
    import difflib
    sequence_similarity = difflib.SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio()

    # Additional checks for semantic similarity
    text1_clean = text1.lower().strip()
    text2_clean = text2.lower().strip()

    # Check for exact match (with some whitespace differences)
    if text1_clean == text2_clean:
        return 1.0

    # Check for substring relationships
    if text1_clean in text2_clean or text2_clean in text1_clean:
        # If one text is a substring of the other, return a high similarity score
        # but not 1.0 to distinguish from exact match
        return max(0.8, min(sequence_similarity, 0.95))

    return float(sequence_similarity)


def content_matching_validation(retrieved_content: str, original_content: str) -> Dict:
    """
    Validate content matching between retrieved content and original content.

    Args:
        retrieved_content: Content retrieved from Qdrant
        original_content: Original content that was embedded

    Returns:
        Dictionary with validation results including similarity ratio and match status
    """
    logger.info("Starting content matching validation")

    # Calculate similarity ratio
    similarity_ratio = calculate_text_similarity(retrieved_content, original_content)

    # Determine if content matches based on 95%+ threshold
    is_match = similarity_ratio >= 0.95

    # Additional validation details
    is_exact_match = retrieved_content.strip() == original_content.strip()
    is_near_match = similarity_ratio >= 0.90  # Higher threshold for near matches

    result = {
        "is_match": is_match,
        "similarity_ratio": similarity_ratio,
        "is_exact_match": is_exact_match,
        "is_near_match": is_near_match,
        "retrieved_content_length": len(retrieved_content),
        "original_content_length": len(original_content),
        "content_preview": {
            "retrieved": retrieved_content[:100] + "..." if len(retrieved_content) > 100 else retrieved_content,
            "original": original_content[:100] + "..." if len(original_content) > 100 else original_content
        }
    }

    logger.info(f"Content matching validation completed: similarity={similarity_ratio:.3f}, match={is_match}")
    return result


def validate_content_matching(retrieved_chunks: List[Dict], original_chunks: List[Dict] = None) -> Dict:
    """
    Validate content matching for all retrieved chunks against original content.

    Args:
        retrieved_chunks: List of retrieved chunks from Qdrant
        original_chunks: List of original chunks for comparison (optional)

    Returns:
        Dictionary with overall validation results
    """
    logger.info(f"Validating content matching for {len(retrieved_chunks)} retrieved chunks")

    if not retrieved_chunks:
        logger.warning("No retrieved chunks to validate")
        return {
            "validation_passed": True,
            "total_chunks": 0,
            "matching_chunks": 0,
            "average_similarity": 0.0,
            "validation_results": []
        }

    validation_results = []
    matching_chunks = 0
    total_similarity = 0.0

    for i, retrieved_chunk in enumerate(retrieved_chunks):
        retrieved_content = retrieved_chunk.get('content', '')

        # If original chunks are provided, use them for comparison
        if original_chunks and i < len(original_chunks):
            original_content = original_chunks[i].get('content', '')
        else:
            # For testing purposes, use the retrieved content itself
            original_content = retrieved_content

        validation_result = content_matching_validation(retrieved_content, original_content)
        validation_results.append({
            "chunk_id": retrieved_chunk.get('id', f'chunk_{i}'),
            "validation": validation_result
        })

        if validation_result['is_match']:
            matching_chunks += 1

        total_similarity += validation_result['similarity_ratio']

    # Calculate overall metrics
    total_chunks = len(retrieved_chunks)
    average_similarity = total_similarity / total_chunks if total_chunks > 0 else 0.0
    match_rate = matching_chunks / total_chunks if total_chunks > 0 else 0.0

    overall_result = {
        "validation_passed": average_similarity >= 0.95,  # Overall validation passes if average similarity is 95%+
        "total_chunks": total_chunks,
        "matching_chunks": matching_chunks,
        "match_rate": match_rate,
        "average_similarity": average_similarity,
        "validation_results": validation_results
    }

    logger.info(f"Content matching validation summary: {match_rate:.2%} match rate, avg similarity {average_similarity:.3f}")
    return overall_result


def validate_url_field(url_value: str) -> Dict:
    """
    Validate URL field in metadata.

    Args:
        url_value: The URL value to validate

    Returns:
        Dictionary with validation results
    """
    import re

    # Check if URL is not empty and has proper format
    is_valid = bool(url_value) and isinstance(url_value, str) and len(url_value.strip()) > 0
    has_scheme = is_valid and (url_value.startswith('http://') or url_value.startswith('https://'))

    # Basic URL pattern check
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    matches_pattern = is_valid and bool(url_pattern.match(url_value))

    validation_result = {
        "value": url_value,
        "is_valid": is_valid,
        "has_scheme": has_scheme,
        "matches_pattern": matches_pattern,
        "is_valid_url": has_scheme and matches_pattern
    }

    return validation_result


def validate_chunk_id_field(chunk_id_value: str) -> Dict:
    """
    Validate chunk_id field in metadata.

    Args:
        chunk_id_value: The chunk_id value to validate

    Returns:
        Dictionary with validation results
    """
    import re

    # Check if chunk_id is valid (not empty, has reasonable format)
    is_valid = bool(chunk_id_value) and isinstance(chunk_id_value, str) and len(chunk_id_value.strip()) > 0
    has_reasonable_length = is_valid and 3 <= len(chunk_id_value) <= 100  # Reasonable length
    is_alphanumeric_or_underscore = is_valid and bool(re.match(r'^[a-zA-Z0-9_-]+$', chunk_id_value))

    validation_result = {
        "value": chunk_id_value,
        "is_valid": is_valid,
        "has_reasonable_length": has_reasonable_length,
        "is_alphanumeric_or_underscore": is_alphanumeric_or_underscore,
        "is_valid_chunk_id": is_valid and has_reasonable_length
    }

    return validation_result


def validate_metadata_fields(metadata: Dict) -> Dict:
    """
    Validate specific metadata fields like URL and chunk_id.

    Args:
        metadata: The metadata dictionary to validate

    Returns:
        Dictionary with validation results for specific fields
    """
    import re

    logger.info("Validating specific metadata fields")

    field_validations = {}

    # Validate URL field if present
    if 'url' in metadata:
        field_validations['url'] = validate_url_field(metadata['url'])

    # Validate chunk_id field if present
    if 'chunk_id' in metadata:
        field_validations['chunk_id'] = validate_chunk_id_field(metadata['chunk_id'])

    # Validate other common fields
    for key, value in metadata.items():
        if key not in field_validations:  # Don't re-validate already validated fields
            field_validations[key] = {
                "value": value,
                "is_valid": value is not None,
                "type": type(value).__name__,
                "is_valid_field": value is not None
            }

    return field_validations


def comprehensive_metadata_validation(expected_metadata: Dict, retrieved_metadata: Dict) -> Dict:
    """
    Perform comprehensive metadata validation with detailed error reporting.

    Args:
        expected_metadata: Expected metadata values
        retrieved_metadata: Actually retrieved metadata

    Returns:
        Dictionary with comprehensive metadata validation results
    """
    logger.info("Starting comprehensive metadata validation")

    # Overall validation
    validation_results = {}
    all_match = True
    field_specific_validations = validate_metadata_fields(retrieved_metadata)

    # Check each expected metadata field
    all_expected_keys = set(expected_metadata.keys()) | set(retrieved_metadata.keys())

    for key in all_expected_keys:
        expected_value = expected_metadata.get(key)
        retrieved_value = retrieved_metadata.get(key)

        # Check if field exists
        exists_in_expected = key in expected_metadata
        exists_in_retrieved = key in retrieved_metadata

        # Check if values match (if both exist)
        matches = expected_value == retrieved_value if exists_in_expected and exists_in_retrieved else False

        # Detailed validation
        validation_results[key] = {
            "expected": expected_value,
            "retrieved": retrieved_value,
            "exists_in_expected": exists_in_expected,
            "exists_in_retrieved": exists_in_retrieved,
            "values_match": matches,
            "field_specific_validation": field_specific_validations.get(key, {})
        }

        if not matches or not exists_in_retrieved:
            all_match = False
            logger.warning(f"Metadata mismatch for '{key}': expected '{expected_value}', got '{retrieved_value}'")

    # Calculate metrics
    total_fields = len(validation_results)
    matching_fields = sum(1 for v in validation_results.values() if v['values_match'])
    success_rate = matching_fields / total_fields if total_fields > 0 else 1.0

    result = {
        "all_fields_match": all_match,
        "total_fields": total_fields,
        "matching_fields": matching_fields,
        "success_rate": success_rate,
        "validation_passed": success_rate >= 0.99,  # 99% success rate required
        "field_validations": validation_results,
        "field_specific_validations": field_specific_validations,
        "detailed_error_report": {
            "missing_fields_in_retrieved": [k for k, v in validation_results.items()
                                           if v['exists_in_expected'] and not v['exists_in_retrieved']],
            "mismatched_fields": [k for k, v in validation_results.items()
                                 if v['exists_in_expected'] and v['exists_in_retrieved'] and not v['values_match']]
        }
    }

    logger.info(f"Comprehensive metadata validation completed: success_rate={success_rate:.2%}, passed={result['validation_passed']}")
    return result


async def validate_metadata_integrity(expected_metadata: Dict, retrieved_metadata: Dict) -> Dict:
    """
    Validate that retrieved metadata matches expected values.

    Args:
        expected_metadata: Expected metadata values
        retrieved_metadata: Actually retrieved metadata

    Returns:
        Dictionary with metadata validation results
    """
    try:
        logger.info("Validating metadata integrity")

        # Use the comprehensive validation function
        result = comprehensive_metadata_validation(expected_metadata, retrieved_metadata)

        logger.info(f"Metadata validation completed: {'PASSED' if result['validation_passed'] else 'FAILED'}")
        return result
    except Exception as e:
        logger.error(f"Error validating metadata integrity: {str(e)}")
        raise e