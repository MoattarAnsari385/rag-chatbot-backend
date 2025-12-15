"""
Google Gemini Agent configuration and tools for the RAG agent system.
Handles agent initialization, query processing, and response generation.
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import google.generativeai as genai
import os
from .config import GEMINI_API_KEY, DEFAULT_TEMPERATURE
from .models import RetrievedChunk

logger = logging.getLogger(__name__)

# Configure Google Generative AI
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the model - using an available model
model = genai.GenerativeModel('gemini-2.5-flash')

# Agent configuration
DEFAULT_AGENT_MODEL = "gemini-2.5-flash"  # Using Gemini 2.5 Flash model


def initialize_agent():
    """
    Initialize and configure the Google Gemini agent with necessary tools.

    Returns:
        Configured Gemini agent object with retrieval tool attached
    """
    try:
        logger.info("Initializing Google Gemini agent with retrieval capabilities")

        # Define the configuration that will be used when processing queries
        agent_config = {
            "model": DEFAULT_AGENT_MODEL,
            "temperature": DEFAULT_TEMPERATURE,
            "generation_config": {
                "temperature": DEFAULT_TEMPERATURE,
                "max_output_tokens": 1000
            },
            "system_instruction": """You are a helpful assistant that answers questions based only on the provided context.
            Do not make up information that is not in the provided context.
            If the context doesn't contain enough information to answer the question, say so.
            Always cite the sources of information you use in your response."""
        }

        logger.info("Google Gemini agent initialized successfully")
        return agent_config

    except Exception as e:
        logger.error(f"Error initializing Google Gemini agent: {str(e)}")
        raise e


async def query_agent(agent: Dict, query: str, retrieved_chunks: List[Dict], temperature: float = DEFAULT_TEMPERATURE) -> str:
    """
    Process a query with the agent using the provided context chunks.

    Args:
        agent: The agent configuration
        query: The user's query text
        retrieved_chunks: List of retrieved context chunks with content and metadata
        temperature: Temperature parameter for response generation

    Returns:
        The agent's response as a string
    """
    try:
        logger.info(f"Processing query with agent: '{query[:50]}...'")
        start_time = time.time()

        # Construct the context from retrieved chunks
        context_str = ""
        sources_info = []

        for i, chunk in enumerate(retrieved_chunks):
            content = chunk.get('content', '')
            metadata = chunk.get('metadata', {})
            source_url = metadata.get('url', 'Unknown source')

            context_str += f"\n\nContext Chunk {i+1}:\n{content}\nSource: {source_url}\n"
            sources_info.append({
                "chunk_id": chunk.get('id', f'chunk_{i}'),
                "url": source_url,
                "content_preview": content[:100] + "..." if len(content) > 100 else content
            })

        # Construct the prompt with instructions and context
        prompt = f"""You are a helpful assistant for the Physical AI & Humanoid Robotics textbook.
        Answer the user's query based ONLY on the provided context.
        Do not use any prior knowledge or make up information.
        If the context doesn't contain enough information to answer the query, clearly state this.
        Always cite the sources of information you use in your response.

        CONTEXT:
        {context_str}

        QUERY:
        {query}

        RESPONSE:"""

        # Generate content using Gemini
        generation_config = agent.get('generation_config', {
            "temperature": temperature,
            "max_output_tokens": 1000
        })

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": generation_config["temperature"],
                "max_output_tokens": generation_config["max_output_tokens"]
            }
        )

        # Extract the agent's response
        agent_response = response.text

        processing_time = time.time() - start_time
        logger.info(f"Agent processed query in {processing_time:.2f}s")

        return agent_response

    except Exception as e:
        logger.error(f"Error processing query with agent: {str(e)}")
        raise e


def validate_agent_response(response: str, context_chunks: List[Dict]) -> Dict[str, Any]:
    """
    Validate that the agent response is grounded in the provided context.

    Args:
        response: The agent's generated response
        context_chunks: List of context chunks used to generate the response

    Returns:
        Dictionary with validation results including grounding check and confidence score
    """
    try:
        logger.info("Validating agent response grounding in context")

        # Extract context content to check against
        context_content = " ".join([chunk.get('content', '') for chunk in context_chunks])
        context_lower = context_content.lower()

        # Check if the response contains information from the context
        response_lower = response.lower()

        # Simple validation: check if key terms from context appear in response
        # This is a basic check - in production, you'd want more sophisticated semantic validation
        context_terms = set(context_lower.split()[:50])  # Take first 50 terms as representative
        response_terms = set(response_lower.split())

        matching_terms = context_terms.intersection(response_terms)
        overlap_ratio = len(matching_terms) / len(context_terms) if context_terms else 0

        # Check if response contains phrases that indicate uncertainty (meaning it couldn't find info in context)
        uncertainty_indicators = ['i don\'t know', 'not in context', 'no information', 'not mentioned', 'not specified']
        has_uncertainty = any(indicator in response_lower for indicator in uncertainty_indicators)

        validation_result = {
            "is_grounded": overlap_ratio > 0.1 or has_uncertainty,  # If overlap is low but agent indicates uncertainty, it's still valid
            "context_overlap_ratio": overlap_ratio,
            "has_uncertainty_indicators": has_uncertainty,
            "confidence_score": min(overlap_ratio * 2, 1.0),  # Scale up but cap at 1.0
            "validation_notes": f"Context-response overlap: {overlap_ratio:.2%}"
        }

        logger.info(f"Response validation: {validation_result['validation_notes']}, grounded: {validation_result['is_grounded']}")
        return validation_result

    except Exception as e:
        logger.error(f"Error validating agent response: {str(e)}")
        raise e