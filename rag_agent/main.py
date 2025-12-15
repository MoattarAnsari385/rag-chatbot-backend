"""
FastAPI application for the RAG agent system.
Exposes a query endpoint that accepts user queries and returns structured responses.
"""
import os
import time
import uuid
import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import ValidationError
import asyncio
from fastapi.middleware.cors import CORSMiddleware

from .config import validate_config
from .models import QueryRequest, QueryResponse, AgentResponse, RetrievedChunk, TokenUsage, ErrorResponse
from .agent import initialize_agent, query_agent
from .retrieval_tool import retrieve_relevant_chunks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Agent API",
    description="API for querying the RAG agent system",
    version="1.0.0"
)

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:3003",
        "http://localhost:3004", "http://localhost:3005", "http://localhost:3006", "http://localhost:3007",
        "https://*.vercel.app",  # Allow requests from Vercel deployments
        "https://physical-ai-and-humanoid-robotics-t-xi.vercel.app"  # Your specific Vercel deployment
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance (initialized on startup)
agent_instance = None

@app.on_event("startup")
async def startup_event():
    """Initialize the agent when the application starts"""
    global agent_instance

    # Validate configuration
    if not validate_config():
        logger.error("Configuration validation failed. Exiting.")
        raise RuntimeError("Invalid configuration")

    # Perform comprehensive startup checks
    from .config import print_startup_report
    startup_ok = print_startup_report()
    if not startup_ok:
        logger.warning("Some startup checks failed, but continuing startup...")

    logger.info("Starting up RAG Agent API...")
    try:
        # Initialize the agent
        agent_instance = initialize_agent()
        logger.info("RAG Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Agent: {str(e)}")
        raise


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Query endpoint that accepts user queries and returns structured responses.

    Args:
        request: QueryRequest containing the user's query and optional parameters

    Returns:
        QueryResponse with the agent's answer or error information
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    logger.info(f"Received query request {request_id}: '{request.query[:50]}...'")

    try:
        # Check if the query is a greeting
        greeting_keywords = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "good night"]
        query_lower = request.query.lower().strip()

        # Check for greeting-like messages
        is_greeting = any(keyword in query_lower for keyword in greeting_keywords) or query_lower in ["hello", "hi", "hey"]

        if is_greeting:
            # Return a friendly greeting response without using the RAG system
            import random
            greeting_responses = [
                "Hello! I'm your AI assistant for the Physical AI & Humanoid Robotics textbook. How can I help you with your studies today?",
                "Hi there! I'm here to help you with the Physical AI & Humanoid Robotics curriculum. What would you like to learn about?",
                "Greetings! I'm your AI tutor for Physical AI & Humanoid Robotics. Ask me anything about the textbook content!",
                "Hello! I'm ready to help you explore the Physical AI & Humanoid Robotics textbook. What topic would you like to discuss?"
            ]
            greeting_answer = random.choice(greeting_responses)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Create successful response for greeting
            token_usage = TokenUsage(
                input_tokens=len(request.query.split()),  # Approximate
                output_tokens=len(greeting_answer.split()),  # Approximate
                total_tokens=len(request.query.split()) + len(greeting_answer.split())  # Approximate
            )

            agent_response = AgentResponse(
                answer=greeting_answer,
                sources=[],
                confidence=1.0,  # High confidence for greeting responses
                processing_time=processing_time,
                tokens_used=token_usage,
                timestamp=time.time()
            )

            success_response = QueryResponse(
                success=True,
                data=agent_response,
                error=None,
                request_id=request_id
            )

            logger.info(f"Greeting request {request_id} completed successfully in {processing_time:.2f}s")
            return success_response

        # Validate the request parameters
        if not request.query or not request.query.strip():
            error_response = QueryResponse(
                success=False,
                error=ErrorResponse(
                    type="validation_error",
                    message="Query cannot be empty",
                    details={"field": "query"}
                ),
                request_id=request_id
            )
            logger.warning(f"Query request {request_id} failed validation: empty query")
            return error_response

        if request.top_k < 1 or request.top_k > 20:
            error_response = QueryResponse(
                success=False,
                error=ErrorResponse(
                    type="validation_error",
                    message=f"top_k must be between 1 and 20, got {request.top_k}",
                    details={"field": "top_k", "value": request.top_k}
                ),
                request_id=request_id
            )
            logger.warning(f"Query request {request_id} failed validation: invalid top_k")
            return error_response

        # Retrieve relevant chunks from Qdrant
        try:
            retrieved_chunks = await retrieve_relevant_chunks(
                query_text=request.query,
                top_k=request.top_k
            )

            # Filter out chunks with empty content to prevent validation errors
            filtered_chunks = []
            for chunk in retrieved_chunks:
                content = chunk.get('content', '').strip()
                if content:  # Only include chunks with non-empty content
                    filtered_chunks.append(chunk)

            retrieved_chunks = filtered_chunks

        except Exception as e:
            error_response = QueryResponse(
                success=False,
                error=ErrorResponse(
                    type="retrieval_error",
                    message=f"Failed to retrieve relevant content: {str(e)}",
                    details={"error_type": type(e).__name__}
                ),
                request_id=request_id
            )
            logger.error(f"Query request {request_id} retrieval failed: {str(e)}")
            return error_response

        # Process the query with the agent
        try:
            agent_answer = await query_agent(
                agent=agent_instance,
                query=request.query,
                retrieved_chunks=retrieved_chunks,
                temperature=request.temperature
            )
        except Exception as e:
            error_response = QueryResponse(
                success=False,
                error=ErrorResponse(
                    type="agent_error",
                    message=f"Failed to process query with agent: {str(e)}",
                    details={"error_type": type(e).__name__}
                ),
                request_id=request_id
            )
            logger.error(f"Query request {request_id} agent processing failed: {str(e)}")
            return error_response

        # Calculate processing time
        processing_time = time.time() - start_time

        # Create successful response
        token_usage = TokenUsage(
            input_tokens=len(request.query.split()),  # Approximate
            output_tokens=len(agent_answer.split()),  # Approximate
            total_tokens=len(request.query.split()) + len(agent_answer.split())  # Approximate
        )

        agent_response = AgentResponse(
            answer=agent_answer,
            sources=retrieved_chunks,
            confidence=0.9,  # Placeholder - in a real implementation, this would be calculated
            processing_time=processing_time,
            tokens_used=token_usage,
            timestamp=time.time()
        )

        success_response = QueryResponse(
            success=True,
            data=agent_response,
            error=None,
            request_id=request_id
        )

        logger.info(f"Query request {request_id} completed successfully in {processing_time:.2f}s")
        return success_response

    except ValidationError as ve:
        error_response = QueryResponse(
            success=False,
            error=ErrorResponse(
                type="validation_error",
                message=f"Request validation failed: {str(ve)}",
                details={"validation_errors": ve.errors()}
            ),
            request_id=request_id
        )
        logger.warning(f"Query request {request_id} validation failed: {str(ve)}")
        return error_response

    except HTTPException:
        # Re-raise FastAPI HTTP exceptions
        raise

    except Exception as e:
        error_response = QueryResponse(
            success=False,
            error=ErrorResponse(
                type="internal_error",
                message=f"Unexpected error processing query: {str(e)}",
                details={"error_type": type(e).__name__}
            ),
            request_id=request_id
        )
        logger.error(f"Query request {request_id} failed with unexpected error: {str(e)}")
        return error_response


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {
        "status": "healthy",
        "service": "RAG Agent API",
        "timestamp": time.time()
    }


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)