"""
Pydantic models for request/response schemas based on data-model.md
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    """
    Model for the query request body
    """
    query: str = Field(..., description="The user's question or query text", min_length=1)
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top results to retrieve")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature parameter for agent response")

    @field_validator('query')
    @classmethod
    def query_not_empty_or_whitespace(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty or only whitespace')
        return v.strip()


class RetrievedChunk(BaseModel):
    """
    Model for a single retrieved context chunk
    """
    id: str = Field(..., description="Unique identifier for the chunk in the vector database")
    content: str = Field(..., description="The actual text content of the chunk", min_length=1)
    metadata: Dict = Field(default_factory=dict, description="Associated metadata (URL, source document, position, etc.)")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score between query and chunk (0.0-1.0)")
    position: int = Field(..., ge=0, description="Position in the ranked results (0-indexed)")


class TokenUsage(BaseModel):
    """
    Model for token usage information
    """
    input_tokens: int = Field(..., ge=0, description="Number of input tokens used")
    output_tokens: int = Field(..., ge=0, description="Number of output tokens generated")
    total_tokens: int = Field(..., ge=0, description="Total tokens used")


class AgentResponse(BaseModel):
    """
    Model for the agent's response
    """
    answer: str = Field(..., description="The agent's answer to the user's query", min_length=1)
    sources: List[RetrievedChunk] = Field(default_factory=list, description="List of source chunks used to generate the answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the response (0.0-1.0)")
    processing_time: float = Field(..., ge=0.0, description="Time taken to process the request in seconds")
    tokens_used: TokenUsage = Field(..., description="Token usage information")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the response was generated")


class ErrorResponse(BaseModel):
    """
    Model for error responses
    """
    type: str = Field(..., description="Error type (e.g., 'retrieval_error', 'api_unavailable')")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict] = Field(None, description="Additional error details")


class QueryResponse(BaseModel):
    """
    Model for the API response
    """
    success: bool = Field(..., description="Whether the request was processed successfully")
    data: Optional[AgentResponse] = Field(None, description="The agent response object (if successful)")
    error: Optional[ErrorResponse] = Field(None, description="Error information if the request failed")
    request_id: str = Field(..., description="Unique identifier for the request for tracking purposes")

    @model_validator(mode='after')
    def validate_data_and_error(self):
        # Ensure either data or error is present, but not both
        if self.data is not None and self.error is not None:
            raise ValueError('Both data and error cannot be present simultaneously')
        return self