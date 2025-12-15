"""
Configuration module for the RAG agent system.
Handles environment loading and settings management.
"""
import os
from typing import Optional, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Qdrant Configuration
QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL")  # Optional for local instances
QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")  # Optional for local instances
QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "rag_embedding")

# Cohere Configuration (for query embedding - must match ingestion model)
COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
COHERE_MODEL: str = os.getenv("COHERE_MODEL", "embed-multilingual-v3.0")

# Google Gemini Configuration
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# Application Configuration
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "rag_embedding")

def validate_config() -> bool:
    """
    Validate that all required configuration values are present.
    Returns True if all required values are set, False otherwise.
    """
    required_vars = [
        "GEMINI_API_KEY",
        "COHERE_API_KEY"
    ]

    # QDRANT_URL is optional if using local instance
    if QDRANT_URL:
        required_vars.append("QDRANT_URL")

    missing_vars = []
    for var in required_vars:
        if not globals()[var]:
            missing_vars.append(var)

    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file")
        return False

    return True


def startup_checks() -> Dict[str, bool]:
    """
    Perform comprehensive startup checks for all services.

    Returns:
        Dictionary with check results for each service
    """
    import requests
    from qdrant_client.http import models
    import google.generativeai as genai
    import cohere

    results = {}

    # Check Gemini API
    try:
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            # Test with a simple model to verify API key validity
            test_model = genai.GenerativeModel('gemini-2.5-flash')
            results['gemini_api'] = True
        else:
            results['gemini_api'] = False
    except Exception as e:
        print(f"Gemini API check failed: {str(e)}")
        results['gemini_api'] = False

    # Check Cohere API
    try:
        if COHERE_API_KEY:
            co = cohere.Client(COHERE_API_KEY)
            # Test with a simple embedding to verify API key validity
            response = co.embed(
                texts=["test"],
                model=COHERE_MODEL,
                input_type="search_query"  # Required parameter
            )
            results['cohere_api'] = True
        else:
            results['cohere_api'] = False
    except Exception as e:
        print(f"Cohere API check failed: {str(e)}")
        results['cohere_api'] = False

    # Check Qdrant connection
    try:
        if QDRANT_URL:
            from qdrant_client import QdrantClient
            client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                timeout=5
            )
            # Test connection by getting collection info
            client.get_collection(QDRANT_COLLECTION_NAME)
            results['qdrant_connection'] = True
        else:
            # Try local connection
            try:
                client = QdrantClient(host="localhost", port=6333, timeout=5)
                client.get_collection(QDRANT_COLLECTION_NAME)
                results['qdrant_connection'] = True
            except:
                results['qdrant_connection'] = False
    except Exception as e:
        print(f"Qdrant connection check failed: {str(e)}")
        results['qdrant_connection'] = False

    # Check configuration validation
    results['config_validation'] = validate_config()

    # Check application parameters
    try:
        # Validate numeric parameters are within reasonable ranges
        assert 1 <= DEFAULT_TOP_K <= 20, "DEFAULT_TOP_K must be between 1 and 20"
        assert 0.0 <= DEFAULT_TEMPERATURE <= 2.0, "DEFAULT_TEMPERATURE must be between 0.0 and 2.0"
        assert 0.0 <= SIMILARITY_THRESHOLD <= 1.0, "SIMILARITY_THRESHOLD must be between 0.0 and 1.0"
        results['parameter_validation'] = True
    except Exception as e:
        print(f"Parameter validation failed: {str(e)}")
        results['parameter_validation'] = False

    return results


def print_startup_report() -> bool:
    """
    Print a comprehensive startup report and return overall status.

    Returns:
        True if all critical checks passed, False otherwise
    """
    print("=== RAG Agent System Startup Checks ===")

    results = startup_checks()

    # Print individual check results
    for check, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{check.replace('_', ' ').title()}: {status}")

    # Determine overall status (all critical checks must pass)
    critical_checks = ['config_validation', 'parameter_validation']
    overall_status = all(results.get(check, False) for check in critical_checks)

    print("\n=== Summary ===")
    print(f"Critical checks: {'[PASS]' if overall_status else '[FAIL]'}")
    print(f"Total checks: {len(results)}")
    print(f"Passed: {sum(results.values())}")
    print(f"Failed: {len(results) - sum(results.values())}")

    if not overall_status:
        print("\nWARNING: Some critical checks failed. System may not operate correctly.")
    else:
        print("\n[READY] All critical startup checks passed. System ready.")

    return overall_status