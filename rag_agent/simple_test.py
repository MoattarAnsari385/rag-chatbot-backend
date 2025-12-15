"""
Simple test to verify the end-to-end pipeline works with the available Gemini API quota.
"""
import asyncio
import time
from typing import List

# Add the backend directory to the path to import rag_agent modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_agent.pipeline_test import PipelineTestConfig, run_pipeline_tests, log_retrieval_correctness


async def main():
    """
    Main function to run a simple end-to-end pipeline test.
    """
    print("Starting simple end-to-end pipeline test...")

    # Define minimal test configuration to avoid rate limits
    test_config = PipelineTestConfig(
        queries=[
            "What are ROS2 concepts?"
        ],
        top_k_values=[3],  # Just one top_k value
        temperature_values=[0.7]
    )

    # Run pipeline tests
    results = await run_pipeline_tests(test_config)

    # Log retrieval correctness and errors
    log_retrieval_correctness(results)

    print("Simple pipeline test completed.")


if __name__ == "__main__":
    asyncio.run(main())