import os
import requests
import time
import logging
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import re


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')  # Optional for local instances
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')  # Optional for local instances

# Initialize clients
co = cohere.Client(COHERE_API_KEY)

# Initialize Qdrant client (will be configured in create_collection)
qdrant_client = None

if QDRANT_URL:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=10
    )
else:
    # Connect to local Qdrant instance
    qdrant_client = QdrantClient(host="localhost", port=6333)

# Constants
CHUNK_SIZE = 1000  # Default chunk size in characters
COHERE_MODEL = "embed-multilingual-v3.0"
VECTOR_SIZE = 1024  # Cohere's multilingual model returns 1024-dimensional vectors
COLLECTION_NAME = "rag_embedding"


def retry_api_call(func, max_retries=3, delay=1, backoff=2):
    """
    Decorator to retry API calls with exponential backoff
    """
    def wrapper(*args, **kwargs):
        retries = 0
        current_delay = delay

        while retries < max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                if retries == max_retries:
                    logger.error(f"Failed after {max_retries} retries: {str(e)}")
                    raise e

                logger.warning(f"Attempt {retries} failed: {str(e)}. Retrying in {current_delay} seconds...")
                time.sleep(current_delay)
                current_delay *= backoff

    return wrapper


def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


@retry_api_call
def get_all_urls(base_url: str) -> List[str]:
    """
    Discover and return all accessible URLs from a Docusaurus site
    First tries to get URLs from sitemap.xml, then falls back to crawling
    """
    urls = set()

    # Try to get URLs from sitemap first
    sitemap_url = f"{base_url.rstrip('/')}/sitemap.xml"
    try:
        logger.info(f"Attempting to fetch sitemap from {sitemap_url}")
        response = requests.get(sitemap_url, timeout=10)
        if response.status_code == 200:
            from xml.etree import ElementTree as ET
            root = ET.fromstring(response.content)

            # Handle both sitemapindex and urlset
            if root.tag.endswith('sitemapindex'):
                # This is a sitemap index, need to fetch individual sitemaps
                for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                    sitemap_loc = sitemap.text
                    logger.info(f"Fetching individual sitemap: {sitemap_loc}")
                    sitemap_resp = requests.get(sitemap_loc, timeout=10)
                    if sitemap_resp.status_code == 200:
                        sitemap_root = ET.fromstring(sitemap_resp.content)
                        for url_elem in sitemap_root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                            url = url_elem.text
                            if url and base_url in url:
                                urls.add(url)
            else:
                # This is a regular sitemap with URLs
                for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                    url = url_elem.text
                    if url and base_url in url:
                        urls.add(url)
            logger.info(f"Found {len(urls)} URLs from sitemap")
        else:
            logger.info("Sitemap not available, attempting to crawl the site")
    except Exception as e:
        logger.warning(f"Sitemap processing failed: {str(e)}, falling back to crawling")

    # If sitemap approach didn't work or returned no URLs, try basic crawling
    if not urls:
        logger.info(f"Starting to crawl {base_url}")
        try:
            response = requests.get(base_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Find all links on the page
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(base_url, href)

                    # Only add URLs from the same domain
                    if urlparse(full_url).netloc == urlparse(base_url).netloc:
                        if full_url.startswith(('http://', 'https://')):
                            urls.add(full_url)

                logger.info(f"Found {len(urls)} URLs through crawling")
        except Exception as e:
            logger.error(f"Crawling failed: {str(e)}")

    return list(urls)


@retry_api_call
def extract_text_from_url(url: str) -> Dict[str, str]:
    """
    Extract clean text content from a single URL
    """
    try:
        logger.info(f"Extracting text from {url}")
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()

        # Try to find the main content area in Docusaurus sites
        # Common selectors for Docusaurus content containers
        content_selectors = [
            'article',  # Most common content container
            '.main-wrapper',  # Docusaurus main content wrapper
            '.container',  # General container
            '.markdown',  # Docusaurus markdown content
            '.theme-doc-markdown',  # Docusaurus specific class
            '.docItemContainer',  # Docusaurus doc item container
            '.docs-wrapper',  # Docusaurus docs wrapper
            'main',  # HTML5 main element
        ]

        content_element = None
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                break

        # If no specific content container found, use body
        if not content_element:
            content_element = soup.find('body')

        # Get text content and clean it up
        if content_element:
            text_content = content_element.get_text(separator=' ', strip=True)
            # Clean up excessive whitespace
            text_content = re.sub(r'\s+', ' ', text_content)
        else:
            # Fallback: get all text from body
            text_content = soup.get_text(separator=' ', strip=True)
            text_content = re.sub(r'\s+', ' ', text_content)

        # Get the page title
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else urlparse(url).path.split('/')[-1] or 'Untitled'

        return {
            'title': title,
            'content': text_content,
            'url': url
        }
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {str(e)}")
        return {
            'title': urlparse(url).path.split('/')[-1] or 'Error',
            'content': '',
            'url': url
        }


def chunk_text(content: str, chunk_size: int = CHUNK_SIZE) -> List[Dict[str, str]]:
    """
    Split content into manageable chunks for embedding
    """
    if not content:
        return []

    chunks = []
    start = 0

    while start < len(content):
        end = start + chunk_size

        # Try to break at sentence boundaries if possible
        if end < len(content):
            # Look for sentence endings near the chunk boundary
            sentence_end = -1
            for delimiter in ['.', '!', '?', '\n']:
                pos = content.rfind(delimiter, start, end)
                if pos > sentence_end:
                    sentence_end = pos

            # If found a sentence boundary, use it; otherwise, use the chunk boundary
            if sentence_end > start:
                end = sentence_end + 1
            else:
                # If no sentence boundary found, try to break at word boundary
                word_end = content.rfind(' ', start, end)
                if word_end > start:
                    end = word_end

        chunk_text = content[start:end].strip()
        if chunk_text:  # Only add non-empty chunks
            chunks.append({
                'text': chunk_text,
                'start_pos': start,
                'end_pos': end
            })

        start = end

    logger.info(f"Content chunked into {len(chunks)} pieces")
    return chunks


@retry_api_call
def embed(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks using Cohere
    """
    if not texts:
        return []

    logger.info(f"Generating embeddings for {len(texts)} text chunks")

    try:
        response = co.embed(
            texts=texts,
            model=COHERE_MODEL,
            input_type="search_document"  # Appropriate for document search
        )

        embeddings = response.embeddings
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise e


def create_collection(collection_name: str = COLLECTION_NAME):
    """
    Create a vector collection in Qdrant if it doesn't exist
    """
    try:
        # Check if collection already exists
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if collection_name in collection_names:
            logger.info(f"Collection '{collection_name}' already exists")
            return

        # Create the collection
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,  # Cohere's embedding size
                distance=models.Distance.COSINE  # Cosine distance for embeddings
            )
        )

        logger.info(f"Created collection '{collection_name}' with {VECTOR_SIZE}-dimension vectors")
    except Exception as e:
        logger.error(f"Error creating collection '{collection_name}': {str(e)}")
        raise e


def save_chunk_to_qdrant(chunk_data: Dict, embedding: List[float], collection_name: str = COLLECTION_NAME):
    """
    Save a chunk with its embedding to Qdrant
    """
    try:
        # Generate a unique ID for this record
        import uuid
        record_id = str(uuid.uuid4())

        # Prepare the payload with metadata
        payload = {
            'url': chunk_data.get('url', ''),
            'title': chunk_data.get('title', ''),
            'text': chunk_data.get('text', ''),
            'start_pos': chunk_data.get('start_pos', 0),
            'end_pos': chunk_data.get('end_pos', 0),
            'created_at': time.time()
        }

        # Upsert the record to Qdrant
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=record_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )

        logger.info(f"Saved chunk to Qdrant with ID: {record_id}")
        return record_id
    except Exception as e:
        logger.error(f"Error saving chunk to Qdrant: {str(e)}")
        raise e


def main():
    """
    Main execution function that orchestrates the complete pipeline
    """
    logger.info("Starting embedding pipeline for https://physical-ai-and-humanoid-robotics-t-seven.vercel.app/")

    # Create the collection in Qdrant
    logger.info("Creating Qdrant collection...")
    create_collection()

    # Get all URLs from the target site
    logger.info("Discovering URLs from the target site...")
    target_url = "https://physical-ai-and-humanoid-robotics-t-seven.vercel.app/"
    urls = get_all_urls(target_url)

    logger.info(f"Found {len(urls)} URLs to process")

    total_chunks_processed = 0

    for i, url in enumerate(urls):
        logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")

        try:
            # Extract text from the URL
            page_data = extract_text_from_url(url)

            if not page_data['content']:
                logger.warning(f"No content extracted from {url}, skipping...")
                continue

            # Chunk the content
            chunks = chunk_text(page_data['content'])

            if not chunks:
                logger.warning(f"No chunks created from {url}, skipping...")
                continue

            # Prepare text chunks for embedding
            text_list = [chunk['text'] for chunk in chunks]

            # Generate embeddings
            embeddings = embed(text_list)

            # Save each chunk with its embedding to Qdrant
            for chunk, embedding in zip(chunks, embeddings):
                # Add URL and title to chunk data for metadata
                chunk_with_metadata = {
                    'url': page_data['url'],
                    'title': page_data['title'],
                    'text': chunk['text'],
                    'start_pos': chunk['start_pos'],
                    'end_pos': chunk['end_pos']
                }

                save_chunk_to_qdrant(chunk_with_metadata, embedding)
                total_chunks_processed += 1

        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            continue  # Continue with the next URL

    logger.info(f"Pipeline completed! Processed {total_chunks_processed} chunks from {len(urls)} URLs")


if __name__ == "__main__":
    main()