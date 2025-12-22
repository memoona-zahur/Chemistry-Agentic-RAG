import os
import cohere
from qdrant_client import QdrantClient
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

cohere_client = cohere.Client(api_key=COHERE_API_KEY)

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

def get_embeddings(text, input_type="search_query"):
    """
    Generate embeddings for a query or document using Cohere API.
    
    Args:
        text (str): The text to embed (can be a query or document)
        input_type (str): Either "search_query" for queries or "search_document" for documents
        
    Returns:
        list: Embedding vector (list of floats)
    """
    response = cohere_client.embed(
        texts=[text],
        model="embed-english-v3.0",
        input_type=input_type
    )
    
    return response.embeddings[0]

def retrieve(query, collection_name="Chemistry", top_k=5):
    """
    Retrieve relevant document chunks from Qdrant based on a query.
    
    Args:
        query (str): The user's question or search query
        collection_name (str): Name of the Qdrant collection to search
        top_k (int): Number of top results to return
        
    Returns:
        list: List of dictionaries containing retrieved chunks with text, metadata, and scores
    """
    # Generate embedding for the query
    query_embedding = get_embeddings(query, input_type="search_query")
    
    # Search Qdrant for similar vectors
    search_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k
    ).points
    
    # Format results
    results = []
    for hit in search_results:
        results.append({
            "text": hit.payload.get("text", ""),
            "metadata": hit.payload.get("metadata", {}),
            "score": hit.score
        })
    
    return results

print(retrieve("What is the Chemical Equilibrium?"))

