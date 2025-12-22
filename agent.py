import os
import cohere

from qdrant_client import QdrantClient
from agents import (
    Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, 
    enable_verbose_stdout_logging, set_tracing_disabled, function_tool
    )

from dotenv import load_dotenv, find_dotenv

enable_verbose_stdout_logging()
set_tracing_disabled(disabled=True)

load_dotenv(find_dotenv())

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

cohere_client = cohere.Client(api_key=COHERE_API_KEY)

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client,  
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

@function_tool
def search_chemistry(query: str) -> str:
    """
    Search for information about chemistry topics from the Chemistry-10th textbook.
    
    Args:
        query: The chemistry topic or question to search for
        
    Returns:
        Relevant chemistry information from the textbook with page references
    """
    results = retrieve(query, top_k=5)
    
    if not results:
        return "No relevant information found in the chemistry textbook."
    
    # Format results with page numbers
    formatted_results = []
    for i, result in enumerate(results, 1):
        page = result.get("metadata", {}).get("page", "N/A")
        text = result.get("text", "")
        score = result.get("score", 0)
        formatted_results.append(f"[Source {i} - Page {page}, Relevance: {score:.2f}]\n{text}")
    
    return "\n\n".join(formatted_results)

agent = Agent(
    name="Chemistry Tutor Agent",
    instructions="""You are an expert Chemistry tutor specializing in 10th-grade chemistry concepts. Your role is to help students understand chemistry topics clearly and accurately.

**Your Responsibilities:**
1. **Use the search_chemistry tool** to find relevant information from the Chemistry-10th textbook before answering any question
2. **Provide accurate, textbook-based answers** - Always base your responses on the retrieved information
3. **Cite your sources** - Always mention the page numbers from which you're drawing information
4. **Explain step-by-step** - Break down complex concepts into simpler, digestible parts
5. **Use examples** - When appropriate, provide real-world examples to illustrate concepts
6. **Check understanding** - Encourage students to ask follow-up questions if anything is unclear

**Response Format:**
- Start with a clear, direct answer to the question
- Provide detailed explanation using information from the textbook
- Include relevant examples or applications when helpful
- Always cite page numbers at the end (e.g., "Source: Pages 12, 15")
- If the information is not in the textbook, clearly state that

**Tone:**
- Be friendly, patient, and encouraging
- Use clear, simple language appropriate for 10th-grade students
- Avoid overly technical jargon unless necessary, and explain it when used

**Important:**
- Always call the search_chemistry tool first before answering
- Never make up information - only use what's retrieved from the textbook
- If you're unsure or the information isn't available, admit it honestly""",
    model=model,
    tools=[search_chemistry],
)

result = Runner.run_sync(
    agent,
    "What is the Chemical Equilibrium?"
)

print(result.final_output)