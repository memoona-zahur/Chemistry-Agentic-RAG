import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import cohere
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

PDF_PATH = "Chemistry-10th.pdf"
COLLECTION_NAME = "Chemistry"

cohere_client = cohere.Client(COHERE_API_KEY)
EMBED_MODEL = "embed-english-v3.0"

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

def extract_text_from_pdf(pdf_path="Chemistry-10th.pdf"):
    """
    Extract text from PDF using PyPDFLoader.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of Document objects, each containing page content and metadata
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # Returns list of Document objects with .page_content and .metadata
    return pages

def chunk_text(documents, chunk_size=1000, chunk_overlap=200):
    """
    Chunk documents into smaller segments using RecursiveCharacterTextSplitter.
    This method intelligently splits on semantic boundaries (paragraphs, sentences, words).
    
    Args:
        documents (list): List of Document objects from PyPDFLoader
        chunk_size (int): Maximum size of each chunk in characters
        chunk_overlap (int): Number of characters to overlap between chunks
        
    Returns:
        list: List of chunked Document objects with preserved metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Split on paragraphs, then sentences, then words
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

def generate_embeddings(chunks, batch_size=96):
    """
    Generate embeddings for text chunks using Cohere API with batch processing.
    
    Args:
        chunks (list): List of Document objects with .page_content
        batch_size (int): Number of chunks to process per API call (Cohere limit: 96)
        
    Returns:
        list: List of embedding vectors (each is a list of floats)
    """
    # Extract text content from Document objects
    texts = [chunk.page_content for chunk in chunks]
    
    all_embeddings = []
    
    # Process in batches to avoid rate limits and memory issues
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            # Generate embeddings using Cohere
            # input_type="search_document" is used for documents to be stored/indexed
            response = cohere_client.embed(
                texts=batch,
                model=EMBED_MODEL,
                input_type="search_document"  # Use "search_query" for query embeddings
            )
            all_embeddings.extend(response.embeddings)
            print(f"  Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        except Exception as e:
            print(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
            raise
    
    return all_embeddings

def store_in_qdrant(chunks, embeddings, collection_name=COLLECTION_NAME):
    """
    Store document chunks and their embeddings in Qdrant vector database.
    
    Args:
        chunks (list): List of Document objects with .page_content and .metadata
        embeddings (list): List of embedding vectors
        collection_name (str): Name of the Qdrant collection
        
    Returns:
        bool: True if successful
    """
    # Get embedding dimension from first embedding
    vector_size = len(embeddings[0])
    
    # Create collection if it doesn't exist
    try:
        qdrant.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists. Adding new points...")
    except Exception:
        print(f"Creating new collection '{collection_name}'...")
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
    
    # Prepare points for insertion with unique UUIDs
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        # Use UUID to prevent ID collisions on re-runs
        unique_id = str(uuid.uuid4())
        point = PointStruct(
            id=unique_id,
            vector=embedding,
            payload={
                "text": chunk.page_content,
                "metadata": chunk.metadata
            }
        )
        points.append(point)
    
    # Upload points to Qdrant
    qdrant.upsert(
        collection_name=collection_name,
        points=points
    )
    
    print(f"Successfully stored {len(points)} chunks in Qdrant collection '{collection_name}'")
    return True

def build_rag_pipeline(pdf_path="Chemistry-10th.pdf", chunk_size=1000, chunk_overlap=200):
    """
    Complete RAG pipeline: Extract PDF → Chunk → Embed → Store in Qdrant.
    
    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Size of each text chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        dict: Summary of the pipeline execution
    """
    print("=" * 60)
    print("Starting RAG Pipeline")
    print("=" * 60)
    
    # Validate PDF file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        # Step 1: Extract text from PDF
        print("\n[1/4] Extracting text from PDF...")
        pages = extract_text_from_pdf(pdf_path)
        if not pages:
            raise ValueError("No pages extracted from PDF. File may be empty or corrupted.")
        print(f"✓ Extracted {len(pages)} pages")
        
        # Step 2: Chunk the documents
        print("\n[2/4] Chunking documents...")
        chunks = chunk_text(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            raise ValueError("No chunks created. Documents may be empty.")
        print(f"✓ Created {len(chunks)} chunks")
        
        # Step 3: Generate embeddings
        print("\n[3/4] Generating embeddings with Cohere...")
        embeddings = generate_embeddings(chunks)
        if len(embeddings) != len(chunks):
            raise ValueError(f"Embedding count mismatch: {len(embeddings)} != {len(chunks)}")
        print(f"✓ Generated {len(embeddings)} embeddings (dimension: {len(embeddings[0])})")
        
        # Step 4: Store in Qdrant
        print("\n[4/4] Storing in Qdrant...")
        store_in_qdrant(chunks, embeddings)
        
        print("\n" + "=" * 60)
        print("RAG Pipeline Complete!")
        print("=" * 60)
        
        return {
            "pages": len(pages),
            "chunks": len(chunks),
            "embeddings": len(embeddings),
            "embedding_dim": len(embeddings[0])
        }
    
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    # Run the pipeline
    result = build_rag_pipeline()
    print(f"\nPipeline Summary: {result}")
