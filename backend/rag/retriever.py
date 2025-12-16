import logging
from typing import Optional, List, Dict, Any

import chromadb
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    get_response_synthesizer,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding

# Reuse constants
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "knowledge_base"

# Ensure settings are initialized
Settings.embedding_model = OpenAIEmbedding()

logger = logging.getLogger(__name__)

def get_index():
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
    )
    return index

def query_knowledge_base(query_text: str) -> str:
    """
    Query the RAG knowledge base.
    """
    try:
        index = get_index()
        query_engine = index.as_query_engine(streaming=False) # streaming handled by agent? for now block
        response = query_engine.query(query_text)
        return str(response)
    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}")
        return "I encountered an error accessing my memory banks."

def find_relevant_document_content(query: str) -> Optional[str]:
    """
    Find a document by semantic search and return its full content (if possible) 
    or best matching chunk for reading.
    For full article reading, we might need a better way to store full text 
    mapped to title, but for RAG chunks are usually separate.
    
    Approach: Retrieve top k chunks, check if they belong to the same doc, 
    or just return the most relevant chunk text.
    
    Better approach for 'reading': 
    We probably need to store the full text separately or fetch it again if we have the URL.
    But for this MVP, we might just return the concatenated text of top chunks 
    or rely on what we have.
    """
    # TODO: Improve this for full document reading. 
    # Current implementation: Return top 3 chunks.
    try:
        index = get_index()
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(query)
        if not nodes:
            return None
        
        # Concatenate text from nodes
        content = "\n\n".join([node.node.get_text() for node in nodes])
        return content
    except Exception as e:
        logger.error(f"Error finding document: {e}")
        return None

