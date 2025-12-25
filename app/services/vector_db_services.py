from ..schema import  RepoChunksResponse
import chromadb
from chromadb.config import Settings
import os
from typing import Optional


CHROMA_PERSISTANT_DIR = os.getenv("CHROMA_PERSISTANT_DIR","./.chroma_db")

#################################################################################################################
#################################################################################################################

def get_client() -> chromadb.Client:
    """
    Return a singleton Chroma client configured with persistent storage.

    Phase-1 guarantees:
    - Single client instance per process
    - Stable persistence directory
    - No hidden side effects
    """

    global _client

    if _client is None:
        _client = chromadb.Client(
            Settings(
                persist_directory=CHROMA_PERSISTANT_DIR
            )
        )

    return _client


#################################################################################################################
#################################################################################################################

def _normalize_collection_name(repo_name: str) -> str:
    """
    Normalize repo name into a safe, deterministic collection name.
    Example:
        'facebook/react' -> 'repo__facebook__react'
    """
    return f"repo_{repo_name.replace('/','_')}"

#################################################################################################################

def get_collection(repo_name: str, embedding_dim: Optional[int] = None):
    """
    Return the vector collection associated with a repository.

    This function provides a stable, repo-scoped namespace in the vector DB.
    Collections are created lazily and reused across calls.

    Phase-1 guarantees:
    - One collection per repository
    - Deterministic, safe collection naming
    - Optional validation of embedding dimensionality
    """

    client = get_client()
    collection_name = _normalize_collection_name(repo_name=repo_name)

    collection = client.get_or_create_collection(
    name=collection_name,
    metadata={"repo_name": repo_name} if embedding_dim is None else {
        "repo_name": repo_name,
        "embedding_dim": embedding_dim
    }
)

    # if embedding_dim is present, validate against the value in metadata

    if embedding_dim is not None:
        stored_dim = collection.meta_data.get("embedding_dim") 
        if stored_dim is not None and stored_dim != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch for repo '{repo_name}'. "
                f"Expected {stored_dim}, got {embedding_dim}."
            )
        
    return collection

    

#################################################################################################################
#################################################################################################################

def store_repo_embedding(repo_name: str, chunks: RepoChunksResponse):
    
    

#################################################################################################################
#################################################################################################################

def search_repo(repo_name: str, chunk_id: int, content: str, top_k : int = 5):
    pass

#################################################################################################################
#################################################################################################################