from ..schema import  RepoChunksResponse
import chromadb
from chromadb.config import Settings
import os
from typing import Optional
import numpy as np
from typing import List
from .embedding_services import embed_text

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

def _normalize_vector(vec: list[float]) -> list[float]:
    """
    L2 normalization a vector . Raise issue if vector is invalid
    """
    arr = np.asarray(vec,dtype=np.float32)
    arr_norm = np.linalg.norm(arr)

    if arr_norm == 0 or np.isnan(arr_norm):
        raise ValueError("Invalid embedding vector (Zerop or NaN norm)")
    
    return (arr / arr_norm).tolist()

#################################################################################################################

def store_repo_embedding(repo_name: str, chunks: RepoChunksResponse, embedding_dim: int, embedding_provider: str):
    """
    Store chunk embeddings for a repository in the vector database.

    Responsibilities:
    - Assign deterministic vector IDs
    - Normalize embeddings
    - Attach retrieval-critical metadata
    - Persist vectors into the repo-scoped collection

    This function assumes embeddings are already computed
    and attached to each RepoChunk.
    """

    client = get_client()
    collection = get_collection(repo_name,embedding_dim)
    ids: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []

    for chunk in chunks.chunks:
        emb_resp = embed_text(chunk.content, provider= embedding_provider)
        vector = _normalize_vector(emb_resp["embedding"])

        vector_id = f"{repo_name}::{chunk.chunk_id}"
        embeddings.append(vector)

        # 3. Metadata required for retrieval + context expansion
        metadatas.append({
            "repo_name": repo_name,
            "chunk_id": chunk.chunk_id,
            "file_path": chunk.file_path,
            "local_index": chunk.local_index
        })

    # 4. Persist into vector DB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )

    client.persist()


    

#################################################################################################################
#################################################################################################################

def search_repo(repo_name: str, chunk_id: int, content: str, top_k : int = 5):
    pass

#################################################################################################################
#################################################################################################################