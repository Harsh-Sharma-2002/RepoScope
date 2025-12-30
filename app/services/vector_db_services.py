from ..schema import  RepoChunksResponse, ContextChunk, RepoChunk, VectorSearchResponse , VectorSearchResult
import chromadb
from chromadb.config import Settings
import os
from typing import Optional
import numpy as np
from typing import Dict
from .embedding_services import embed_text



CHROMA_PERSISTANT_DIR = os.getenv("CHROMA_PERSISTANT_DIR","./.chroma_db")
_client = None


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
    metadata={
    "repo_name": repo_name,
    "embedding_dim": embedding_dim
    }

)

    # if embedding_dim is present, validate against the value in metadata

    if embedding_dim is not None:
        stored_dim = collection.metadata.get("embedding_dim") 
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
    Index repository chunks into the vector database.

    Phase-1 behavior:
    - Embeddings are computed inline using the specified provider
    - Each RepoChunk is embedded independently
    - Embeddings are L2-normalized before storage
    - Vector metadata stores only retrieval-critical fields
    - The vector database is treated as disposable and rebuildable

    This function is a write-only command:
    - It performs side effects (vector DB writes)
    - It returns nothing on success
    - It fails loudly on errors

    Phase-2 note:
    - Embedding computation may be moved upstream
    - Incremental updates will replace full re-indexing
    """
    if not chunks.chunks:
        raise ValueError(
            f"No chunks to index for repo '{repo_name}'. "
            "Indexing aborted."
    )

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
        ids.append(vector_id)

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

def _distance_to_score(distance: float) -> float:
    """Convert cosine distance to similarity score (higher is better)."""
    return 1.0 - distance

#################################################################################################################

def _build_chunk_lookup(
    chunks: RepoChunksResponse,
) -> dict[str, dict[int, RepoChunk]]:
    """
    Canonical lookup:
        file_path -> local_index -> RepoChunk

    NOTE:
    This is an index over existing RepoChunk objects.
    No data is copied.
    """
    lookup: dict[str, dict[int, RepoChunk]] = {}
    for c in chunks.chunks:
        lookup.setdefault(c.file_path, {})[c.local_index] = c
    return lookup

#################################################################################################################

def _expand_window(
    file_chunks: dict[int, RepoChunk],
    center_index: int,
    window_size: int,
) -> list[RepoChunk]:
    """
    Expand a strictly file-local context window.

    IMPORTANT:
    - Expansion NEVER crosses file boundaries
    - Missing indices are simply skipped
    """
    start = center_index - window_size
    end = center_index + window_size

    out: list[RepoChunk] = []
    for idx in range(start, end + 1):
        if idx in file_chunks:
            out.append(file_chunks[idx])

    return out

#################################################################################################################

def search_repo(
    repo_name: str,
    query: str,
    current_file_path: str,
    chunks: RepoChunksResponse,
    embedding_provider: str,
    top_k: int = 5,
    window_size: int = 2,
) -> VectorSearchResponse:
    """
    Retrieve external repository context relevant to the query.

    Phase-1 guarantees:
    - Full current file is assumed to be included elsewhere
    - Same-file vector hits are filtered out
    - External anchors are dynamically over-fetched
    - Context windows are strictly file-local
    - Single-chunk files return only that chunk
    - Partial external recall is allowed
    - Results are deterministic, bounded, and deduplicated
    """

    # -------------------------
    # Phase-1 constants
    # -------------------------
    INITIAL_FETCH = top_k * 2
    FETCH_MULTIPLIER = 2
    MAX_FETCH = 50
    MAX_CONTEXT_CHUNKS = 25

    # -------------------------
    # Embed query
    # -------------------------
    emb = embed_text(query, provider=embedding_provider)
    query_vec = _normalize_vector(emb["embedding"])

    collection = get_collection(repo_name=repo_name)

    # Canonical lookup for expansion
    chunk_lookup = _build_chunk_lookup(chunks)

    # -------------------------
    # Dynamic over-fetch loop
    # -------------------------
    desired_external = top_k
    k = INITIAL_FETCH
    external_anchors: list[tuple[dict, float]] = []

    while k <= MAX_FETCH:
        raw = collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            include=["metadatas", "distances"],
        )

        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        if not metadatas:
            break

        filtered: list[tuple[dict, float]] = []
        for meta, dist in zip(metadatas, distances):
            # Exclude current file — it is already fully included
            if meta.get("file_path") == current_file_path:
                continue
            filtered.append((meta, dist))

        # Accept partial recall — do NOT fail hard
        external_anchors = filtered

        if len(filtered) >= desired_external:
            break

        k *= FETCH_MULTIPLIER

    # Graceful degradation: no external context found
    if not external_anchors:
        return VectorSearchResponse(results=[])

    # Sort by similarity (distance ascending)
    external_anchors.sort(key=lambda x: x[1])
    selected_anchors = external_anchors[:top_k]

    # -------------------------
    # Expand + deduplicate
    # -------------------------
    results: list[VectorSearchResult] = []
    seen_chunks: set[tuple[str, int]] = set()  # (file_path, local_index)
    used_context = 0

    for meta, dist in selected_anchors:
        file_path = meta["file_path"]
        local_index = meta["local_index"]

        file_chunks = chunk_lookup.get(file_path)
        if not file_chunks:
            continue

        # Rule: if file has only one chunk, return only that chunk
        if len(file_chunks) == 1:
            expanded = list(file_chunks.values())
        else:
            expanded = _expand_window(
                file_chunks=file_chunks,
                center_index=local_index,
                window_size=window_size,
            )

        context_chunks: list[ContextChunk] = []

        for c in sorted(expanded, key=lambda x: x.local_index):
            key = (c.file_path, c.local_index)

            if key in seen_chunks:
                continue
            if used_context >= MAX_CONTEXT_CHUNKS:
                break

            seen_chunks.add(key)
            used_context += 1

            context_chunks.append(
                ContextChunk(
                    chunk_id=c.chunk_id,
                    file_path=c.file_path,
                    local_index=c.local_index,
                    content=c.content,
                )
            )

        if not context_chunks:
            continue

        results.append(
            VectorSearchResult(
                anchor_chunk_id=meta["chunk_id"],
                file_path=file_path,
                score=_distance_to_score(dist),
                context_chunks=context_chunks,
            )
        )

        if used_context >= MAX_CONTEXT_CHUNKS:
            break

    return VectorSearchResponse(results=results)

#################################################################################################################
#################################################################################################################

def reindex_repo(
    repo_name: str,
    chunks: RepoChunksResponse,
    embedding_dim: int,
    embedding_provider: str,
):
    """
    Reindex the repository at the current working HEAD.

    Phase-1 behavior:
    - Deletes the existing vector collection for the repo (if present)
    - Creates a fresh collection
    - Rebuilds the vector index from scratch

    This enforces the invariant:
    - Vector DB always represents the current repo state
    """

    client = get_client()
    collection_name = _normalize_collection_name(repo_name)

    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
    except Exception:
        # Collection may not exist — this is fine in Phase-1
        pass

    # Create fresh collection with correct embedding_dim
    get_collection(
        repo_name=repo_name,
        embedding_dim=embedding_dim
    )

    # Store embeddings into the new collection
    store_repo_embedding(
        repo_name=repo_name,
        chunks=chunks,
        embedding_dim=embedding_dim,
        embedding_provider=embedding_provider,
    )


#################################################################################################################
#################################################################################################################
