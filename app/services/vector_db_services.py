from ..schema import  RepoChunksResponse, RepoChunk, VectorSearchResponse , VectorSearchResult, ContextChunk
import chromadb
from chromadb.config import Settings
import os
from typing import Optional
import numpy as np
from typing import Dict
from .embedding_services import embed_text
from .chunk_services import chunk_repo_contents
from .repo_index_services import index_repo_clone

"""
================================================================================
Vector Database Services — Phase-1 Design & Invariants
================================================================================

This module implements all vector-database–related logic for repository indexing
and semantic retrieval. It is intentionally strict, deterministic, and rebuildable.

The vector database is treated as a **derived cache**, never as a source of truth.

-------------------------------------------------------------------------------
CORE PRINCIPLES
-------------------------------------------------------------------------------

1. Determinism
   - Same inputs always produce the same vector IDs and collection layout.
   - No hidden randomness or auto-fallback behavior.

2. Explicitness
   - Embedding provider must always be specified.
   - No silent defaults or implicit behavior.

3. Rebuildability
   - Vector indices are disposable.
   - Reindexing always starts from a clean state.

4. Separation of Concerns
   - Chunking, embedding, indexing, and retrieval are strictly separated.
   - This module does NOT contain LLM logic.

5. Safety First
   - Fail fast on invalid state.
   - Never silently ignore missing data.

-------------------------------------------------------------------------------
FUNCTION-LEVEL INVARIANTS
-------------------------------------------------------------------------------

--------------------------------------------------------------------------------
normalize_repo_id(owner: str, repo: str) -> str
--------------------------------------------------------------------------------
Purpose:
    Produce a canonical identifier for a GitHub repository.

Invariants:
    - Same (owner, repo) → same output, always.
    - Different owners never collide.
    - Output contains no slashes.
    - Output is filesystem-safe and DB-safe.
    - Pure function (no I/O, no side effects).

Must Never:
    - Depend on branch or commit.
    - Produce different IDs across runs.

--------------------------------------------------------------------------------
normalize_vector(vec: list[float]) -> list[float]
--------------------------------------------------------------------------------
Purpose:
    Ensure embeddings behave correctly under cosine similarity.

Invariants:
    - Output vector has L2 norm == 1.
    - Input must be numeric and finite.
    - Zero or NaN vectors raise immediately.

Must Never:
    - Silently return unnormalized vectors.
    - Mutate input data.
    - Catch numerical errors.

--------------------------------------------------------------------------------
build_chunk_lookup(chunks: RepoChunksResponse) -> dict
--------------------------------------------------------------------------------
Purpose:
    Enable fast, deterministic context expansion.

Structure:
    file_path → local_index → RepoChunk

Invariants:
    - Does not copy chunk data.
    - Lookup is read-only.
    - Ordering derives solely from local_index.

Must Never:
    - Merge chunks from different files.
    - Modify chunks.

--------------------------------------------------------------------------------
expand_file_window(file_chunks, center_index, window_size) -> list
--------------------------------------------------------------------------------
Purpose:
    Extract contiguous, file-local context.

Invariants:
    - Expansion never crosses file boundaries.
    - Missing indices are skipped.
    - Output order is ascending local_index.

Must Never:
    - Pull chunks from other files.
    - Pad with unrelated content.

--------------------------------------------------------------------------------
get_client() -> chromadb.Client
--------------------------------------------------------------------------------
Purpose:
    Provide a singleton persistent Chroma client.

Invariants:
    - Exactly one client instance per process.
    - Always uses the same persistent directory.
    - Directory must exist before client creation.

Must Never:
    - Create multiple clients.
    - Fall back to in-memory storage silently.

--------------------------------------------------------------------------------
get_or_create_collection(repo_id: str)
--------------------------------------------------------------------------------
Purpose:
    Obtain a writable collection for indexing.

Invariants:
    - Collection name == repo_id.
    - Idempotent creation.
    - Safe to call multiple times.

Must Never:
    - Delete existing data.
    - Create multiple collections for same repo.

--------------------------------------------------------------------------------
get_existing_collection(repo_id: str)
--------------------------------------------------------------------------------
Purpose:
    Obtain a read-only collection handle.

Invariants:
    - Fails if collection does not exist.
    - Never creates collections.

Must Never:
    - Hide missing index errors.

--------------------------------------------------------------------------------
delete_collection_if_exists(repo_id: str)
--------------------------------------------------------------------------------
Purpose:
    Reset vector state safely.

Invariants:
    - Idempotent.
    - Deletes entire repo index.
    - No effect if collection missing.

Must Never:
    - Partially delete data.
    - Affect other repositories.

--------------------------------------------------------------------------------
vector_index_service(...)
--------------------------------------------------------------------------------
Purpose:
    Build the vector index for a repository at HEAD.

Invariants:
    - Index always reflects current repo state.
    - Old index is fully discarded.
    - Vector count == chunk count.
    - Embedding provider is explicit and consistent.
    - Vector IDs are deterministic.

Must Never:
    - Perform incremental updates in Phase-1.
    - Mix embedding providers.
    - Return success if zero vectors written.

--------------------------------------------------------------------------------
vector_search_service(...)
--------------------------------------------------------------------------------
Purpose:
    Retrieve external semantic context for LLM usage.

Invariants:
    - Search is read-only.
    - Same-file matches are excluded.
    - Context windows are file-local.
    - Query resolution:
        • If query provided → use it.
        • Else → use full current file.
    - Results are deterministic.

Must Never:
    - Modify vector DB.
    - Cross file boundaries.
    - Hide missing collection errors.

--------------------------------------------------------------------------------
get_vector_status(owner, repo)
--------------------------------------------------------------------------------
Purpose:
    Inspect index health.

Invariants:
    - Returns indexed_vectors = 0 if missing.
    - No side effects.

Must Never:
    - Create or mutate collections.

-------------------------------------------------------------------------------
PHASE NOTES
-------------------------------------------------------------------------------

Phase-1:
    - Full rebuild indexing only.
    - No incremental updates.
    - No async execution.
    - No caching.

Phase-2 (planned):
    - Incremental reindexing.
    - Symbolic + semantic hybrid retrieval.
    - Async execution.
    - Query-aware context selection.

================================================================================
"""


#################################################################################################################
#################################################################################################################

def normalize_repo_id(owner: str, repo: str) -> str:
    """
    Create a safe, deterministic identifier for a GitHub repository.

    Invariants:
    - Same (owner, repo) → same ID
    - Different owners never collide
    - Output is filesystem- and DB-safe

    Example:
        owner="facebook", repo="react"
        → "repo__facebook__react"
    """
    if not owner or not repo:
        raise ValueError("Owner and repo must be non-empty")

    safe_owner = owner.replace("/", "__")
    safe_repo = repo.replace("/", "__")

    return f"repo__{safe_owner}__{safe_repo}"

#################################################################################################################

def normalize_vector(vec: list[float]) -> list[float]:
    """
    L2-normalize an embedding vector.

    Raises:
    - ValueError if vector is zero-length or invalid
    """
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)

    if norm == 0 or np.isnan(norm):
        raise ValueError("Invalid embedding vector (zero or NaN norm)")

    return (arr / norm).tolist()

#################################################################################################################

def build_chunk_lookup(
    chunks: RepoChunksResponse,
) -> dict[str, dict[int, RepoChunk]]:
    """
    Build a file-local lookup for chunk access.

    Structure:
        file_path → local_index → RepoChunk

    Notes:
    - This does NOT copy data
    - It only indexes existing RepoChunk objects
    """
    lookup: dict[str, dict[int, RepoChunk]] = {}

    for chunk in chunks.chunks:
        lookup.setdefault(chunk.file_path, {})[chunk.local_index] = chunk

    return lookup

#################################################################################################################

def expand_file_window(file_chunks: dict[int, RepoChunk],center_index: int,window_size: int,) -> list[RepoChunk]:
    """
    Expand a context window around a chunk index.

    Guarantees:
    - NEVER crosses file boundaries
    - Missing indices are skipped
    - Order is preserved by local_index
    """
    start = center_index - window_size
    end = center_index + window_size

    out: list[RepoChunk] = []

    for idx in range(start, end + 1):
        if idx in file_chunks:
            out.append(file_chunks[idx])

    return out


#################################################################################################################
## Collection Helper


def get_or_create_collection(repo_id: str):
    client = get_client()
    return client.get_or_create_collection(name=repo_id)


def get_existing_collection(repo_id: str):
    client = get_client()
    return client.get_collection(name=repo_id)


def delete_collection_if_exists(repo_id: str):
    client = get_client()
    try:
        client.delete_collection(repo_id)
    except Exception:
        pass

#################################################################################################################
#################################################################################################################

_client: Optional[chromadb.Client] = None


_client: Optional[chromadb.Client] = None
_chroma_dir: Optional[str] = None

import chromadb
import os
from typing import Optional

_client: Optional[chromadb.Client] = None

def get_client() -> chromadb.Client:
    global _client

    if _client is None:
        chroma_dir = os.getenv("CHROMA_PERSISTENT_DIR")

        if not chroma_dir:
            raise RuntimeError(
                "CHROMA_PERSISTENT_DIR is not set"
            )

        os.makedirs(chroma_dir, exist_ok=True)

        _client = chromadb.PersistentClient(
            path=chroma_dir
        )

        print("✅ CHROMA PERSISTENT CLIENT INITIALIZED")
        print("📂 PATH:", chroma_dir)
        print("📁 CONTENTS:", os.listdir(chroma_dir))

    return _client



####
#################################################################################################################
#################################################################################################################


# -------------------------------------------------------------------
# SERVICE 1: Index (initial + reindex)
# -------------------------------------------------------------------

def vector_index_service(
    owner: str,
    repo: str,
    branch: str,
    embedding_provider: str,
) -> dict:
    """
    Build or rebuild vector index for a repository at current HEAD.
    """

    repo_id = normalize_repo_id(owner, repo)

    # 1. Clone repo
    repo_index = index_repo_clone(owner, repo, branch)

    # 2. Chunk repo
    repo_chunks: RepoChunksResponse = chunk_repo_contents(repo_index)

    if not repo_chunks.chunks:
        raise ValueError("No chunks produced from repository")

    # 3. Reset collection
    delete_collection_if_exists(repo_id)
    collection = get_or_create_collection(repo_id)

    # 4. Embed + store
    ids = []
    embeddings = []
    metadatas = []

    for chunk in repo_chunks.chunks:
        emb = embed_text(chunk.content, provider=embedding_provider)
        vec = normalize_vector(emb["embedding"])

        vector_id = f"{repo_id}::{chunk.chunk_id}"

        ids.append(vector_id)
        embeddings.append(vec)
        metadatas.append({
            "file_path": chunk.file_path,
            "local_index": chunk.local_index,
            "chunk_id": chunk.chunk_id,
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return {
        "repo": repo_id,
        "indexed_vectors": len(ids),
        "branch": branch,
    }


# -------------------------------------------------------------------
# SERVICE 2: Search
# -------------------------------------------------------------------
def vector_search_service(
    owner: str,
    repo: str,
    query: str,
    current_file_path: str,
    embedding_provider: str,
    top_k: int,
    window_size: int,
) -> VectorSearchResponse:
    """
    Phase-1 vector search with expanded context windows.
    """

    # Resolve repo id + collection (READ-ONLY)
    repo_id = normalize_repo_id(owner, repo)
    collection = get_existing_collection(repo_id)

    # Load repo + chunks (Phase-1 correctness)
    repo_index = index_repo_clone(owner, repo, branch="main")
    repo_chunks: RepoChunksResponse = chunk_repo_contents(repo_index)

    if not repo_chunks.chunks:
        raise ValueError("Repository produced zero chunks")

    # Resolve query text
    if query and query.strip():
        query_text = query
    else:
        file_chunk_objs = [
            c for c in repo_chunks.chunks if c.file_path == current_file_path
        ]

        if not file_chunk_objs:
            raise ValueError(f"No chunks found for file '{current_file_path}'")

        file_chunk_objs_sorted = sorted(file_chunk_objs, key=lambda c: c.local_index)
        query_text = "\n".join(c.content for c in file_chunk_objs_sorted)

    # Embed query
    emb = embed_text(query_text, provider=embedding_provider)
    query_vec = normalize_vector(emb["embedding"])

    # Vector DB search
    raw = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["metadatas", "distances"],
    )

    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    # Build lookup for window expansion
    chunk_lookup = build_chunk_lookup(repo_chunks)

    # Expand windows + build response
    results: list[VectorSearchResult] = []

    for meta, dist in zip(metadatas, distances):
        file_path = meta["file_path"]

        # 🚫 FIX 1: Skip same-file matches
        if file_path == current_file_path:
            continue

        local_index = meta["local_index"]

        file_chunks_map = chunk_lookup.get(file_path)
        if not file_chunks_map:
            continue

        expanded_chunks = expand_file_window(
            file_chunks=file_chunks_map,
            center_index=local_index,
            window_size=window_size,
        )

        if not expanded_chunks:
            continue

        context_chunks = [
            ContextChunk(
                chunk_id=c.chunk_id,
                file_path=c.file_path,
                local_index=c.local_index,
                content=c.content,
            )
            for c in sorted(expanded_chunks, key=lambda x: x.local_index)
        ]

        results.append(
            VectorSearchResult(
                anchor_chunk_id=meta["chunk_id"],
                file_path=file_path,
                score=1.0 - dist,
                context_chunks=context_chunks,
            )
        )

    return VectorSearchResponse(results=results)



# -------------------------------------------------------------------
# SERVICE 3: Status
# -------------------------------------------------------------------

def get_vector_status(owner: str, repo: str) -> dict:
    repo_id = normalize_repo_id(owner, repo)

    try:
        collection = get_existing_collection(repo_id)
        count = collection.count()
    except Exception:
        # Collection truly does not exist → treat as 0 if number of vectors 0 then route will say 404 doesnt exist
        count = 0

    return {
        "repo": repo_id,
        "indexed_vectors": count,
    }
