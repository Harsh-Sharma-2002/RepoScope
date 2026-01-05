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

CHROMA_PERSISTENT_DIR = os.getenv("CHROMA_PERSISTENT_DIR")
_client: Optional[chromadb.Client] = None


def get_client() -> chromadb.Client:
    global _client
    if _client is None:
        _client = chromadb.Client(
            Settings(persist_directory=CHROMA_PERSISTENT_DIR)
        )
    return _client

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

# def vector_search_service(
#     owner: str,
#     repo: str,
#     query: str,
#     current_file_path: str,
#     embedding_provider: str,
#     top_k: int,
#     window_size: int,
# ) -> VectorSearchResponse:
#     """
#     Phase-1 vector search with expanded context windows.
#     """

#     # Resolve repo id + collection
#     repo_id = normalize_repo_id(owner, repo)
#     collection = get_or_create_collection(repo_id)  # or use your get_existing_collection helper

#     # Load repo + chunks (Phase-1 correctness)
#     repo_index = index_repo_clone(owner, repo, branch="main")
#     repo_chunks: RepoChunksResponse = chunk_repo_contents(repo_index)

#     if not repo_chunks.chunks:
#         raise ValueError("Repository produced zero chunks")

#     # Resolve query text
#     if query and query.strip():
#         query_text = query
#     else:
#         # Keep chunk objects (not just content) so we can sort by local_index
#         file_chunk_objs = [
#             c for c in repo_chunks.chunks if c.file_path == current_file_path
#         ]

#         if not file_chunk_objs:
#             raise ValueError(f"No chunks found for file '{current_file_path}'")

#         # sort by the chunk's local_index, then join their contents
#         file_chunk_objs_sorted = sorted(file_chunk_objs, key=lambda c: c.local_index)
#         query_text = "\n".join(c.content for c in file_chunk_objs_sorted)

#     # Embed query
#     emb = embed_text(query_text, provider=embedding_provider)
#     query_vec = normalize_vector(emb["embedding"])

#     # Vector DB search
#     raw = collection.query(
#         query_embeddings=[query_vec],
#         n_results=top_k,
#         include=["metadatas", "distances"],
#     )

#     metadatas = raw.get("metadatas", [[]])[0]
#     distances = raw.get("distances", [[]])[0]

#     # Build lookup for window expansion (map file_path -> {local_index: chunk})
#     chunk_lookup = build_chunk_lookup(repo_chunks)

#     # Expand windows + build response
#     results: list[VectorSearchResult] = []

#     for meta, dist in zip(metadatas, distances):
#         file_path = meta["file_path"]
#         local_index = meta["local_index"]

#         file_chunks_map = chunk_lookup.get(file_path)
#         if not file_chunks_map:
#             continue

#         expanded_chunks = expand_file_window(
#             file_chunks=file_chunks_map,
#             center_index=local_index,
#             window_size=window_size,
#         )

#         if not expanded_chunks:
#             continue

#         context_chunks = [
#             ContextChunk(
#                 chunk_id=c.chunk_id,
#                 file_path=c.file_path,
#                 local_index=c.local_index,
#                 content=c.content,
#             )
#             for c in sorted(expanded_chunks, key=lambda x: x.local_index)
#         ]

#         results.append(
#             VectorSearchResult(
#                 anchor_chunk_id=meta["chunk_id"],
#                 file_path=file_path,
#                 score=1.0 - dist,
#                 context_chunks=context_chunks,
#             )
#         )

#     return VectorSearchResponse(results=results)



# -------------------------------------------------------------------
# SERVICE 3: Status
# -------------------------------------------------------------------

def get_vector_status(owner: str, repo: str) -> dict:
    repo_id = normalize_repo_id(owner, repo)

    try:
        collection = get_existing_collection(repo_id)
        count = collection.count()
    except Exception:
        count = 0

    return {
        "repo": repo_id,
        "indexed_vectors": count,
    }