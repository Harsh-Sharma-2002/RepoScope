from fastapi import APIRouter, HTTPException
from ..schema import VectorRepoInitRequest, RepoChunksResponse
from ..services.repo_index_services import index_repo_clone
from ..services.chunk_services import chunk_repo_contents
from ..services.vector_db_services import reindex_repo


router = APIRouter(tags=["vector_services"])


##############################################################################################
##############################################################################################

@router.post("/reindex")
def reindex_vector_repo(req: VectorRepoInitRequest):
    """
    Reindex repository vectors at the current head of the given branch
    """

    try:
        repo_index = index_repo_clone(owner = req.owner,repo = req.repo, branch = req.branch)

        repo_chunks: RepoChunksResponse = chunk_repo_contents(repo_index)

        reindex_repo(repo_name= f'{req.owner}/{req.repo}' , chunks= repo_chunks, embedding_dim=None, embedding_provider=req.embedding_provider)

        return {
            "status": "success",
            "repo": f"{req.owner}/{req.repo}",
            "branch": req.branch
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

##############################################################################################
##############################################################################################

from ..schema import VectorSearchRequest, VectorSearchResponse
from ..services.vector_db_services import search_repo


@router.post("/search", response_model=VectorSearchResponse)
def search_vector_repo(req: VectorSearchRequest):
    """
    Search vector DB for external code context.
    """

    try:
        results = search_repo(
            repo_name=req.repo_name,
            query=req.query,
            current_file_path=req.current_file_path,
            chunks=req.repo_chunks,
            embedding_provider=req.embedding_provider,
            top_k=req.top_k
        )

        return results

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

##############################################################################################
##############################################################################################

from ..services.vector_db_services import get_collection


@router.get("/status")
def vector_status(repo_name: str):
    """
    Check vector DB status for a repository.
    """

    try:
        collection = get_collection(repo_name)
        count = collection.count()

        return {
            "repo": repo_name,
            "indexed_vectors": count
        }

    except Exception:
        return {
            "repo": repo_name,
            "indexed_vectors": 0
        }
