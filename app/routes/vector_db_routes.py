from fastapi import APIRouter, HTTPException
from ..schema import VectorRepoInitRequest, VectorSearchRequest, VectorSearchResponse
from ..services.vector_db_services import get_vector_status, vector_search_service, vector_index_service


router = APIRouter(tags=["vector_services"])


##############################################################################################
##############################################################################################

@router.post("/index")
def reindex_vector_repo(req: VectorRepoInitRequest):
    """
    Reindex repository vectors at the current HEAD of the given branch.

    Route responsibilities:
    - Accept request
    - Call service
    - Translate errors to HTTP
    """
    try:
        return vector_index_service(
        owner=req.owner,
        repo=req.repo,
        branch=req.branch,
        embedding_provider=req.embedding_provider,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



##############################################################################################
##############################################################################################

@router.post("/search", response_model=VectorSearchResponse)
def search_vector_repo(req: VectorSearchRequest):
    """
    Retrieve external repository context.
    """
    try:
        return vector_search_service(
        repo_name=req.repo_name,
        query=req.query,
        embedding_provider=req.embedding_provider,
        top_k=req.top_k,
    )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

##############################################################################################
##############################################################################################

@router.get("/status")
def vector_status(owner: str, repo: str):
    return get_vector_status(owner, repo)


