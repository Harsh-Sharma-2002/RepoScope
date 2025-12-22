from fastapi import APIRouter
from ..schema import *
from ..services.chunk_services import chunk_repo_contents
from ..services.repo_index_services import index_repo_clone


router = APIRouter(tags=["Chunk Routes"])

##############################################################################################
##############################################################################################

@router.get("/chunk_repo",response_model= RepoChunksResponse)
def chunk_repo_route(owner: str,repo : str, branch: str = "main") -> RepoChunksResponse :
    
    repo_index = index_repo_clone(owner, repo, branch)
    return chunk_repo_contents(repo_index)

##############################################################################################
##############################################################################################
