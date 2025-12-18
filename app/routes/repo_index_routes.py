from fastapi import APIRouter
from ..schema import *
from ..services.repo_index_services import index_repo, fetch_repo_tree, index_repo_clone



router = APIRouter(tags=["repo_index services"])



##############################################################################################
##############################################################################################

@router.get("/fetch_repo_tree",response_model=RepoTreeResponse)
def fetch_repo_tree_route(owner: str, repo: str, branch: str = "main"):
    """
    API route to fetch the repository tree for a given branch.
    """
    return fetch_repo_tree(owner, repo, branch)

##############################################################################################
##############################################################################################

@router.get("/index_repo_crawl",response_model=RepoIndexResponse)
def index_repo_crawl(owner: str, repo: str, branch: str = "main"):
    """
    Indexes the repository by fetching all files and their contents.
    Returns a list of RepoIndexItem with path and content.

    Note: 1. To be used for repo indexing under 100 files if using the free token
          2. For larger repos, use the the clone method.
    """
    return index_repo(owner, repo, branch)

##############################################################################################
##############################################################################################

@router.get("/index_repo_clone",response_model=RepoIndexResponse) 
def index_repo_clone_route(owner: str, repo: str, branch: str = "main"):
    """
    Indexes the repository by cloning it and reading all files and their contents.
    Returns a list of RepoIndexItem with path and content.

    Note: Suitable for larger repositories.
    """
    return index_repo_clone(owner, repo, branch)

##############################################################################################
##############################################################################################

