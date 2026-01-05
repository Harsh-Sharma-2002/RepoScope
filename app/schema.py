from typing import List, Optional
from pydantic import BaseModel, conint


####################################################################################################
# Model for ONE changed file (metadata only)

class FileChange(BaseModel):
    filename: str
    status: str
    patch: Optional[str] = None
    contents_url: Optional[str] = None



# Response for /fetch_pr_files

class PRFilesResponse(BaseModel):
    files: list[FileChange]


####################################################################################################
# Model for a single file's decoded content

class FileContent(BaseModel):
    filename: str
    content: Optional[str]  # may be None for binary files


####################################################################################################
# Model for a file with diff + content
# Used in /fetch_all_file_contents
class ExpandedFile(BaseModel):
    filename: str
    patch: Optional[str] = None
    content: Optional[str] = None


# Response for /fetch_all_file_contents

class AllFilesContentResponse(BaseModel):
    files: list[ExpandedFile]

####################################################################################################
# Model for a single item in the repo tree
class RepoTreeItem(BaseModel):
    path: str
    type: str       # "blob" or "tree"
    sha: str
    mode: Optional[str] = None
    size: Optional[int] = None
    url: Optional[str] = None

# Response for /fetch_repo_tree
class RepoTreeResponse(BaseModel):
    tree: list[RepoTreeItem]
    truncated: Optional[bool] = None


####################################################################################################

# Model for a single item in the repo index
class RepoIndexItem(BaseModel):
    path: str
    content: str

# Response for /index_repo
class RepoIndexResponse(BaseModel):
    items: list[RepoIndexItem]

####################################################################################################

# Model for repo chunks
class RepoChunk(BaseModel):
    file_path: str
    chunk_id: int
    content: str
    local_index: int

# Response for repo chunks
class RepoChunksResponse(BaseModel):
    chunks: list[RepoChunk]


####################################################################################################

# Embedding request, response schemas

class EmbedRequest(BaseModel):
    text: str
    provider: str

class BatchEmbedRequest(BaseModel):
    texts: List[str]
    provider: str 

class EmbedResponse(BaseModel):
    embedding: List[float]
    provider: str

class BatchEmbedResponse(BaseModel):
    embeddings: List[EmbedResponse]

####################################################################################################

# Vector DB schemas

# Request to index a repo into the vector database at current HEAD
class VectorRepoInitRequest(BaseModel):
    owner: str
    repo: str
    branch: str = "main"
    embedding_provider: str
    


# Request to search a repository vector index for relevant external context.
# NOTE: prefer passing owner + repo (not "owner/repo") so services can normalize cleanly.
class VectorSearchRequest(BaseModel):
    owner: str
    repo: str
    query: str
    top_k: conint(ge=1) = 5
    embedding_provider: str
    window_size: conint(ge=0) = 2
    current_file_path: str


# A single canonical code chunk included in the expanded context window.
class ContextChunk(BaseModel):
    chunk_id: int
    file_path: str
    local_index: int
    content: str


# A single search hit: the external anchor + expanded, file-local context chunks
class VectorSearchResult(BaseModel):
    anchor_chunk_id: int
    file_path: str
    score: float
    context_chunks: List[ContextChunk]


class VectorSearchResponse(BaseModel):
    results: List[VectorSearchResult]



    
