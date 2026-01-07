from typing import List, Optional, Literal
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



# LLM Schemas (RAG + Review Layer)


LLMProvider = Literal["llama", "openai", "claude", "gemini"]

# Explain File (Phase-1)
class ExplainFileRequest(BaseModel):
    """
    Request to explain a file in repository context.

    The backend:
    - infers intent
    - performs retrieval
    - builds the prompt
    - dispatches to the selected LLM provider

    The client MUST explicitly choose a provider and supply its API key.
    """
    owner: str
    repo: str
    file_path: str

    llm_provider: LLMProvider
    llm_api_key: str  # REQUIRED for all providers


class ExplainFileResponse(BaseModel):
    """
    LLM-generated explanation of the file.
    """
    explanation: str
    provider: str



# Review File 


class ReviewFileRequest(BaseModel):
    """
    Request to review a file using repository context.
    """
    owner: str
    repo: str
    file_path: str

    llm_provider: LLMProvider
    llm_api_key: str  # REQUIRED


class ReviewComment(BaseModel):
    message: str
    severity: Optional[str] = None


class ReviewFileResponse(BaseModel):
    summary: str
    comments: list[ReviewComment]
    provider: str


# -----------------------------------------------------------------------------
# Memory Control (Phase-2)
# -----------------------------------------------------------------------------

class ResetMemoryResponse(BaseModel):
    status: str
    message: str




    
