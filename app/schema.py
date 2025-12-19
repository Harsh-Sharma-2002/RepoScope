from typing import List, Optional
from pydantic import BaseModel


####################################################################################################
# Model for ONE changed file (metadata only)

class FileChange(BaseModel):
    filename: str
    status: str
    patch: Optional[str] = None
    contents_url: Optional[str] = None



# Response for /fetch_pr_files

class PRFilesResponse(BaseModel):
    files: List[FileChange]


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
    files: List[ExpandedFile]

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
    tree: List[RepoTreeItem]
    truncated: Optional[bool] = None


####################################################################################################

# Model for a single item in the repo index
class RepoIndexItem(BaseModel):
    path: str
    content: str

# Response for /index_repo
class RepoIndexResponse(BaseModel):
    items: List[RepoIndexItem]

####################################################################################################

# Model for repo chunks
class RepoChunk(BaseModel):
    file_path: str
    chunk_id: int
    content: str

# Response for repo chunks
class RepoChunksResponse(BaseModel):
    chunks: List[RepoChunk]


####################################################################################################

# Embedding request, response schemas

class EmbedRequest(BaseModel):
    text: str
    provider: Optional[str] = None

class BatchEmbedRequest(BaseModel):
    texts:List[str]
    provider: Optional[str] = None

class EmbedResponse(BaseModel):
    embedding: List[float]
    provider: str

class BatchEmbedResponse(BaseModel):
    embeddings: List[EmbedResponse]

####################################################################################################