from typing import List, Optional
from pydantic import BaseModel

from typing import List, Optional
from pydantic import BaseModel



# Model for ONE changed file

class FileChange(BaseModel):
    filename: str
    status: str                      # modified, added, removed, renamed
    patch: Optional[str] = None      # diff text (can be None for binary or deleted files)
    contents_url: Optional[str] = None   # URL to get full content (None for removed files)


# Response model for /fetch_pr_files

class PRFilesResponse(BaseModel):
    files: List[FileChange]



# Model for file content
# (used in /fetch_file_content)

class FileContent(BaseModel):
    file_path: str                   # name/path of the file
    content: str                     # decoded text content



# Response model for /fetch_all_contents (future step)
class PRFullFilesResponse(BaseModel):
    files: List[FileContent]

