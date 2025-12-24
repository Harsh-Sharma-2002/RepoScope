from ..schema import  RepoChunksResponse
import chromadb
from chromadb.config import Settings
import os


CHROMA_PERSISTANT_DIR = os.getenv("CHROMA_PERSISTANT_DIR","./.chroma_db")

#################################################################################################################
#################################################################################################################

def get_client() -> chromadb.Client:
    """
    Return a singleton Chroma client configured with persistent storage.

    Phase-1 guarantees:
    - Single client instance per process
    - Stable persistence directory
    - No hidden side effects
    """

    global _client

    if _client is None:
        _client = chromadb.Client(
            Settings(
                persist_directory=CHROMA_PERSISTANT_DIR
            )
        )

    return _client


#################################################################################################################
#################################################################################################################

def get_collection():
    pass

#################################################################################################################
#################################################################################################################

def store_repo_embedding(repo_name: str, chunks: RepoChunksResponse):
    pass

#################################################################################################################
#################################################################################################################

def search_repo(repo_name: str, chunk_id: int, content: str, top_k : int = 5):
    pass

#################################################################################################################
#################################################################################################################