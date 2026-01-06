# app/routes/llm_routes.py

from fastapi import APIRouter, HTTPException
from ..schema import (
    ExplainContextRequest,
    ExplainContextResponse,
)
from ..services.llm_explain_services import (
    explain_context_llama,
    reset_llm_memory,
)

router = APIRouter(tags=["llm"])


@router.post("/explain-context", response_model=ExplainContextResponse)
def explain_context(req: ExplainContextRequest):
    """
    Explain why the retrieved files are relevant to the query
    and the current file.

    This is a pure RAG explanation endpoint.
    """

    try:
        return explain_context_llama(
            query=req.query,
            current_file_path=req.current_file_path,
            results=req.results.results,
            max_files=req.max_files,
            max_chars_per_file=req.max_chars_per_file,
        )

    except ValueError as e:
        # User / input error
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # True server failure
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/reset-memory")
def reset_memory(owner: str, repo: str):
    """
    Reset LLM working memory for a repository.
    Useful when starting a fresh review.
    """

    reset_llm_memory(owner, repo)

    return {
        "status": "ok",
        "message": "LLM working memory cleared"
    }
