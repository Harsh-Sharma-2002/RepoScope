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


@router.post("/explain-context",response_model=ExplainContextResponse,)
def explain_context(req: ExplainContextRequest):
    """
    Explain why the retrieved code context is relevant to the query
    and the current file.

    This endpoint:
    - does NOT perform vector search
    - does NOT store memory
    - only explains existing retrieved context
    """

    try:
        return explain_context_llama(
            query=req.query,
            current_file_path=req.current_file_path,
            vector_search_response=req.results,
        )

    except ValueError as e:
        # user/input error
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # genuine server error
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
