# app/routes/llm_routes.py

from fastapi import APIRouter, HTTPException
from ..schema import LLMRequest,LLMResponse
    

from ..services.llm_explain_services import (
    explain_context_llama,
    reset_llm_memory,
)

router = APIRouter(tags=["llm"])


@router.post("/run", response_model=LLMResponse)
def run_llm_route(req: LLMRequest):
    """
    Execute an LLM call with a fully constructed prompt.

    The prompt is assumed to be built by the service layer.
    """

    try:
        output = run_llm(
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        return LLMResponse(output=output)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
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
