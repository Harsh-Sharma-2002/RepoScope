from fastapi import APIRouter, HTTPException

from ..schema import (
    ExplainFileRequest,
    ExplainFileResponse,
)
from ..services.llm_services import explain_file_service

router = APIRouter(tags=["llm"])


@router.post("/explain-file", response_model=ExplainFileResponse)
def explain_file(req: ExplainFileRequest):
    """
    Explain a file in the context of its repository.

    The client provides:
    - repo identity
    - file path
    - explicit LLM provider
    - API key for that provider

    The backend:
    - infers intent
    - performs vector search
    - builds the prompt
    - calls the selected LLM
    """

    try:
        explanation = explain_file_service(
            owner=req.owner,
            repo=req.repo,
            file_path=req.file_path,
            llm_provider=req.llm_provider,
            llm_api_key=req.llm_api_key,
        )

        return ExplainFileResponse(
            explanation=explanation,
            provider=req.llm_provider,
        )

    except ValueError as e:
        # User / input / configuration error
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        # Genuine server failure
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
