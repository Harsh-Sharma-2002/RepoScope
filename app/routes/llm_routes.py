from fastapi import APIRouter, HTTPException

from ..schema import (
    ExplainFileRequest,
    ExplainFileResponse,
    ReviewFileRequest,
    ReviewFileResponse,
    ReviewRepoRequest,
    ReviewRepoResponse
)
from ..services.llm_services import explain_file_service,review_file_service,review_repo_service

router = APIRouter(tags=["llm"])


@router.post("/explain-file", response_model=ExplainFileResponse)
def explain_file(req: ExplainFileRequest):
    """
    Explain a file in the context of its repository.

    Client provides:
    - repo identity
    - file path
    - LLM provider
    - API key for that provider (BYOK)

    Backend:
    - infers intent
    - performs vector search
    - builds the prompt
    - dispatches to selected LLM
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
            detail=str(e),
        )

    except Exception as e:
        import traceback
        traceback.print_exc()   # 👈 prints full stack trace to terminal

        raise HTTPException(
            status_code=500,
            detail=repr(e),     # 👈 repr instead of str
        )
    


@router.post("/review-file", response_model=ReviewFileResponse)
def review_file(req: ReviewFileRequest):
    """
    Review a file in the context of its repository.
    """

    try:
        result = review_file_service(
            owner=req.owner,
            repo=req.repo,
            file_path=req.file_path,
            llm_provider=req.llm_provider,
            llm_api_key=req.llm_api_key,
        )

        return ReviewFileResponse(
            summary=result["summary"],
            comments=result["comments"],
            provider=req.llm_provider,
        )

    except ValueError as e:
        # User / input / configuration error
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),     
        )


@router.post("/review-repo", response_model=ReviewRepoResponse)
def review_repo(req: ReviewRepoRequest):
    """
    Review an entire repository at a high level.
    """

    try:
        result = review_repo_service(
            owner=req.owner,
            repo=req.repo,
            llm_provider=req.llm_provider,
            llm_api_key=req.llm_api_key,
        )

        return ReviewRepoResponse(
            summary=result["summary"],
            key_risks=result["key_risks"],
            design_observations=result["design_observations"],
            provider=req.llm_provider,
        )

    except ValueError as e:
        # User / input / configuration error
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception:
        # Genuine server failure
        raise HTTPException(
            status_code=500,
            detail="Internal server error while reviewing repository",
        )
