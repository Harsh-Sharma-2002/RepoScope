from fastapi import APIRouter, HTTPException
from ..services.embedding_services import embed_text
from ..schema import EmbedRequest,EmbedResponse,BatchEmbedRequest,BatchEmbedResponse

router = APIRouter(tags=["embedding"])



# #############################################################################################################################
@router.post("/single",response_model=EmbedResponse)
def embed_single_route(req: EmbedRequest):
    """
    Embed a single text string using user-selected or auto-selected provider.
    """
    result = embed_text(req.text, provider=req.provider)

    if "error" in result:
        raise HTTPException(status_code=500,detail=result)
    
    return EmbedResponse(embedding=result["embedding"], provider=result["provider"])

# #############################################################################################################################

@router.post("/batch", response_model=BatchEmbedResponse)
def embed_batch(req: BatchEmbedRequest):
    """
    Embed a list of texts.
    Provider selection applies to all texts.
    """

    responses = []

    for t in req.texts:
        result = embed_text(t, provider=req.provider)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result)

        responses.append(
            EmbedResponse(
                embedding=result["embedding"],
                provider=result["provider"]
            )
        )

    return BatchEmbedResponse(embeddings=responses)

# #############################################################################################################################