from fastapi import APIRouter, Request

router = APIRouter()

@router.post("/")
async def webhook_handler(request: Request):
    payload = await request.json()
    print("Received webhook payload:", payload)
    return {"status": "ok"}
