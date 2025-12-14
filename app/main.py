from fastapi import FastAPI
# from app.routes.webhook import router as webhook_router
import requests
from dotenv import load_dotenv
import os

load_dotenv()


app = FastAPI()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

#app.include_router(webhook_router, prefix="/webhook")

@app.get("/")
async def read_root():
    return {"Home Page for the PR Reviewer Application"}

@app.get("/test_pr")
async def test_pr():
    url = "https://api.github.com/repos/facebook/react/pulls/1"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(url, headers=headers)
    return response.json()

@app.get("/test_files")
async def test_files():
    owner = "facebook"
    repo = "react"
    pr_number = 1
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(url, headers=headers)
    return response.json()