from fastapi import FastAPI
# from app.routes.webhook import router as webhook_router
import requests
from dotenv import load_dotenv
import os
import base64
import requests
from app.services import github_service

load_dotenv()


app = FastAPI()

app.include_router(github_service.router)

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



@app.get("/fetch_file_content")
async def fetch_file_content(contents_url: str):
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    response = requests.get(contents_url, headers=headers)
    
    if response.status_code != 200:
        return {"error": f"Failed: {response.text}"}
    
    data = response.json()  # THIS MUST BE A DICT
    
    encoded = data.get("content")
    if not encoded:
        return {"error": "No 'content' key — you passed wrong URL"}
    
    decoded = base64.b64decode(encoded).decode("utf-8", errors="replace")
    
    return {
        "file_path": data.get("path"),
        "file_content": decoded
    }

# @app.get("/fetch_pr_files")
# async def get_pr_files(pr_number: int, owner: str, repo: str):
#     url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
#     headers = {
#         "Authorization" : f"Bearer " + GITHUB_TOKEN,
#         "Accept": "application/vnd.github.v3+json"
#     }
#     response = requests.get(url, headers=headers)
#     if response.status_code != 200: 
#         return {"error": f"Failed to fetch files: {response.text}"}
#     raw_data = response.json()

