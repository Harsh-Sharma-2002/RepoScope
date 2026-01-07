from fastapi import FastAPI
import requests
from dotenv import load_dotenv
import os
import requests
from app.routes import pr_routes, repo_index_routes, chunk_routes, embedding_routes, vector_db_routes, llm_routes

load_dotenv()


app = FastAPI()


app.include_router(pr_routes.router, prefix="/pr")
app.include_router(repo_index_routes.router, prefix="/repo_index")
app.include_router(chunk_routes.router, prefix="/chunk")
app.include_router(embedding_routes.router,prefix="/embed")
app.include_router(vector_db_routes.router,prefix="/vector")
app.include_router(llm_routes.router, prefix="/llm")


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

#app.include_router(webhook_router, prefix="/webhook")

@app.get("/")
def read_root():
    return {"Home Page for the PR Reviewer Application"}

@app.get("/test_pr")
def test_pr():
    url = "https://api.github.com/repos/facebook/react/pulls/1"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(url, headers=headers)
    return response.json()

@app.get("/test_files")
def test_files():
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




