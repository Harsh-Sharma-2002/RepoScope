from fastapi import HTTPException
from ..schema import *
import requests, os
from ..utils import fetch_file_content
import tempfile
import subprocess
import shutil
import os




#################################################################################################################
#################################################################################################################


def fetch_repo_tree(owner: str, repo: str, branch: str = "main"):

    """
    Fetches the repository tree for a given branch.
    """
    # Get the reference for the branch to find the latest commit
    ref_url = f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch}"
    headers = {
        "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
        "Accept": "application/vnd.github.v3+json"
    }
    ref_response = requests.get(url=ref_url, headers=headers)
    if ref_response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"GitHub API error (ref): {ref_response.text}"
        )
    
    ref_data = ref_response.json()


    # Get the commit URL from the ref data
    commit_url = ref_data["object"]["url"]
    commit_response = requests.get(url=commit_url, headers=headers)
    if commit_response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"GitHub API error (commit): {commit_response.text}"
        )
    commit_data = commit_response.json()

    # Get the tree URL from the commit data
    tree_url = commit_data["tree"]["url"] + "?recursive=1"
    tree_response = requests.get(url=tree_url, headers=headers)
    if tree_response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"GitHub API error (tree): {tree_response.text}"
        )

    tree_data = tree_response.json()

    tree_items = []
    for item in tree_data.get("tree", []):
        tree_items.append(
            RepoTreeItem(
                path=item["path"],
                type=item["type"],
                sha=item["sha"],
                mode=item.get("mode"),
                size=item.get("size"),
                url=item.get("url")
            )
        )


    return RepoTreeResponse(tree=tree_items, truncated=tree_data.get("truncated"))


#################################################################################################################
#################################################################################################################


def index_repo(owner: str, repo: str, branch: str = "main"):
    """
    Indexes the repository by fetching all files and their contents.
    Returns a list of RepoIndexItem with path and content.

    Note: 1. To be used for repo indexing under 100 files if using the free token
          2. For larger repos, use the the clone method.
    """

    # 1. Fetch tree
    tree_response = fetch_repo_tree(owner, repo, branch)
    tree_items = tree_response.tree

    index_items = []

    for item in tree_items:

        # Skip directories 1
        if item.type != "blob":
            continue

        # Build contents API URL
        contents_url = (
            f"https://api.github.com/repos/{owner}/{repo}/contents/{item.path}"
            f"?ref={branch}"
        )

        # Fetch and decode file content
        try:
            file_data = fetch_file_content(contents_url)
        except Exception:
            continue

        index_items.append(
            RepoIndexItem(
                path=item.path,
                content=file_data.get("file_content", "")
            )
        )

    return RepoIndexResponse(items=index_items)



#################################################################################################################
#################################################################################################################

def index_repo_clone(owner: str, repo: str, branch: str = "main"):
    """
    Clone a GitHub repository at the given branch and return indexable files.

    Phase-1 guarantees:
    - Non-interactive git execution
    - Deterministic failure on auth issues
    - Temporary clone cleaned up on exit
    """

    temp_dir = tempfile.mkdtemp()
    repo_url = f"https://github.com/{owner}/{repo}.git"

    # 🔑 Critical: disable interactive git prompts
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"

    try:
        subprocess.run(
            [
                "git", "clone",
                "--branch", branch,
                "--single-branch",
                "--depth", "1",
                repo_url,
                temp_dir
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=(
                "Git clone failed. Ensure GitHub credentials are configured "
                "(HTTPS + PAT or existing VS Code auth).\n\n"
                f"{e.stderr.strip()}"
            )
        )

    index_items = []

    SKIP_DIRS = {
        ".git", "node_modules", "venv", "env", "__pycache__",
        "dist", "build", "target", ".idea", ".vscode"
    }

    BINARY_EXTS = {
        ".png", ".jpg", ".jpeg", ".gif", ".ico",
        ".pdf", ".zip", ".tar", ".gz", ".exe", ".dll",
        ".so", ".dylib", ".7z", ".mp4", ".mp3"
    }

    MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB

    try:
        for root, dirs, files in os.walk(temp_dir):
            # Prune unwanted directories in-place (faster + safer)
            dirs[:] = [
                d for d in dirs
                if d not in SKIP_DIRS
            ]

            for filename in files:
                if any(filename.lower().endswith(ext) for ext in BINARY_EXTS):
                    continue

                file_path = os.path.join(root, filename)

                try:
                    if os.path.getsize(file_path) > MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue

                rel_path = os.path.relpath(file_path, temp_dir)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception:
                    continue

                index_items.append(
                    RepoIndexItem(
                        path=rel_path,
                        content=content
                    )
                )

    finally:
        # Always clean up temp dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    return RepoIndexResponse(items=index_items)


#################################################################################################################
#################################################################################################################


