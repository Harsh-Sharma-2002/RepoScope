from __future__ import annotations

import json
import os
from typing import List

import requests

from ..schema import VectorSearchResponse
from .vector_db_services import vector_search_service

"""
LLM Services — Phase 1/2

ARCHITECTURE:
- Client can choose:
    - local  -> Ollama (your Qwen model)
    - openai -> BYOK
    - claude -> BYOK
    - gemini -> BYOK

- Service:
    1. Infers intent from file_path / repo
    2. Retrieves external context via vector search
    3. Builds the full prompt internally
    4. Dispatches prompt to selected LLM provider

- Client never sends prompts or vector data
- No memory yet (added later)
"""

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MAX_TOKENS = 400
DEFAULT_TEMPERATURE = 0.2

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))


# =============================================================================
# Helper 1: Extract semantic content from vector search output
# =============================================================================

def extract_content(res: VectorSearchResponse) -> List[str]:
    """
    Convert VectorSearchResponse into pure semantic text windows.
    Drops all vector metadata.
    """
    windows: List[str] = []

    for result in res.results:
        content = "\n\n".join(chunk.content for chunk in result.context_chunks)
        windows.append(content)

    return windows


# =============================================================================
# Helper 2: Build prompts
# =============================================================================

def build_explain_prompt(*, file_path: str, context_windows: List[str]) -> str:
    """
    Build the full prompt for explaining a file in repo context.
    """
    sections = []

    for i, window in enumerate(context_windows, start=1):
        sections.append(
            f"""
CONTEXT WINDOW {i}:
{window}
""".strip()
        )

    prompt = f"""
You are a senior software engineer reviewing a codebase.

FILE UNDER REVIEW:
{file_path}

Explain the role and responsibilities of this file in the repository.
Describe how it interacts with the related code shown below.

Focus on architecture, responsibilities, and data flow.
Do NOT repeat code verbatim.

{chr(10).join(sections)}
""".strip()

    return prompt


def build_review_file_prompt(
    *,
    file_path: str,
    context_windows: List[str],
) -> str:
    """
    Build a structured prompt for reviewing a file in repository context.
    """
    sections = []

    for i, window in enumerate(context_windows, start=1):
        sections.append(
            f"""
CONTEXT WINDOW {i}:
{window}
""".strip()
        )

    prompt = f"""
You are a senior software engineer performing a professional code review.

FILE UNDER REVIEW:
{file_path}

REVIEW GOALS:
- Identify bugs, edge cases, and potential failures
- Point out design or architectural issues
- Suggest improvements or refactors
- Comment on readability and maintainability
- Consider consistency with surrounding repository code

OUTPUT FORMAT (STRICT JSON):
Return ONLY valid JSON in the following structure:

{{
  "summary": "<one-paragraph overall assessment>",
  "comments": [
    {{
      "message": "<specific review comment>",
      "severity": "<low | medium | high>"
    }}
  ]
}}

RULES:
- Do NOT include explanations outside JSON
- Do NOT repeat code verbatim
- If no issues are found, return an empty comments list

{chr(10).join(sections)}
""".strip()

    return prompt


def build_review_repo_prompt(
    *,
    repo_name: str,
    context_windows: List[str],
) -> str:
    """
    Build a structured prompt for reviewing an entire repository.
    """
    sections = []

    for i, window in enumerate(context_windows, start=1):
        sections.append(
            f"""
CONTEXT WINDOW {i}:
{window}
""".strip()
        )

    prompt = f"""
You are a senior software engineer performing a high-level repository review.

REPOSITORY:
{repo_name}

REVIEW GOALS:
- Assess overall architecture and design
- Identify systemic risks or technical debt
- Highlight maintainability and scalability concerns
- Point out cross-cutting issues (duplication, coupling, complexity)
- Suggest high-level improvements

OUTPUT FORMAT (STRICT JSON):
Return ONLY valid JSON in the following structure:

{{
  "summary": "<one-paragraph high-level assessment>",
  "key_risks": [
    {{
      "message": "<description of a risk>",
      "severity": "<low | medium | high>"
    }}
  ],
  "design_observations": [
    "<architectural or design observation>",
    "<another observation>"
  ]
}}

RULES:
- Do NOT include explanations outside JSON
- Do NOT repeat code verbatim
- Focus on system-level issues, not line-by-line comments

{chr(10).join(sections)}
""".strip()

    return prompt


# =============================================================================
# Ollama local runner
# =============================================================================

def run_ollama_local(
    *,
    prompt: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool = False,
) -> str:
    """
    Run local inference through Ollama.

    Uses:
    - POST /api/generate
    - stream=False
    - response field from the returned JSON

    For structured outputs:
    - set format="json"
    """
    url = f"{OLLAMA_HOST}/api/generate"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    if json_mode:
        payload["format"] = "json"

    resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
    resp.raise_for_status()

    data = resp.json()
    text = data.get("response", "")

    if not text or not isinstance(text, str):
        raise RuntimeError(f"Ollama returned empty or invalid response: {data}")

    return text.strip()


# =============================================================================
# BYOK runners
# =============================================================================

def run_openai(
    *,
    prompt: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """
    OpenAI chat completion runner.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a senior software engineer."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return resp.choices[0].message.content


def run_claude(
    *,
    prompt: str,
    api_key: str,
    max_tokens: int,
) -> str:
    """
    Anthropic Claude runner.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    msg = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    if hasattr(msg, "content"):
        try:
            return msg.content[0].text
        except Exception:
            pass

    return str(msg)


def run_gemini(
    *,
    prompt: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """
    Google Gemini runner (google.generativeai).
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        },
    )

    if hasattr(resp, "text"):
        return resp.text

    return str(resp)


# =============================================================================
# Provider dispatcher
# =============================================================================

def run_llm_with_provider(
    *,
    provider: str,
    prompt: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool = False,
) -> str:
    """
    Dispatch prompt execution to the selected LLM provider.

    Provider mapping:
    - local  -> Ollama/Qwen
    - openai -> OpenAI BYOK
    - claude -> Anthropic BYOK
    - gemini -> Google BYOK
    """
    provider = provider.lower().strip()

    if provider == "local":
        return run_ollama_local(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )

    if not api_key or not api_key.strip():
        raise ValueError("API key must not be empty")

    if provider == "openai":
        return run_openai(
            prompt=prompt,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    if provider == "claude":
        return run_claude(
            prompt=prompt,
            api_key=api_key,
            max_tokens=max_tokens,
        )

    if provider == "gemini":
        return run_gemini(
            prompt=prompt,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported LLM provider '{provider}'")


# =============================================================================
# Parsing helpers
# =============================================================================

def parse_review_file_output(raw_output: str) -> dict:
    """
    Parse structured review output from the LLM.
    """
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        raise ValueError("LLM did not return valid JSON for review output.")

    if not isinstance(data, dict):
        raise ValueError("Invalid review output format.")

    if "summary" not in data or "comments" not in data:
        raise ValueError("Missing required fields in review output.")

    if not isinstance(data["comments"], list):
        raise ValueError("Review comments must be a list.")

    return {
        "summary": data["summary"],
        "comments": data["comments"],
    }


def parse_review_repo_output(raw_output: str) -> dict:
    """
    Parse structured repository review output from the LLM.
    """
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        raise ValueError("LLM did not return valid JSON for repository review.")

    if not isinstance(data, dict):
        raise ValueError("Invalid repository review output format.")

    required_fields = {"summary", "key_risks", "design_observations"}
    if not required_fields.issubset(data.keys()):
        raise ValueError("Missing required fields in repository review output.")

    if not isinstance(data["key_risks"], list):
        raise ValueError("key_risks must be a list.")

    if not isinstance(data["design_observations"], list):
        raise ValueError("design_observations must be a list.")

    return {
        "summary": data["summary"],
        "key_risks": data["key_risks"],
        "design_observations": data["design_observations"],
    }


# =============================================================================
# Service: Explain File
# =============================================================================

def explain_file_service(
    *,
    owner: str,
    repo: str,
    file_path: str,
    llm_provider: str,
    llm_api_key: str,
) -> str:
    """
    Explain a file in repository context using a selected LLM.
    """
    query = f"Explain the role and responsibilities of {file_path} in the repository."

    vector_results = vector_search_service(
        owner=owner,
        repo=repo,
        query=query,
        current_file_path=file_path,
        embedding_provider="local",
        top_k=5,
        window_size=2,
    )

    context_windows = extract_content(vector_results)

    if not context_windows:
        raise ValueError("No relevant external context found.")

    prompt = build_explain_prompt(
        file_path=file_path,
        context_windows=context_windows,
    )

    return run_llm_with_provider(
        provider=llm_provider,
        prompt=prompt,
        api_key=llm_api_key,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        json_mode=False,
    )


# =============================================================================
# Service: Review File
# =============================================================================

def review_file_service(
    *,
    owner: str,
    repo: str,
    file_path: str,
    llm_provider: str,
    llm_api_key: str,
) -> dict:
    """
    Review a file in repository context.
    """
    query = f"Review {file_path} for bugs, design issues, and improvements."

    vector_results = vector_search_service(
        owner=owner,
        repo=repo,
        query=query,
        current_file_path=file_path,
        embedding_provider="local",
        top_k=5,
        window_size=2,
    )

    context_windows = extract_content(vector_results)

    if not context_windows:
        raise ValueError("No relevant external context found for review.")

    prompt = build_review_file_prompt(
        file_path=file_path,
        context_windows=context_windows,
    )

    raw_output = run_llm_with_provider(
        provider=llm_provider,
        prompt=prompt,
        api_key=llm_api_key,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        json_mode=True,
    )

    return parse_review_file_output(raw_output)


# =============================================================================
# Service: Review Repo
# =============================================================================

def review_repo_service(
    *,
    owner: str,
    repo: str,
    llm_provider: str,
    llm_api_key: str,
) -> dict:
    """
    Review an entire repository using retrieved context.
    """
    query = f"Review the overall architecture and design of the repository {repo}."

    vector_results = vector_search_service(
        owner=owner,
        repo=repo,
        query=query,
        current_file_path="",
        embedding_provider="local",
        top_k=10,
        window_size=2,
    )

    context_windows = extract_content(vector_results)

    if not context_windows:
        raise ValueError("No relevant repository context found for review.")

    prompt = build_review_repo_prompt(
        repo_name=f"{owner}/{repo}",
        context_windows=context_windows,
    )

    raw_output = run_llm_with_provider(
        provider=llm_provider,
        prompt=prompt,
        api_key=llm_api_key,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        json_mode=True,
    )

    return parse_review_repo_output(raw_output)