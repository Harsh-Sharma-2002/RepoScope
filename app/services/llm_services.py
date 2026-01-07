import os
import requests
from typing import List


from ..schema import VectorSearchResponse
from .vector_db_services import vector_search_service
from .embedding_services import embed_text


"""
LLM Services — Phase 1 (Explain File)

ARCHITECTURE:
- Client sends: owner, repo, file_path, llm_provider, llm_api_key
- Service:
    1. Infers intent from file_path
    2. Retrieves external context via vector search
    3. Builds the full prompt internally
    4. Dispatches prompt to selected LLM provider
- Client never sends prompts or vector data
- No memory yet (added in review phase)
"""


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MAX_TOKENS = 400
DEFAULT_TEMPERATURE = 0.2


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
        content = "\n\n".join(
            chunk.content for chunk in result.context_chunks
        )
        windows.append(content)

    return windows


# =============================================================================
# Helper 2: Build explain-file prompt
# =============================================================================

def build_explain_prompt(
    *,
    file_path: str,
    context_windows: List[str],
) -> str:
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


# =============================================================================
# Provider-specific LLM runners
# =============================================================================

def run_llama_hf(
    *,
    prompt: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """
    CodeLLaMA via Hugging Face Inference API.
    """
    HF_API_URL = os.getenv("HUGGINGFACE_API_URL")
    if not HF_API_URL:
        raise RuntimeError("HUGGINGFACE_API_URL is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "return_full_text": False,
        },
    }

    response = requests.post(
        HF_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()

    data = response.json()

    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"]

    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]

    return str(data)


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

    return msg.content[0].text


def run_gemini(
    *,
    prompt: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """
    Google Gemini runner.
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

    return resp.text


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
) -> str:
    """
    Dispatch prompt execution to the selected LLM provider.
    """

    if provider == "llama":
        return run_llama_hf(
            prompt=prompt,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )

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

    PIPELINE:
    1. Infer intent from file_path
    2. Retrieve external context via vector search
    3. Build prompt internally
    4. Dispatch to selected LLM provider
    """

    # 1. Infer implicit query
    query = f"Explain the role and responsibilities of {file_path} in the repository."

    # 2. Vector search (external context only)
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

    # 3. Build prompt
    prompt = build_explain_prompt(
        file_path=file_path,
        context_windows=context_windows,
    )

    # 4. Run selected LLM
    return run_llm_with_provider(
        provider=llm_provider,
        prompt=prompt,
        api_key=llm_api_key,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
    )
