import os
import requests
import json
from sentence_transformers import SentenceTransformer
from typing import Optional


# ========================================
# Embedding Services — Design Notes
# ========================================

# PURPOSE
# -------
# This module is responsible for generating vector embeddings from raw text
# (code chunks or queries). These embeddings are used exclusively for semantic
# similarity search in the vector database.

# Embeddings are NOT:
# - reversible
# - shown to the LLM
# - used for reasoning or generation

# The LLM always receives the original text, never vectors.

# ----------------------------------------
# KEY DESIGN DECISIONS
# ----------------------------------------

# 1. EXPLICIT PROVIDER REQUIREMENT
# --------------------------------
# All embedding calls require an explicit `provider` argument
# (e.g. "local", "openai", "gemini").

# Rationale:
# - Vector similarity is only meaningful when all vectors in a collection
#   are generated using the same embedding model.
# - Auto-selection or fallback can silently mix embeddings from different
#   providers, which breaks retrieval quality and is extremely hard to debug.

# Invariant:
# - A single repository must use exactly ONE embedding provider.
# - Provider selection is a repo-level decision, not a per-call decision.

# ----------------------------------------

# 2. NO AUTO-SELECTION IN PRODUCTION PATHS
# ----------------------------------------
# Automatic fallback (OpenAI → Gemini → Local) was intentionally removed from
# production embedding paths.

# Rationale:
# - Auto-selection hides failures and introduces nondeterministic behavior.
# - Silent fallback can cause indexing and querying to use different models.

# Behavior:
# - All embedding failures fail fast and raise exceptions.
# - Misuse is surfaced immediately instead of degrading silently.

# ----------------------------------------

# 3. SEPARATION OF CONCERNS
# -------------------------
# This module is intentionally pure and stateless.

# It does NOT:
# - know about repositories
# - know about vector databases
# - know about LLMs
# - store configuration or state

# It ONLY:
# - takes (text, provider)
# - returns raw model embeddings

# This keeps the system modular, testable, and easy to evolve.

# ----------------------------------------

# 4. LOCAL FALLBACK STRATEGY
# --------------------------
# A local SentenceTransformer model (`all-MiniLM-L6-v2`) is used as a fallback
# provider when no external API key is available.

# Rationale:
# - Allows the system to work out-of-the-box.
# - Enables fast local development and testing.
# - Provides deterministic, offline embeddings.

# ----------------------------------------

# 5. NORMALIZATION POLICY
# -----------------------
# Embeddings are NOT normalized in this module.

# Normalization is applied:
# - at the vector database boundary
# - immediately before storing vectors
# - immediately before querying vectors

# Rationale:
# - Normalization is a vector-DB concern, not a model concern.
# - Prevents accidental double-normalization.
# - Allows different similarity metrics or DBs to be used later.

# ----------------------------------------

# SUMMARY
# -------
# This embedding layer was deliberately hardened early to enforce strong
# invariants around provider consistency and debuggability. These constraints
# prevent subtle retrieval bugs later and align the system with production
# RAG architectures.

# ========================================


# Load local model once
_local_model = SentenceTransformer("all-MiniLM-L6-v2")


########################################################################################################
# Helper: wrap output
def _wrap(embedding: list[float], provider: str):
    return {
        "embedding": embedding,
        "provider": provider
    }

########################################################################################################
# LOCAL FALLBACK
def embed_local(text: str):
    """Local fallback embedding using MiniLM."""
    emb = _local_model.encode(text).tolist()
    return _wrap(emb, "local")


########################################################################################################
# OPENAI
def embed_openai(text: str, api_key: str):
    url = "https://api.openai.com/v1/embeddings"
    payload = {"model": "text-embedding-3-large", "input": text}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload))
        resp.raise_for_status()
        emb = resp.json()["data"][0]["embedding"]
        return _wrap(emb, "openai")
    except Exception as e:
        raise Exception(f"OpenAI embedding error → {resp.text if 'resp' in locals() else str(e)}")


########################################################################################################
# GEMINI
def embed_gemini(text: str, api_key: str):
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        "text-embedding-004:embedText?key="
        + api_key
    )
    payload = {"text": text}

    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        emb = resp.json()["embedding"]["values"]
        return _wrap(emb, "gemini")
    except Exception as e:
        raise Exception(f"Gemini embedding error → {resp.text if 'resp' in locals() else str(e)}")


########################################################################################################
# CLAUDE (NO REAL EMBEDDINGS)
def embed_claude(text: str, api_key: str):
    """
    Anthropic has no embedding endpoint.
    We use local embedding but label provider as 'claude-fallback'
    """
    try:
        emb = _local_model.encode(text).tolist()
        return _wrap(emb, "claude-fallback")
    except Exception as e:
        raise Exception(f"Claude fallback embedding error → {str(e)}")


########################################################################################################
# MAIN EMBEDDING LOGIC

def embed_text(text: str, provider: str):
    """
    Generate embeddings using an explicitly specified provider.

    This function MUST be called with a provider argument.
    Auto-selection is intentionally disallowed to guarantee
    embedding consistency within a repository.

    Returns:
    {
        "embedding": [...],
        "provider": "openai" | "gemini" | "claude-fallback" | "local"
    }
    """

    if not provider:
        raise ValueError("Embedding provider must be explicitly specified.")

    provider = provider.lower()

    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    claude_key = os.getenv("ANTHROPIC_API_KEY")

    try:
        if provider == "openai":
            if not openai_key:
                raise ValueError("Missing OPENAI_API_KEY")
            return embed_openai(text, openai_key)

        if provider == "gemini":
            if not gemini_key:
                raise ValueError("Missing GEMINI_API_KEY")
            return embed_gemini(text, gemini_key)

        if provider == "claude":
            # Claude has no embedding endpoint → fallback is explicit
            return embed_claude(text, claude_key)

        if provider == "local":
            return embed_local(text)

        raise ValueError(f"Unknown embedding provider '{provider}'")

    except Exception as e:
        # Fail loudly — do NOT silently fall back
        raise RuntimeError(
            f"Embedding failed using provider '{provider}': {str(e)}"
        )

