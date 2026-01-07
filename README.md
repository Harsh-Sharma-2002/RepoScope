RepoScope — Repository-Aware Code Intelligence & Review System
==============================================================

RepoScope is a repository-aware code intelligence backend that uses
retrieval-augmented generation (RAG) to explain and review code with
full awareness of its surrounding codebase.

Unlike naive “chat with code” tools, RepoScope indexes entire repositories,
retrieves semantically relevant cross-file context, and uses that context
to power file-level explanations, code reviews, and system-level
repository reviews.

This project emphasizes correct system design, retrieval quality,
and real-world LLM integration constraints, rather than brittle demos.


Key Features
------------

Repository-Aware Retrieval (RAG)
- Indexes full repositories into a persistent vector database
- Retrieves semantically relevant code chunks across files
- Enforces strict file-local context expansion
- Deterministic, bounded retrieval suitable for production systems

Code Understanding
- Explain the role and responsibility of any file in its repository context
- Describe how files interact with surrounding code
- Focuses on architecture and data flow (not verbatim code repetition)

Code Review
- Structured file-level code review
- Identifies bugs, design issues, and maintainability concerns
- Outputs machine-readable JSON with severity-tagged comments

Repository-Level Review
- High-level architectural review of an entire codebase
- Identifies systemic risks and technical debt
- Useful for onboarding and architectural assessment

Pluggable LLM Backends (BYOK)
- Bring-Your-Own-Key (BYOK) LLM integration
- Clean abstraction over multiple providers:
  - OpenAI
  - Anthropic
  - Google Gemini
  - Hugging Face (local or dedicated inference)
- No vendor lock-in at the API level


System Architecture
-------------------

Client
  |
  v
FastAPI Routes
  |
  v
Service Layer
  |-- Retrieval (Vector DB)
  |-- Prompt Construction
  |-- LLM Dispatch
  |-- Output Parsing
  v
LLM Backend (Pluggable)

Design Principles
- Retrieval correctness > prompt size
- File-local context expansion only
- Explicit, structured outputs
- Fail loudly on invalid assumptions


Project Structure
-----------------

app/
├── main.py
├── schema.py
├── routes/
│   ├── llm_routes.py
│   ├── vector_db_routes.py
│   ├── repo_index_routes.py
│   ├── chunk_routes.py
│   ├── embedding_routes.py
│   └── pr_routes.py
├── services/
│   ├── vector_db_services.py
│   ├── repo_index_services.py
│   ├── chunk_services.py
│   ├── embedding_services.py
│   ├── llm_services.py
│   └── pr_services.py
└── utils.py


Retrieval Pipeline
------------------

1. Repository Indexing
- Repository cloned or crawled
- Only source files are indexed
- Binary files and large assets are skipped

2. Semantic Chunking
- Language-agnostic, line-based chunking
- Prefers logical boundaries (functions, classes)
- Chunk sizes tuned for safe prompt assembly
- No static overlap; locality preserved via indices

3. Vector Storage
- Persistent ChromaDB storage
- One collection per repository
- Deterministic, rebuildable vector index

4. Context Retrieval
- Semantic vector search for external anchors
- Same-file matches excluded
- File-local window expansion
- Bounded, deduplicated context windows


LLM Integration Notes
---------------------

RepoScope abstracts LLM usage via a provider interface.

IMPORTANT:
Hugging Face’s free serverless inference is non-guaranteed and
region-dependent.

Large open-source models such as CodeLLaMA, DeepSeek-Coder,
Qwen-Coder, and Evo-8k require:
- Local inference (vLLM / TGI), or
- A dedicated Hugging Face Inference Endpoint

RepoScope supports these backends cleanly, but free HF inference
may be unavailable depending on region and load.
This is an infrastructure constraint, not a project limitation.


API Endpoints
-------------

Explain a File
POST /llm/explain-file

Request:
{
  "owner": "facebook",
  "repo": "react",
  "file_path": "packages/react/src/ReactHooks.js",
  "llm_provider": "openai",
  "llm_api_key": "YOUR_API_KEY"
}

Response:
{
  "explanation": "This file defines core React Hooks and their integration with the reconciler...",
  "provider": "openai"
}


Review a File
POST /llm/review-file

Response:
{
  "summary": "The file is well-structured but contains several maintainability risks...",
  "comments": [
    {
      "message": "Implicit dependency on global state may cause subtle bugs.",
      "severity": "medium"
    }
  ],
  "provider": "openai"
}


Review an Entire Repository
POST /llm/review-repo

Response:
{
  "summary": "The repository follows a modular design but exhibits tight coupling in core services.",
  "key_risks": [
    {
      "message": "Business logic and infrastructure concerns are intertwined.",
      "severity": "high"
    }
  ],
  "design_observations": [
    "Clear separation between API and service layers.",
    "Lack of abstraction around persistence layer."
  ],
  "provider": "openai"
}


Getting Started
---------------

1. Install Dependencies
pip install -r requirements.txt

2. Set Environment Variables
export GITHUB_TOKEN=your_github_pat
export CHROMA_PERSISTENT_DIR=./chromaDB

3. Run the Server
uvicorn app.main:app --reload


Why RepoScope Is Different
--------------------------

- Correct RAG design (not prompt stuffing)
- Deterministic retrieval logic
- Explicit handling of real-world LLM infra limitations
- Clean service boundaries
- Machine-readable outputs
- Phase-aware design (no premature features)

RepoScope prioritizes engineering correctness over flashy demos.


Roadmap
-------

Phase 1 (Completed)
- Repository indexing & semantic chunking
- Vector search with file-local context expansion
- File explanation service
- File review service
- Repository-level review service
- Pluggable LLM backends (BYOK)

Phase 2 (Planned)
- PR diff-aware reviews
- Local inference backend (vLLM / TGI)
- Symbol-level explanations (functions/classes)
- Memory-aware follow-up reviews
- Performance optimizations (repo cache)


License
-------

MIT


Acknowledgements
----------------

Built to explore real-world retrieval-augmented code intelligence,
with an emphasis on correctness, clarity, and practical system constraints.
