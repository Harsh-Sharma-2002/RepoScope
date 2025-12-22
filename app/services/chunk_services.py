from ..schema import *


##################################################################################################################
##################################################################################################################

def chunk_text(text: str):
    """
    Semantic, line-based chunking for code and text.

    Properties:
    - Variable-length chunks
    - Prefer ending at logical boundaries
    - Never cuts mid-line
    - Language-agnostic (indent + braces heuristics)

    Chunk sizes are chosen based on downstream LLM context constraints rather than arbitrary limits.
    Chunks are structurally aware and variable-length, preferring logical boundaries (functions, classes, blocks) while enforcing a hard maximum size to keep token usage predictable.
    The max chunk size (~1000 characters) is derived from worst-case prompt assembly: the system retrieves the top-K relevant chunks and performs limited local context expansion (neighboring chunks). With this cap, even in the worst case, the expanded context fits safely within an ~8k token window.
    This approach balances semantic coherence, retrieval quality, and prompt reliability, while avoiding brittle assumptions about model context limits.
    """

    target_size = 850
    min_size = 700
    max_size = 1000

    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    chunks = []
    buffer = []
    buffer_len = 0
    chunk_id = 0

    indent_stack = [0]   # for indentation-based languages
    brace_depth = 0      # for brace-based languages

    def is_strong_boundary(line: str, prev_brace_depth: int) -> bool:
        stripped = line.strip()

        # Python: end of function or class definition
        if stripped.startswith(("def ", "class ")):
            return True

        # Optional: explicit main guard
        if stripped.startswith("if __name__"):
            return True

        # JS / C-like: closing brace of top-level block
        if stripped == "}" and prev_brace_depth == 1:
            return True

        return False

    for line in lines:
        buffer.append(line)
        buffer_len += len(line) + 1  # +1 for newline

        # Track indentation (Python-style)
        current_indent = len(line) - len(line.lstrip())
        if current_indent > indent_stack[-1]:
            indent_stack.append(current_indent)
        elif current_indent < indent_stack[-1]:
            while indent_stack and indent_stack[-1] > current_indent:
                indent_stack.pop()

        # ---- FIX: brace depth off-by-one ----
        prev_brace_depth = brace_depth
        brace_depth += line.count("{")
        brace_depth -= line.count("}")

        # Decide when to flush
        if (
            buffer_len >= min_size
            and is_strong_boundary(line, prev_brace_depth)
            and buffer_len <= max_size
        ) or buffer_len >= max_size:

            chunk = "\n".join(buffer).strip()
            if chunk:
                chunks.append((chunk_id, chunk))
                chunk_id += 1

            buffer = []
            buffer_len = 0

    # Flush remainder
    if buffer:
        chunk = "\n".join(buffer).strip()
        if chunk:
            chunks.append((chunk_id, chunk))

    return chunks




##################################################################################################################
##################################################################################################################

def chunk_repo_contents(repo_index: RepoIndexResponse) -> RepoChunksResponse:
    """
    Chunks repository contents into structurally coherent, LLM-safe code segments.

    Only source code files are indexed. Non-code artifacts are excluded to
    preserve retrieval precision and avoid polluting the vector index with
    low-signal data.
    """

    repo_chunks = []
    global_chunk_id = 0

    CODE_EXTENSIONS = {
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".go", ".rs", ".cpp", ".c",
        ".h", ".hpp", ".cs"
    }

    for item in repo_index.items:
        file_path = item.path
        content = item.content

        if not content.strip():
            continue

        filename = file_path.split("/")[-1]
        extension = "." + filename.split(".")[-1] if "." in filename else ""

        # Allowlist-based filtering
        if extension not in CODE_EXTENSIONS:
            continue

        if len(content) > 200_000:
            continue

        chunks = chunk_text(content)

        for local_id, chunk_content in chunks:
            if len(chunk_content) < 200:
                continue

            repo_chunks.append(
                RepoChunk(
                    file_path=file_path,
                    chunk_id=global_chunk_id,
                    content=chunk_content
                )
            )
            global_chunk_id += 1

    return RepoChunksResponse(chunks=repo_chunks)




##################################################################################################################
##################################################################################################################