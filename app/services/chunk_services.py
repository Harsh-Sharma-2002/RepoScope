from ..schema import *


##################################################################################################################
##################################################################################################################

def chunk_text(text: str,target_size: int = 850,min_size: int = 700,max_size: int = 1000):
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

    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    chunks = []
    buffer = []
    buffer_len = 0
    chunk_id = 0

    indent_stack = [0]   # for indentation-based languages
    brace_depth = 0      # for brace-based languages

    def is_strong_boundary(line: str) -> bool:
        stripped = line.strip()

        # End of Python function / class (dedent to top-level)
        if stripped and not line.startswith((" ", "\t")):
            return True

        # End of JS/C-like block at top-level
        if stripped == "}" and brace_depth == 0:
            return True

        return False


    for line in lines:
        buffer.append(line)
        buffer_len += len(line) + 1  # +1 for newline

        # Update indentation stack (Python-style)
        current_indent = len(line) - len(line.lstrip())
        if current_indent > indent_stack[-1]:
            indent_stack.append(current_indent)
        elif current_indent < indent_stack[-1]:
            while indent_stack and indent_stack[-1] > current_indent:
                indent_stack.pop()

        # Update brace depth (JS / C-like)
        brace_depth += line.count("{")
        brace_depth -= line.count("}")

        # Decide when to flush
        if (
            buffer_len >= min_size
            and is_strong_boundary(line)
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

def chunk_repo_contents(repo_index: RepoIndexResponse,chunk_size: int = 800,overlap: int = 200) -> RepoChunksResponse:
    
    repo_chunks = []
    global_chunk_id = 0

    for item in repo_index.items:

        file_path = item.path
        content = item.content

        # Skip empty files
        if not content.strip():
            continue

        # Skip extremely large files
        if len(content) > 200_000:
            continue

        chunks = chunk_text(content, chunk_size, overlap)

        for local_id, chunk_content in chunks:
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