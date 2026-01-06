class WorkingMemory:
    def __init__(self, max_items: int = 5):
        self.max_items = max_items
        self.items: list[str] = []

    def add(self, summary: str):
        self.items.append(summary.strip())
        self.items = self.items[-self.max_items:]

    def render(self) -> str:
        if not self.items:
            return ""

        return (
            "WORKING CONTEXT (from earlier analysis):\n"
            + "\n".join(f"- {item}" for item in self.items)
        )


# simple in-memory store (Phase-1)
_memory_store: dict[str, WorkingMemory] = {}


def get_memory(repo_id: str) -> WorkingMemory:
    if repo_id not in _memory_store:
        _memory_store[repo_id] = WorkingMemory()
    return _memory_store[repo_id]


def reset_memory(repo_id: str):
    _memory_store.pop(repo_id, None)
