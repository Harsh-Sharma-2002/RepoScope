from ..services.vector_db_services import vector_search_service
from ..schema import VectorSearchResponse

def extract_content(res: VectorSearchResponse) -> list[str]:
    windows = []

    for result in res.results:
        content = "\n\n".join(
            chunk.content for chunk in result.context_chunks
        )
        windows.append(content)

    return windows



