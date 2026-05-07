import os

import httpx
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

_OLLAMA_CLOUD = "https://ollama.com/api"

TOOLS = ToolsSchema(
    standard_tools=[
        FunctionSchema(
            name="web_search",
            description=(
                "Tìm kiếm thông tin trên internet. Dùng khi cần dữ liệu cập nhật, "
                "tin tức hiện tại, hoặc thông tin ngoài kiến thức sẵn có."
            ),
            properties={
                "query": {
                    "type": "string",
                    "description": "Từ khóa hoặc câu hỏi cần tìm kiếm",
                }
            },
            required=["query"],
        ),
        FunctionSchema(
            name="web_fetch",
            description="Lấy nội dung từ một URL cụ thể trên internet.",
            properties={
                "url": {
                    "type": "string",
                    "description": "URL đầy đủ cần lấy nội dung",
                }
            },
            required=["url"],
        ),
    ]
)


def _api_key() -> str | None:
    return os.getenv("OLLAMA_API_KEY")


async def handle_web_search(params: FunctionCallParams):
    query = params.arguments.get("query", "")
    logger.info(f"[web_search] query={query!r}")

    key = _api_key()
    if not key:
        await params.result_callback("Lỗi: OLLAMA_API_KEY chưa được cấu hình trong .env")
        return

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{_OLLAMA_CLOUD}/web_search",
                headers={"Authorization": f"Bearer {key}"},
                json={"query": query, "max_results": 3},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.exception(f"[web_search] error: {e}")
        await params.result_callback(f"Lỗi tìm kiếm: {e}")
        return

    results = data.get("results", [])
    if not results:
        await params.result_callback("Không tìm thấy kết quả nào.")
        return

    parts = []
    for r in results[:3]:
        title = r.get("title", "")
        content = r.get("content", "")[:500]
        url = r.get("url", "")
        parts.append(f"{title}: {content} ({url})")

    await params.result_callback(" | ".join(parts))


async def handle_web_fetch(params: FunctionCallParams):
    url = params.arguments.get("url", "")
    logger.info(f"[web_fetch] url={url!r}")

    key = _api_key()
    if not key:
        await params.result_callback("Lỗi: OLLAMA_API_KEY chưa được cấu hình trong .env")
        return

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                f"{_OLLAMA_CLOUD}/web_fetch",
                headers={"Authorization": f"Bearer {key}"},
                json={"url": url},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.exception(f"[web_fetch] error: {e}")
        await params.result_callback(f"Lỗi fetch: {e}")
        return

    title = data.get("title", "")
    content = data.get("content", "")
    snippet = content[:1500] + ("..." if len(content) > 1500 else "")
    await params.result_callback(f"{title}\n\n{snippet}" if snippet else "Không lấy được nội dung.")
