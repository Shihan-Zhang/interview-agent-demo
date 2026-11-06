"""解析 qdrant-find 的返回:fastmcp 2.7 把 list[str] 序列化成一段 JSON 数组文本。"""

import json
import re
from xml.sax.saxutils import unescape


def find_text(result) -> str:
    items = result if isinstance(result, list) else getattr(result, "content", [])
    raw = "\n".join(c.text for c in items if getattr(c, "text", None))
    try:
        parts = json.loads(raw)
        return "\n".join(parts) if isinstance(parts, list) else str(parts)
    except json.JSONDecodeError:
        return raw


def chunk_ids(text: str) -> list[str]:
    """按返回顺序取出每条结果 metadata 里的 chunk_id。"""
    return [json.loads(unescape(m))["chunk_id"] for m in re.findall(r"<metadata>(.*?)</metadata>", text, re.S)]
