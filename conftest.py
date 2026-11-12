import os
import tempfile

os.environ["LLM_ENABLED"] = "0"  # keep tests offline and deterministic
os.environ["FAQ_ENABLED"] = "0"  # FAQ cache/vector tests enable it explicitly
os.environ["GEMINI_API_KEY"] = ""  # never call Gemini from tests; chunking falls back to the offline splitter
os.environ["MCP_SERVER_URL"] = ""  # use the in-process MCP server, not a running service
os.environ["VALKEY_URL"] = ""  # never use a developer's real Valkey cache in tests
os.environ["DB_ENABLED"] = "0"  # tests use the JSON dummy data, not MySQL
os.environ["SESSION_DB"] = os.path.join(tempfile.mkdtemp(), "sessions.db")
os.environ["REGISTERED_USERS_FILE"] = os.path.join(tempfile.mkdtemp(), "registered_users.json")
# Tests run against Qdrant's in-memory embedded mode with a deterministic embedder.
os.environ.pop("QDRANT_URL", None)
os.environ["QDRANT_ENABLED"] = "1"
os.environ["QDRANT_PATH"] = ":memory:"

import json  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import zlib  # noqa: E402

from qdrant_client import models  # noqa: E402

from mcp_server import embeddings, vectordb  # noqa: E402

STOPWORDS = {
    "what", "is", "the", "my", "i", "do", "how", "a", "an", "to", "for", "if", "me", "in", "of", "and", "can",
    "does", "are", "am", "or", "with", "your", "you", "on", "it", "be",
}


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Hashed bag-of-words so semantic tests run offline: texts sharing words are close in cosine distance."""
    out = []
    for text in texts:
        vec = [0.0] * embeddings.DIMENSIONS
        for word in set(vectordb.normalize(text).split()) - STOPWORDS:
            vec[zlib.crc32(word.encode()) % embeddings.DIMENSIONS] = 1.0
        out.append(vec if any(vec) else [0.0] * (embeddings.DIMENSIONS - 1) + [0.001])
    return out


embeddings.set_embedder(fake_embed)
vectordb.ensure_ready()
vectordb.ingest_document("test-policy", "policy.txt", [["P10 says about late payment: the grace period is 30 days and quarterly premiums are $375."]])
_settings = json.loads((Path(__file__).resolve().parents[1] / "app" / "data" / "plan_settings.json").read_text(encoding="utf-8"))
vectordb._upsert(vectordb.SETTINGS, [  # test fixture only: the app inserts no settings
    models.PointStruct(id=vectordb._id("setting", k), vector=[1.0],
                       payload={"key": k, "value": v["value"], "description": v.get("description", "")})
    for k, v in _settings.items()
])

