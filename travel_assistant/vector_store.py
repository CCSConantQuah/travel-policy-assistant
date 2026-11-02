"""OpenAI embeddings and one active, persistent local Chroma knowledge base."""

import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from travel_assistant.documents import PolicyError, chunk_pages, extract_pdf

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
DEFAULT_STORE = Path(__file__).resolve().parents[1] / "data" / "chroma_db"
# Serialize publication, retrieval and old-collection deletion across Streamlit sessions.
# This local proof of concept supports one app process, not multiple server workers.
_STORE_LOCK = threading.RLock()


def embed_texts(texts: list[str], api_key: str) -> list[list[float]]:
    """Use only the embeddings endpoint; bounded batches and sanitized failures."""
    if not api_key.strip():
        raise PolicyError("Set OPENAI_API_KEY in .env or Streamlit secrets, then restart the app.")
    vectors = []
    try:
        with OpenAI(api_key=api_key, timeout=60.0, max_retries=1) as client:
            for start in range(0, len(texts), 64):
                batch = texts[start:start + 64]
                response = client.embeddings.create(
                    model=MODEL, input=batch, encoding_format="float",
                )
                items = sorted(response.data, key=lambda item: item.index)
                if [item.index for item in items] != list(range(len(batch))):
                    raise ValueError("Incomplete embeddings response")
                for item in items:
                    vector = item.embedding
                    if len(vector) != DIMENSIONS or not all(math.isfinite(v) for v in vector):
                        raise ValueError("Invalid embedding vector")
                    vectors.append(vector)
    except Exception:
        raise PolicyError("Embedding request failed. Check your API key, quota, and connection, then retry. The active policy has not been changed.") from None
    return vectors


def _client(store_dir: Path):
    return chromadb.PersistentClient(
        path=str(store_dir), settings=Settings(anonymized_telemetry=False),
    )


def _read_active(store_dir: Path) -> dict | None:
    path = store_dir / "active_policy.json"
    if not path.exists():
        return None
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(info["collection"], str)
                or not info["collection"].startswith("policy_")
                or info["model"] != MODEL
                or info["status"] != "Ready"
                or not isinstance(info["processed_at"], str)
                or not isinstance(info["filename"], str)
                or not isinstance(info["page_count"], int)
                or not isinstance(info["chunk_count"], int)
                or not 1 <= info["text_page_count"] <= info["page_count"]
                or info["chunk_count"] < 1):
            raise ValueError("Invalid manifest")
        return info
    except Exception:
        raise PolicyError("The active policy record is unreadable. Check the local data directory before processing another policy.") from None


def _active_collection(client, info: dict):
    collection = client.get_collection(info["collection"], embedding_function=None)
    if collection.count() != info["chunk_count"]:
        raise ValueError("Stored chunk count does not match the active policy")
    return collection


def get_active_policy(store_dir: Path = DEFAULT_STORE) -> dict | None:
    """Read persisted status and verify the referenced collection still exists."""
    with _STORE_LOCK:
        info = _read_active(store_dir)
        if info is not None:
            try:
                _active_collection(_client(store_dir), info)
            except Exception:
                raise PolicyError("The active policy database could not be read. Check local storage and permissions.") from None
        return info


def _publish(store_dir: Path, info: dict) -> None:
    """Atomically replace only the manifest after the new collection is complete."""
    temporary = store_dir / f"active_{uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, store_dir / "active_policy.json")
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass  # A cleanup error must never undo a published policy.


def process_policy(pdf_bytes: bytes, filename: str, api_key: str,
                   store_dir: Path = DEFAULT_STORE) -> dict:
    """Build in isolation; switch the active pointer only after storage succeeds."""
    pages, page_count = extract_pdf(pdf_bytes, filename)
    chunks = chunk_pages(pages)
    if not chunks:
        raise PolicyError("No usable text chunks were found in this PDF.")
    with _STORE_LOCK:
        # Do not overwrite an unreadable active record.
        previous = get_active_policy(store_dir)
        vectors = embed_texts([chunk.page_content for chunk in chunks], api_key)
        client = None
        collection_name = "policy_" + uuid4().hex
        info = {
            "collection": collection_name,
            "filename": pages[0].metadata["source"],
            "page_count": page_count,
            "text_page_count": len(pages),
            "chunk_count": len(chunks),
            "model": MODEL,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "status": "Ready",
        }
        try:
            store_dir.mkdir(parents=True, exist_ok=True)
            client = _client(store_dir)
            collection = client.create_collection(
                collection_name, embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}},
            )
            for start in range(0, len(chunks), 64):
                batch = chunks[start:start + 64]
                collection.add(
                    ids=[str(i) for i in range(start, start + len(batch))],
                    documents=[chunk.page_content for chunk in batch],
                    metadatas=[chunk.metadata for chunk in batch],
                    embeddings=vectors[start:start + len(batch)],
                )
            if collection.count() != len(chunks):
                raise ValueError("Incomplete storage")
            _publish(store_dir, info)
        except Exception:
            if client is not None:
                try:
                    client.delete_collection(collection_name)
                except Exception:
                    pass  # Inactive staging data never participates in retrieval.
            raise PolicyError("Policy storage failed. Check disk space and permissions. The previous active policy is unchanged.") from None
        # Publication succeeded. Cleanup failure must not invalidate the new policy.
        if previous:
            try:
                client.delete_collection(previous["collection"])
            except Exception:
                info["cleanup_warning"] = "The new policy is active, but inactive data could not be removed from disk."
        return info


def retrieve_policy(query: str, api_key: str, store_dir: Path = DEFAULT_STORE,
                    *, expected_collection: str | None = None) -> list[dict]:
    """Return up to three chunks; optionally require a specific policy version."""
    query = query.strip()
    if not query or len(query) > 1000:
        raise PolicyError("Enter a search query between 1 and 1,000 characters.")
    with _STORE_LOCK:
        info = get_active_policy(store_dir)
        if info is None:
            raise PolicyError("Process a synthetic policy PDF before testing retrieval.")
        if expected_collection is not None and info["collection"] != expected_collection:
            raise PolicyError("The active policy changed. Please ask your question again.")
        query_vector = embed_texts([query], api_key)[0]
        try:
            collection = _active_collection(_client(store_dir), info)
            results = collection.query(
                query_embeddings=[query_vector], n_results=min(3, info["chunk_count"]),
                include=["documents", "metadatas", "distances"],
            )
            return [
                {"text": text, **metadata, "distance": distance}
                for text, metadata, distance in zip(
                    results["documents"][0], results["metadatas"][0], results["distances"][0],
                )
            ]
        except Exception:
            raise PolicyError("Policy retrieval failed. Check the local database and retry.") from None
