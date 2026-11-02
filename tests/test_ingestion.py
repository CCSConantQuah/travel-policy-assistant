"""Synthetic PDFs, real local Chroma, and mocked embedding API calls."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pymupdf
import pytest

from travel_assistant import vector_store as store
from travel_assistant.documents import PolicyError, chunk_pages, extract_pdf


def make_pdf(*texts):
    """Build small fictional PDFs in memory; no document fixtures are committed."""
    with pymupdf.open() as pdf:
        for text in texts:
            page = pdf.new_page()
            if text:
                page.insert_text((50, 60), text)
        return pdf.tobytes()


def fake_embeddings(texts, api_key):
    """Deterministic semantic stand-in: flights, hotels, approvals."""
    vectors = []
    for text in texts:
        lower = text.lower()
        vector = [float(word in lower) for word in ("flight", "hotel", "approval")]
        vectors.append(vector + [1.0] + [0.0] * (store.DIMENSIONS - 4))
    return vectors


@pytest.fixture
def embedded_store(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "embed_texts", fake_embeddings)
    return tmp_path / "chroma_db"


def test_extract_preserves_physical_pages_and_filename():
    pages, count = extract_pdf(make_pdf("Fictional flight policy", "", "Hotel policy"), "folder/policy.pdf")
    assert count == 3
    assert [page.metadata for page in pages] == [
        {"source": "policy.pdf", "page": 1}, {"source": "policy.pdf", "page": 3},
    ]
    assert "flight" in pages[0].page_content


@pytest.mark.parametrize("content,filename", [
    (b"broken PDF", "broken.pdf"), (b"", "empty.pdf"),
    (make_pdf(""), "scan.pdf"), (make_pdf("text"), "policy.txt"),
])
def test_invalid_or_textless_pdf(content, filename):
    with pytest.raises(PolicyError):
        extract_pdf(content, filename)


def test_password_protected_pdf():
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((50, 60), "Fictional policy")
        encrypted = pdf.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="demo", owner_pw="owner")
    with pytest.raises(PolicyError, match="password"):
        extract_pdf(encrypted, "locked.pdf")


def test_chunk_metadata_and_overlap():
    pages, _ = extract_pdf(make_pdf("Flight rules", "Hotel rules"), "policy.pdf")
    pages[0].page_content = "A" * 2300
    chunks = chunk_pages(pages)
    assert len(chunks) == 4
    assert all(0 < len(chunk.page_content) <= 1000 for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(4))
    assert [chunk.metadata["page"] for chunk in chunks] == [1, 1, 1, 2]
    assert all(chunk.metadata["source"] == "policy.pdf" for chunk in chunks)
    assert sum(len(chunk.page_content) for chunk in chunks[:3]) > 2300


def test_retrieval_top_three_and_persisted_metadata(embedded_store):
    info = store.process_policy(make_pdf("Flight economy rules", "Hotel nightly rates", "Approval required", "General travel"),
                                "policy.pdf", "mock-key", embedded_store)
    assert info["page_count"] == info["chunk_count"] == 4
    assert store.get_active_policy(embedded_store) == info
    results = store.retrieve_policy("hotel", "mock-key", embedded_store)
    assert len(results) == 3
    assert results[0]["page"] == 2
    assert "Hotel" in results[0]["text"]
    assert all(result["source"] == "policy.pdf" for result in results)


def test_replacement_uses_only_new_policy(embedded_store):
    old = store.process_policy(make_pdf("Flight old policy"), "old.pdf", "mock-key", embedded_store)
    new = store.process_policy(make_pdf("Hotel replacement", "Approval replacement"), "new.pdf", "mock-key", embedded_store)
    assert store.get_active_policy(embedded_store) == new
    results = store.retrieve_policy("flight", "mock-key", embedded_store)
    assert len(results) == 2
    assert all(item["source"] == "new.pdf" for item in results)
    names = [collection.name for collection in store._client(embedded_store).list_collections()]
    assert names == [new["collection"]]
    assert old["collection"] not in names


@pytest.mark.parametrize("failure", ["embeddings", "storage", "publication"])
def test_failed_replacement_preserves_old_policy(embedded_store, monkeypatch, failure):
    old = store.process_policy(make_pdf("Flight original"), "original.pdf", "mock-key", embedded_store)
    with monkeypatch.context() as patch:
        if failure == "embeddings":
            patch.setattr(store, "embed_texts", MagicMock(side_effect=PolicyError("Embedding request failed.")))
        elif failure == "storage":
            real_client = store._client(embedded_store)
            wrapped = MagicMock(wraps=real_client)
            def create(*args, **kwargs):
                collection = real_client.create_collection(*args, **kwargs)
                broken = MagicMock(wraps=collection)
                broken.add.side_effect = OSError("Simulated disk failure")
                return broken
            wrapped.create_collection.side_effect = create
            patch.setattr(store, "_client", lambda path: wrapped)
        else:
            patch.setattr(store.os, "replace", MagicMock(side_effect=OSError("Simulated publication failure")))
        with pytest.raises(PolicyError):
            store.process_policy(make_pdf("Hotel replacement"), "new.pdf", "mock-key", embedded_store)
    assert store.get_active_policy(embedded_store) == old
    assert store.retrieve_policy("flight", "mock-key", embedded_store)[0]["source"] == "original.pdf"
    assert len(store._client(embedded_store).list_collections()) == 1


def test_missing_key_and_api_failure_are_sanitized(monkeypatch):
    constructor = MagicMock(side_effect=RuntimeError("sensitive-provider-details"))
    monkeypatch.setattr(store, "OpenAI", constructor)
    with pytest.raises(PolicyError, match="OPENAI_API_KEY"):
        store.embed_texts(["Fictional policy"], "")
    constructor.assert_not_called()
    with pytest.raises(PolicyError) as error:
        store.embed_texts(["Fictional policy"], "mock-key")
    assert "sensitive-provider-details" not in str(error.value)
    assert "mock-key" not in str(error.value)


def test_embedding_model_batching_and_response_order(monkeypatch):
    client = MagicMock()
    vector = [0.5] * store.DIMENSIONS
    def response(**kwargs):
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=vector)
                                     for i in reversed(range(len(kwargs["input"])))])
    client.embeddings.create.side_effect = response
    constructor = MagicMock()
    constructor.return_value.__enter__.return_value = client
    monkeypatch.setattr(store, "OpenAI", constructor)
    assert len(store.embed_texts(["Fictional policy"] * 65, "mock-key")) == 65
    calls = client.embeddings.create.call_args_list
    assert [len(call.kwargs["input"]) for call in calls] == [64, 1]
    assert all(call.kwargs["model"] == "text-embedding-3-small" for call in calls)


def test_no_policy_or_empty_query_never_calls_api(tmp_path, monkeypatch):
    embed = MagicMock()
    monkeypatch.setattr(store, "embed_texts", embed)
    with pytest.raises(PolicyError, match="before testing"):
        store.retrieve_policy("hotel", "mock-key", tmp_path)
    with pytest.raises(PolicyError, match="search query"):
        store.retrieve_policy(" ", "mock-key", tmp_path)
    embed.assert_not_called()


def test_cleanup_failure_does_not_undo_new_policy(embedded_store, monkeypatch):
    store.process_policy(make_pdf("Flight original"), "old.pdf", "mock-key", embedded_store)
    client = MagicMock(wraps=store._client(embedded_store))
    client.delete_collection.side_effect = OSError("Simulated cleanup failure")
    monkeypatch.setattr(store, "_client", lambda path: client)
    new = store.process_policy(make_pdf("Hotel replacement"), "new.pdf", "mock-key", embedded_store)
    assert "cleanup_warning" in new
    assert store.get_active_policy(embedded_store)["filename"] == "new.pdf"
    assert store.retrieve_policy("flight", "mock-key", embedded_store)[0]["source"] == "new.pdf"
