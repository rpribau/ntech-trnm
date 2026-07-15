from config.settings import Settings
from ntech_agent.ingest.chunking import _chunk_file


def test_chunk_python_metadata(tmp_path):
    f = tmp_path / "module.py"
    f.write_text("def foo():\n    return 1\n\n\n" * 40, encoding="utf-8")

    docs = list(_chunk_file(f, "myrepo", "module.py", Settings()))
    assert docs, "debería producir al menos un chunk"
    meta = docs[0].metadata
    assert meta["repo"] == "myrepo"
    assert meta["path"] == "module.py"
    assert meta["language"] == "python"
    assert meta["source_type"] == "code"
    assert "id" in meta and isinstance(meta["id"], str)
    assert meta["breadcrumb"].startswith("myrepo > module.py")


def test_chunk_ids_are_deterministic(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n" * 100, encoding="utf-8")

    ids1 = [d.metadata["id"] for d in _chunk_file(f, "r", "a.py", Settings())]
    ids2 = [d.metadata["id"] for d in _chunk_file(f, "r", "a.py", Settings())]
    assert ids1 == ids2


def test_skip_unknown_extension(tmp_path):
    f = tmp_path / "image.png"
    f.write_bytes(b"\x89PNG\r\n")
    assert list(_chunk_file(f, "r", "image.png", Settings())) == []
