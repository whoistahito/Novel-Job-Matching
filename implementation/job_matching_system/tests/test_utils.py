from utils import chunk_markdown


def test_chunk_markdown_empty():
    assert chunk_markdown("", chunk_size=10) == []


def test_chunk_markdown_single_small_chunk():
    text = "# Title\nHello world"
    chunks = chunk_markdown(text, chunk_size=10_000)
    assert chunks == [text]


def test_chunk_markdown_splits_on_headings():
    text = "# A\nOne\n## B\nTwo\n### C\nThree\n"
    chunks = chunk_markdown(text, chunk_size=3)  # chars_per_chunk=12
    assert len(chunks) >= 2
    assert "# A" in chunks[0]
    assert any("## B" in c or "### C" in c for c in chunks)


def test_chunk_markdown_fallback_chunks_by_size_when_too_large():
    chunk_size = 2  # chars_per_chunk=8
    text = "x" * 50
    chunks = chunk_markdown(text, chunk_size=chunk_size)

    assert all(len(c) <= chunk_size * 4 for c in chunks)
    assert "".join(chunks) == text
