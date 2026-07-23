import pytest

from src.chunking import read_and_split_text


def test_read_and_split_text_returns_paragraphs(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "This is the first paragraph with enough characters to count.\n\n"
        "This is the second paragraph, also long enough to be kept.\n\n"
        "Short.\n\n"  # should be dropped -- below min_chars
        "This is the third and final paragraph in the test corpus."
    )

    paragraphs = read_and_split_text(corpus, min_chars=20)

    assert len(paragraphs) == 3
    assert "first paragraph" in paragraphs[0]
    assert "Short." not in paragraphs


def test_read_and_split_text_collapses_internal_whitespace(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("This paragraph\nspans   multiple\nlines with   odd spacing here.")

    paragraphs = read_and_split_text(corpus, min_chars=10)

    assert len(paragraphs) == 1
    assert "\n" not in paragraphs[0]
    assert "  " not in paragraphs[0]


def test_read_and_split_text_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises(FileNotFoundError):
        read_and_split_text(missing)


def test_read_and_split_text_empty_corpus_raises(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n\n\n")
    with pytest.raises(ValueError):
        read_and_split_text(corpus)
