"""
Turns a raw text corpus into a list of clean, retrievable chunks.

The source notebooks split on newlines with no further cleaning. That's
fine for a tidy corpus but breaks on real-world text (multi-line
paragraphs, stray whitespace, empty lines). This version is a bit more
defensive so it survives being pointed at messier documents later.
"""

from pathlib import Path
from typing import List


def read_and_split_text(filepath: str | Path, min_chars: int = 20) -> List[str]:
    """
    Read a text file and split it into paragraph-level chunks.

    Paragraphs are separated by blank lines (double newline), which
    matches how the corpus.txt in this project is authored. Chunks
    shorter than `min_chars` are dropped -- they're usually stray
    whitespace or section headers with no retrievable content.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Corpus file not found: {filepath}")

    text = filepath.read_text(encoding="utf-8")

    raw_paragraphs = text.split("\n\n")
    paragraphs = [_clean(p) for p in raw_paragraphs]
    paragraphs = [p for p in paragraphs if len(p) >= min_chars]

    if not paragraphs:
        raise ValueError(
            f"No usable paragraphs found in {filepath}. "
            "Check that the file has blank-line-separated paragraphs."
        )

    return paragraphs


def _clean(paragraph: str) -> str:
    """
    Collapse internal newlines and repeated whitespace into single spaces.

    This turns a multi-line paragraph into one clean line, so downstream
    embedding and display code never has to deal with stray formatting.
    """
    return " ".join(paragraph.split())
