"""
extraction.py
=============
Functions for extracting skill-related phrases from text using GLiNER.

GLiNER works like a flexible search engine for concepts: instead of
matching a fixed list of known skills, you provide human-readable *label
names* and the model finds text spans that match those concepts.

Labels used here:
  - "skill"          – general competency (e.g. "project management")
  - "software tool"  – named application or platform (e.g. "Jira", "Excel")
  - "technology"     – technology, framework, language (e.g. "Python", "AWS")
  - "method"         – approach or process (e.g. "agile", "A/B testing")
  - "certification"  – formal qualification (e.g. "PMP", "AWS Certified")

The model runs each sentence through a neural network and returns spans
(start char, end char, text) together with a confidence score.
"""

from typing import List

# Labels that GLiNER will search for
SKILL_LABELS = [
    "skill",
    "software tool",
    "technology",
    "method",
    "certification",
]

# Only accept spans with at least this confidence (0–1 scale)
MIN_CONFIDENCE = 0.40


def extract_phrases(model, text: str) -> List[str]:
    """
    Run GLiNER on *text* and return a deduplicated list of skill-related phrases.

    Parameters
    ----------
    model : GLiNER instance (loaded via models.load_gliner)
    text  : raw string from the JD or CV

    Returns
    -------
    List[str] – unique, stripped phrases in the order they appear.

    Notes
    -----
    - GLiNER has a token limit (~512 tokens). We split long texts into
      overlapping windows so no content is skipped.
    - Results are lowercased and deduplicated to make comparisons easier.
    """
    if not text or not text.strip():
        return []

    # Split into overlapping chunks to handle long documents
    chunks = _split_into_chunks(text, chunk_size=400, overlap=50)

    seen: set = set()
    phrases: List[str] = []

    for chunk in chunks:
        try:
            entities = model.predict_entities(
                chunk,
                labels=SKILL_LABELS,
                threshold=MIN_CONFIDENCE,
            )
        except Exception:
            # If a chunk fails (e.g. encoding error), skip it gracefully
            continue

        for ent in entities:
            phrase = ent["text"].strip().lower()
            # Skip very short or very long spans (likely noise)
            if len(phrase) < 2 or len(phrase) > 80:
                continue
            if phrase not in seen:
                seen.add(phrase)
                phrases.append(phrase)

    return phrases


def _split_into_chunks(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Split *text* into word-based chunks of at most *chunk_size* words,
    with *overlap* words shared between adjacent chunks.

    This prevents a skill phrase that spans a chunk boundary from being lost.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap  # step forward, keeping some overlap

    return chunks
