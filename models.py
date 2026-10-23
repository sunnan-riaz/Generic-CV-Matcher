"""
models.py
=========
Loads and caches the two pretrained models used by the app:

  1. GLiNER (urchade/gliner_small-v2.1)
     - Named-entity recognition model fine-tuned for open-label extraction.
     - Used here to pull skill-related phrases from free text.

  2. Sentence-Transformers (sentence-transformers/all-MiniLM-L6-v2)
     - Encodes sentences into fixed-length vectors so we can measure
       how similar two pieces of text are in meaning.

Both models are downloaded from Hugging Face Hub on first run (~100 MB total)
and then cached locally by their respective libraries.

The @st.cache_resource decorator tells Streamlit to load each model once per
server session and reuse the same object on every interaction — this avoids
reloading 100 MB of weights every time the user clicks a button.
"""

import streamlit as st

# ── Suppress noisy INFO logs from underlying libraries ─────────────────────
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


@st.cache_resource(show_spinner=False)
def load_gliner():
    """
    Download (first run) and cache the GLiNER small model.

    Returns the model object, or None if loading fails.

    GLiNER is a zero-shot NER model: you give it a list of *label names* and
    it finds spans in the text that match those concepts — no fixed skill list
    needed.
    """
    try:
        from gliner import GLiNER  # type: ignore
        model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
        return model
    except Exception as exc:
        st.error(
            f"Failed to load GLiNER: {exc}\n\n"
            "Check your internet connection. On first run the model (~50 MB) "
            "is downloaded from Hugging Face Hub."
        )
        return None


@st.cache_resource(show_spinner=False)
def load_sentence_transformer():
    """
    Download (first run) and cache the Sentence-Transformer model.

    Returns the SentenceTransformer object, or None if loading fails.

    The model converts a piece of text into a 384-dimensional vector.
    Two texts with similar meaning will have vectors that point in roughly
    the same direction (high cosine similarity).
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return model
    except Exception as exc:
        st.error(
            f"Failed to load Sentence-Transformer: {exc}\n\n"
            "Check your internet connection. On first run the model (~50 MB) "
            "is downloaded from Hugging Face Hub."
        )
        return None
