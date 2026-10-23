"""
scoring.py
==========
All scoring logic for the CV–Job Match app.

Scoring overview (prototype – unvalidated weights)
---------------------------------------------------
Step 1  Direct-mention check
        Search the CV text for the requirement phrase (case-insensitive).
        If found:
          • Check for limitation words (e.g. "no experience", "unfamiliar with").
          • Credit = 1.0 if no limitation, 0.0 if limitation detected.

Step 2  Semantic fallback
        Encode the requirement and every CV sentence with the Sentence-
        Transformer. If the best cosine-similarity ≥ threshold, credit = 0.5
        and the matching CV sentence is stored as "evidence" for human review.

Step 3  Weighted coverage
        coverage = 100 × Σ(credit × weight) / Σ(weight)
        where weight = 2 for essential, 1 for desirable.

Step 4  Meaning similarity
        Encode the full JD in 5-sentence windows, do the same for the CV,
        find the best-matching CV window for each JD window, and average
        the cosine similarities.

Step 5  Final rating
        rating = 0.70 × coverage + 0.30 × (meaning_similarity × 100)

All weights and thresholds are *unvalidated design choices* for demonstration.
"""

from __future__ import annotations

import re
from typing import List, Tuple, Dict, Any

import numpy as np

# ── Words that suggest a candidate *lacks* the skill ──────────────────────────
LIMITATION_PATTERNS = [
    r"\bno\s+experience\b",
    r"\bnot\s+experienced\b",
    r"\bunfamiliar\s+with\b",
    r"\bno\s+knowledge\b",
    r"\blimited\s+experience\b",
    r"\bnever\s+used\b",
    r"\bnot\s+proficient\b",
    r"\black\s+of\b",
    r"\bwithout\s+experience\b",
]
_LIMITATION_RE = re.compile("|".join(LIMITATION_PATTERNS), re.IGNORECASE)

# Heuristic keywords that suggest a JD phrase is "essential" rather than desirable
ESSENTIAL_KEYWORDS = [
    "required", "must", "essential", "mandatory", "necessary",
    "minimum", "qualification", "experience in", "proficiency",
]


# ══════════════════════════════════════════════════════════════════════════════
# classify_requirements
# ══════════════════════════════════════════════════════════════════════════════

def classify_requirements(
    jd_phrases: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Split extracted JD phrases into Essential and Desirable lists.

    This is a heuristic shortcut: phrases that contain common "essential"
    keyword substrings are placed in the Essential list; everything else
    goes to Desirable.  The user can override this in the UI.

    Parameters
    ----------
    jd_phrases : List[str]  – lowercased phrases from extract_phrases()

    Returns
    -------
    (essential, desirable) – two lists of strings
    """
    essential, desirable = [], []
    for phrase in jd_phrases:
        if any(kw in phrase for kw in ESSENTIAL_KEYWORDS):
            essential.append(phrase)
        else:
            desirable.append(phrase)

    # If nothing was classified as essential (common for short JDs), treat
    # the first half as essential so the UI always has both columns populated.
    if not essential and desirable:
        mid = max(1, len(desirable) // 2)
        essential, desirable = desirable[:mid], desirable[mid:]

    return essential, desirable


# ══════════════════════════════════════════════════════════════════════════════
# _split_sentences
# ══════════════════════════════════════════════════════════════════════════════

def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences, lines, and bullet points.
    Works for standard prose as well as bulleted CV formats.
    """
    if not text or not text.strip():
        return []
    # Split on sentence terminators (. ! ?) followed by whitespace, or on newlines
    raw_chunks = re.split(r'(?:(?<=[.!?])\s+|\r?\n+)', text.strip())
    cleaned = []
    for chunk in raw_chunks:
        # Strip common bullet characters and extra whitespace
        c = re.sub(r'^[•\-\*\u2022\u2023\u25E6\u2043\u2219]\s*', '', chunk.strip()).strip()
        if c and len(c) > 1:
            cleaned.append(c)
    return cleaned if cleaned else [text.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# score_cv_against_requirements
# ══════════════════════════════════════════════════════════════════════════════

def score_cv_against_requirements(
    requirements: List[Tuple[str, str]],  # (phrase, "essential"|"desirable")
    cv_text: str,
    st_model,
    threshold: float,
) -> List[Dict[str, Any]]:
    """
    For each requirement, check the CV for a direct mention or a semantic match.

    Parameters
    ----------
    requirements : list of (phrase, priority) tuples
    cv_text      : full CV text string
    st_model     : SentenceTransformer model (from load_sentence_transformer)
    threshold    : cosine-similarity cutoff for semantic fallback (e.g. 0.60)

    Returns
    -------
    List of dicts, one per requirement:
        {
          "requirement" : str,
          "priority"    : "essential" | "desirable",
          "status"      : "direct" | "possible" | "missing" | "limited",
          "credit"      : float  (1.0, 0.5, or 0.0),
          "evidence"    : str    (supporting CV passage, or ""),
        }
    """
    cv_sentences = _split_sentences(cv_text)
    # Safety guard: fall back to full CV as a single "sentence" if empty
    if not cv_sentences:
        cv_sentences = [cv_text.strip()] if cv_text.strip() else [""]

    # Pre-encode all CV sentences once (expensive) rather than once per requirement
    cv_embeddings = None
    if st_model is not None and any(s.strip() for s in cv_sentences):
        try:
            valid_sentences = [s for s in cv_sentences if s.strip()]
            if valid_sentences:
                cv_embeddings = st_model.encode(
                    valid_sentences,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                cv_sentences = valid_sentences
        except Exception:
            cv_embeddings = None  # Embedding failed; fallback to direct-mention only

    results: List[Dict[str, Any]] = []

    for phrase, priority in requirements:
        result = _score_single_requirement(
            phrase=phrase,
            priority=priority,
            cv_text=cv_text,
            cv_sentences=cv_sentences,
            cv_embeddings=cv_embeddings,
            st_model=st_model,
            threshold=threshold,
        )
        results.append(result)

    return results


def _score_single_requirement(
    phrase: str,
    priority: str,
    cv_text: str,
    cv_sentences: List[str],
    cv_embeddings,   # np.ndarray | None
    st_model,
    threshold: float,
) -> Dict[str, Any]:
    """Score one requirement against the CV and return a result dict."""

    # ── Step 1: Direct mention ────────────────────────────────────────────────
    phrase_clean = phrase.strip()
    phrase_lower = phrase_clean.lower()

    # Word-boundary check: handle symbols (e.g. c++, c#, .net) gracefully
    prefix = r"\b" if re.match(r"^\w", phrase_lower) else r"(?:\s|\A)"
    suffix = r"\b" if re.search(r"\w$", phrase_lower) else r"(?:\s|[.,;!?]|\Z)"
    pattern = re.compile(prefix + re.escape(phrase_lower) + suffix, re.IGNORECASE)

    # Find matching sentences
    matching_sentences = [
        s for s in cv_sentences
        if pattern.search(s) or (len(phrase_lower) >= 3 and phrase_lower in s.lower())
    ]

    if matching_sentences:
        # Check surrounding context (up to 3 sentences) for limitation words
        context = " ".join(matching_sentences[:3])
        if _LIMITATION_RE.search(context):
            return {
                "requirement": phrase,
                "priority": priority,
                "status": "limited",
                "credit": 0.0,
                "evidence": context[:300],
            }
        else:
            return {
                "requirement": phrase,
                "priority": priority,
                "status": "direct",
                "credit": 1.0,
                "evidence": matching_sentences[0][:300],
            }

    # ── Step 2: Semantic fallback ─────────────────────────────────────────────
    if cv_embeddings is not None and st_model is not None:
        try:
            req_embedding = st_model.encode(
                [phrase_clean],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            # Cosine similarity: dot product of normalized vectors
            similarities = np.dot(cv_embeddings, req_embedding[0])
            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            if np.isfinite(best_sim) and best_sim >= threshold:
                evidence_text = cv_sentences[best_idx][:300] if best_idx < len(cv_sentences) else ""
                return {
                    "requirement": phrase,
                    "priority": priority,
                    "status": "possible",
                    "credit": 0.5,
                    "evidence": evidence_text,
                    "similarity": round(best_sim, 3),
                }
        except Exception:
            pass  # If encoding fails, fall through to "missing"

    # ── Step 3: Not found ─────────────────────────────────────────────────────
    return {
        "requirement": phrase,
        "priority": priority,
        "status": "missing",
        "credit": 0.0,
        "evidence": "",
    }


# ══════════════════════════════════════════════════════════════════════════════
# compute_meaning_similarity
# ══════════════════════════════════════════════════════════════════════════════

def compute_meaning_similarity(jd_text: str, cv_text: str, st_model) -> float:
    """
    Measure overall meaning similarity between JD and CV using Sentence-Transformers.

    Method:
      1. Split JD into overlapping 5-sentence windows.
      2. Encode each JD window and all CV sentences.
      3. For each JD window, find the CV sentence with the highest cosine similarity.
      4. Return the average of those best-match scores.

    Returns a float in [0, 1].  Multiply by 100 to get a percentage.
    """
    jd_sentences = _split_sentences(jd_text)
    cv_sentences = _split_sentences(cv_text)

    if not jd_sentences or not cv_sentences or st_model is None:
        return 0.0

    # Build overlapping JD windows (groups of 5 sentences)
    jd_windows = _build_windows(jd_sentences, window_size=5, step=3)
    if not jd_windows:
        jd_windows = [jd_text.strip()]

    try:
        jd_embeddings = st_model.encode(
            jd_windows,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        cv_embeddings = st_model.encode(
            cv_sentences,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # For each JD window, find the best-matching CV sentence
        similarity_matrix = np.dot(jd_embeddings, cv_embeddings.T)  # shape: (n_jd, n_cv)
        best_per_jd = similarity_matrix.max(axis=1)                  # shape: (n_jd,)

        score = float(np.mean(best_per_jd))
        if np.isnan(score) or not np.isfinite(score):
            return 0.0
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


def _build_windows(sentences: List[str], window_size: int, step: int) -> List[str]:
    """Join groups of *window_size* sentences into strings, stepping by *step*."""
    windows = []
    for i in range(0, len(sentences), step):
        window = " ".join(sentences[i: i + window_size])
        if window.strip():
            windows.append(window)
    return windows


# ══════════════════════════════════════════════════════════════════════════════
# compute_final_rating
# ══════════════════════════════════════════════════════════════════════════════

def compute_final_rating(
    score_rows: List[Dict[str, Any]],
    meaning_score: float,  # 0–1
) -> Tuple[float, float]:
    """
    Compute weighted coverage and the final combined rating.

    Weights:
        essential = 2,  desirable = 1

    Formula:
        coverage = 100 × Σ(credit × weight) / Σ(weight)
        final    = 0.70 × coverage + 0.30 × (meaning_score × 100)

    Returns
    -------
    (coverage_score, final_rating) – both in [0, 100]
    """
    weight_map = {"essential": 2, "desirable": 1}

    total_weighted_credit = 0.0
    total_weight = 0.0

    for row in score_rows:
        w = weight_map.get(row.get("priority", "desirable"), 1)
        total_weighted_credit += float(row.get("credit", 0.0)) * w
        total_weight += w

    if total_weight == 0:
        coverage = 0.0
    else:
        coverage = 100.0 * total_weighted_credit / total_weight

    if np.isnan(meaning_score) or not np.isfinite(meaning_score):
        meaning_score = 0.0

    final = 0.70 * coverage + 0.30 * (meaning_score * 100.0)

    if np.isnan(coverage) or not np.isfinite(coverage):
        coverage = 0.0
    if np.isnan(final) or not np.isfinite(final):
        final = 0.0

    coverage = max(0.0, min(100.0, coverage))
    final = max(0.0, min(100.0, final))

    return round(coverage, 2), round(final, 2)
