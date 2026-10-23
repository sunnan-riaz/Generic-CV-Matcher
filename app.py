"""
CV–Job Match App
================
A beginner-friendly Streamlit app that compares a CV against a job description (JD)
using two pretrained Hugging Face models:

  • GLiNER  (urchade/gliner_small-v2.1)       – extracts skill-related phrases
  • Sentence-Transformers (all-MiniLM-L6-v2)  – measures semantic similarity

IMPORTANT DISCLAIMERS (built into the app and this file):
  - Scores estimate *text alignment* only; they do not verify skills, qualifications,
    experience, certificates, or hiring probability.
  - Weights and thresholds are unvalidated design choices – they have NOT been
    calibrated against real hiring outcomes.
  - A direct phrase match does NOT prove proficiency.
  - Semantic similarity can confuse related but different skills (e.g. Java vs JavaScript).
  - Uploaded CV text is processed in memory only and is NOT stored or logged.
"""

import streamlit as st

# ── Page config must be the first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="CV–Job Match",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

import re
import io
import datetime
from pathlib import Path

# ── Local modules ──────────────────────────────────────────────────────────────
from models import load_gliner, load_sentence_transformer
from extraction import extract_phrases
from scoring import (
    classify_requirements,
    score_cv_against_requirements,
    compute_meaning_similarity,
    compute_final_rating,
)
from file_reader import read_uploaded_file
from report import build_text_report
from samples import SAMPLE_JD, SAMPLE_CV

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

MAX_CHARS = 15_000  # Guard against very large pastes that could OOM the models


def _check_length(text: str, label: str) -> bool:
    """Return True if text is within the allowed limit, else show an error."""
    if len(text) > MAX_CHARS:
        st.error(
            f"⚠️  The {label} is too long ({len(text):,} characters). "
            f"Please paste at most {MAX_CHARS:,} characters to keep memory usage "
            "reasonable on Streamlit Community Cloud."
        )
        return False
    return True


def _jd_or_cv_changed() -> bool:
    """
    Detect whether the JD or CV text has changed since the last extraction.
    Returns True when re-extraction is needed.
    """
    ss = st.session_state
    jd_changed = ss.get("jd_text", "") != ss.get("last_extracted_jd", None)
    cv_changed = ss.get("cv_text", "") != ss.get("last_extracted_cv", None)
    return jd_changed or cv_changed


def _reset_results() -> None:
    """Clear all downstream results so stale scores are never shown."""
    for key in [
        "jd_phrases", "cv_phrases",
        "essential_reqs", "desirable_reqs",
        "score_rows", "coverage_score", "meaning_score", "final_rating",
        "last_extracted_jd", "last_extracted_cv",
    ]:
        st.session_state.pop(key, None)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar – settings & disclaimers
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> float:
    """Render the sidebar and return the user-chosen similarity threshold."""
    with st.sidebar:
        st.title("⚙️ Settings")

        threshold = st.slider(
            "Semantic-similarity threshold",
            min_value=0.40,
            max_value=0.95,
            value=0.60,
            step=0.05,
            help=(
                "CV phrases scoring above this cosine-similarity value (but not found "
                "by direct-mention search) are flagged as *possible* matches. "
                "Lower values surface more candidates but increase false positives."
            ),
        )

        st.markdown("---")
        st.subheader("ℹ️ About this app")
        st.markdown(
            """
This prototype compares CV and JD **text** only.

**It does NOT:**
- verify your skills or qualifications
- predict hiring success
- access any external database

**Scoring note:** Weights (0.70 / 0.30) and thresholds are *unvalidated design
choices* made for demonstration. They have not been calibrated against real
hiring outcomes.

**Privacy:** Uploaded file text is processed in memory only. Nothing is stored
or logged by this app.

**Scanned PDFs** (images of text) cannot be read; you would need a separate OCR
step first.
            """
        )
        st.markdown("---")
        st.caption("Models: GLiNER small-v2.1 · all-MiniLM-L6-v2")
    return threshold


# ══════════════════════════════════════════════════════════════════════════════
# Section 1 – Input
# ══════════════════════════════════════════════════════════════════════════════

def render_input_section() -> None:
    """
    Render JD and CV text areas plus file upload.
    Stores text in session state under 'jd_text' and 'cv_text'.
    """
    st.header("1 · Paste or Upload Your Documents")

    col_jd, col_cv = st.columns(2)

    # ── Job Description ──────────────────────────────────────────────────────
    with col_jd:
        st.subheader("Job Description (JD)")

        if st.button("Load sample JD", key="btn_sample_jd"):
            st.session_state["jd_text"] = SAMPLE_JD
            _reset_results()

        jd_input = st.text_area(
            "Paste the full job description here",
            value=st.session_state.get("jd_text", ""),
            height=300,
            key="jd_area",
            placeholder="Paste the job posting text here …",
        )

        # Sync text area → session state (detect changes)
        if jd_input != st.session_state.get("jd_text", ""):
            st.session_state["jd_text"] = jd_input
            _reset_results()

    # ── CV ────────────────────────────────────────────────────────────────────
    with col_cv:
        st.subheader("Your CV / Résumé")

        if st.button("Load sample CV", key="btn_sample_cv"):
            st.session_state["cv_text"] = SAMPLE_CV
            _reset_results()

        uploaded = st.file_uploader(
            "Upload a PDF, DOCX, or TXT file (optional)",
            type=["pdf", "docx", "txt"],
            key="cv_upload",
        )

        if uploaded is not None:
            with st.spinner("Reading file …"):
                extracted, error = read_uploaded_file(uploaded)

            if error:
                st.error(f"Could not read file: {error}")
            else:
                st.success("File read successfully. You can edit the text below before extracting.")
                st.session_state["cv_text"] = extracted
                _reset_results()

        cv_input = st.text_area(
            "CV text (edit if needed after file upload)",
            value=st.session_state.get("cv_text", ""),
            height=300,
            key="cv_area",
            placeholder="Paste your CV text here, or upload a file above …",
        )

        if cv_input != st.session_state.get("cv_text", ""):
            st.session_state["cv_text"] = cv_input
            _reset_results()


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 – Extraction
# ══════════════════════════════════════════════════════════════════════════════

def render_extraction_section(gliner_model) -> None:
    """
    Run GLiNER extraction on JD and CV.
    Results are stored in session state so they survive re-renders.
    """
    st.header("2 · Extract Skill-Related Phrases")

    st.info(
        "GLiNER scans both documents for broad skill-related phrases using labels: "
        "*skill, software tool, technology, method, certification*. "
        "No fixed skill list is used."
    )

    # Warn if text has changed since last extraction
    if st.session_state.get("last_extracted_jd") is not None and _jd_or_cv_changed():
        st.warning(
            "⚠️  Your JD or CV text has changed since the last extraction. "
            "Please click **Extract phrases** again before scoring."
        )

    if st.button("🔍 Extract phrases", key="btn_extract"):
        jd_text = st.session_state.get("jd_text", "").strip()
        cv_text = st.session_state.get("cv_text", "").strip()

        # ── Validation ───────────────────────────────────────────────────────
        ok = True
        if not jd_text:
            st.error("Please paste or type a Job Description first.")
            ok = False
        if not cv_text:
            st.error("Please paste, upload, or type your CV text first.")
            ok = False
        if ok and not _check_length(jd_text, "Job Description"):
            ok = False
        if ok and not _check_length(cv_text, "CV"):
            ok = False

        if ok:
            with st.spinner("Extracting phrases (this may take ~20–60 s on CPU) …"):
                jd_phrases = extract_phrases(gliner_model, jd_text)
                cv_phrases = extract_phrases(gliner_model, cv_text)

            st.session_state["jd_phrases"] = jd_phrases
            st.session_state["cv_phrases"] = cv_phrases

            # Auto-classify into essential / desirable
            essential, desirable = classify_requirements(jd_phrases)
            st.session_state["essential_reqs"] = essential
            st.session_state["desirable_reqs"] = desirable

            # Record the text that was extracted so we can detect later changes
            st.session_state["last_extracted_jd"] = jd_text
            st.session_state["last_extracted_cv"] = cv_text

            # Clear any old scores
            for k in ["score_rows", "coverage_score", "meaning_score", "final_rating"]:
                st.session_state.pop(k, None)

            st.success(
                f"Found **{len(jd_phrases)}** JD phrase(s) and "
                f"**{len(cv_phrases)}** CV phrase(s)."
            )

    # Show extracted phrases if available
    if st.session_state.get("jd_phrases") is not None:
        col_jd_ph, col_cv_ph = st.columns(2)
        with col_jd_ph:
            st.subheader("JD phrases detected")
            for p in st.session_state["jd_phrases"]:
                st.markdown(f"- `{p}`")
        with col_cv_ph:
            st.subheader("CV phrases detected")
            st.caption(
                "These are *detected mentions* in the text, not verified abilities."
            )
            for p in st.session_state["cv_phrases"]:
                st.markdown(f"- `{p}`")


# ══════════════════════════════════════════════════════════════════════════════
# Section 3 – Requirement Review
# ══════════════════════════════════════════════════════════════════════════════

def render_requirement_review_section() -> None:
    """
    Show editable Essential and Desirable requirement lists.
    The user can add missed requirements, remove irrelevant ones, or change priority.
    """
    if st.session_state.get("essential_reqs") is None:
        return  # Nothing to show yet

    st.header("3 · Review & Edit Requirements")
    st.info(
        "The app has auto-classified JD phrases into Essential and Desirable lists. "
        "You can **add missed requirements**, **remove irrelevant ones**, and "
        "**move items between lists**. These edited lists are what will be scored."
    )

    col_ess, col_des = st.columns(2)

    # ── Essential ─────────────────────────────────────────────────────────────
    with col_ess:
        st.subheader("🔴 Essential requirements  (weight 2)")
        essential = st.session_state.get("essential_reqs", [])

        # Text area for editing – one requirement per line
        ess_text = st.text_area(
            "Edit essential requirements (one per line)",
            value="\n".join(essential),
            height=200,
            key="ess_edit",
        )
        new_ess = [r.strip() for r in ess_text.splitlines() if r.strip()]
        st.session_state["essential_reqs"] = new_ess

        st.caption(f"{len(new_ess)} item(s)")

    # ── Desirable ─────────────────────────────────────────────────────────────
    with col_des:
        st.subheader("🟡 Desirable requirements  (weight 1)")
        desirable = st.session_state.get("desirable_reqs", [])

        des_text = st.text_area(
            "Edit desirable requirements (one per line)",
            value="\n".join(desirable),
            height=200,
            key="des_edit",
        )
        new_des = [r.strip() for r in des_text.splitlines() if r.strip()]
        st.session_state["desirable_reqs"] = new_des

        st.caption(f"{len(new_des)} item(s)")


# ══════════════════════════════════════════════════════════════════════════════
# Section 4 – Scoring
# ══════════════════════════════════════════════════════════════════════════════

def render_scoring_section(st_model, threshold: float) -> None:
    """
    Score the CV against the confirmed requirements and display results.
    """
    essential = st.session_state.get("essential_reqs")
    desirable = st.session_state.get("desirable_reqs")
    if essential is None:
        return  # Extraction hasn't happened yet

    st.header("4 · Score & Results")

    # Warn if the input text has changed since extraction
    if _jd_or_cv_changed():
        st.warning(
            "⚠️  JD or CV changed after extraction – scores shown may be stale. "
            "Go back to Step 2 and re-extract before scoring."
        )

    st.markdown(
        """
**Scoring method (prototype – unvalidated weights):**

| Situation | Credit |
|---|---|
| Direct mention, no limitation wording detected | 1.0 |
| Possible semantic match above threshold + supporting passage | 0.5 *(review this)* |
| Missing or limitation/negative wording detected | 0.0 |

- Essential weight = **2**, Desirable weight = **1**
- Weighted coverage = `100 × Σ(credit × weight) / Σ(weight)`
- Meaning similarity = average best-cosine-similarity across JD passages vs CV passages
- **Final rating = 0.70 × weighted coverage + 0.30 × meaning similarity**

> ⚠️ Weights and thresholds are unvalidated design choices. The rating estimates
> text alignment; it does **not** verify skills, qualifications, experience,
> certificates, or hiring probability.
        """
    )

    if st.button("📊 Calculate score", key="btn_score"):
        jd_text = st.session_state.get("jd_text", "").strip()
        cv_text = st.session_state.get("cv_text", "").strip()

        all_reqs = (
            [(r, "essential") for r in essential]
            + [(r, "desirable") for r in desirable]
        )

        if not all_reqs:
            st.warning("There are no requirements to score against. Add some in Step 3.")
            return

        with st.spinner("Scoring (embedding + direct search) …"):
            score_rows = score_cv_against_requirements(
                requirements=all_reqs,
                cv_text=cv_text,
                st_model=st_model,
                threshold=threshold,
            )
            meaning_score = compute_meaning_similarity(
                jd_text=jd_text,
                cv_text=cv_text,
                st_model=st_model,
            )
            coverage_score, final_rating = compute_final_rating(score_rows, meaning_score)

        st.session_state["score_rows"] = score_rows
        st.session_state["coverage_score"] = coverage_score
        st.session_state["meaning_score"] = meaning_score
        st.session_state["final_rating"] = final_rating

    # ── Display results ───────────────────────────────────────────────────────
    if st.session_state.get("final_rating") is not None:
        fr = st.session_state["final_rating"]
        cov = st.session_state["coverage_score"]
        msim = st.session_state["meaning_score"]

        st.markdown("---")

        # Big metric
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Final Rating", f"{fr:.1f} / 100")
        col_b.metric("Weighted Coverage", f"{cov:.1f} / 100")
        col_c.metric("Meaning Similarity", f"{msim * 100:.1f} / 100")

        # Progress bar
        st.progress(min(int(fr), 100))

        # Colour hint
        if fr >= 75:
            st.success("Strong text alignment — the CV mentions most requirements.")
        elif fr >= 50:
            st.info("Moderate alignment — some requirements may need strengthening.")
        else:
            st.warning("Weak alignment — the CV may be missing key requirement language.")

        st.caption(
            "Reminder: this rating measures text alignment only. "
            "It does not prove skills or predict hiring success."
        )

        # ── Evidence table ────────────────────────────────────────────────────
        st.subheader("Requirement-by-Requirement Evidence")

        import pandas as pd  # only needed for display

        rows_display = []
        for row in st.session_state["score_rows"]:
            status_icon = {
                "direct": "✅ Direct mention",
                "possible": "🔶 Possible match (review)",
                "missing": "❌ Not found",
                "limited": "⚠️  Limitation detected",
            }.get(row["status"], row["status"])

            rows_display.append({
                "Requirement": row["requirement"],
                "Priority": row["priority"].capitalize(),
                "Status": status_icon,
                "Credit": row["credit"],
                "Supporting CV passage": row.get("evidence", "—"),
            })

        df = pd.DataFrame(rows_display)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Suggestions ───────────────────────────────────────────────────────
        st.subheader("💡 Improvement Suggestions")
        st.caption(
            "Suggestions are based only on missing or weak text matches. "
            "Only add examples if they are **truthful**. "
            "Do not invent achievements."
        )

        missing_reqs = [
            r for r in st.session_state["score_rows"]
            if r["status"] in ("missing", "limited")
        ]
        possible_reqs = [
            r for r in st.session_state["score_rows"]
            if r["status"] == "possible"
        ]

        if not missing_reqs and not possible_reqs:
            st.success(
                "All requirements have direct mentions — no immediate changes suggested. "
                "Still review the evidence table to confirm accuracy."
            )
        else:
            if missing_reqs:
                st.markdown("**Missing requirements — consider adding (if truthful):**")
                for r in missing_reqs:
                    st.markdown(
                        f"- **{r['requirement']}** ({r['priority']}) — "
                        "not found in your CV. If you have this skill/experience, "
                        "add a specific, truthful example (e.g. a project, tool used, result achieved)."
                    )

            if possible_reqs:
                st.markdown("**Possible matches — please review:**")
                for r in possible_reqs:
                    st.markdown(
                        f"- **{r['requirement']}** ({r['priority']}) — "
                        f"possibly related to: *\"{r.get('evidence', '')}\"*. "
                        "If this genuinely covers the requirement, consider making the connection "
                        "more explicit in your CV."
                    )

        # ── Download report ───────────────────────────────────────────────────
        st.subheader("📥 Download Report")
        report_text = build_text_report(
            jd_text=st.session_state.get("jd_text", ""),
            cv_text=st.session_state.get("cv_text", ""),
            score_rows=st.session_state["score_rows"],
            coverage_score=cov,
            meaning_score=msim,
            final_rating=fr,
            threshold=threshold,
        )
        st.download_button(
            label="Download plain-text report (.txt)",
            data=report_text.encode("utf-8"),
            file_name=f"cv_match_report_{datetime.date.today()}.txt",
            mime="text/plain",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.title("📄 CV–Job Match")
    st.markdown(
        """
Compare your CV against a job description using AI-powered phrase extraction
and semantic similarity — entirely local, no paid API required.

> **Disclaimer:** This tool estimates text alignment only. It does not verify
> skills, qualifications, experience, certificates, or hiring probability.
> Scores are produced by an *unvalidated prototype*; treat them as a
> starting point for reflection, not as a final judgement.
        """
    )

    # ── Load models (cached – loaded only once per session) ───────────────────
    threshold = render_sidebar()

    with st.spinner("Loading AI models (first run downloads ~100 MB) …"):
        gliner_model = load_gliner()
        st_model = load_sentence_transformer()

    if gliner_model is None or st_model is None:
        st.error(
            "❌ Model loading failed. Check your internet connection (models are "
            "downloaded from Hugging Face on first run). See the README for "
            "troubleshooting steps."
        )
        st.stop()

    # ── Sections ──────────────────────────────────────────────────────────────
    render_input_section()
    st.markdown("---")
    render_extraction_section(gliner_model)
    st.markdown("---")
    render_requirement_review_section()
    st.markdown("---")
    render_scoring_section(st_model, threshold)

    # Footer
    st.markdown("---")
    st.caption(
        "CV–Job Match · prototype · models: GLiNER small-v2.1 & all-MiniLM-L6-v2 · "
        "no data is stored or shared"
    )


if __name__ == "__main__":
    main()
