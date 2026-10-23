"""
CV–Job Match App  —  Full Width & User-Friendly Edition
======================================================
A step-by-step wizard that guides the user through:

  STEP 1  →  Paste your Job Description + CV (or Upload file with auto-extract)
  STEP 2  →  Let AI find skill phrases  (one button click)
  STEP 3  →  Check / edit the requirement list
  STEP 4  →  Calculate match score + suggestions

DISCLAIMERS:
  • Scores measure text alignment only — NOT verified skills or hiring probability.
  • Scoring weights are unvalidated prototype values.
  • Uploaded text is processed in memory only — never stored or logged.
"""

import streamlit as st

# ── Page config — must be the FIRST Streamlit call ────────────────────────────
st.set_page_config(
    page_title="CV–Job Match",
    page_icon="📄",
    layout="wide",  # Full width of screen
    initial_sidebar_state="collapsed",
)

import datetime
import re
import pandas as pd

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

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_CHARS = 15_000   # roughly 2,500 words — safe for Streamlit free tier

# ── Custom CSS for full-width polish ──────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Full width container optimization */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 100% !important;
    }
    /* Breadcrumb badge styles */
    .step-badge {
        text-align: center;
        padding: 8px 12px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .step-badge-done {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }
    .step-badge-active {
        background-color: #dbeafe;
        color: #1d4ed8;
        border: 2px solid #3b82f6;
    }
    .step-badge-pending {
        background-color: #f3f4f6;
        color: #6b7280;
        border: 1px solid #e5e7eb;
    }
    /* Section containers */
    .wizard-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions for state resets
# ══════════════════════════════════════════════════════════════════════════════

def _reset_all() -> None:
    """Wipe everything downstream so stale results never show."""
    for key in [
        "jd_text", "cv_text", "jd_textarea", "cv_textarea",
        "_last_uploaded_file_id", "cv_upload",
        "jd_phrases", "cv_phrases",
        "essential_reqs", "desirable_reqs",
        "ess_edit", "des_edit",
        "score_rows", "coverage_score", "meaning_score", "final_rating",
        "last_extracted_jd", "last_extracted_cv",
    ]:
        st.session_state.pop(key, None)


def _reset_from_extraction() -> None:
    """Clear only extraction + scoring results (keep the pasted text)."""
    for key in [
        "jd_phrases", "cv_phrases",
        "essential_reqs", "desirable_reqs",
        "ess_edit", "des_edit",
        "score_rows", "coverage_score", "meaning_score", "final_rating",
        "last_extracted_jd", "last_extracted_cv",
    ]:
        st.session_state.pop(key, None)


def _reset_scores_only() -> None:
    """Clear only the score so editing requirements forces re-scoring."""
    for key in ["score_rows", "coverage_score", "meaning_score", "final_rating"]:
        st.session_state.pop(key, None)


def _text_changed_since_extraction() -> bool:
    """Return True if the user edited JD or CV after the last extraction."""
    ss = st.session_state
    if ss.get("last_extracted_jd") is None or ss.get("last_extracted_cv") is None:
        return False
    jd_ok = ss.get("jd_text", "").strip() == ss.get("last_extracted_jd", "").strip()
    cv_ok = ss.get("cv_text", "").strip() == ss.get("last_extracted_cv", "").strip()
    return not (jd_ok and cv_ok)


def _check_length(text: str, label: str) -> bool:
    """Show an error and return False if text exceeds MAX_CHARS."""
    if len(text) > MAX_CHARS:
        st.error(
            f"⚠️ Your {label} is too long ({len(text):,} characters). "
            f"Please use at most {MAX_CHARS:,} characters (~2,500 words) "
            "to keep memory use reasonable."
        )
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Progress bar at the top
# ══════════════════════════════════════════════════════════════════════════════

def _render_progress(current_step: int) -> None:
    """Show visual breadcrumbs spanning the full width."""
    steps = [
        ("1. Add Documents", "📋"),
        ("2. Find Skills", "🔍"),
        ("3. Edit Requirements", "✏️"),
        ("4. Calculate Score", "📊"),
    ]
    cols = st.columns(4, gap="medium")
    for i, (col, (label, icon)) in enumerate(zip(cols, steps), start=1):
        if i < current_step:
            col.markdown(
                f"<div class='step-badge step-badge-done'>✅ {icon} {label}</div>",
                unsafe_allow_html=True,
            )
        elif i == current_step:
            col.markdown(
                f"<div class='step-badge step-badge-active'>▶️ {icon} {label}</div>",
                unsafe_allow_html=True,
            )
        else:
            col.markdown(
                f"<div class='step-badge step-badge-pending'>{icon} {label}</div>",
                unsafe_allow_html=True,
            )
    st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Add your documents
# ══════════════════════════════════════════════════════════════════════════════

def render_step1() -> int:
    """
    Collect the Job Description and CV text.
    Handles upload with automatic text extraction into the text box below.
    Returns 2 if both JD and CV are present, else 1.
    """
    st.markdown("### Step 1 — Add Your Documents")
    st.markdown(
        "Paste or load the **Job Description** on the left. On the right, **upload your CV** "
        "(PDF, Word DOCX, or TXT) and its text will automatically extract into the box below."
    )

    # ── Sample Loaders Expander ────────────────────────────────────────────────
    with st.expander("🎯 Need test data? Load built-in samples with one click", expanded=not bool(st.session_state.get("jd_text") and st.session_state.get("cv_text"))):
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            if st.button("✨ Load Both Sample JD + CV", key="btn_sample_both", use_container_width=True):
                st.session_state["jd_text"] = SAMPLE_JD
                st.session_state["jd_textarea"] = SAMPLE_JD
                st.session_state["cv_text"] = SAMPLE_CV
                st.session_state["cv_textarea"] = SAMPLE_CV
                _reset_from_extraction()
                st.rerun()
        with col_s2:
            if st.button("📋 Load Sample JD Only", key="btn_sample_jd", use_container_width=True):
                st.session_state["jd_text"] = SAMPLE_JD
                st.session_state["jd_textarea"] = SAMPLE_JD
                _reset_from_extraction()
                st.rerun()
        with col_s3:
            if st.button("📄 Load Sample CV Only", key="btn_sample_cv", use_container_width=True):
                st.session_state["cv_text"] = SAMPLE_CV
                st.session_state["cv_textarea"] = SAMPLE_CV
                _reset_from_extraction()
                st.rerun()

    st.markdown("---")

    # ── Two columns: JD on left, CV on right ──────────────────────────────────
    col_jd, col_cv = st.columns(2, gap="large")

    # ── Job Description ───────────────────────────────────────────────────────
    with col_jd:
        st.markdown("#### 📋 Job Description")
        st.caption("Paste the full job posting below.")

        if "jd_text" not in st.session_state:
            st.session_state["jd_text"] = ""
        if "jd_textarea" not in st.session_state:
            st.session_state["jd_textarea"] = st.session_state["jd_text"]

        jd_new = st.text_area(
            "Job Description text",
            value=st.session_state.get("jd_text", ""),
            height=300,
            key="jd_textarea",
            placeholder="Paste the job posting here…\n\nExample:\n  'We are looking for a Data Analyst proficient in Python, SQL, and data visualization with Tableau…'",
            label_visibility="collapsed",
        )
        if jd_new != st.session_state.get("jd_text", ""):
            st.session_state["jd_text"] = jd_new
            _reset_from_extraction()

        char_count_jd = len(st.session_state.get("jd_text", ""))
        if char_count_jd > 0:
            colour = "red" if char_count_jd > MAX_CHARS else "green"
            st.caption(f":{colour}[{char_count_jd:,} / {MAX_CHARS:,} characters]")
        else:
            st.caption("No JD text yet.")

    # ── CV / Résumé ───────────────────────────────────────────────────────────
    with col_cv:
        st.markdown("#### 📄 Your CV / Résumé")
        st.caption("Upload a file — text will be **automatically extracted** into the box below.")

        if "cv_text" not in st.session_state:
            st.session_state["cv_text"] = ""
        if "cv_textarea" not in st.session_state:
            st.session_state["cv_textarea"] = st.session_state["cv_text"]

        uploaded = st.file_uploader(
            "Upload PDF, DOCX, or TXT file",
            type=["pdf", "docx", "txt"],
            key="cv_upload",
            label_visibility="visible",
        )

        # Automatic file extraction
        if uploaded is not None:
            file_id = f"{uploaded.name}_{uploaded.size}"
            if st.session_state.get("_last_uploaded_file_id") != file_id:
                with st.spinner("📄 Automatically extracting text from uploaded file…"):
                    extracted, error = read_uploaded_file(uploaded)
                if error:
                    st.error(f"❌ Could not read file: {error}")
                else:
                    st.session_state["cv_text"] = extracted
                    st.session_state["cv_textarea"] = extracted
                    st.session_state["_last_uploaded_file_id"] = file_id
                    _reset_from_extraction()
                    st.toast("✅ CV text extracted automatically into the box below!")
                    st.rerun()

        cv_new = st.text_area(
            "CV text",
            value=st.session_state.get("cv_text", ""),
            height=240,
            key="cv_textarea",
            placeholder="Paste your CV here…\n\nOr upload a PDF/DOCX/TXT above — the text extracts automatically here.",
            label_visibility="collapsed",
        )
        if cv_new != st.session_state.get("cv_text", ""):
            st.session_state["cv_text"] = cv_new
            _reset_from_extraction()

        char_count_cv = len(st.session_state.get("cv_text", ""))
        if char_count_cv > 0:
            colour = "red" if char_count_cv > MAX_CHARS else "green"
            st.caption(f":{colour}[{char_count_cv:,} / {MAX_CHARS:,} characters]")
        else:
            st.caption("No CV text yet.")

    # ── Check completion ──────────────────────────────────────────────────────
    jd_ready = bool(st.session_state.get("jd_text", "").strip())
    cv_ready = bool(st.session_state.get("cv_text", "").strip())

    st.markdown("")
    if jd_ready and cv_ready:
        st.success("✅ Both documents ready! Scroll down to **Step 2** to find skill phrases.")
        return 2
    else:
        missing = []
        if not jd_ready:
            missing.append("Job Description")
        if not cv_ready:
            missing.append("CV")
        st.info(f"👆 Please add your **{' and '.join(missing)}** above to continue.")
        return 1


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Extract skill phrases
# ══════════════════════════════════════════════════════════════════════════════

def render_step2(gliner_model) -> int:
    """
    Run GLiNER on both documents and display the detected phrases.
    Returns 3 if phrases are available, else 2.
    """
    st.markdown("### Step 2 — Find Skill Phrases")
    st.markdown(
        "The zero-shot NER model (**GLiNER**) scans both documents and extracts "
        "skill-related phrases (tools, programming languages, qualifications, certifications)."
    )

    # Stale text warning
    if _text_changed_since_extraction():
        st.warning(
            "⚠️ You edited your JD or CV text after the last extraction. "
            "Click **Find Skill Phrases** again to refresh the requirements."
        )

    col_btn, col_note = st.columns([1, 4])
    with col_btn:
        extract_clicked = st.button(
            "🔍 Find Skill Phrases",
            key="btn_extract",
            use_container_width=True,
            type="primary",
        )
    with col_note:
        st.caption(
            "Reads every sentence using GLiNER. Typically takes 15–45 seconds on CPU."
        )

    if extract_clicked:
        jd_text = st.session_state.get("jd_text", "").strip()
        cv_text = st.session_state.get("cv_text", "").strip()

        ok = True
        if not _check_length(jd_text, "Job Description"):
            ok = False
        if not _check_length(cv_text, "CV"):
            ok = False

        if ok:
            with st.spinner("🤖 AI is extracting skill phrases from your documents…"):
                jd_phrases = extract_phrases(gliner_model, jd_text)
                cv_phrases = extract_phrases(gliner_model, cv_text)

            if not jd_phrases:
                st.warning(
                    "⚠️ No skill phrases could be extracted from the Job Description. "
                    "Try adding more specific requirements, or load the sample JD."
                )
            else:
                essential, desirable = classify_requirements(jd_phrases)
                st.session_state["jd_phrases"] = jd_phrases
                st.session_state["cv_phrases"] = cv_phrases
                st.session_state["essential_reqs"] = essential
                st.session_state["desirable_reqs"] = desirable
                st.session_state["ess_edit"] = "\n".join(essential)
                st.session_state["des_edit"] = "\n".join(desirable)
                st.session_state["last_extracted_jd"] = jd_text
                st.session_state["last_extracted_cv"] = cv_text
                _reset_scores_only()
                st.toast("✅ Skills extracted successfully!")
                st.rerun()

    # ── Display extracted phrases if available ───────────────────────────────
    if st.session_state.get("jd_phrases") is not None:
        jd_phrases = st.session_state.get("jd_phrases", [])
        cv_phrases = st.session_state.get("cv_phrases", [])

        col_j, col_c = st.columns(2, gap="large")

        with col_j:
            st.markdown(f"**Found in JD ({len(jd_phrases)} phrases)**")
            st.caption("These form the baseline requirements for Step 3.")
            if jd_phrases:
                p_items = "  \n".join(f"• {p}" for p in jd_phrases)
                st.markdown(p_items)
            else:
                st.caption("No phrases found.")

        with col_c:
            st.markdown(f"**Found in your CV ({len(cv_phrases)} phrases)**")
            st.caption("⚠️ Detected mentions only — not verified proficiency.")
            if cv_phrases:
                p_items = "  \n".join(f"• {p}" for p in cv_phrases)
                st.markdown(p_items)
            else:
                st.caption("No phrases found.")

        st.success("✅ Extraction complete! Scroll down to **Step 3** to verify requirements.")
        return 3

    return 2


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Review requirements
# ══════════════════════════════════════════════════════════════════════════════

def render_step3() -> int:
    """
    Review and edit Essential and Desirable requirement lists.
    Guarantees state synchronization so requirements are never lost.
    Returns 4 if requirements exist, else 3.
    """
    st.markdown("### Step 3 — Check & Edit Requirement List")
    st.markdown(
        "The requirements are organized into two tiers. "
        "**Essential** requirements carry double weight (2×). "
        "**Desirable** requirements carry single weight (1×).\n\n"
        "👇 Feel free to add missing skills or delete incorrect lines (one item per line)."
    )

    # Initialize edit states if missing
    if "ess_edit" not in st.session_state:
        st.session_state["ess_edit"] = "\n".join(st.session_state.get("essential_reqs", []))
    if "des_edit" not in st.session_state:
        st.session_state["des_edit"] = "\n".join(st.session_state.get("desirable_reqs", []))

    col_ess, col_des = st.columns(2, gap="large")

    # ── Essential ─────────────────────────────────────────────────────────────
    with col_ess:
        st.markdown("#### 🔴 Essential  *(Weight = 2)*")
        st.caption("Must-have requirements. Edit or add lines below:")
        ess_text = st.text_area(
            "Essential requirements — one per line",
            value=st.session_state.get("ess_edit", ""),
            height=220,
            key="ess_edit",
            label_visibility="collapsed",
        )
        new_ess = [r.strip() for r in ess_text.splitlines() if r.strip()]
        if new_ess != st.session_state.get("essential_reqs", []):
            st.session_state["essential_reqs"] = new_ess
            _reset_scores_only()
        st.caption(f"{len(new_ess)} essential requirement(s)")

    # ── Desirable ─────────────────────────────────────────────────────────────
    with col_des:
        st.markdown("#### 🟡 Desirable  *(Weight = 1)*")
        st.caption("Nice-to-have requirements. Edit or add lines below:")
        des_text = st.text_area(
            "Desirable requirements — one per line",
            value=st.session_state.get("des_edit", ""),
            height=220,
            key="des_edit",
            label_visibility="collapsed",
        )
        new_des = [r.strip() for r in des_text.splitlines() if r.strip()]
        if new_des != st.session_state.get("desirable_reqs", []):
            st.session_state["desirable_reqs"] = new_des
            _reset_scores_only()
        st.caption(f"{len(new_des)} desirable requirement(s)")

    total = len(st.session_state.get("essential_reqs", [])) + len(st.session_state.get("desirable_reqs", []))
    if total == 0:
        st.warning("⚠️ Both requirement lists are empty. Please type at least one requirement to score against.")
        return 3

    st.success(f"✅ {total} requirement(s) configured. Scroll down to **Step 4** to calculate your match score.")
    return 4


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Score & Results
# ══════════════════════════════════════════════════════════════════════════════

def render_step4(st_model, threshold: float) -> None:
    """
    Score the CV against the requirements, displaying results across the full width.
    """
    st.markdown("### Step 4 — Calculate Match Score & Recommendations")
    st.markdown(
        "Each requirement is matched against your CV text using both exact keyword matching "
        "and Sentence-Transformers semantic similarity."
    )

    if _text_changed_since_extraction():
        st.warning(
            "⚠️ Your JD or CV text changed after the last extraction. "
            "For highest accuracy, re-run Step 2 to refresh the phrases."
        )

    col_btn, col_note = st.columns([1, 4])
    with col_btn:
        score_clicked = st.button(
            "📊 Calculate My Score",
            key="btn_score",
            use_container_width=True,
            type="primary",
        )
    with col_note:
        st.caption("Embeds and compares sentences using MiniLM. Takes about 5–20 seconds.")

    if score_clicked:
        essential = st.session_state.get("essential_reqs", [])
        desirable = st.session_state.get("desirable_reqs", [])
        all_reqs = [(r, "essential") for r in essential] + [(r, "desirable") for r in desirable]

        if not all_reqs:
            st.error("⚠️ No requirements to score. Please add at least one requirement in Step 3.")
            return

        jd_text = st.session_state.get("jd_text", "").strip()
        cv_text = st.session_state.get("cv_text", "").strip()

        if not cv_text:
            st.error("⚠️ Your CV text is empty. Please paste or upload your CV in Step 1.")
            return

        with st.spinner("🤖 Scoring your CV against requirements…"):
            try:
                score_rows = score_cv_against_requirements(
                    requirements=all_reqs,
                    cv_text=cv_text,
                    st_model=st_model,
                    threshold=threshold,
                )
                meaning_score = compute_meaning_similarity(jd_text, cv_text, st_model)
                coverage_score, final_rating = compute_final_rating(score_rows, meaning_score)

                st.session_state["score_rows"] = score_rows
                st.session_state["coverage_score"] = coverage_score
                st.session_state["meaning_score"] = meaning_score
                st.session_state["final_rating"] = final_rating
                st.toast("🎉 Score calculation complete!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Score calculation failed: {e}")
                return

    # ── Display results if calculated ─────────────────────────────────────────
    if st.session_state.get("final_rating") is None:
        return

    fr = float(st.session_state["final_rating"])
    cov = float(st.session_state["coverage_score"])
    msim = float(st.session_state["meaning_score"])
    rows = st.session_state.get("score_rows", [])

    st.markdown("---")

    # ── Score Banner ──────────────────────────────────────────────────────────
    st.markdown("#### 🎯 Overall Match Score")

    col_score_card, col_metrics = st.columns([1, 2], gap="large")

    with col_score_card:
        if fr >= 75:
            st.success(f"## {fr:.0f} / 100 — Strong Match 🟢")
            st.markdown("Your CV mentions most of the job requirements with matching phrasing.")
        elif fr >= 50:
            st.info(f"## {fr:.0f} / 100 — Moderate Match 🟡")
            st.markdown("Your CV covers some requirements well, but notable gaps exist.")
        else:
            st.warning(f"## {fr:.0f} / 100 — Weak Match 🔴")
            st.markdown("Your CV is missing key requirement language from the job posting.")

        st.progress(min(max(int(fr), 0), 100))

    with col_metrics:
        m_c1, m_c2, m_c3 = st.columns(3)
        m_c1.metric(
            "Requirement Coverage",
            f"{cov:.0f}%",
            help="Weighted score of how many JD requirements were found in your CV (Essential counts 2×).",
        )
        m_c2.metric(
            "Meaning Similarity",
            f"{msim * 100:.0f}%",
            help="Semantic cosine similarity between overall JD and CV language windows.",
        )
        m_c3.metric(
            "Combined Rating",
            f"{fr:.0f} / 100",
            help="70% Weighted Coverage + 30% Overall Meaning Similarity.",
        )

        with st.expander("📐 How the score formula works", expanded=False):
            st.markdown(
                """
**Formula:** `Final Score = 0.70 × Coverage + 0.30 × (Meaning Similarity × 100)`

- **Coverage:** Evaluates each requirement phrase. Direct mention = 1.0 credit, semantic match = 0.5 credit, limitation detected = 0.0 credit.
- **Meaning:** Measures vocabulary and context alignment across the full document text.
- **Weights:** Essential items count **twice (2×)** as much as Desirable items.

> ⚠️ **Disclaimer:** All weights are prototype design choices. Scores reflect **text similarity only**, not verified candidate competence or hiring probability.
                """
            )

    st.markdown("---")

    # ── Table breakdown ───────────────────────────────────────────────────────
    st.markdown("#### 📋 Requirement-by-Requirement Breakdown")
    st.caption("Review how each requirement was matched against your CV text.")

    icon_map = {
        "direct": "✅ Found in CV",
        "possible": "🔶 Possible (Verify)",
        "limited": "⚠️ Limitation Detected",
        "missing": "❌ Not Found",
    }

    rows_display = []
    for r in rows:
        evidence = r.get("evidence", "") or "—"
        if len(evidence) > 150:
            evidence = evidence[:150] + "…"
        rows_display.append({
            "Requirement": r.get("requirement", ""),
            "Priority": "🔴 Essential" if r.get("priority") == "essential" else "🟡 Desirable",
            "Status": icon_map.get(r.get("status", ""), r.get("status", "")),
            "Credit": f"{float(r.get('credit', 0.0)):.1f}",
            "CV Evidence Passage": evidence,
        })

    df = pd.DataFrame(rows_display)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Suggestions ────────────────────────────────────────────────────────────
    st.markdown("#### 💡 Actionable Improvement Suggestions")
    st.caption(
        "Only add experiences or keywords if they are **100% genuine and truthful**. "
        "Never misrepresent skills."
    )

    missing = [r for r in rows if r.get("status") in ("missing", "limited")]
    possible = [r for r in rows if r.get("status") == "possible"]

    col_sugg1, col_sugg2 = st.columns(2, gap="large")

    with col_sugg1:
        st.markdown("**❌ Missing Requirements (Add if truthful)**")
        if missing:
            for r in missing:
                p_label = "Essential" if r.get("priority") == "essential" else "Desirable"
                st.markdown(
                    f"- **{r.get('requirement')}** *({p_label})*\n"
                    f"  If you possess this skill, highlight a specific project, tool version, or outcome."
                )
        else:
            st.success("🎉 No missing requirements detected!")

    with col_sugg2:
        st.markdown("**🔶 Possible Semantic Matches (Make more explicit)**")
        if possible:
            for r in possible:
                ev = r.get("evidence", "")
                st.markdown(
                    f"- **{r.get('requirement')}**\n"
                    f"  CV excerpt: *\"{ev[:120]}…\"*\n"
                    f"  Consider using the job posting's exact terminology if appropriate."
                )
        else:
            st.info("No borderline matches requiring verification.")

    st.markdown("---")

    # ── Download Report ───────────────────────────────────────────────────────
    st.markdown("#### 📥 Export Match Report")
    report_text = build_text_report(
        jd_text=st.session_state.get("jd_text", ""),
        cv_text=st.session_state.get("cv_text", ""),
        score_rows=rows,
        coverage_score=cov,
        meaning_score=msim,
        final_rating=fr,
        threshold=threshold,
    )

    st.download_button(
        label="⬇️ Download Full Report (.txt)",
        data=report_text.encode("utf-8"),
        file_name=f"cv_match_report_{datetime.date.today()}.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — Settings & Help
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> float:
    with st.sidebar:
        st.header("⚙️ Configuration")
        threshold = st.slider(
            "Semantic match threshold",
            min_value=0.40,
            max_value=0.95,
            value=0.60,
            step=0.05,
            help="Cosine similarity threshold for semantic matches (Sentence-Transformers). Higher = stricter.",
        )
        st.caption(f"Sensitivity: {threshold:.2f}")

        st.markdown("---")
        st.markdown("**❓ Quick Help & FAQs**")
        with st.expander("Score interpretations"):
            st.markdown(
                """
- **75–100:** Strong alignment with JD terminology
- **50–74:** Moderate match, some missing requirements
- **0–49:** Low alignment with JD requirements

*Scores measure text similarity only — not verified candidate ability.*
                """
            )
        with st.expander("Status icons"):
            st.markdown(
                """
- ✅ **Found** — exact keyword phrase found in CV
- 🔶 **Possible** — semantic match found via embeddings (verify manually)
- ⚠️ **Limited** — phrase found near negative phrasing ("no experience")
- ❌ **Not Found** — phrase not identified in CV
                """
            )
        with st.expander("Data privacy"):
            st.markdown(
                "All text and files are processed **strictly in-memory** during your session. "
                "No database, storage, or external logging is used."
            )

        st.markdown("---")
        if st.button("🔄 Reset App & Start Over", key="btn_reset_all", use_container_width=True):
            _reset_all()
            st.rerun()

        st.caption("AI Models: GLiNER small-v2.1 · all-MiniLM-L6-v2")
    return threshold


# ══════════════════════════════════════════════════════════════════════════════
# Main application entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    st.title("📄 CV–Job Match Analysis")
    st.markdown(
        "**Evaluate how well your CV aligns with any Job Description** — "
        "powered by open-source Hugging Face models (**GLiNER** & **Sentence-Transformers**)."
    )
    st.caption(
        "⚠️ *Ethical AI Notice:* This system estimates text alignment and vocabulary coverage. "
        "It does not assess human competence, experience quality, or predict hiring decisions."
    )
    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    threshold = render_sidebar()

    # ── Load pretrained models ────────────────────────────────────────────────
    with st.spinner("⏳ Loading AI models (GLiNER & Sentence-Transformers)…"):
        gliner_model = load_gliner()
        st_model = load_sentence_transformer()

    if gliner_model is None or st_model is None:
        st.error(
            "❌ AI models could not be loaded. Please verify your internet connection "
            "so weights can download from Hugging Face Hub."
        )
        st.stop()

    # ── Determine active wizard step ───────────────────────────────────────────
    if st.session_state.get("final_rating") is not None:
        progress_step = 4
    elif st.session_state.get("jd_phrases") is not None and (
        st.session_state.get("essential_reqs") or st.session_state.get("desirable_reqs")
    ):
        progress_step = 3
    elif st.session_state.get("jd_text", "").strip() and st.session_state.get("cv_text", "").strip():
        progress_step = 2
    else:
        progress_step = 1

    _render_progress(progress_step)

    # ══ STEP 1: Add Documents ═════════════════════════════════════════════════
    step1_result = render_step1()
    if step1_result < 2:
        return

    # ══ STEP 2: Find Skills ═══════════════════════════════════════════════════
    st.markdown("---")
    step2_result = render_step2(gliner_model)
    if step2_result < 3:
        return

    # ══ STEP 3: Review Requirements ═══════════════════════════════════════════
    st.markdown("---")
    step3_result = render_step3()
    if step3_result < 4:
        return

    # ══ STEP 4: Calculate Score ═══════════════════════════════════════════════
    st.markdown("---")
    render_step4(st_model, threshold)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "CV–Job Match · GLiNER small-v2.1 · all-MiniLM-L6-v2 · "
        "Strictly In-Memory · Designed for Streamlit Cloud"
    )


if __name__ == "__main__":
    main()
