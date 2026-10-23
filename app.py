"""
CV–Job Match App  —  User-Friendly Edition
===========================================
A step-by-step wizard that guides the user through:

  STEP 1  →  Paste your Job Description + CV
  STEP 2  →  Let AI find skill phrases  (one button click)
  STEP 3  →  Check / edit the requirement list
  STEP 4  →  Get your match score + suggestions

Design principle: each step only appears AFTER the previous one is complete,
so the user always knows exactly what to do next.

DISCLAIMERS (also shown in the app):
  • Scores measure text alignment only — NOT verified skills or hiring probability.
  • Scoring weights are unvalidated prototype values.
  • Uploaded text is processed in memory only — never stored or logged.
"""

import streamlit as st

# ── Page config — must be the FIRST Streamlit call ────────────────────────────
st.set_page_config(
    page_title="CV–Job Match",
    page_icon="📄",
    layout="centered",          # centred is friendlier for a wizard flow
    initial_sidebar_state="collapsed",
)

import datetime
import re

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


# ══════════════════════════════════════════════════════════════════════════════
# Small helper functions
# ══════════════════════════════════════════════════════════════════════════════

def _reset_all() -> None:
    """Wipe everything downstream so stale results never show."""
    for key in [
        "jd_text", "cv_text",
        "jd_phrases", "cv_phrases",
        "essential_reqs", "desirable_reqs",
        "score_rows", "coverage_score", "meaning_score", "final_rating",
        "last_extracted_jd", "last_extracted_cv",
    ]:
        st.session_state.pop(key, None)


def _reset_from_extraction() -> None:
    """Clear only extraction + scoring results (keep the pasted text)."""
    for key in [
        "jd_phrases", "cv_phrases",
        "essential_reqs", "desirable_reqs",
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
    jd_ok = ss.get("jd_text", "") == ss.get("last_extracted_jd")
    cv_ok = ss.get("cv_text", "") == ss.get("last_extracted_cv")
    return not (jd_ok and cv_ok)


def _check_length(text: str, label: str) -> bool:
    """Show an error and return False if text exceeds MAX_CHARS."""
    if len(text) > MAX_CHARS:
        st.error(
            f"⚠️  Your {label} is too long ({len(text):,} characters). "
            f"Please use at most {MAX_CHARS:,} characters (~2,500 words) "
            "to keep memory use reasonable."
        )
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Progress bar at the top
# ══════════════════════════════════════════════════════════════════════════════

def _render_progress(current_step: int) -> None:
    """
    Show a simple visual breadcrumb so the user always knows where they are.
    current_step: 1, 2, 3, or 4
    """
    steps = ["📋 Add Documents", "🔍 Find Skills", "✏️ Edit Requirements", "📊 Get Score"]
    cols = st.columns(4)
    for i, (col, label) in enumerate(zip(cols, steps), start=1):
        if i < current_step:
            col.markdown(f"<div style='text-align:center;color:#22c55e;font-weight:bold'>✅ {label}</div>", unsafe_allow_html=True)
        elif i == current_step:
            col.markdown(f"<div style='text-align:center;color:#3b82f6;font-weight:bold;border-bottom:3px solid #3b82f6;padding-bottom:4px'>▶ {label}</div>", unsafe_allow_html=True)
        else:
            col.markdown(f"<div style='text-align:center;color:#9ca3af'>{label}</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Add your documents
# ══════════════════════════════════════════════════════════════════════════════

def render_step1() -> int:
    """
    Collect the Job Description and CV text.
    Returns the current wizard step (1 if inputs missing, 2 if both present).
    """
    jd_text = st.session_state.get("jd_text", "")
    cv_text = st.session_state.get("cv_text", "")

    # ── Friendly intro ─────────────────────────────────────────────────────────
    st.markdown("### Step 1 — Add Your Documents")
    st.markdown(
        "Paste (or load the sample) **Job Description** on the left and your "
        "**CV** on the right. You can also upload a PDF, Word, or text file for your CV."
    )

    # ── Try the sample ─────────────────────────────────────────────────────────
    with st.expander("🎯 New here? Try the built-in sample first!", expanded=(not jd_text and not cv_text)):
        st.markdown(
            "Click the button below to load a fictional **Data Analyst** job and CV. "
            "This lets you see the full app flow without using your own documents."
        )
        if st.button("✨ Load sample Job Description + CV", key="btn_sample_both"):
            st.session_state["jd_text"] = SAMPLE_JD
            st.session_state["cv_text"] = SAMPLE_CV
            _reset_from_extraction()
            st.rerun()

    st.markdown("---")

    # ── Two columns: JD on left, CV on right ──────────────────────────────────
    col_jd, col_cv = st.columns(2, gap="large")

    # ── Job Description ───────────────────────────────────────────────────────
    with col_jd:
        st.markdown("#### 📋 Job Description")
        st.caption("Copy the full job posting and paste it here.")

        jd_new = st.text_area(
            "Job Description text",
            value=jd_text,
            height=280,
            key="jd_textarea",
            placeholder="Paste the job posting here…\n\nExample:\n  'We are looking for a Data Analyst with Python and SQL experience…'",
            label_visibility="collapsed",
        )
        # Detect changes
        if jd_new != jd_text:
            st.session_state["jd_text"] = jd_new
            _reset_from_extraction()

        # Character counter + hint
        char_count = len(st.session_state.get("jd_text", ""))
        if char_count > 0:
            colour = "red" if char_count > MAX_CHARS else "green"
            st.caption(f":{colour}[{char_count:,} / {MAX_CHARS:,} characters]")
        else:
            st.caption("No text yet.")

    # ── CV ────────────────────────────────────────────────────────────────────
    with col_cv:
        st.markdown("#### 📄 Your CV / Résumé")
        st.caption("Upload a file **or** paste your CV text directly below.")

        uploaded = st.file_uploader(
            "Upload PDF, DOCX, or TXT",
            type=["pdf", "docx", "txt"],
            key="cv_upload",
            label_visibility="visible",
        )

        if uploaded is not None:
            with st.spinner("Reading your file…"):
                extracted, error = read_uploaded_file(uploaded)
            if error:
                st.error(f"❌ Could not read file — {error}")
            else:
                st.success("✅ File read! You can edit the text below if anything looks wrong.")
                if extracted != st.session_state.get("cv_text", ""):
                    st.session_state["cv_text"] = extracted
                    _reset_from_extraction()

        cv_new = st.text_area(
            "CV text",
            value=st.session_state.get("cv_text", ""),
            height=230,
            key="cv_textarea",
            placeholder="Paste your CV here…\n\nOr upload a file above — the text will appear here automatically.",
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
            st.caption("No text yet.")

    # ── Decide what step we are on ────────────────────────────────────────────
    jd_ready = bool(st.session_state.get("jd_text", "").strip())
    cv_ready = bool(st.session_state.get("cv_text", "").strip())

    st.markdown("")  # breathing room

    if jd_ready and cv_ready:
        st.success("✅ Both documents added! Scroll down to **Step 2** to extract skill phrases.")
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
    Returns 3 if extraction succeeded, else 2.
    """
    st.markdown("### Step 2 — Find Skill Phrases")
    st.markdown(
        "The AI (called **GLiNER**) will scan both documents and pull out "
        "skill-related words and phrases — things like tools, technologies, "
        "methods, and certifications."
    )

    # Warn if text changed after a previous extraction
    if (
        st.session_state.get("last_extracted_jd") is not None
        and _text_changed_since_extraction()
    ):
        st.warning(
            "⚠️  You edited your JD or CV after the last extraction. "
            "Click **Find Skill Phrases** again to get fresh results."
        )

    col_btn, col_note = st.columns([1, 3])
    with col_btn:
        extract_clicked = st.button(
            "🔍 Find Skill Phrases",
            key="btn_extract",
            use_container_width=True,
            type="primary",
        )
    with col_note:
        st.caption(
            "This may take **20–60 seconds** on CPU. "
            "The AI reads every sentence — please be patient."
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
            with st.spinner("🤖 AI is reading your documents… (may take up to 60 s)"):
                jd_phrases = extract_phrases(gliner_model, jd_text)
                cv_phrases = extract_phrases(gliner_model, cv_text)

            if not jd_phrases:
                st.warning(
                    "⚠️  No skill phrases found in the Job Description. "
                    "Try adding more detail, or load the sample JD to test the app."
                )
            else:
                essential, desirable = classify_requirements(jd_phrases)
                st.session_state.update({
                    "jd_phrases": jd_phrases,
                    "cv_phrases": cv_phrases,
                    "essential_reqs": essential,
                    "desirable_reqs": desirable,
                    "last_extracted_jd": jd_text,
                    "last_extracted_cv": cv_text,
                })
                _reset_scores_only()
                st.success(
                    f"✅ Found **{len(jd_phrases)}** phrase(s) in the JD and "
                    f"**{len(cv_phrases)}** phrase(s) in your CV."
                )

    # ── Show results if extraction was done ───────────────────────────────────
    if st.session_state.get("jd_phrases") is not None:
        jd_phrases = st.session_state["jd_phrases"]
        cv_phrases = st.session_state["cv_phrases"]

        col_j, col_c = st.columns(2)

        with col_j:
            st.markdown(f"**JD skills found ({len(jd_phrases)})**")
            st.caption("These become the requirements in Step 3.")
            if jd_phrases:
                phrase_text = "  \n".join(f"• {p}" for p in jd_phrases)
                st.markdown(phrase_text)
            else:
                st.caption("None found.")

        with col_c:
            st.markdown(f"**Your CV skills found ({len(cv_phrases)})**")
            st.caption("⚠️ These are *detected mentions* — not verified abilities.")
            if cv_phrases:
                phrase_text = "  \n".join(f"• {p}" for p in cv_phrases)
                st.markdown(phrase_text)
            else:
                st.caption("None found.")

        st.success("✅ Extraction done! Scroll down to **Step 3** to review the requirements.")
        return 3

    return 2


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Review requirements
# ══════════════════════════════════════════════════════════════════════════════

def render_step3() -> int:
    """
    Let the user review and edit the Essential / Desirable requirement lists.
    Returns 4 when the user is ready to score.
    """
    st.markdown("### Step 3 — Check the Requirement List")
    st.markdown(
        "The app sorted the JD phrases into two groups. "
        "**Essential** requirements count double in the score. "
        "**Desirable** requirements count once.\n\n"
        "👇 Edit the boxes below — add anything that's missing, "
        "delete anything wrong, one item per line."
    )

    col_ess, col_des = st.columns(2, gap="large")

    # ── Essential ─────────────────────────────────────────────────────────────
    with col_ess:
        st.markdown("#### 🔴 Essential  *(counts double)*")
        st.caption("Must-have requirements. Add, remove, or correct lines.")
        ess_text = st.text_area(
            "Essential requirements — one per line",
            value="\n".join(st.session_state.get("essential_reqs", [])),
            height=220,
            key="ess_edit",
            label_visibility="collapsed",
        )
        new_ess = [r.strip() for r in ess_text.splitlines() if r.strip()]
        if new_ess != st.session_state.get("essential_reqs", []):
            st.session_state["essential_reqs"] = new_ess
            _reset_scores_only()
        st.caption(f"{len(new_ess)} item(s)")

    # ── Desirable ─────────────────────────────────────────────────────────────
    with col_des:
        st.markdown("#### 🟡 Desirable  *(counts once)*")
        st.caption("Nice-to-have requirements. Add, remove, or correct lines.")
        des_text = st.text_area(
            "Desirable requirements — one per line",
            value="\n".join(st.session_state.get("desirable_reqs", [])),
            height=220,
            key="des_edit",
            label_visibility="collapsed",
        )
        new_des = [r.strip() for r in des_text.splitlines() if r.strip()]
        if new_des != st.session_state.get("desirable_reqs", []):
            st.session_state["desirable_reqs"] = new_des
            _reset_scores_only()
        st.caption(f"{len(new_des)} item(s)")

    # ── Tips ──────────────────────────────────────────────────────────────────
    with st.expander("💡 Tips for editing the lists"):
        st.markdown(
            """
- **Add a missing skill:** type it on a new line in the right box.
- **Remove something wrong:** delete that line.
- **Change priority:** cut a line from one box and paste it into the other.
- **Tip:** Essential = "must have" in the job ad. Desirable = "nice to have" or "advantage".
            """
        )

    total = len(st.session_state.get("essential_reqs", [])) + len(st.session_state.get("desirable_reqs", []))
    if total == 0:
        st.warning("⚠️  Both lists are empty. Add at least one requirement to score against.")
        return 3

    st.success(f"✅ {total} requirement(s) ready. Scroll down to **Step 4** to get your score.")
    return 4


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Score & Results
# ══════════════════════════════════════════════════════════════════════════════

def render_step4(st_model, threshold: float) -> None:
    """
    Score the CV against the requirements, then display results clearly.
    """
    st.markdown("### Step 4 — Get Your Match Score")
    st.markdown(
        "The app will compare each requirement from Step 3 against your CV text. "
        "Click the button and wait about 10–30 seconds."
    )

    # Stale-extraction warning
    if _text_changed_since_extraction():
        st.warning(
            "⚠️  Your JD or CV text changed after the last extraction. "
            "Go back to **Step 2** and click **Find Skill Phrases** again first."
        )

    col_btn, col_note = st.columns([1, 3])
    with col_btn:
        score_clicked = st.button(
            "📊 Calculate My Score",
            key="btn_score",
            use_container_width=True,
            type="primary",
        )
    with col_note:
        st.caption("Takes about 10–30 seconds.")

    if score_clicked:
        essential = st.session_state.get("essential_reqs", [])
        desirable = st.session_state.get("desirable_reqs", [])
        all_reqs = [(r, "essential") for r in essential] + [(r, "desirable") for r in desirable]

        if not all_reqs:
            st.warning("No requirements to score. Go back to Step 3 and add some.")
            return

        jd_text = st.session_state.get("jd_text", "")
        cv_text = st.session_state.get("cv_text", "")

        with st.spinner("🤖 Scoring your CV… (checking each requirement)"):
            score_rows = score_cv_against_requirements(
                requirements=all_reqs,
                cv_text=cv_text,
                st_model=st_model,
                threshold=threshold,
            )
            meaning_score = compute_meaning_similarity(jd_text, cv_text, st_model)
            coverage_score, final_rating = compute_final_rating(score_rows, meaning_score)

        st.session_state.update({
            "score_rows": score_rows,
            "coverage_score": coverage_score,
            "meaning_score": meaning_score,
            "final_rating": final_rating,
        })

    # ── Display results ───────────────────────────────────────────────────────
    if st.session_state.get("final_rating") is None:
        return

    fr  = st.session_state["final_rating"]
    cov = st.session_state["coverage_score"]
    msim= st.session_state["meaning_score"]
    rows = st.session_state["score_rows"]

    st.markdown("---")

    # ── Score banner ──────────────────────────────────────────────────────────
    st.markdown("#### 🎯 Your Match Score")

    if fr >= 75:
        st.success(f"### {fr:.0f} / 100 — Strong Match 🟢")
        st.markdown("Your CV mentions most of the job requirements using similar language.")
    elif fr >= 50:
        st.info(f"### {fr:.0f} / 100 — Moderate Match 🟡")
        st.markdown("Your CV covers some requirements well, but a few gaps remain.")
    else:
        st.warning(f"### {fr:.0f} / 100 — Weak Match 🔴")
        st.markdown("Your CV may be missing key requirement language from the job posting.")

    st.progress(min(int(fr), 100))

    # ── Score breakdown ────────────────────────────────────────────────────────
    with st.expander("📐 How is the score calculated?", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Requirement Coverage", f"{cov:.0f} / 100",
                    help="How many requirements your CV mentions (weighted by priority).")
        col2.metric("Meaning Similarity", f"{msim*100:.0f} / 100",
                    help="How similar the overall writing style and vocabulary is.")
        col3.metric("Final Rating", f"{fr:.0f} / 100",
                    help="70% coverage + 30% meaning similarity.")
        st.markdown(
            """
**Formula:** `Final = 0.70 × Coverage + 0.30 × Meaning`

- **Coverage** checks whether each requirement phrase appears in your CV.
- **Meaning** checks whether your CV and the JD use similar vocabulary overall.
- Essential requirements count **twice** as much as desirable ones.

> ⚠️ These weights are *prototype values* — not proven by research.
> The score estimates **text alignment only**. It does not verify your skills,
> qualifications, experience, or predict hiring success.
            """
        )

    st.caption("⚠️ Reminder: this score measures text similarity, not actual competence.")
    st.markdown("")

    # ── Requirement-by-requirement table ─────────────────────────────────────
    st.markdown("#### 📋 Requirement-by-Requirement Breakdown")
    st.markdown(
        "This table shows how each requirement was matched to your CV. "
        "Click a row to read the supporting quote from your CV."
    )

    import pandas as pd

    icon_map = {
        "direct":   "✅ Found in CV",
        "possible": "🔶 Possible — please check",
        "limited":  "⚠️ Found, but limitation detected",
        "missing":  "❌ Not found",
    }

    rows_display = []
    for r in rows:
        rows_display.append({
            "Requirement":    r["requirement"],
            "Priority":       "🔴 Essential" if r["priority"] == "essential" else "🟡 Desirable",
            "Result":         icon_map.get(r["status"], r["status"]),
            "Score":          f"{r['credit']:.1f}",
            "CV Passage":     r.get("evidence", "—")[:120] + ("…" if len(r.get("evidence","")) > 120 else ""),
        })

    df = pd.DataFrame(rows_display)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Suggestions ────────────────────────────────────────────────────────────
    st.markdown("#### 💡 What You Can Improve")
    st.caption(
        "Only act on suggestions that are **100% truthful** for you. "
        "Never add skills or achievements you don't actually have."
    )

    missing = [r for r in rows if r["status"] in ("missing", "limited")]
    possible = [r for r in rows if r["status"] == "possible"]

    if not missing and not possible:
        st.success(
            "🎉 All requirements were directly matched in your CV! "
            "Double-check the table above to make sure the matches are accurate."
        )
    else:
        if missing:
            st.markdown("**❌ These requirements were not found — add them if truthful:**")
            for r in missing:
                priority_label = "essential" if r["priority"] == "essential" else "desirable"
                with st.container():
                    st.markdown(
                        f"- **{r['requirement']}** *({priority_label})*\n"
                        f"  → If you genuinely have this skill, add a specific example to your CV "
                        f"(e.g. a project name, a tool version, a measurable result)."
                    )

        if possible:
            st.markdown("**🔶 These may be covered — please verify:**")
            for r in possible:
                ev = r.get("evidence", "")
                with st.container():
                    st.markdown(
                        f"- **{r['requirement']}** *(possible match)*\n"
                        f"  → Your CV says: *\"{ev[:150]}\"*\n"
                        f"  → If this genuinely covers the requirement, "
                        f"make the connection more explicit."
                    )

    # ── Download ───────────────────────────────────────────────────────────────
    st.markdown("#### 📥 Download Your Report")
    st.markdown(
        "Save a plain-text version of all these results to review offline "
        "or share with a careers advisor."
    )

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
        label="⬇️ Download Report (.txt)",
        data=report_text.encode("utf-8"),
        file_name=f"cv_match_report_{datetime.date.today()}.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — advanced settings + help
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> float:
    with st.sidebar:
        st.header("⚙️ Settings")
        threshold = st.slider(
            "Similarity sensitivity",
            min_value=0.40,
            max_value=0.95,
            value=0.60,
            step=0.05,
            help=(
                "How similar a CV sentence needs to be to count as a 'possible match'. "
                "Lower = more lenient (more possible matches, more false positives). "
                "Higher = stricter (fewer matches, fewer false positives)."
            ),
        )
        st.caption(f"Current: {threshold:.2f}")

        st.markdown("---")
        st.markdown("**❓ Quick Help**")
        with st.expander("What do the scores mean?"):
            st.markdown(
                """
| Score | Meaning |
|-------|---------|
| 75–100 | Strong text match |
| 50–74 | Moderate match |
| 0–49 | Weak match |

The score measures **text similarity only**.
It does NOT verify your actual skills.
                """
            )
        with st.expander("What are ✅ / 🔶 / ❌?"):
            st.markdown(
                """
- ✅ **Found** — that exact phrase appears in your CV
- 🔶 **Possible** — a similar phrase was found (AI matched it — you should verify)
- ⚠️ **Limited** — phrase found but near words like "no experience"
- ❌ **Missing** — not found at all
                """
            )
        with st.expander("My PDF shows no text?"):
            st.markdown(
                "Your PDF is probably **scanned** (a photo of paper). "
                "This app can't read images — only text-based PDFs. "
                "Try exporting your CV as a text PDF, or paste the text manually."
            )
        with st.expander("Privacy"):
            st.markdown(
                "Your CV and JD text is processed **in memory only** during your session. "
                "Nothing is saved to a database or sent to any third-party service."
            )

        st.markdown("---")
        if st.button("🔄 Start Over", key="btn_reset"):
            _reset_all()
            st.rerun()

        st.caption("Models: GLiNER small-v2.1 · all-MiniLM-L6-v2")
    return threshold


# ══════════════════════════════════════════════════════════════════════════════
# Main app
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    st.title("📄 CV–Job Match")
    st.markdown(
        "**Find out how well your CV matches a job description** — "
        "using AI, in 4 simple steps. No sign-up, no paid API, nothing stored."
    )
    st.caption(
        "⚠️ This tool measures *text similarity* only — not your actual skills, "
        "qualifications, or hiring chances. Treat the score as a reflection tool."
    )
    st.markdown("---")

    # ── Settings (sidebar) ─────────────────────────────────────────────────────
    threshold = render_sidebar()

    # ── Load models (once per session) ────────────────────────────────────────
    with st.spinner("⏳ Loading AI models — first run downloads ~140 MB, please wait…"):
        gliner_model = load_gliner()
        st_model     = load_sentence_transformer()

    if gliner_model is None or st_model is None:
        st.error(
            "❌ Could not load the AI models. "
            "Check your internet connection — the models are downloaded from "
            "Hugging Face Hub on first run. See README.md for troubleshooting."
        )
        st.stop()

    # ── Determine which step the user is on ───────────────────────────────────
    # We figure out the furthest completed step to show the progress bar correctly.
    if st.session_state.get("jd_phrases") is not None and st.session_state.get("essential_reqs") is not None:
        progress_step = 3
    elif st.session_state.get("jd_text", "").strip() and st.session_state.get("cv_text", "").strip():
        progress_step = 2
    else:
        progress_step = 1

    _render_progress(progress_step)

    # ══ STEP 1 ════════════════════════════════════════════════════════════════
    step1_result = render_step1()
    if step1_result < 2:
        return  # Nothing more to show until both documents are added

    # ══ STEP 2 ════════════════════════════════════════════════════════════════
    st.markdown("---")
    step2_result = render_step2(gliner_model)
    if step2_result < 3:
        return  # Extraction not done yet

    # ══ STEP 3 ════════════════════════════════════════════════════════════════
    st.markdown("---")
    step3_result = render_step3()
    if step3_result < 4:
        return  # No requirements yet

    # ══ STEP 4 ════════════════════════════════════════════════════════════════
    st.markdown("---")
    render_step4(st_model, threshold)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "CV–Job Match · prototype · GLiNER small-v2.1 · all-MiniLM-L6-v2 · "
        "no data stored · not a hiring tool"
    )


if __name__ == "__main__":
    main()
