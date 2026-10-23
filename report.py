"""
report.py
=========
Builds a plain-text report that the user can download.

The report contains:
  - Metadata (date, score, thresholds used)
  - Important disclaimers
  - Requirement-by-requirement evidence table
  - Improvement suggestions
  - Privacy statement

All wording is intentionally cautious: we never claim to verify skills
or predict hiring success.
"""

from __future__ import annotations

import datetime
from typing import List, Dict, Any

# Width of the text report (characters)
LINE_WIDTH = 80


def build_text_report(
    jd_text: str,
    cv_text: str,
    score_rows: List[Dict[str, Any]],
    coverage_score: float,
    meaning_score: float,
    final_rating: float,
    threshold: float,
) -> str:
    """
    Assemble and return a plain-text report string.

    Parameters
    ----------
    jd_text        : original JD text (for character count / reference)
    cv_text        : original CV text (for character count / reference)
    score_rows     : list of dicts from score_cv_against_requirements()
    coverage_score : weighted requirement coverage (0–100)
    meaning_score  : semantic similarity (0–1)
    final_rating   : 0.70×coverage + 0.30×meaning×100 (0–100)
    threshold      : cosine-similarity threshold used for semantic fallback

    Returns
    -------
    str – multi-line plain text suitable for a .txt download
    """
    lines: List[str] = []

    def hr(char: str = "=") -> None:
        lines.append(char * LINE_WIDTH)

    def section(title: str) -> None:
        lines.append("")
        hr()
        lines.append(f"  {title}")
        hr()

    def wrap(text: str, indent: int = 0) -> None:
        """Very simple word-wrap for long strings."""
        prefix = " " * indent
        words = text.split()
        current_line: List[str] = []
        current_len = indent

        for word in words:
            if current_len + len(word) + 1 > LINE_WIDTH:
                lines.append(prefix + " ".join(current_line))
                current_line = [word]
                current_len = indent + len(word)
            else:
                current_line.append(word)
                current_len += len(word) + 1

        if current_line:
            lines.append(prefix + " ".join(current_line))

    # ── Header ────────────────────────────────────────────────────────────────
    hr()
    lines.append("  CV–JOB MATCH REPORT")
    lines.append(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    hr()

    # ── Disclaimer ────────────────────────────────────────────────────────────
    section("IMPORTANT DISCLAIMERS")
    wrap(
        "This report was produced by a prototype tool that estimates TEXT ALIGNMENT "
        "between a CV and a job description.  It does NOT verify skills, "
        "qualifications, experience, certificates, or hiring probability."
    )
    lines.append("")
    wrap(
        "Scoring weights (0.70 / 0.30) and similarity thresholds are UNVALIDATED "
        "DESIGN CHOICES made for demonstration.  They have not been calibrated "
        "against real hiring outcomes."
    )
    lines.append("")
    wrap(
        "A direct phrase match does NOT prove proficiency.  Semantic similarity can "
        "confuse related but different skills (e.g. Java vs JavaScript).  Treat all "
        "results as a starting point for reflection, not a final judgement."
    )
    lines.append("")
    wrap(
        "PRIVACY: Uploaded CV text was processed in memory only.  Nothing is stored "
        "or logged by this app."
    )

    # ── Score summary ─────────────────────────────────────────────────────────
    section("SCORE SUMMARY")
    lines.append(f"  Final Rating          : {final_rating:.1f} / 100")
    lines.append(f"  Weighted Coverage     : {coverage_score:.1f} / 100")
    lines.append(f"  Meaning Similarity    : {meaning_score * 100:.1f} / 100")
    lines.append("")
    lines.append("  Scoring formula:")
    lines.append("    coverage = 100 × Σ(credit × weight) / Σ(weight)")
    lines.append("      where weight: essential=2, desirable=1")
    lines.append("    final = 0.70 × coverage + 0.30 × meaning_similarity × 100")
    lines.append("")
    lines.append(f"  Semantic threshold used : {threshold}")
    lines.append(f"  JD length               : {len(jd_text):,} characters")
    lines.append(f"  CV length               : {len(cv_text):,} characters")

    # ── Credit legend ─────────────────────────────────────────────────────────
    section("CREDIT LEGEND")
    lines.append("  direct   1.0  – phrase found in CV, no limitation wording detected")
    lines.append("  possible 0.5  – semantic match above threshold; PLEASE REVIEW")
    lines.append("  limited  0.0  – phrase found but near limitation/negative wording")
    lines.append("  missing  0.0  – no match found")

    # ── Evidence table ────────────────────────────────────────────────────────
    section("REQUIREMENT-BY-REQUIREMENT EVIDENCE")

    essential_rows = [r for r in score_rows if r["priority"] == "essential"]
    desirable_rows = [r for r in score_rows if r["priority"] == "desirable"]

    for group_label, group_rows in [
        ("ESSENTIAL (weight 2)", essential_rows),
        ("DESIRABLE (weight 1)", desirable_rows),
    ]:
        if not group_rows:
            continue
        lines.append("")
        lines.append(f"  {group_label}")
        lines.append("  " + "-" * (LINE_WIDTH - 2))

        for row in group_rows:
            status_map = {
                "direct": "DIRECT  ",
                "possible": "POSSIBLE",
                "limited": "LIMITED ",
                "missing": "MISSING ",
            }
            status_str = status_map.get(row["status"], row["status"].upper().ljust(8))
            credit_str = f"credit={row['credit']}"

            lines.append(f"  [{status_str}] [{credit_str}] {row['requirement']}")

            evidence = row.get("evidence", "").strip()
            if evidence:
                lines.append(f"    Supporting passage: \"{evidence[:250]}\"")
            lines.append("")

    # ── Suggestions ───────────────────────────────────────────────────────────
    section("IMPROVEMENT SUGGESTIONS")
    wrap(
        "Suggestions are based only on missing or weak text matches. "
        "Only act on suggestions that are TRUTHFUL for you.  Do not invent achievements."
    )
    lines.append("")

    missing = [r for r in score_rows if r["status"] in ("missing", "limited")]
    possible = [r for r in score_rows if r["status"] == "possible"]

    if not missing and not possible:
        lines.append("  All requirements have direct mentions – no changes suggested.")
        lines.append("  Still review the evidence table to confirm accuracy.")
    else:
        if missing:
            lines.append("  Missing / Limited requirements (consider adding if truthful):")
            for r in missing:
                lines.append(f"  • {r['requirement']} ({r['priority']})")
                wrap(
                    "If you genuinely have this skill or experience, add a specific, "
                    "truthful example (e.g. a project, tool used, measurable result).",
                    indent=4,
                )
                lines.append("")

        if possible:
            lines.append("  Possible matches – please review:")
            for r in possible:
                ev = r.get("evidence", "")
                lines.append(f"  • {r['requirement']} ({r['priority']})")
                if ev:
                    lines.append(f"    Possibly related to: \"{ev[:200]}\"")
                wrap(
                    "If this genuinely covers the requirement, consider making the "
                    "connection more explicit in your CV.",
                    indent=4,
                )
                lines.append("")

    # ── Footer ────────────────────────────────────────────────────────────────
    lines.append("")
    hr()
    lines.append("  End of report – CV-Job Match prototype")
    lines.append("  Models: GLiNER small-v2.1 · sentence-transformers/all-MiniLM-L6-v2")
    hr()

    return "\n".join(lines)
