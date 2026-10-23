# CV–Job Match

A beginner-friendly web app that compares a CV against a job description (JD)
using two pretrained Hugging Face models — no paid API required.

> **Disclaimer:** This tool estimates *text alignment* only. It does **not**
> verify skills, qualifications, experience, certificates, or hiring probability.
> Scores are produced by an *unvalidated prototype* and should be treated as a
> starting point for reflection, not a final judgement.

---

## Table of Contents

1. [What the App Does](#what-the-app-does)
2. [Project File Structure](#project-file-structure)
3. [Models Used](#models-used)
4. [Local Setup (Antigravity / VS Code)](#local-setup)
5. [Running Locally](#running-locally)
6. [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
7. [How the Score is Calculated](#how-the-score-is-calculated)
8. [Limitations](#limitations)
9. [Privacy](#privacy)
10. [Troubleshooting](#troubleshooting)
11. [Suggested Evaluation Plan](#suggested-evaluation-plan)
12. [Beginner's Presentation Guide](#beginners-presentation-guide)

---

## What the App Does

1. **Accept inputs** — paste a JD and CV as text, or upload a PDF / DOCX / TXT file.
2. **Extract skill phrases** — GLiNER scans both documents for broad skill-related
   concepts (skills, tools, technologies, methods, certifications) without using a
   fixed skill list.
3. **Review requirements** — an editable "Essential" and "Desirable" list lets you
   add missed items, remove irrelevant ones, or change priorities.
4. **Score the CV** — direct phrase mentions score 1.0; possible semantic matches
   score 0.5 (flagged for human review); missing phrases score 0.0.
5. **Show evidence** — every score is backed by a quoted CV passage so you can
   verify why a requirement was matched.
6. **Suggest improvements** — the app lists missing or weak requirements and
   reminds you to add examples only if they are truthful.
7. **Download a report** — a plain-text summary you can share or keep.

---

## Project File Structure

```
Generic-CV-Matcher/
├── app.py            # Main Streamlit application
├── models.py         # Model loading and caching
├── extraction.py     # GLiNER phrase extraction
├── scoring.py        # Direct-mention + semantic scoring logic
├── file_reader.py    # PDF / DOCX / TXT file reading
├── report.py         # Plain-text report builder
├── samples.py        # Sample JD and CV for demonstration
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## Models Used

| Model | Purpose | Size |
|-------|---------|------|
| [`urchade/gliner_small-v2.1`](https://huggingface.co/urchade/gliner_small-v2.1) | Zero-shot NER — extracts skill phrases using your own label names | ≈ 50 MB |
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Encodes text into vectors for semantic similarity | ≈ 90 MB |

Both models are downloaded automatically from Hugging Face Hub on first run
(~140 MB total). They run on CPU; a GPU will speed them up but is not required.

**These models were not trained for this project.** They are general-purpose
pretrained models used as-is.

---

## Local Setup

### Prerequisites

- Python 3.10 or 3.11 (recommended)
- `pip` package manager
- Git
- Antigravity IDE (or any terminal)

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/Generic-CV-Matcher.git
cd Generic-CV-Matcher
```

### Step 2 — Create a virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

> **Antigravity tip:** If you open the project folder in Antigravity IDE,
> it will usually detect and activate the virtual environment automatically.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This downloads PyTorch, GLiNER, Sentence-Transformers, Streamlit, and the
file-reading libraries. Expect **5–15 minutes** depending on your connection.

> **GPU (optional):** If you have an NVIDIA GPU and CUDA installed, replace
> the `torch` line in `requirements.txt` with the matching CUDA wheel from
> https://pytorch.org/get-started/locally/ — the app automatically detects
> the GPU at runtime.

---

## Running Locally

```bash
# Make sure your virtual environment is active, then:
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it
in your browser.

**First run:** The two models are downloaded from Hugging Face (~140 MB).
This takes 1–5 minutes and happens only once. Subsequent runs use the
cached models.

---

## Deploying to Streamlit Community Cloud

> **Honest note:** Deployment instructions below are accurate as of October 2025.
> The steps assume you have a GitHub account and a Streamlit Community Cloud account.
> This guide describes the process; it does not confirm that deployment succeeded
> for this specific project.

### Step 1 — Push to GitHub

```bash
git add .
git commit -m "Initial commit – CV Job Match app"
git push origin main
```

Make sure your repository is **public**, or grant Streamlit access to a private
repository during the next step.

### Step 2 — Sign in to Streamlit Community Cloud

Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

### Step 3 — Deploy the app

1. Click **"New app"**.
2. Select your repository (`Generic-CV-Matcher`), branch (`main`), and main file (`app.py`).
3. Click **"Deploy"**.

Streamlit Community Cloud will:
- Install all packages from `requirements.txt` (allow 5–10 minutes).
- Download the models on first boot (~140 MB; allow a further 2–5 minutes).
- Start the app and give you a public URL.

### Step 4 — Memory considerations

The Streamlit Community Cloud **free tier** provides approximately **1 GB RAM**.
Peak usage during scoring is ~800 MB – 1.2 GB.

If the app crashes with an out-of-memory error:
- Keep JD and CV inputs under 15 000 characters each.
- The app enforces this limit automatically with a user-facing error.
- If you need larger inputs, consider upgrading to a paid Streamlit plan or
  running the app locally.

### Secrets

This project does **not** require any API keys or secrets. No
`secrets.toml` file is needed.

---

## How the Score is Calculated

> All weights and thresholds are **unvalidated design choices** made for
> demonstration. They have not been calibrated against real hiring data.

### Step 1 — Phrase extraction (GLiNER)

GLiNER scans the JD and CV using broad concept labels (skill, software tool,
technology, method, certification). It returns text spans — not a lookup against
a fixed skill dictionary.

### Step 2 — Requirement classification

JD phrases are auto-sorted into **Essential** (weight 2) and **Desirable**
(weight 1) lists using keyword heuristics. You can override this in the UI.

### Step 3 — Per-requirement credit

| Situation | Credit |
|-----------|--------|
| Direct phrase match, no limitation wording | 1.0 |
| Semantic match ≥ threshold, with supporting passage | 0.5 *(review this)* |
| Missing or limitation/negative wording | 0.0 |

**Limitation wording** includes phrases like "no experience", "unfamiliar with",
"limited experience", etc.

**Semantic matching** uses cosine similarity between normalized Sentence-Transformer
embeddings. The default threshold is 0.60 (adjustable in the sidebar).

### Step 4 — Weighted coverage

```
coverage = 100 × Σ(credit × weight) / Σ(weight)
```

### Step 5 — Meaning similarity

Each JD window (5 sentences) is encoded; the best-matching CV sentence is found;
scores are averaged.

### Step 6 — Final rating

```
final_rating = 0.70 × coverage + 0.30 × (meaning_similarity × 100)
```

---

## Limitations

| Limitation | Detail |
|-----------|--------|
| Text alignment only | The score measures how similarly the JD and CV are *worded*, not whether you actually have the skills. |
| Unvalidated weights | The 0.70/0.30 split and threshold values are prototype choices, not empirically calibrated. |
| Semantic confusion | The model can give high similarity to related-but-different skills (e.g. Java vs JavaScript). |
| No OCR | Scanned PDFs (images) cannot be read. Export your CV as a text-based PDF or paste the text. |
| Token limit | GLiNER processes text in 400-word windows. Very dense bullet lists may produce duplicate or missed phrases. |
| English-only | Models perform best on English text. |
| Memory | Free Streamlit tier has ~1 GB RAM. Very long documents may cause out-of-memory errors. |

---

## Privacy

- Uploaded CV text is **processed in memory only** during your browser session.
- Nothing is written to disk, stored in a database, or sent to any third-party API.
- When you close or refresh the browser tab, all session data is discarded.
- Model weights are downloaded from Hugging Face Hub to the server's local cache
  (standard behaviour for all Hugging Face models).

---

## Troubleshooting

### "Failed to load GLiNER / Sentence-Transformer"
- Check your internet connection; models are downloaded from Hugging Face on first run.
- On Streamlit Community Cloud, the download happens during app startup — wait
  for the spinner to finish.
- If the error persists, check whether Hugging Face Hub is reachable from your network.

### App is very slow
- Both models run on CPU by default on Streamlit Community Cloud (~20–60 s for extraction).
- Reduce the JD/CV length to stay well under 15 000 characters.
- Running locally with a GPU is significantly faster.

### Out-of-memory crash on Streamlit Cloud
- The free tier has ~1 GB RAM. Peak usage can approach or exceed this.
- Keep both inputs short (< 10 000 characters each is safest).
- Consider a paid Streamlit plan or run locally.

### PDF shows "no text extracted"
- Your PDF is probably scanned (image-only). Open it in Acrobat and check
  whether you can select/copy text. If not, you need OCR software first.

### DOCX shows garbled text
- Tables and text boxes inside DOCX files are sometimes read in a different
  order than they appear visually. Edit the extracted text in the text area
  before extraction.

### Phrases look wrong after editing the JD/CV
- The app detects text changes and warns you to re-run extraction.
  Always click **"Extract phrases"** again after editing either document.

### `pip install` fails with a C++ build error
- Make sure you have the Microsoft C++ Build Tools installed (Windows).
- Alternatively, use the pre-built wheels: `pip install --prefer-binary -r requirements.txt`

---

## Suggested Evaluation Plan

> Do **not** describe basic functional checks as accuracy benchmarks.
> The following is a rigorous evaluation plan for future work.

### What would a proper evaluation look like?

1. **Collect labeled examples:** Gather 50–100 real (or carefully constructed)
   JD–CV pairs where a human recruiter has rated each CV against each JD on a
   scale (e.g. "strong match", "partial match", "weak match").

2. **Split into development and held-out sets:** Use 70% to tune thresholds
   and weights; keep 30% strictly unseen until final evaluation.

3. **Compute metrics:**
   - Rank-correlation (Spearman ρ) between the app's score and human rating.
   - Precision/recall on per-requirement "matched" vs "not matched" labels.
   - False-positive rate for semantic matches (cases where the model thought
     a phrase matched but the human disagreed).

4. **Test edge cases:**
   - CVs with explicit limitation phrases ("no experience with Python").
   - Synonyms and abbreviations (e.g. "ML" vs "machine learning").
   - Very short CVs and very long JDs.
   - Non-English text.

5. **Document findings honestly:** Report what was tested, the dataset size,
   who labeled it, and any observed failure modes.

### Current status

This prototype has **not** been evaluated against human-labeled data.
The scoring weights are unvalidated. Do not cite any numbers from this
tool in research or hiring decisions.

---

## Beginner's Presentation Guide

Use this section to prepare for a viva, class demo, or project presentation.

---

### 1. What the project does (30-second version)

> "I built a web app that compares a CV to a job description using two
> pretrained AI models. One model finds skill-related phrases; the other
> measures how similar the overall meaning of the two documents is.
> The app gives a score, shows evidence for each requirement, and
> suggests CV improvements — all without using a paid API."

---

### 2. Which libraries and models it uses

| Component | Library / Model | Purpose |
|-----------|----------------|---------|
| Web interface | Streamlit | Turns Python code into an interactive web page |
| Phrase extraction | GLiNER `gliner_small-v2.1` | Finds skill phrases using flexible label names |
| Semantic similarity | Sentence-Transformers `all-MiniLM-L6-v2` | Encodes text as vectors for meaning comparison |
| Numerical computing | NumPy | Vector dot products for cosine similarity |
| Data display | pandas | DataFrame for the evidence table |
| PDF reading | pdfplumber | Extracts text from text-based PDFs |
| DOCX reading | python-docx | Extracts paragraphs from Word documents |
| Deep learning runtime | PyTorch | Runs the neural network models on CPU or GPU |

---

### 3. How data flows through the app

```
User pastes JD + CV text
        │
        ▼
[ GLiNER ] ──► Skill phrases from JD  ──► Essential / Desirable lists (editable)
             ──► Skill phrases from CV  ──► Displayed as detected mentions
        │
        ▼
[ Direct mention search ] ──────────────────────────────────────────────► Credit 1.0
[ Sentence-Transformer cosine similarity above threshold ] ──────────► Credit 0.5
[ No match ] ────────────────────────────────────────────────────────► Credit 0.0
        │
        ▼
Weighted coverage = 100 × Σ(credit × weight) / Σ(weight)
Meaning similarity = average best cosine-similarity (JD windows vs CV)
Final rating = 0.70 × coverage + 0.30 × (meaning similarity × 100)
        │
        ▼
Evidence table + Improvement suggestions + Downloadable report
```

---

### 4. How the score is calculated

Speak through this step by step:

1. **Extraction** — GLiNER finds phrases like "Python", "A/B testing", "AWS".
2. **Classification** — JD phrases split into Essential (weight 2) and Desirable (weight 1).
3. **Per-requirement credit** — direct mention → 1.0; semantic match → 0.5; missing → 0.0.
4. **Weighted coverage** — `100 × sum(credit × weight) / sum(weights)`.
5. **Meaning similarity** — JD and CV encoded as vectors; best-match similarity averaged.
6. **Final** — `0.70 × coverage + 0.30 × meaning`.

> "These weights are design choices I made for the prototype. They haven't been
> tested against real hiring outcomes, so they should be seen as a demonstration,
> not a calibrated benchmark."

---

### 5. The main limitations

Be honest — examiners appreciate this:

- "The score measures *text similarity*, not actual skills or experience."
- "Semantic similarity can confuse related but different things, for example Java and JavaScript."
- "Scanned PDFs cannot be read because they contain images, not text."
- "The weights (70%/30%) and threshold (0.60) are arbitrary starting points, not validated values."
- "The tool has not been evaluated against a labeled dataset of real CVs."

---

### 6. Likely teacher / viva questions — with accurate answers

**Q: Why did you choose GLiNER instead of spaCy or a keyword list?**

> GLiNER is a zero-shot NER model — I give it label names like "skill" or
> "certification" and it finds matching spans in the text without needing a
> pre-built list of skill names. This makes it flexible for any job sector.
> spaCy would need a custom-trained model for skill extraction, and a keyword
> list would miss synonyms and phrasing variations.

**Q: What is cosine similarity and why use it here?**

> When you encode a sentence with a Sentence-Transformer, you get a list of
> numbers (a vector) that represents its meaning. Cosine similarity measures
> the angle between two vectors — vectors pointing in the same direction score
> close to 1.0 (similar meaning) and vectors pointing in opposite directions
> score close to 0.0 (different meaning). I use it to find CV sentences that
> are similar in meaning to a JD requirement, even when they don't use the
> same words.

**Q: How did you decide on 0.70 and 0.30 as the final score weights?**

> I chose them as a starting point based on the idea that explicit requirement
> coverage is more informative than overall document similarity. They are
> unvalidated; a proper calibration would require a labeled dataset and
> optimisation against human judgements.

**Q: What happens if the CV is a scanned image PDF?**

> pdfplumber can only read text-layer PDFs. A scanned PDF contains only an
> image, so pdfplumber returns nothing. The app detects this and tells the
> user that OCR software (like Tesseract) would be needed first — but that is
> outside the scope of this project.

**Q: Is this suitable for real hiring decisions?**

> No. The tool is a prototype that demonstrates how NLP techniques can be
> applied to text comparison. It has not been validated, it cannot verify
> qualifications, and it would need extensive testing before being used in
> any real recruitment process.

**Q: How would you improve it?**

> Collect a labeled dataset of JD–CV pairs rated by human recruiters, then
> tune the threshold and weights to maximise rank-correlation with those
> ratings. Adding OCR support, multi-language models, and a proper held-out
> evaluation set would all be meaningful improvements.

**Q: How does Streamlit session state work here?**

> Streamlit re-runs the entire Python script from top to bottom every time
> the user interacts with the page. Session state (`st.session_state`) is a
> dictionary that persists between re-runs, so I store the extracted phrases,
> requirements, and scores there. I also track the text that was active when
> extraction last ran, and warn the user if it has changed — preventing stale
> scores from being displayed alongside new text.

**Q: Why cache the models with `@st.cache_resource`?**

> Each model is ~50–90 MB and takes several seconds to load. Without caching,
> Streamlit would reload the models from disk on every button click, making
> the app unusably slow. `@st.cache_resource` tells Streamlit to load the
> model once and reuse the same object in memory for the entire server session.

---

*End of README — CV–Job Match prototype*
