# CV–Job Match — Presentation Guide
### Complete Slide-by-Slide Content for Student Presentation / Viva

> **How to use this file:**
> Each "SLIDE" section = one PowerPoint / Google Slides slide.
> Speaker notes are included under every slide — read them aloud.
> Estimated time: **12–15 minutes** + Q&A.

---

# SLIDE 1 — TITLE SLIDE

## CV–Job Match
### An AI-Powered CV Analyser Using Pretrained NLP Models

**Your Name · Course / Module · Date**

> *"How well does your CV actually match the job you're applying for?"*

**Speaker notes:**
Start with the question on screen. Pause 3 seconds. Then say:
*"Most people guess the answer. This project lets you measure it — using real AI
models, no paid API, and a live web app you can open right now."*

---

# SLIDE 2 — THE PROBLEM

## The Problem Every Job Seeker Faces

- 📋 You find a job description with 15+ requirements
- 📄 You have a CV you wrote months ago
- ❓ You don't know how well they actually match
- ⏱️ Manually checking every requirement takes 30+ minutes
- 😟 You might miss keywords that ATS scanners look for

> **ATS = Applicant Tracking System** — software companies use to filter CVs
> automatically before any human reads them.

**Speaker notes:**
*"Every job seeker faces this. ATS software filters CVs by keyword matching —
if your CV doesn't use the right language, it gets rejected before a human sees it.
This project gives you a tool to measure that gap quickly and honestly."*

---

# SLIDE 3 — THE SOLUTION

## What CV–Job Match Does

| Step | What Happens |
|------|-------------|
| 1 | Paste or upload your JD and CV |
| 2 | AI extracts skill-related phrases from both |
| 3 | You review and edit the requirement list |
| 4 | App scores your CV against each requirement |
| 5 | You see evidence, a rating, and improvement tips |
| 6 | Download a plain-text report |

✅ Runs in a browser · ✅ No paid API · ✅ No data stored

**Speaker notes:**
*"The app works in six steps. Give it the job description and your CV.
An AI model extracts skill phrases. You can edit the list.
Then the app scores each requirement and shows you exactly which CV sentence
it matched — or tells you it's missing. Finally, download a report."*

---

# SLIDE 4 — TECHNOLOGY STACK

## Technology Used

### Web Framework
- **Streamlit** — turns Python code into an interactive web app

### AI Models (pretrained, free, no API key)

| Model | Source | Purpose |
|-------|--------|---------|
| `urchade/gliner_small-v2.1` | Hugging Face | Extracts skill phrases |
| `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face | Measures meaning similarity |

### Supporting Libraries

| Library | Purpose |
|---------|---------|
| PyTorch | Runs the neural networks on CPU/GPU |
| pdfplumber | Reads text from PDF files |
| python-docx | Reads text from Word (DOCX) files |
| NumPy | Vector maths for cosine similarity |
| pandas | Displays the evidence table |

**Speaker notes:**
*"Streamlit lets me build a browser UI in pure Python — no HTML or JavaScript needed.
The two AI models are pretrained by their original researchers and downloaded from
Hugging Face Hub. I am using them as tools, not claiming to have trained them."*

---

# SLIDE 5 — MODEL 1: GLiNER

## What is GLiNER?

**GLiNER = Generalist and Lightweight model for Named Entity Recognition**

### How it works
1. Give it a piece of text
2. Give it **label names** (e.g. `"skill"`, `"technology"`, `"certification"`)
3. It finds text spans that match those concepts
4. **No fixed skill list needed** — it understands meaning from context

### Example
```
Text:    "Experience with Python and A/B testing required."
Labels:  ["skill", "technology", "method"]

Output:  "Python"      → technology  (confidence 0.92)
         "A/B testing" → method      (confidence 0.87)
```

### Why not a keyword list?
| Keyword List | GLiNER |
|-------------|--------|
| Misses synonyms ("ML" vs "machine learning") | Understands meaning |
| Needs manual updates per sector | Works on any job type |
| Fixed, brittle | Flexible, context-aware |

**Speaker notes:**
*"Named Entity Recognition normally finds people's names and cities in news articles.
GLiNER was trained to find any concept you describe by name.
The key advantage: I don't need a list of 50,000 known skills —
I just tell it 'find skills and technologies' and it does."*

---

# SLIDE 6 — MODEL 2: SENTENCE-TRANSFORMER

## What is a Sentence-Transformer?

### Core idea: turning words into numbers
- Model reads a sentence → outputs a **vector** (list of 384 numbers)
- Sentences with **similar meaning** → vectors that **point in the same direction**
- Similarity measured with **cosine similarity** (0 = unrelated, 1 = identical meaning)

### Visual analogy
```
"I have Python experience"   →  [0.12, -0.45, 0.83, ...]  ─┐
                                                              ├─ similarity = 0.91 ✅
"Proficient in Python"       →  [0.14, -0.43, 0.81, ...]  ─┘

"I enjoy hiking on weekends" →  [-0.67, 0.22, -0.31, ...] ──── similarity = 0.08 ❌
```

### Why this matters for CV matching
- JD says: *"machine learning"* — CV says: *"scikit-learn and model training"*
- Direct word search misses this connection
- Semantic similarity catches it

**Speaker notes:**
*"Think of it like Google Search — if you type 'car' it finds results about 'automobile'
because it understands meaning, not just spelling. Each sentence becomes a point in
mathematical space; similar sentences are close together. Cosine similarity measures
how close they are."*

---

# SLIDE 7 — DATA FLOW DIAGRAM

## How Data Flows Through the App

```
┌───────────────────────────────────────────────┐
│                  USER INPUT                   │
│  Job Description (paste/text)                 │
│  CV (paste / upload PDF / DOCX / TXT)         │
└────────────────┬──────────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
  ┌───────────┐    ┌─────────────┐
  │  GLiNER   │    │ File Reader │
  │ Extracts  │    │ PDF/DOCX    │
  │ JD phrases│    │ → plain text│
  └─────┬─────┘    └─────────────┘
        │
        ▼
┌───────────────────────┐
│ Requirement Classifier│
│ Essential (weight 2)  │  ← User can edit these lists
│ Desirable (weight 1)  │
└───────────┬───────────┘
            │
            ▼
┌─────────────────────────────────────────────┐
│               SCORING ENGINE                │
│                                             │
│  For each requirement:                      │
│  1. Direct mention search                   │
│     → found, no limit words → credit 1.0   │
│     → found, limit words    → credit 0.0   │
│  2. Semantic fallback (Sentence-Transformer)│
│     → similarity ≥ threshold → credit 0.5  │
│     → below threshold       → credit 0.0   │
└───────────┬─────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────┐
│               RESULTS DISPLAY               │
│  Final Rating  │  Evidence Table            │
│  Coverage      │  Suggestions               │
│  Similarity    │  Downloadable Report       │
└─────────────────────────────────────────────┘
```

**Speaker notes:**
*"Walk through each box. Inputs arrive at the top.
GLiNER processes the JD; the file reader converts uploads to text.
Phrases are classified — the user can override.
The scoring engine runs two checks per requirement: direct search first,
semantic fallback second. Results are displayed with evidence."*

---

# SLIDE 8 — SCORING: CREDITS

## Scoring — Part 1: Per-Requirement Credit

| Situation | Credit | Label |
|-----------|--------|-------|
| Phrase found in CV, **no** limitation words nearby | **1.0** | ✅ Direct mention |
| Phrase not found, semantic similarity **≥ threshold** | **0.5** | 🔶 Possible — review |
| Phrase found near limitation words | **0.0** | ⚠️ Limitation detected |
| Phrase not found at all | **0.0** | ❌ Missing |

### Limitation words detected (examples)
> "no experience", "unfamiliar with", "limited experience",
> "never used", "not proficient", "lack of"

### Why 0.5 for semantic matches (not 1.0)?
- Semantic similarity can confuse related-but-different skills
- Example: "Java" and "JavaScript" — similar words, very different skills
- Human review is required before trusting the match

**Speaker notes:**
*"The graded credit system is more nuanced than a simple yes/no.
A direct mention scores full credit — but if the surrounding text says
'no experience with Python', that is a limitation and scores zero.
Semantic matches score half and are flagged for the user to verify —
because the model can be fooled by similar-sounding but different skills."*

---

# SLIDE 9 — SCORING: FORMULA

## Scoring — Part 2: The Formula

### Weighted Coverage

```
Essential requirements → weight = 2
Desirable requirements → weight = 1

Coverage = 100 × Σ(credit × weight)
                  ─────────────────
                       Σ(weight)
```

**Worked example:**

| Requirement | Priority | Credit | Weight | Credit × Weight |
|-------------|---------|--------|--------|-----------------|
| Python | Essential | 1.0 | 2 | 2.0 |
| SQL | Essential | 1.0 | 2 | 2.0 |
| AWS | Desirable | 0.0 | 1 | 0.0 |
| Agile | Desirable | 0.5 | 1 | 0.5 |
| **Total** | | | **6** | **4.5** |

**Coverage = 100 × 4.5 / 6 = 75.0**

### Meaning Similarity
Encode JD in 5-sentence windows → find best-matching CV sentence for each → average

### Final Rating
```
Final Rating = 0.70 × Coverage  +  0.30 × (Meaning Similarity × 100)
```

> ⚠️ **0.70 / 0.30 are unvalidated design choices — not proven values.**

**Speaker notes:**
*"Essential requirements count double because missing them is more damaging.
The 70/30 split is a prototype choice — I chose it as a starting point.
A proper calibration would need labeled data from real recruiters.
I am transparent about this in the app and the report."*

---

# SLIDE 10 — LIVE DEMO

## Live Demo

### Steps I will show:
1. **Load sample** — fictional Data Analyst JD and CV
2. **Extract phrases** — GLiNER runs live on CPU (~20–40 s)
3. **Edit requirements** — move a phrase between Essential / Desirable
4. **Score** — see the rating, evidence table, and suggestions
5. **Download report** — open the .txt file

### Key things to point out:
- 🔶 "Possible match — review" flags with supporting CV passage
- ⚠️ Warning banner when text is edited after extraction
- Sidebar threshold slider
- Privacy notice (no data stored)

**Speaker notes:**
*"Switch to browser with app running. Load sample data. Click Extract — while
it runs explain the GLiNER chunking. Point out the phrase lists. Edit one requirement.
Score it. Point to the evidence column — show a specific matching sentence.
Download the report and open it."*

---

# SLIDE 11 — STREAMLIT & DEPLOYMENT

## Streamlit and Cloud Deployment

### Why Streamlit?
- Python only — no HTML, CSS, or JavaScript needed
- Reruns the whole script on every interaction
- `st.session_state` persists data between reruns

### Key features used

| Feature | Purpose |
|---------|---------|
| `@st.cache_resource` | Load models once (avoids reloading 140 MB per click) |
| `st.session_state` | Store phrases and scores between interactions |
| `st.file_uploader` | Accept PDF/DOCX/TXT |
| `st.download_button` | Provide the report file |

### Deployment (Streamlit Community Cloud — free)
```
1. Push code to GitHub
2. Go to share.streamlit.io → New App
3. Select repo, branch (main), file (app.py)
4. Click Deploy
5. Streamlit installs requirements.txt automatically
6. Models download on first boot (~5 min, ~140 MB)
7. App is live at a public URL
```

No API keys required. No secrets file needed.

**Speaker notes:**
*"The cache decorator is critical. Without it the 140 MB model would reload
every time the user clicks anything — making the app unusable.
With it, the model loads once and stays in memory for the whole session."*

---

# SLIDE 12 — FILE STRUCTURE

## How the Code Is Organised

```
Generic-CV-Matcher/
├── app.py          ← UI layout and section rendering
├── models.py       ← Load + cache both AI models
├── extraction.py   ← GLiNER phrase extraction with chunking
├── scoring.py      ← Direct search + semantic scoring + formula
├── file_reader.py  ← PDF / DOCX / TXT reading (with error handling)
├── report.py       ← Plain-text report builder
├── samples.py      ← Sample JD and CV for demo
└── requirements.txt← All Python dependencies
```

### Why split into modules?
- **Readable** — each file has one job
- **Testable** — test scoring without running the UI
- **Maintainable** — change the formula in `scoring.py` without touching `app.py`

**Speaker notes:**
*"This is modular software design. If a teacher asks 'how does scoring work',
I point to one file. If they ask 'how are files read', I point to another.
Each module is independent and documented."*

---

# SLIDE 13 — LIMITATIONS

## Honest Limitations

| Limitation | Why It Matters |
|-----------|---------------|
| Text alignment only | High score ≠ actually qualified |
| Cannot verify qualifications | Fake claims score the same as real ones |
| Semantic confusion | "Java" ≈ "JavaScript" to the model |
| No OCR | Scanned PDFs (image-only) return empty |
| Unvalidated weights | 0.70/0.30 are prototype choices |
| English-only | Models work best on English text |
| Memory limits | Free Streamlit tier ≈ 1 GB RAM |
| No real-world evaluation | Not tested against recruiter judgements |

### What it IS useful for
✅ Finding language gaps between your CV and a JD
✅ Identifying requirements you forgot to mention
✅ A starting point for CV editing — never a final verdict

**Speaker notes:**
*"Being honest about limitations is as important as showing what works.
The biggest limitation: text alignment is not the same as competence.
A perfectly copied JD would score 100. The app cannot detect dishonesty.
The semantic confusion issue is real — always check the evidence column."*

---

# SLIDE 14 — WHAT I LEARNED

## Challenges Faced and Solved

| Challenge | Solution Applied |
|-----------|----------------|
| GLiNER token limit on long documents | Split into overlapping 400-word windows |
| Models reloading on every click | `@st.cache_resource` decorator |
| numpy 1.x had no Python 3.14 wheel | Upgraded requirement to numpy 2.x |
| Stale scores after text edits | Track last-extracted text; warn user to re-extract |
| Scanned PDFs returning empty text | Detect empty output; show clear explanation |
| Pip installing to wrong environment | Used `python -m pip` explicitly with venv |

### Technical Skills Gained
- Using pretrained NLP models without training them myself
- How cosine similarity works mathematically
- Streamlit session state and caching patterns
- Multi-module Python project structure
- Dependency management for scientific Python on Windows

**Speaker notes:**
*"The numpy issue was a real-world dependency management problem — Python 3.14 is
so new that numpy 1.x had not yet published binary wheels for it.
Upgrading to numpy 2.x solved it instantly.
These are exactly the kinds of problems professional developers encounter."*

---

# SLIDE 15 — VIVA QUESTIONS & ANSWERS

## 10 Teacher Questions — Prepared Answers

---

**Q1: Why GLiNER instead of a keyword list or spaCy?**
> A keyword list misses synonyms and needs manual updates. spaCy's built-in NER
> finds people and places — not skills. GLiNER is zero-shot: I give it label names
> and it finds those concepts in any text, for any job sector, with no training data.

---

**Q2: What is cosine similarity?**
> Each sentence is encoded as a vector (384 numbers). Cosine similarity measures
> the angle between two vectors. Vectors pointing the same direction (similar
> meaning) score near 1.0; vectors pointing away from each other score near 0.0.

---

**Q3: How did you choose 0.70 and 0.30?**
> They are a starting point — coverage feels more informative than overall
> similarity, so I weighted it higher. A proper calibration needs a labeled dataset
> of JD-CV pairs rated by real recruiters. I state clearly they are unvalidated.

---

**Q4: Can this tell if someone is actually qualified?**
> No. It measures text alignment only. A CV that copies JD language word-for-word
> scores high. The app cannot verify competence, experience, or honesty.
> This is stated prominently in the app and report.

---

**Q5: What happens with a scanned PDF?**
> Scanned PDFs are images — pdfplumber finds no text characters. The app detects
> the empty result and tells the user that OCR (like Tesseract) would be needed
> first. OCR is outside the scope of this project.

---

**Q6: How does Streamlit session state work?**
> Streamlit reruns the entire script on every interaction. Session state is a
> dictionary that persists between reruns. I store phrases, requirements, and
> scores there. I also track which text was extracted — if it changes, I warn
> the user to re-extract before scoring.

---

**Q7: Why @st.cache_resource?**
> Each model is 50–90 MB and takes 10–30 s to load. Without caching, every button
> click would reload it. The decorator runs the load function once and returns the
> same model object on every subsequent call.

---

**Q8: Difference between direct mention and possible match?**
> Direct mention: exact phrase found in CV text using word-boundary regex. Possible
> match: phrase not found directly, but Sentence-Transformer found a CV sentence
> with cosine similarity above the threshold. Possible matches need human review
> because semantic similarity can confuse similar-sounding but different skills.

---

**Q9: If someone copies the JD into their CV, do they score 100?**
> Yes. The app cannot detect dishonesty — it only measures text alignment.
> This is a fundamental limitation of any text-comparison approach and is why
> the app includes strong disclaimers and is described as a prototype tool.

---

**Q10: How would you properly evaluate this app?**
> Collect 50–100 JD-CV pairs rated by human recruiters. Split into development
> and held-out sets. Measure Spearman rank-correlation between app scores and
> human ratings. Measure precision/recall on per-requirement match labels.
> Currently the weights are unvalidated — I state this clearly.

---

# SLIDE 16 — CONCLUSION

## Summary

### What was built
A working web app that compares a CV to a job description using two pretrained
AI models, graded scoring with evidence, and an honest set of disclaimers.

### This project vs a simple keyword checker

| Simple Keyword Checker | CV–Job Match |
|------------------------|-------------|
| Fixed skill list | Open-label GLiNER extraction |
| Yes / No match | Graded credit (1.0 / 0.5 / 0.0) |
| No evidence | Supporting CV passage for every match |
| No explanation | Requirement-by-requirement table |
| No output file | Downloadable plain-text report |

### Honest assessment
> Useful for finding language gaps and reflecting on CV content.
> Not a hiring decision system. The value is in showing *where* the
> gap is and *what* the model matched — not in the number itself.

**Speaker notes:**
*"This project taught me to connect real AI models to a usable interface,
to be honest about what a system can and cannot do, and to handle real
engineering problems like dependency conflicts and session management.
Thank you."*

---

# APPENDIX — QUICK CONCEPT CARDS

*Use during viva if asked to explain a concept on the spot.*

| Concept | Simple Explanation |
|---------|-------------------|
| **NER** | Automatically finding and labelling things (people, skills, places) in text |
| **Vector / Embedding** | A list of numbers representing the meaning of a sentence |
| **Cosine similarity** | Score 0–1: how similar two meaning-vectors are; 1 = same direction = same meaning |
| **Pretrained model** | A neural network already trained by researchers; you download and use it |
| **Zero-shot learning** | Using a model on a new task by describing it in plain language — no retraining |
| **@st.cache_resource** | Stores an expensive object (like a loaded model) in memory; reuses it every call |
| **Session state** | A dictionary that survives Streamlit reruns so data is not lost between clicks |
| **Weighted average** | Average where some items count more (essential weight=2, desirable weight=1) |
| **Threshold** | A minimum similarity score; below it = no match, above it = possible match |
| **OCR** | Optical Character Recognition — converting an image of text into actual text |

---

*End of Presentation Guide — CV–Job Match*
