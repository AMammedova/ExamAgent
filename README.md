# ExamAgent

A local exam-preparation system for a combined **Machine Learning + Deep Learning**
university final, built for a 7-day run-up.

It is not a tutor chatbot. It behaves like an examiner: it asks before it explains,
marks strictly, records every mistake, and spends your remaining time on whatever is
most likely to cost you marks.

---

## Table of contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Offline mode vs LLM mode](#offline-mode-vs-llm-mode)
- [Architecture](#architecture)
- [Environment variables](#environment-variables)
- [Uploading materials](#uploading-materials)
- [Running a study session](#running-a-study-session)
- [Running a mock exam](#running-a-mock-exam)
- [How the scoring works](#how-the-scoring-works)
- [How RAG works](#how-rag-works)
- [How the adaptive planner works](#how-the-adaptive-planner-works)
- [Chat commands](#chat-commands)
- [Testing](#testing)
- [Project layout](#project-layout)

---

## What it does

| Capability | Detail |
|---|---|
| **Calculation engine** | 21 generators produce randomised numeric problems *and their exact solutions* — MLP backprop, CNN shapes/params, receptive fields, attention, PCA, K-Means, entropy, metrics, Bayes, and more. Graded part-by-part with real partial credit. |
| **Assertion–Reason engine** | 87 curated items plus LLM generation. Every item stores `assertion_truth`, `reason_truth`, `reason_explains_assertion`; the A–E answer is *derived*, never hand-typed. Includes the examiner's traps (both true but unrelated; false assertion with a true reason). |
| **Strict evaluation** | 0–10 with what you got right, what you missed, what was wrong, what the examiner expects, a model answer, and one sentence on turning your answer into an exam answer. |
| **Weakness detection** | Six dimensions per topic (concept / calculation / reasoning / comparison / application / confidence). Detects the case that matters: *concept 85%, calculation 31% → stop reading, start drilling*. |
| **Adaptive 7-day planner** | Time follows exam damage, not syllabus size. Mastered topics get dropped; weak prerequisites get promoted. |
| **Mock exams** | Timed, no hints, no feedback until submission, blueprint modelled on the exam samples. Full performance report afterwards. |
| **RAG knowledge base** | Your PDFs/slides/notes, chunked with page-accurate citations. University material outranks Udemy material for exam questions. |
| **Spaced repetition** | Measured in *hours*, not days — a 7-day horizon needs it. Critical mistakes come back within a few hours. |

---

## Quick start

```bash
# from the project root
pip install -r requirements.txt
streamlit run app.py
```

Then open <http://localhost:8501>.

Windows shortcuts are included:

```powershell
.\run.ps1          # install deps if needed, then launch
```

```bash
./run.sh           # same, for bash / git-bash
```

**No API key is required to start.** See the next section.

A starter document is included at `sample_materials/exam_pattern_reference.md`. Ingest
it as **EXAM_SAMPLES** on the Materials page to give the app question-style calibration
before your own past papers are uploaded — it records every question format from the
previous midterm plus the worked 2-2-1 MLP backpropagation pattern, the CNN shape and
receptive-field formulas, and the transformer patterns.

---

## Offline mode vs LLM mode

The app is designed to degrade gracefully, because a broken API key the night before
an exam must not stop you studying.

| | Offline (no key) | With an API key |
|---|---|---|
| Calculation problems | ✅ full (deterministic engine) | ✅ same engine |
| Assertion–Reason | ✅ 87-item bank | ✅ bank + generated |
| Seed question bank | ✅ 58 hand-written items | ✅ + generated |
| Marking calculations | ✅ exact | ✅ exact |
| Marking A–R / MCQ | ✅ exact | ✅ exact |
| Marking open answers | ⚠️ concept-coverage heuristic | ✅ full examiner rubric |
| Lessons | ⚠️ structured from your material + topic registry | ✅ generated, grounded, cited |
| New question generation | ⚠️ templates for uncovered types | ✅ unlimited, grounded |
| Planner, scoring, weaknesses, mock exams | ✅ full | ✅ full |

Add a key in **Settings → LLM provider**, or in `.env`. The toggle in the sidebar
forces offline mode at any time (faster, free).

---

## Architecture

Deliberately simple: one Streamlit process, one SQLite file, one local vector store.
No services, no containers, no auth.

```
                      ┌─────────────────────────────┐
                      │        Streamlit UI          │
                      │  Dashboard · Study · Quiz    │
                      │  Mock · Chat · Weaknesses    │
                      │  Map · Progress · Materials  │
                      └──────────────┬───────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
┌───────▼────────┐        ┌──────────▼─────────┐       ┌──────────▼─────────┐
│  Tutor         │        │  Question Gen      │       │  Planner           │
│  lessons, chat │        │  ┌──────────────┐  │       │  7-day plan        │
│  commands      │        │  │ calc engine  │  │       │  next best action  │
└───────┬────────┘        │  │ A–R engine   │  │       │  session builder   │
        │                 │  │ seed bank    │  │       └──────────┬─────────┘
        │                 │  │ LLM + RAG    │  │                  │
        │                 │  └──────────────┘  │                  │
        │                 └──────────┬─────────┘                  │
        │                            │                            │
┌───────▼────────┐        ┌──────────▼─────────┐       ┌──────────▼─────────┐
│  Retriever     │        │  Evaluator         │       │  Progress /        │
│  TF-IDF/Chroma │        │  strict marking    │──────▶│  Weakness tracker  │
│  source rank   │        │  error typing      │       │  readiness score   │
└───────┬────────┘        └────────────────────┘       └──────────┬─────────┘
        │                                                          │
┌───────▼──────────────────────────────────────────────────────────▼─────────┐
│  SQLite (topics, attempts, mistakes, mock exams, sessions, documents)       │
│  + local vector store (chunks.jsonl)                                        │
└────────────────────────────────────────────────────────────────────────────┘
```

Components share one LLM client and one database session factory. Every component
works without the LLM.

---

## Environment variables

Copy `.env.example` to `.env`. Nothing is required to run.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `openai`, or `none` |
| `ANTHROPIC_API_KEY` | — | required for `anthropic` |
| `OPENAI_API_KEY` | — | required for `openai` |
| `MODEL_NAME` | `claude-sonnet-4-5` | model id |
| `MAX_TOKENS` / `TEMPERATURE` | `2000` / `0.3` | generation settings |
| `VECTOR_BACKEND` | `local` | `local` (TF-IDF) or `chroma` |
| `EMBEDDING_BACKEND` | `local` | retrieval embeddings |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1100` / `180` | ingestion chunking |
| `RETRIEVAL_TOP_K` | `6` | chunks per query |
| `EXAM_DATE` | today + `STUDY_DAYS` | `YYYY-MM-DD` |
| `STUDY_DAYS` | `7` | plan length |
| `READINESS_W_*` | see below | readiness weighting |
| `DATA_DIR` | `data` | database, uploads, vector store |
| `LOG_LEVEL` | `INFO` | logging |

Credentials are never hardcoded. The Settings page writes to `.env` and reloads config.

**ChromaDB is optional.** `pip install chromadb` and set `VECTOR_BACKEND=chroma`. If it
is missing or fails to load, the app logs a warning and falls back to the local
TF-IDF store automatically — retrieval keeps working either way.

---

## Uploading materials

**Materials → Upload.** Choose a source category, optionally name the lecture, drop
files in.

Supported: **PDF, TXT, MD, DOCX, PPTX, TEX, CSV**. Images are accepted but only
contribute text if an OCR engine is installed (`pip install pytesseract pillow` plus
the Tesseract binary); otherwise they are recorded with a clear "no extractable text"
status rather than silently ignored.

Categories drive retrieval priority:

| Category | Used for |
|---|---|
| `UNIVERSITY_ML` / `UNIVERSITY_DL` | **Highest trust** — what the exam expects |
| `EXAM_SAMPLES` | Question style and difficulty calibration |
| `STUDENT_NOTES` | Your own summaries |
| `UDEMY_ML` / `UDEMY_DL` | Intuition and practical ML, lower trust |

Duplicate files (by content hash) are skipped, not re-indexed. **Materials → Coverage**
shows which CRITICAL topics still have no supporting material — the highest-value gaps
in your knowledge base.

You can also paste notes directly under **Add notes directly**.

---

## Running a study session

**Study** → pick a mode (or let the planner choose). The flow enforces active recall:

1. **Explanation** — concise, capped, cited when grounded in your material
2. **Intuition**
3. **The mathematics that matters** — the formulas that get examined
4. **Exam points** — phrased the way you should write them
5. **Example**
6. **Explain it back** — *you* write first
7. **Strict evaluation** of your explanation
8. **Harder reasoning question**
9. **Calculation question** where the topic supports one
10. **Profile updated** — dimension scores, next review, mistakes logged

After every answer the next question adapts: ≥8/10 → harder or a new dimension;
5–8 → consolidate at the same level; <5 → one level easier on the same idea.

Session modes: Quick (15m), 30 Minute, 60 Minute, Deep (90m), Rapid Revision (20m),
Weakness Repair (45m), Exam Simulation (75m).

---

## Running a mock exam

**Mock Exam** → set length and time limit → *Generate exam*.

Exam conditions are real: a visible countdown, no hints, no per-question feedback,
mixed ML and DL, difficulty 4–6 only. The blueprint follows the exam samples:

```
assertion_reason 5 · calculation 4 · conceptual 3 · what-happens-if 2
comparison 2 · scenario 1 · architecture interpretation 1
```

On submission you get: total, ML vs DL, per-dimension and per-question-type breakdown,
top 5 weaknesses, top 5 strengths, the most dangerous knowledge gaps (CRITICAL topics
you failed), topics needing immediate revision, and a revision plan. Every answer also
updates your topic profile.

---

## How the scoring works

### Per-answer

- **Calculation** — graded part by part. A recognised wrong path (recall instead of
  precision, missing bias term, dividing by `d_k` instead of `√d_k`) earns method
  credit and a specific diagnosis. Errors are typed: *Arithmetic, Formula, Conceptual,
  Dimension, Reasoning, Terminology, Incomplete*.
- **Assertion–Reason / MCQ** — exact, with a breakdown of each truth flag.
- **Open answers** — LLM rubric when configured; otherwise concept coverage combined
  with a technical-vocabulary signal. Calibrated so full-mark answers average ~8.4/10
  and vague answers score ~0.

### Per-topic

Six dimensions, updated as difficulty-weighted moving averages. A level-6 question
moves the estimate more than a level-2 one. **Untested dimensions are excluded from
the average** rather than counted as zero, so a topic is not punished for questions you
have not seen — they are reported separately as coverage gaps.

### Exam readiness

Not a flat average. Default weights (configurable in Settings or `.env`):

```
30%  critical topic mastery      20%  calculation ability
20%  reasoning ability           15%  exam question performance
10%  coverage                     5%  confidence
```

Two extra rules:

- **Prerequisites propagate.** A topic's effective score is discounted by its weakest
  tested prerequisite — shaky `chain_rule` drags down `backpropagation`.
- **Unknowns count as risk.** Untested CRITICAL topics lower readiness, because you
  cannot be ready for what you have never attempted.

---

## How RAG works

1. **Ingest** — load → clean (de-hyphenate PDF line breaks) → chunk (~1100 chars,
   180 overlap). Chunks never span a page or section boundary, so page citations stay
   truthful.
2. **Tag** — keyword matching against the 98-topic registry gives each chunk topic tags.
3. **Store** — local TF-IDF (or ChromaDB) with full metadata: `source_type`,
   `source_name`, `lecture`, `page`, `section`, `topics`.
4. **Retrieve** — over-fetch, then re-rank by source trust (university ×1.30, exam
   samples ×1.25, notes ×1.05, Udemy ×0.95).
5. **Ground or refuse** — `RetrievalResult.grounded` gates every course-specific claim.
   When retrieval finds nothing, the app **says the source material does not establish
   the answer** instead of inventing one.

A **source filter is a hard constraint**: asking for exam samples only will never
return university content relabelled. **Materials → Search** shows the university and
Udemy treatments side by side and states which one to follow for the exam.

---

## How the adaptive planner works

Each topic gets a **damage score**:

```
damage = gap × priority_weight × (0.35 + 0.65 × exam_relevance) × centrality
```

where `gap` blends the effective score with the **worst measured dimension** (so
concept 85% / calculation 10% is not hidden by a good average), `centrality` grows with
the number of dependent topics, and untested topics carry a deliberate risk premium.

Time is allocated in proportion to damage, capped at 35 min per block. Then:

- Mastered topics are **dropped** from the plan entirely.
- A topic is not scheduled on consecutive days (rotation is enforced).
- The final day switches to live weaknesses plus a mock exam.
- If your biggest gap depends on a broken prerequisite, the planner **redirects you to
  the prerequisite** and tells you why.

The **next best action** on the dashboard prefers, in order: an unresolved
high-severity mistake → a weak prerequisite → the highest-damage due topic.

---

## Chat commands

| Command | Effect |
|---|---|
| `/study [topic]` | Start a session (planner chooses if omitted) |
| `/quiz [topic]` | One practice question |
| `/exam` | One exam-level question |
| `/mock` | Open the mock exam builder |
| `/calculate <topic>` | A calculation problem |
| `/assertion <topic>` | An assertion–reason question |
| `/compare <a> vs <b>` | A comparison question |
| `/review` | Unresolved mistakes |
| `/weakness` | Weakest topics with actions |
| `/progress` | Readiness breakdown |
| `/plan` | The study plan |
| `/explain <topic>` | A grounded explanation |
| `/rapid_review` | Rapid active recall |
| `/help` | List commands |

Plain English works too: *"teach me PCA"*, *"quiz me on CNN"*, *"give me a
backpropagation calculation"*, *"what should I study now"*, *"show my weakest topics"*,
*"give me a 30 minute session"*, *"explain attention like I am a beginner"*.

---

## Testing

```bash
python -m pytest tests/ -q
```

200 tests covering ingestion, chunking, retrieval and source priority, question
generation, assertion–reason validity, answer evaluation and calibration, score
updates, spaced repetition, weakness detection, study planning, mock exam scoring,
persistence, the full student journey, and a UI smoke test that renders every page
through Streamlit's `AppTest` harness (including generating a question, submitting an
answer, and checking it was marked and stored).

Tests run against a throwaway data directory and never touch your real progress.

---

## Project layout

```
app.py                      Streamlit entry point, navigation, first-run flow
examagent/
  config.py                 env-driven settings, logging
  models/
    schemas.py              Pydantic domain models
    db.py                   SQLAlchemy models + session handling
  data/
    topics.py               98-topic knowledge graph, priorities, day themes
    seed_questions.py       58 hand-written exam-style questions
  services/
    calc_engine.py          21 problem generators + exact grading
    assertion_engine.py     87-item A–R bank + generation + grading
    question_gen.py         unified generator with graceful degradation
    evaluator.py            strict marking (deterministic / heuristic / LLM)
    progress.py             score updates, spaced repetition, readiness
    weakness.py             weakness analysis, error log, knowledge map
    planner.py              adaptive 7-day plan, next action, session builder
    mock_exam.py            exam construction, scoring, reporting
    tutor.py                lessons, chat, command routing
    ingest.py               document loading and chunking
    vectorstore.py          local TF-IDF store + optional ChromaDB
    rag.py                  retrieval with source priority and citations
    materials.py            ingestion pipeline and library status
    llm.py                  provider-agnostic client
  ui/                       one module per page + shared widgets
tests/                      200 tests
sample_materials/           starter exam-pattern reference to ingest
data/                       SQLite, uploads, vector store (created on first run)
```

---

## Notes and limitations

- Open-answer marking without an API key is a **heuristic**. It is deliberately harsh
  on vague answers and calibrated against the seed bank, but it cannot verify semantics
  the way the LLM rubric can. Calculation and assertion–reason marking are exact either
  way.
- Scanned PDFs and images need an OCR engine to contribute text; the app tells you
  when a file produced nothing rather than pretending it worked.
- Initial topic priorities are the brief's; they update as exam samples and materials
  are ingested and as your performance data accumulates.
