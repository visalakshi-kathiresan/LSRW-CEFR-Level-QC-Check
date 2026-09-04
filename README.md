# LSRW QC Check
 
Automated quality-control pipelines for language-learning assessment items — **L**istening, **S**peaking, **R**eading, and **W**riting. Each skill has its own standalone tool, but all four share the same design: run deterministic rule-based checks plus an LLM "judge" pass (CEFR level fit, accuracy, grammar, clarity, construction, and fairness/bias), then roll everything up into a final `select` / `review` / `reject` decision with a full audit trail.
 
Every tool outputs a CSV/JSON report and a searchable, filterable HTML review dashboard, so QC reviewers don't need to touch the raw data files.
 
## Project structure
 
```
LSRW QC Check/
├── listening project/
│   ├── listeningproject.py
│   ├── listening.json          # sample input
│   └── resultsofListening.*    # sample output (csv/html/checkpoint)
├── reading project/
│   ├── readingproject.py
│   ├── reading.json
│   └── resultsofReading.*
├── speaking project/
│   ├── speakingproject.py
│   ├── speaking.json
│   └── resultofSpeaking.*
└── writing project/
    ├── writingproject.py
    ├── writing.json
    └── resultsofWriting.*
```
 
Each subfolder is self-contained and has its own `README.md` with tool-specific details.
 
## How it works
 
1. **Input** — a batch of items as `.json` (a list of item objects, or nested documents with a `questions` list) or `.csv` (MCQ options pipe-`|`-delimited).
2. **Rule-based checks** — fast, deterministic validation (formatting, structure, answerability, etc.), no API calls.
3. **LLM judge pass** — items are scored by a Gemini model (`gemma-4-26b-a4b-it`) via [`deepeval`](https://github.com/confident-ai/deepeval)'s `GeminiModel`, covering accuracy, grammar, clarity, completeness, CEFR-level fit, and fairness/bias.
4. **Decision rollup** — every item gets a `final_decision` of `select`, `review`, or `reject`, with the reasoning behind it.
5. **Optional reform** — items flagged for review can be automatically rewritten and re-checked (`--reform`).
6. **Output** — a `.csv`/`.json` results file, a reviewer-facing HTML dashboard, and a resumable checkpoint file.
## Requirements
 
- Python 3.12+
- [`deepeval`](https://pypi.org/project/deepeval/) and `google-genai`
- A Google AI Studio API key for the Gemini judge model
```bash
pip install deepeval google-genai
export GOOGLE_API_KEY="your-gemini-api-key"
```
 
## Quick start
 
Run the QC tool for the skill you need, from inside that skill's folder:
 
```bash
# Reading
cd "reading project"
python readingproject.py --input reading.json --output results.csv
 
# Writing
cd "writing project"
python writingproject.py --input writing.json --output results.csv
 
# Speaking
cd "speaking project"
python speakingproject.py --input speaking.json --output results.json
 
# Listening
cd "listening project"
python listeningproject.py --input listening.json --output results.csv
```
 
Every script supports:
 
| Flag | What it does |
|---|---|
| `--input` | Path to input `.json` or `.csv` (required) |
| `--output` | Path to output `.json` or `.csv` (required) |
| `--skip-llm` | Rule-based checks only — fast, free, no API calls |
| `--reform` | Auto-rewrite items flagged for review and re-check them |
| `--reform-attempts` | Max rewrite attempts per item (default: `2`) |
| `--level-tolerance` | Allowed drift between target and detected CEFR level |
| `--skill-filter` | Restrict LLM QC to specific skill(s) |
| `--workers` | Concurrent worker threads for LLM calls |
| `--resume` | Resume an interrupted run using the checkpoint file |
| `--timing` | Print per-stage timing info |
| `--html-output` / `--html-report` | Custom path for the HTML dashboard |
 
See each subfolder's `README.md` for the full, tool-specific flag list and examples — flag names vary slightly between tools (e.g. `--checkpoint` vs `--checkpoint-file`, `--no-html-report`, `--compact`).
 
## Outputs
 
- **Results file** (`.csv`/`.json`) — one row/object per item, every check score, and the `final_decision`.
- **HTML dashboard** — filterable/searchable report for manual review.
- **Checkpoint file** (`.jsonl`) — lets a run be resumed without re-scoring already-processed items.
## Notes
 
- `.venv/` and `.deepeval/` folders (local virtual environments and deepeval telemetry) are included in some subfolders from local development — you'll likely want to exclude these via `.gitignore` before pushing to GitHub (see below).
- All four tools call the same underlying judge model; no local/Ollama path is supported.
## Suggested `.gitignore`
 
```
.venv/
__pycache__/
.deepeval/
*.pyc
.env
```
 
