# speakingproject.py — Item QC Tool

Runs automated quality-control checks on reading/writing/speaking language-learning items. Each item gets a rule-based pass plus an LLM ("judge model") pass that scores accuracy, grammar, clarity, construction, passage strength, and CEFR-level fit, then rolls everything up into a final **SELECT / REVIEW / REJECT** decision. It can optionally rewrite ("reform") items that fail QC and re-check them.

## Setup

**Requirements:**
```bash
pip install deepeval google-genai
```

**Environment variable (required):**
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```
The judge model is Gemini (`gemma-4-26b-a4b-it`) accessed via Google AI Studio, wrapped through `deepeval`'s `GeminiModel`. Every LLM call in the script — level classification, metric scoring, fairness evaluation, and reform rewrites — goes through this one model.

## How to run

```bash
python speakingproject.py --input items.json --output results.json
```

Input can be `.json` (a list of item objects) or `.csv` (with an `options` column pipe-`|`-delimited for MCQ choices). Output format is inferred from the extension the same way (`.json` or `.csv`).

### Common examples

Run everything with defaults:
```bash
python speakingproject.py --input items.csv --output results.csv
```

Rule-based checks only, no LLM calls (fast, free):
```bash
python speakingproject.py --input items.json --output results.json --skip-llm
```

Only QC the speaking items, with 6 concurrent workers, and auto-rewrite anything flagged:
```bash
python speakingproject.py --input items.json --output results.json \
  --skill-filter speaking --workers 6 --reform
```

Resume a long run that got interrupted:
```bash
python speakingproject.py --input items.json --output results.json --resume
```

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to input `.json` or `.csv` file |
| `--output` | *(required)* | Path to output `.json` or `.csv` file |
| `--level-tolerance` | `0` | How many CEFR tiers off is still "within range" (0 = exact match required, 1 = allows e.g. A2↔B1) |
| `--skip-llm` | off | Only run rule-based checks — no judge model calls at all |
| `--reform` | off | For items that FLAG, ask the judge model to rewrite them and re-run QC on the rewrite |
| `--reform-attempts` | `2` | Max reform + re-check cycles per flagged item |
| `--skill-filter` | `reading,writing,speaking` | Comma-separated list of skills to run LLM-based QC on (case-insensitive). Other skills still get rule-based checks but are marked `SKIPPED` for the LLM steps. Use `all` to disable filtering |
| `--html-output` | `<output>.html` | Path for a self-contained HTML review report (SELECT/REVIEW/REJECT summary, filters, search). Pass `none` to skip it |
| `--timing` | off | Record how long each item took to QC in a `time_taken_sec` column |
| `--checkpoint-every` | `10` | Save resumable progress (and rewrite the output + HTML report) every N items. `0` disables checkpointing |
| `--checkpoint-file` | `<output>.checkpoint.json` | Path to the checkpoint file |
| `--resume` | off | Resume from the checkpoint file instead of starting over |
| `--workers` | `1` | Number of items to QC concurrently (I/O-bound network calls to Gemini, safe to parallelize — try 4–8, results are still written in original item order) |

## Input fields

The script reads these canonical fields per item, but auto-detects common alias column names too (so e.g. `question_text`, `prompt`, or `item_text` are all recognized as `text`):

| Canonical field | Recognized aliases |
|---|---|
| `text` | `text`, `question_text`, `prompt`, `question`, `item_text` |
| `type` | `type`, `question_type`, `item_type`, `qtype` |
| `level` | `level`, `cefr_level`, `labeled_level`, `target_level` |
| `answer` | `answer`, `expected_answer`, `correct_answer`, `key` |
| `skill` | `skill`, `skill_type` |
| `options` | `options`, `choices` |

`level` must be one of the CEFR levels: `A1, A2, B1, B2, C1, C2`.

## What gets checked

### 1. Rule-based flags (no LLM, always run)
Fast, deterministic checks producing short flag keywords, e.g.:
- `missing_text`, `answer_whitespace`, `text_whitespace`
- `invalid_level` (not a valid CEFR level)
- `no_blank_marker` (fill-in-the-blank item missing `___`)
- `missing_options` (MCQ item with no options)
- `missing_end_punctuation`
- `answer_too_long_for_level` (A1/A2 answer over 4 words)
- `weak_passage` (item text under 20 words)
- `poor_construction` (repeated words, unbalanced quotes/brackets/parens)

### 2. LLM-scored metrics (per skill)
The judge model scores each item against a skill-specific rubric. Metrics and pass thresholds:

| Metric | Scale | Threshold | Used for |
|---|---|---|---|
| `accuracy_pct` | 0–100% | ≥ 90% | reading, writing, speaking |
| `grammar_errors` | count | ≤ 0 | writing, speaking |
| `clarity_score` | 1–5 | ≥ 4 | reading, writing, speaking |
| `completeness_score` | 1–5 | ≥ 4 | reading only |
| `vocabulary_level_fit` | 1–5 | ≥ 4 | reading only |
| `construction_score` | 1–5 | ≥ 4 | reading, writing, speaking |
| `passage_strength_score` | 1–5 | ≥ 4 | reading, writing, speaking |

The script also predicts the item's CEFR level and compares it against the labeled level (within `--level-tolerance` tiers).

### 3. Fairness gate (independent of content QC)
A separate judge-model call checks four dimensions, each true/false:
- `requires_specialist_or_cultural_knowledge`
- `cultural_bias`
- `gender_bias`
- `sensitive_content`

This produces its own `FAIRNESS_PASS` / `FAIRNESS_FLAG` / `FAIRNESS_ERROR` label, independent of the content QC label — an item can pass one and fail the other.

## Final decision

Content QC and fairness are rolled up into one actionable verdict per item:

- **SELECT** — every gate that ran came back PASS (or, if no LLM gate ran, rule checks were clean). Safe to give to a student as-is.
- **REVIEW** — nothing hard-flagged it, but coverage is incomplete (a gate was skipped/errored) or the only issue was the predicted CEFR level falling outside range — a human should glance at it.
- **REJECT** — content QC or fairness actively flagged the item (excluding a level-only mismatch, which is downgraded to REVIEW instead).

## Output

Written to `--output` (JSON or CSV) with columns including: original item fields, `metric_flags`, `predicted_level`, `level_match`, every LLM metric score, `qc_label`/`qc_reason`, `fairness_flags`/`fairness_qc_label`/`fairness_qc_reason`, and `final_decision`/`final_decision_reason`. (`question`, `answer`, `type`, `completeness_score`, and `vocabulary_level_fit` are dropped from the final output — the last two are only meaningful for reading items.)

A self-contained **HTML report** is also written (unless `--html-output none`) with a SELECT/REVIEW/REJECT summary plus filtering and search, so results can be reviewed without opening the raw JSON/CSV.

## Resumability

Every `--checkpoint-every` items, progress is saved to a checkpoint file (`.checkpoint.json` by default), and the output + HTML report are rewritten together so they stay in sync. If a run is interrupted, re-run the same command with `--resume` to pick up where it left off instead of starting over.
