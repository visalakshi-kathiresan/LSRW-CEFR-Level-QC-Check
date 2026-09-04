# Item QC Pipeline

Runs deterministic rule checks + LLM-based QC (CEFR level classification, rubric
scoring, fairness review, optional auto-rewrite) over a batch of language-learning
items, and produces a CSV/JSON report plus a reviewer-facing HTML report.

The judge model is Gemma (`gemma-4-26b-a4b-it`) called via `deepeval`'s
`GeminiModel` wrapper, over the Google Gemini API. There is no local/Ollama path.

## Setup

```bash
pip install deepeval google-genai
export GOOGLE_API_KEY=your_key_here   # or pass api_key= directly in the script
```

## How to run

```bash
python project.py --input items.csv --output results.csv
```

- `--input` — path to a `.json` or `.csv` file of items (required)
- `--output` — path to write results, `.json` or `.csv` (required)

A CSV input needs an `options` column with choices separated by `|`. Field names
are flexible — the script auto-maps common aliases (e.g. `question_text`,
`prompt`, `question` all map to `text`; `cefr_level`, `target_level` map to
`level`) onto the canonical fields: `text`, `type`, `level`, `answer`, `skill`,
`options`.

By default this also writes an HTML report next to `--output`
(`results.html` in the example above) and a resumable checkpoint file
(`results.csv.checkpoint.jsonl`).

## CLI flags

| Flag | Default | What it does |
|---|---|---|
| `--input` | *(required)* | Path to input `.json` or `.csv` |
| `--output` | *(required)* | Path to output `.json` or `.csv` |
| `--level-tolerance` | `0` | How many CEFR tiers off is still "within range" (0 = exact match, 1 = allows e.g. A2↔B1) |
| `--skip-llm` | off | Only run rule-based checks — no Gemini API calls at all |
| `--skill-filter` | `reading,writing` | Comma-separated skills to run LLM QC on. Other skills still get rule checks but are marked `SKIPPED` for LLM steps. Use `all` to disable filtering |
| `--reform` | off | For items that come back `REVIEW`/`REJECT`, ask Gemma to rewrite them and re-check (up to `--reform-attempts` times) |
| `--reform-attempts` | `2` | Max reform + re-check cycles per flagged item |
| `--timing` | off | Records per-item wall-clock time (`elapsed_sec` column + summary at the end) |
| `--html-output` | `<output>.html` | Path for the HTML report; pass `none` to skip it |
| `--checkpoint` | `<output>.checkpoint.jsonl` | Checkpoint file path (see Resuming, below) |
| `--fresh-start` | off | Ignore any existing checkpoint and reprocess every item from scratch |
| `--save-every` | `5` | Re-writes `--output`/HTML report every N completed items, so a killed run still leaves a readable report. `0` = only write at the very end |

## Pipeline stages, per item

1. **Rule-based flags** (`metric_flags`) — fast, no LLM call. Checks:
   - `missing_text` — empty text field
   - `invalid_level` — level isn't one of A1/A2/B1/B2/C1/C2
   - `missing_options` — MCQ item with no options
   - `no_blank_marker` — fill-in-the-blank item with no `___`/`____`
   - `answer_whitespace` — leading/trailing whitespace in the answer
   - `missing_end_punctuation` — text doesn't end in `.`/`?`/`!`
   - `answer_too_long_for_level` — answer >4 words for an A1/A2 item
2. **Fairness check** (`evaluate_fairness`, LLM) — flags:
   - `requires_specialist_or_cultural_knowledge`
   - `cultural_bias`
   - `gender_bias`
   - `sensitive_content`
   - `fairness_check_failed` — the fairness judge call itself failed/was unparseable (infra issue, not a real fairness verdict)
3. **CEFR classification** (`classify_level`, LLM) — predicts a level and compares it to the item's labeled level, within `--level-tolerance`.
4. **Rubric scoring** (`score_metrics`, LLM/GEval) — scores whatever metrics are configured for the item's skill in `SKILL_METRIC_CONFIG` (see below).
5. **Verdict** (`qc_verdict`) — rolls all of the above into one label + reason.
6. **Optional reform** (`--reform`) — for `REVIEW`/`REJECT` items, asks Gemma to rewrite the item and re-runs steps 1–5 on the rewrite, up to `--reform-attempts` times.

## Gates: flag severity → verdict

Every flag has a severity of `review` or `reject` (`FLAG_SEVERITY`), and any
flag not explicitly listed defaults to `review`:

- **`reject`** — the item is actively wrong or unfit to show a student. This
  is every fairness flag (`requires_specialist_or_cultural_knowledge`,
  `cultural_bias`, `gender_bias`, `sensitive_content`).
- **`review`** — usable but imperfect; needs a human look. This is every
  deterministic rule flag (formatting/structural issues, not wrongness),
  a CEFR level outside tolerance, and `fairness_check_failed` (infra failure,
  not a real fairness judgment).

`qc_verdict` produces one of three labels:

- **`REJECT`** — one or more `reject`-severity issues present. Takes priority over `REVIEW`.
- **`REVIEW`** — no reject-level issues, but one or more `review`-level issues (rule flags, level mismatch, or a rubric metric scoring below its threshold).
- **`SELECT`** — no issues at all.

Rubric metrics also carry their own severity (`review` or `reject`, set per
metric in `SKILL_METRIC_CONFIG`) and threshold; a metric scoring below
threshold routes to whichever bucket that metric is configured for.

> **Note:** `SKILL_METRIC_CONFIG`'s `metrics` lists are currently empty for
> `reading`, `writing`, and `_default` — no rubric metrics are wired up yet.
> Add entries there (each with `key`, `label`, `scale`, `threshold`,
> `compare`, `definition`, and optional `severity`/`definition_no_answer`) to
> turn scoring on for a skill.

## Resuming / checkpointing

Every completed item is appended to the checkpoint file immediately
(flushed + fsynced), keyed by `question_id` if present, otherwise a content
hash. If a run is killed partway through, re-running the **same command**
(same `--output`) reloads the checkpoint and skips every item already done,
instead of re-paying for their LLM calls. Use `--fresh-start` to ignore an
existing checkpoint and start over.

## Output

- `--output`: one row per item with the original fields plus `metric_flags`,
  `predicted_level`, `level_match`, `level_change_reason`, one column per
  configured rubric metric (+ a `_reason` column each), `qc_label`,
  `qc_reason`, and (if `--reform` was used) `original_text`/
  `original_answer`/`original_options`/`reform_change_summary`.
- HTML report: same data, reviewer-facing, written to `--html-output`
  (or `<output>.html`).
