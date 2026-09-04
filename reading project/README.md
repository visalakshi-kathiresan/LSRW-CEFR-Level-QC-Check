# Reading Item QC Tool

Runs automated QC on reading-comprehension items: rule-based checks, an LLM
judge (via deepeval + Gemini) for accuracy/grammar/clarity/completeness,
CEFR level classification, construction checks (distractors, answerability),
and fairness/bias checks. Outputs a CSV/JSON with every check plus a
searchable HTML review dashboard.

## Requirements

- Python 3 with `deepeval` installed
- Credentials configured for deepeval's `GeminiModel` judge (the model used
  is `gemma-4-26b-a4b-it`) — set up per deepeval's docs before running
- Input file as `.json` (nested documents with a `questions` list, or a
  flat list of items) or `.csv`

## Running it

```bash
python readingproject.py --input items.json --output results.csv
```

Common examples:

```bash
# Only run rule-based checks, skip the LLM judge entirely (fast, free)
python readingproject.py --input items.json --output results.csv --skip-llm

# Run LLM QC on every skill, not just "reading"
python readingproject.py --input items.json --output results.json --skill-filter all

# Resume a run that was interrupted, reusing already-scored items
python readingproject.py --input items.json --output results.csv --resume

# Compact output (just the decision columns) + custom HTML report path
python readingproject.py --input items.json --output results.csv --compact --html-output review.html
```

### CLI flags

| Flag | Default | What it does |
|---|---|---|
| `--input` | *(required)* | Path to input `.json` or `.csv` |
| `--output` | *(required)* | Path to output `.json` or `.csv` |
| `--level-tolerance` | `0` | How many CEFR steps off the labeled level the predicted level can be before it's flagged as a mismatch |
| `--skip-llm` | off | Skip all LLM judge calls (level, cognitive skill, metrics, construction, fairness) — only rule-based flags run |
| `--skill-filter` | `reading` | Comma-separated list of `skill` values to run LLM QC on (e.g. `reading,listening`), or `all` |
| `--compact` | off | Write only `question_id, sub_question_id, skill, question_type, final_decision, final_decision_reason` |
| `--html-output` | *(auto)* | Path for the interactive HTML report. Defaults to the output path with `.html`. Set to `none` to disable |
| `--workers` | `8` | Thread-pool size for concurrent item processing |
| `--checkpoint` | *(auto)* | Path to the checkpoint `.jsonl` file. Defaults to `<output>.checkpoint.jsonl` |
| `--resume` | off | Reuse cached results from the checkpoint file for items already processed |
| `--force-recalculate` | off | Ignore the checkpoint cache and reprocess everything |
| `--live-interval` | `10.0` | Seconds between live CSV/HTML writes while running. `0` disables live saving |

## Metrics (LLM-scored, per item)

| Metric | Scale | Passes when | What it checks |
|---|---|---|---|
| `accuracy_pct` | 0–100 | ≥ 90 | The question and its expected answer are factually/grammatically correct together |
| `grammar_errors` | count | = 0 | Grammar/spelling errors in the question text and answer |
| `clarity_score` | 1–5 | ≥ 4 | How unambiguous the question is (5 = completely clear) |
| `completeness_score` | 1–5 | ≥ 4 | Whether all context needed to answer is present, with nothing missing |

## Flags

### Rule-based flags (`metric_flags`, no LLM needed)

| Flag | Meaning |
|---|---|
| `answer_whitespace` / `text_whitespace` | Leading/trailing whitespace on the field |
| `missing_text` / `missing_answer` | Field is empty |
| `invalid_level` | Labeled CEFR level isn't one of A1–C2 |
| `no_blank_marker` | Fill-in-the-blank item has no `___`/`____` |
| `missing_options` | MCQ or True/False item has no options |
| `invalid_true_false_options` | True/False options aren't one of the accepted sets (`True/False[/Not Given]`, `Yes/No[/Not Given]`) |
| `invalid_true_false_answer` | Answer isn't one of the accepted True/False values |
| `missing_end_punctuation` | Text doesn't end in `. ! ?` (ignored when computing the final decision) |
| `double_spacing` | Two or more consecutive spaces/tabs in the text |
| `answer_too_long_for_level` | A1/A2 item has an answer over 4 words |
| `fully_redundant_with:<id>` / `partially_redundant_with:<id>` | Near-duplicate of another item in the same document, by question/answer word overlap |
| `trivial_lookup` | Answer is just a short copy of words already in the question, with no reasoning language |

### Construction flags (`construction_flags`, LLM-judged)

| Flag | Meaning |
|---|---|
| `weak_distractors` | MCQ distractors aren't plausible enough for the item's level |
| `multiple_defensible_answers` | More than one option could reasonably be correct |
| `sensitive_content_review_needed` | Flagged as sensitive by the construction judge |
| `construction_check_error` | The judge call failed/couldn't be parsed |

### Fairness flags (`fairness_flags`, LLM-judged)

| Flag | Meaning |
|---|---|
| `requires_specialist_knowledge` | Needs specialist or cultural knowledge beyond the passage |
| `cultural_bias` | Judge detected cultural bias |
| `gender_bias` | Judge detected gender bias |
| `sensitive_content_review_needed` | Flagged as sensitive by the fairness judge |
| `fairness_check_error` | The judge call failed/couldn't be parsed |

## QC labels & final decision

Each gate (content, construction, fairness) produces its own label
(`PASS`/`FLAG`/`ERROR` variants). `final_decision` combines all three plus
the rule-based flags and the CEFR level match:

- **`REJECT`** — any gate came back flagged
- **`USE`** — all gates passed cleanly (and predicted level matched the label, if any)
- **`REVIEW`** — mixed/incomplete coverage, or predicted level didn't match the labeled level, or only rule flags tripped

## Output

- `--output`: CSV or JSON with one row per item — all flags, scores, QC labels, and `final_decision`
- HTML report (unless disabled): a searchable/filterable dashboard of the same data, grouped by decision, skill, and level
- Checkpoint `.jsonl`: incremental per-item results, used by `--resume`
