# Item QC Pipeline

QC-flags language-learning items (reading/listening comprehension questions)
on **Accuracy, Grammar, Clarity, Completeness, and CEFR Level**, plus
independent **Construction** and **Fairness/Bias** checks, using a Gemini
judge model (`gemma-4-26b-a4b-it`) via `deepeval`.

Every item ends up with a `final_decision` of `select`, `review`, or
`reject`, plus a full trail of *why*.

---

## 1. Setup

```bash
pip install deepeval google-genai
export GOOGLE_API_KEY="your-key-here"     # Google AI Studio key
```

Optional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — | Required. Auth for the Gemini judge model. |
| `DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE` | `180` | Per-call timeout for the judge model. The script sets this itself unless you've already set it. |
| `JUDGE_MAX_CALLS_PER_MINUTE` | `0` (off) | Optional proactive throttle — caps how many judge-model calls start per minute across all worker threads, to stay under a known rate limit. |

---

## 2. Running it

```bash
python project.py --input items.json --output results.csv
```

Input can be `.json` or `.csv`. Output extension (`.json`/`.csv`) controls
the output format. A filterable/searchable HTML review dashboard is written
alongside it automatically (see [Outputs](#5-outputs)).

### Common examples

```bash
# Basic run, only listening items get LLM QC (the default filter)
python project.py --input items.json --output results.csv

# Run LLM QC on every skill, not just "listening"
python project.py --input items.json --output results.csv --skill-filter all

# Rule-based checks only, no judge-model calls (fast, free, no API key needed)
python project.py --input items.json --output results.csv --skip-llm

# Ask the judge model to rewrite anything that FLAGs, then re-check it
python project.py --input items.json --output results.csv --reform

# Resume-safe long run with more parallelism
python project.py --input items.json --output results.csv --workers 15 --checkpoint run1.jsonl
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to input `.json` or `.csv` file. |
| `--output` | *(required)* | Path to output `.json` or `.csv` file. |
| `--level-tolerance` | `0` | How many CEFR tiers off is still "within range" (0 = exact match required). |
| `--skip-llm` | off | Only run rule-based checks — no judge-model calls at all. |
| `--reform` | off | For items that `FLAG` on content QC, ask the judge model to rewrite them and re-check the rewrite. |
| `--reform-attempts` | `2` | Max reform + re-check cycles per flagged item. |
| `--skill-filter` | `listening` | Comma-separated list of skills to run LLM QC on (case-insensitive). Other skills still get rule-based checks but are marked `SKIPPED` for LLM steps. Use `all` to disable filtering. |
| `--html-report` | `<output>.html` | Path for the HTML dashboard. |
| `--no-html-report` | off | Skip writing the HTML dashboard. |
| `--workers` | `10` | Concurrent items processed via a thread pool. Set to `1` for strictly sequential processing. |
| `--checkpoint` | `<output>.checkpoint.jsonl` | JSONL file every finished item is appended to, so an interrupted run can resume by re-running the same command. |
| `--no-checkpoint` | off | Disable checkpointing (no resume, no per-item durability). |
| `--save-every` | `10` | Re-write `--output`/HTML from whatever's done so far every N finished items. `0` = only save at the end. |
| `--timing` | off | Track/report per-item elapsed time and ETA. |

---

## 3. Input format

Each item needs (field-name aliases in parentheses are auto-mapped to the
canonical name):

| Canonical field | Aliases accepted | Meaning |
|---|---|---|
| `text` | `question_text`, `prompt`, `question`, `item_text` | The question (and/or passage+question) text. |
| `type` | `question_type`, `item_type`, `qtype` | `mcq` / `multiple_choice`, `fill_up`, `short_answer`, etc. |
| `level` | `cefr_level`, `labeled_level`, `target_level` | Labeled CEFR level (`A1`–`C2`). |
| `answer` | `expected_answer`, `correct_answer`, `key` | Expected answer. |
| `skill` | `skill_type`, `category` | e.g. `listening`, `reading`. Drives which metric config and audio-check apply. |
| `options` | `choices` | List of MCQ options (CSV: pipe-`\|`-separated). |

Other useful fields: `question_id` / `sub_question_id` (identity — items
sharing a `question_id` are treated as questions on the same passage, for
the redundancy/triviality check), `content` / `transcript` / `audio`
(background material), `cognitive_skill` (labeled cognitive skill, checked
against the predicted one).

A JSON item can also be a **nested passage document** — a dict with a
`questions` list — which gets expanded into one row per sub-question
automatically.

**Per-item override:** set `"skip_audio_check": true` (or
`"no_audio_needed": true`) on an item to suppress the "missing audio/
transcript" flag for that item specifically.

---

## 4. How QC works

Three **independent gates** run per item, then roll up into one
`final_decision`.

### 4a. Rule-based checks (`metric_flags`) — always run, no LLM call

Fast, deterministic checks on every item regardless of `--skip-llm`:

| Flag | Meaning |
|---|---|
| `answer_whitespace` / `text_whitespace` | Leading/trailing whitespace in `answer` / `text`. |
| `missing_text` / `missing_answer` | Field is empty. |
| `invalid_level` | Labeled `level` missing or not one of A1–C2. |
| `no_blank_marker` | Type is `fill_up` but text has no `___` blank. |
| `missing_options` | Type is MCQ but `options` is empty. |
| `missing_audio_or_transcript` | Listening item missing both `audio` and `transcript` (unless overridden — see above). |
| `missing_end_punctuation` | Text doesn't end in `.`/`?`/`!`. **Minor** — reported but never drives a FLAG on its own (see `_IGNORED_MINOR_FLAGS`). |
| `double_spacing` | Text contains a double (or larger) space. |

**Cross-question checks** (compares every pair of questions sharing the
same `question_id`, i.e. same passage):

| Flag | Meaning |
|---|---|
| `fully_redundant_with:<id>` | Same fact tested twice (subject/object swap, or near-identical wording + same answer). One should probably be cut. |
| `partially_redundant_with:<id>` | Meaningful overlap in wording/answer, worth a human glance. |
| `trivial_lookup` | Answer is a bare word/number copied straight from the question — answerable by string-matching, not comprehension. |

### 4b. Content QC gate (`content_qc_label` / `content_qc_reason`)

One combined judge-model call (`classify_and_score`) per item produces:

- **`predicted_level`** — CEFR level of the question itself (A1–C2), compared to the labeled `level` within `--level-tolerance` tiers → `level_match`.
- **`predicted_cognitive_skill`** — one of `recall`, `comprehension`, `vocabulary`, `inference`, compared to labeled `cognitive_skill` (reported independently — a mismatch alone never causes a FLAG).
- **QC metrics**, scored against per-skill targets (`SKILL_METRIC_CONFIG`):

  | Metric | Scale | Target | Applies to |
  |---|---|---|---|
  | `accuracy_pct` | 0–100 | ≥ 90 | all skills |
  | `grammar_errors` | count | ≤ 0 | all skills |
  | `clarity_score` | 1–5 | ≥ 4 | all skills |
  | `completeness_score` | 1–5 | ≥ 4 | all skills |

  `listening`-skill items additionally require `audio`/`transcript` to be
  present (checked in 4a).

`content_qc_label` is `PASS` or `FLAG`. If the *only* reason for flagging is
a CEFR level mismatch, that's tracked separately
(`content_flagged_on_level_only`) and treated more leniently in the final
roll-up (see 4d).

### 4c. Construction & Fairness gates — independent of content QC

**Construction** (`evaluate_construction` → `construction_flags` →
`construction_qc_gate`) judges *how the question is built*, not whether its
content is accurate:

| Flag | Meaning |
|---|---|
| `weak_distractors` | MCQ distractors aren't plausible enough for the item's CEFR level. |
| `multiple_defensible_answers` | More than one MCQ option could be argued correct. |
| `sensitive_content_review_needed` | Question/background touches distressing/inappropriate content. |
| `construction_check_error` | The judge-model call itself failed — not a real verdict; surfaced as `CONSTRUCTION_ERROR`, not `CONSTRUCTION_FLAG`. |

Distractor-plausibility and background-relevance thresholds scale by CEFR
level (looser at A1/A2, stricter at C1/C2) — see
`CONSTRUCTION_THRESHOLDS_BY_LEVEL`.

**Fairness** (`classify_fairness` → `fairness_flags` → `fairness_qc_gate`)
judges four independent bias/appropriateness dimensions:

| Flag | Meaning |
|---|---|
| `requires_specialist_knowledge` | Answering needs knowledge not given in the background material. |
| `cultural_bias` | Text asserts one culture's norms as default, or relies on a stereotype. |
| `gender_bias` | Text relies on a gender stereotype. |
| `sensitive_content_review_needed` | Genuinely distressing/political/graphic content for a general audience. |
| `fairness_check_error` | The judge-model call itself failed → `FAIRNESS_ERROR`, not `FAIRNESS_FLAG`. |

Both gates are calibrated to under-flag routine technical/professional
content — a flag requires specific, citable evidence, not just "the topic is
technical" or "a person has a gender."

### 4d. Final roll-up (`final_decision` / `final_decision_result`)

`compute_final_decision()` combines all three gates:

| `final_decision` | When |
|---|---|
| **`select`** | Every gate that ran came back `PASS` (or, if no LLM gates ran — `--skip-llm` or the skill was filtered out — the rule-based checks came back clean). |
| **`review`** | No gate flagged the item outright, but coverage is incomplete (a gate `SKIPPED`/`ERROR`ed, or rule flags fired with no LLM confirmation) — needs a human look. Also covers the case where content QC's *only* problem is a CEFR level mismatch with construction/fairness both clean (downgraded from reject, since a level disagreement alone isn't necessarily a bad item). |
| **`reject`** | At least one gate actively `FLAG`ged the item (and it isn't the level-only-mismatch case above). |

`final_decision_result` explains the decision in priority order: CEFR
level-change reason and/or a plain-English rule-flag breakdown first;
otherwise whichever gate's own reason drove the flag; otherwise `"pass"`.

---

## 5. Outputs

1. **`--output`** (`.csv` or `.json`) — one row per item with the original
   fields plus every derived QC column (see `PREFERRED_COLUMN_ORDER` in the
   script for the exact order: identity → authored content → rule flags →
   level/cognitive-skill checks → content QC scores → construction →
   fairness → reform trail → final decision).
2. **HTML dashboard** (`<output>.html` by default) — filterable/searchable
   select/review/reject queue with full QC detail per item.
3. **Checkpoint** (`<output>.checkpoint.jsonl` by default) — one JSON line
   per completed item, written as soon as it finishes. Re-running the same
   command reuses anything already in this file instead of re-spending
   judge-model calls, so a crashed/interrupted run resumes cleanly.
   Disable with `--no-checkpoint`.

`--save-every N` (default `10`) periodically re-writes the output + HTML
report from whatever's completed so far, so a killed run still leaves a
usable partial result on disk.

---

## 6. `--reform`

For any item whose `content_qc_label` is `FLAG`, `--reform` asks the judge
model to rewrite `text`/`answer`/`options` so it passes, then re-runs
**content QC only** (not construction/fairness) on the rewrite — up to
`--reform-attempts` times. `original_text`/`original_answer` preserve the
pre-reform version, and `reform_change_summary` records what changed and
why. `final_decision` is recomputed after reform using the (possibly
updated) content QC result.
