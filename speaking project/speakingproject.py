import argparse
import concurrent.futures
import csv
import functools
import json
import os
import re
import sys
import time

# Raise deepeval's per-call timeout (default ~90s) before the judge model is
# constructed below, so a slow-but-fine response isn't indistinguishable from
# a genuinely bad item (see score_metrics()/classify_level() failure paths).
# Only set if the caller hasn't already overridden it.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "180")

from deepeval.models import GeminiModel

# ----------------------------------------------------------------------
# deepeval judge model
# ----------------------------------------------------------------------
# Every LLM call in this file -- classify_level(), score_metrics(),
# evaluate_fairness(), and reform_item() -- goes through the same
# deepeval-wrapped judge model below (_JUDGE_MODEL), the hosted
# gemma-4-26b-a4b-it model via the Gemini API (Google AI Studio). Requires
# GOOGLE_API_KEY to be set in the environment (or pass api_key= directly
# below) and `pip install google-genai`.
_JUDGE_MODEL = GeminiModel(
    model="gemma-4-26b-a4b-it",
    temperature=0.2,
)


_RETRY_DELAY_RE = re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE)
_RATE_LIMIT_MARKERS = ("429", "RESOURCE_EXHAUSTED", "rate limit", "quota")


def _judge_call(prompt: str, retries: int = 6, max_backoff_sec: float = 60.0) -> str:
    """Calls _JUDGE_MODEL through deepeval's DeepEvalBaseLLM.generate()
    interface and returns plain text, with retry/backoff. Callers do their
    own JSON-extraction/parsing on the returned text, since this call has
    no hard JSON-mode guarantee from the provider.

    On a 429/RESOURCE_EXHAUSTED response, Gemini's own error payload tells
    us how long to wait (e.g. "Please retry in 24.01s") -- that number is
    frequently much bigger than a naive fixed backoff, so we parse it and
    sleep that long (plus a small buffer) instead of guessing. Any other
    error falls back to exponential backoff."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            raw = _JUDGE_MODEL.generate(prompt)
            # DeepEvalBaseLLM.generate() implementations differ on whether
            # they return plain text or a (text, cost) tuple depending on
            # provider/schema usage -- normalize to plain text either way.
            if isinstance(raw, tuple):
                raw = raw[0]
            return str(raw).strip()
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            msg = str(e)
            m = _RETRY_DELAY_RE.search(msg)
            if m:
                delay = float(m.group(1)) + 1.0  # small buffer past what the API asked for
            elif any(marker in msg for marker in _RATE_LIMIT_MARKERS):
                delay = min(max_backoff_sec, 10.0 * (2 ** attempt))
            else:
                delay = 1.5 * (attempt + 1)
            delay = min(delay, max_backoff_sec)
            time.sleep(delay)
    raise RuntimeError("Could not reach judge model (Gemini via deepeval): %s" % last_err)


CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

SKILL_METRIC_CONFIG = {
    "reading": {
        "check_audio_transcript": False,
        "metrics": [
            {"key": "accuracy_pct", "label": "accuracy", "scale": "pct", "compare": "min", "threshold": 90,
             "definition": "Is the question and its expected answer factually/grammatically correct together?"},
            {"key": "clarity_score", "label": "clarity", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "How easy is the question to understand, 5 = completely unambiguous, 1 = very confusing."},
            {"key": "completeness_score", "label": "completeness", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "Is all necessary information/context provided to answer the question, with no missing context."},
            {"key": "vocabulary_level_fit", "label": "vocabulary level fit", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "Does the vocabulary/register used match the item's CEFR level (5 = perfect fit for that level, 1 = badly mismatched, e.g. B1 vocabulary in an A1 item)."},
            {"key": "construction_score", "label": "construction", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "Is the item's text well-constructed as a piece of writing -- coherent sentence structure, no fragments/run-ons/garbled phrasing, logical flow between sentences (5 = well-formed throughout, 1 = badly malformed/fragmented construction)."},
            {"key": "passage_strength_score", "label": "passage strength", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "Is the passage/context substantive and well-developed enough to support the question -- enough detail, specificity, and content for the question to be reasonably answerable (5 = strong, well-developed passage, 1 = thin/vague passage with insufficient content)."},
        ],
    },
    "writing": {
        "check_audio_transcript": False,
        "metrics": [
            {"key": "accuracy_pct", "label": "accuracy", "scale": "pct", "compare": "min", "threshold": 90,
             "definition": "Is the prompt correctly aligned with its labeled CEFR level and question type, and is any provided sample answer/rubric key a valid, correct response to the prompt?"},
            {"key": "grammar_errors", "label": "grammar", "scale": "count", "compare": "max", "threshold": 0,
             "definition": "Count ONLY grammar/spelling/punctuation errors in the PROMPT text (the writing task instructions, not a learner's hypothetical response) that actually change, obscure, or misconvey the intended meaning -- i.e. a writer could reasonably misunderstand what they're being asked to produce. Do NOT count minor, cosmetic, or purely stylistic issues (small typos, informal phrasing, missing an Oxford comma, a stray capitalization, a tense slip like is/was/were, a pronoun mismatch like 'What I buy there' instead of 'What you buy there', etc.) that don't change what the prompt is asking."},
            {"key": "clarity_score", "label": "clarity", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "How clearly does the prompt communicate what the writer is being asked to produce? 5 = completely unambiguous, 1 = very confusing."},
            {"key": "construction_score", "label": "construction", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "Is the PROMPT text itself well-constructed -- coherent sentence structure, no fragments/run-ons/garbled phrasing (5 = well-formed, 1 = badly malformed/fragmented construction)."},
            {"key": "passage_strength_score", "label": "passage strength", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "If the prompt gives a scenario/passage/context for the writer to respond to, is it substantive and well-developed enough to write a full response from, rather than thin or vague? Score 5 if there is no scenario/context to evaluate."},
        ],
    },
    "speaking": {
        "check_audio_transcript": False,
        "metrics": [
            {"key": "accuracy_pct", "label": "accuracy", "scale": "pct", "compare": "min", "threshold": 90,
             "definition": "Is the prompt correctly aligned with its labeled CEFR level and question type, and is any provided sample/model response a valid, correct answer to the prompt?"},
            {"key": "grammar_errors", "label": "grammar", "scale": "count", "compare": "max", "threshold": 0,
             "definition": "Count ONLY grammar/spelling/punctuation errors in the PROMPT text (the speaking task instructions, not a learner's hypothetical spoken response) that actually change, obscure, or misconvey the intended meaning -- i.e. a speaker could reasonably misunderstand what they're being asked to say/do. Do NOT count minor, cosmetic, or purely stylistic issues (small typos, informal phrasing, missing an Oxford comma, a stray capitalization, a tense slip like is/was/were, a pronoun mismatch like 'What I buy there' instead of 'What you buy there', etc.) that don't change what the prompt is asking -- this is a speaking task meant to be understood by ear, not graded as formal writing, so minor imperfections that don't distort meaning should NOT be counted as errors."},
            {"key": "clarity_score", "label": "clarity", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "How clearly does the prompt communicate what the speaker is being asked to say/do? 5 = completely unambiguous, 1 = very confusing."},
            {"key": "construction_score", "label": "construction", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "Is the PROMPT text itself well-constructed -- coherent sentence structure, no fragments/run-ons/garbled phrasing (5 = well-formed, 1 = badly malformed/fragmented construction)."},
            {"key": "passage_strength_score", "label": "passage strength", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "If the prompt gives a scenario/passage/context for the speaker to respond to, is it substantive and well-developed enough to speak from, rather than thin or vague? Score 5 if there is no scenario/context to evaluate."},
        ],
    },
    # Fallback used for any skill not explicitly listed above (falls back to
    # the reading config, since reading is the primary skill in scope).
    "_default": {
        "check_audio_transcript": False,
        "metrics": [
            {"key": "accuracy_pct", "label": "accuracy", "scale": "pct", "compare": "min", "threshold": 90,
             "definition": "Is the question and its expected answer factually/grammatically correct together?"},
            {"key": "clarity_score", "label": "clarity", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "How easy is the question to understand, 5 = completely unambiguous, 1 = very confusing."},
            {"key": "completeness_score", "label": "completeness", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "Is all necessary information/context provided to answer the question, with no missing context."},
            {"key": "vocabulary_level_fit", "label": "vocabulary level fit", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "Does the vocabulary/register used match the item's CEFR level (5 = perfect fit for that level, 1 = badly mismatched, e.g. B1 vocabulary in an A1 item)."},
            {"key": "construction_score", "label": "construction", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "Is the item's text well-constructed -- coherent sentence structure, no fragments/run-ons/garbled phrasing (5 = well-formed, 1 = badly malformed/fragmented construction)."},
            {"key": "passage_strength_score", "label": "passage strength", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "Is the passage/context substantive and well-developed enough to support the question, rather than thin or vague? Score 5 if there is no passage/context to evaluate."},
        ],
    },
}

LEVEL_TOLERANCE = 0          # 0 = exact CEFR match required, 1 = allow one tier off (e.g. A2<->B1)

# Minimum word count for item text to count as a substantive passage/prompt
# rather than a thin stub. Applies across all skills (a "passage" here just
# means the item's own text -- the reading/listening/writing/speaking
# context the learner is given), not just the reading skill specifically.
WEAK_PASSAGE_MIN_WORDS = 20

# Union of every metric key across all configured skills, in first-seen
# order. Used so CSV/JSON output has consistent columns across rows even
# though different skills score different metrics (n/a where not applicable).
ALL_METRIC_KEYS = []
for _cfg in SKILL_METRIC_CONFIG.values():
    for _m in _cfg["metrics"]:
        if _m["key"] not in ALL_METRIC_KEYS:
            ALL_METRIC_KEYS.append(_m["key"])


def _skill_config(skill: str) -> dict:
    key = (skill or "").strip().lower()
    return SKILL_METRIC_CONFIG.get(key, SKILL_METRIC_CONFIG["_default"])
# ---------------------------------------------------------------------------


# ----------------------------------------------------------------------
# 1. Deterministic rule-based checks (fast, no LLM needed)
# ----------------------------------------------------------------------

def metric_flags(item: dict) -> list:
    flags = []
    raw_text = (item.get("text") or "")
    text = raw_text.strip()
    answer = (item.get("answer") or "")
    level = (item.get("level") or "").strip().upper()
    itype = (item.get("type") or "").strip().lower()
    skill = (item.get("skill") or "").strip().lower()

    if answer != answer.strip():
        flags.append("answer_whitespace")
    if raw_text != text:
        flags.append("text_whitespace")
    if not text:
        flags.append("missing_text")
    if level not in CEFR_LEVELS:
        flags.append("invalid_level")
    if itype == "fill_up" and "___" not in text and "____" not in text:
        flags.append("no_blank_marker")
    if itype in ("mcq", "multiple_choice") and not item.get("options"):
        flags.append("missing_options")
    if _skill_config(skill).get("check_audio_transcript") and not item.get("audio") and not item.get("transcript"):
        flags.append("missing_audio_or_transcript")
    if text and not re.search(r"[.?!]\s*$", text.replace("___", "").replace("____", "")):
        flags.append("missing_end_punctuation")
    if level in ("A1", "A2") and len(answer.split()) > 4:
        flags.append("answer_too_long_for_level")
    if text and len(text.split()) < WEAK_PASSAGE_MIN_WORDS:
        flags.append("weak_passage")
    construction_problem = _construction_problem(text)
    if construction_problem:
        flags.append("poor_construction")

    return flags


_REPEATED_WORD_RE = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)


def _construction_problem(text: str) -> str:
    """Deterministic, regex-level sentence-construction check -- catches
    obviously garbled/malformed text (immediately repeated words, unbalanced
    quotes/brackets) without needing the judge model. Returns a short
    description of the first problem found, or '' if none. This is
    intentionally narrow (it will NOT catch subtler construction problems
    like run-on sentences or dangling clauses -- those are covered by the
    LLM-judged 'construction_score' metric in score_metrics()/qc_verdict()
    instead)."""
    if not text:
        return ""
    dup = _REPEATED_WORD_RE.search(text)
    if dup:
        return "immediately repeated word: %r" % dup.group(0)
    for open_ch, close_ch, name in (('"', '"', "double quote"), ("(", ")", "parenthesis"), ("[", "]", "bracket")):
        if open_ch == close_ch:
            if text.count(open_ch) % 2 != 0:
                return "unbalanced %s" % name
        else:
            if text.count(open_ch) != text.count(close_ch):
                return "unbalanced %s" % name
    return ""


def _ws_location(raw: str) -> str:
    """Describes exactly where whitespace was found in a raw field value,
    e.g. '2 leading space(s) and 1 trailing space(s)'."""
    lead = len(raw) - len(raw.lstrip())
    trail = len(raw) - len(raw.rstrip())
    parts = []
    if lead:
        parts.append("%d leading space(s)" % lead)
    if trail:
        parts.append("%d trailing space(s)" % trail)
    return " and ".join(parts) if parts else "stray whitespace"


def metric_flag_details(item: dict, flags: list) -> dict:
    """Expands each short rule-flag keyword returned by metric_flags() into
    a full sentence naming (a) which field the problem is in and (b) what,
    concretely, is wrong there -- e.g. quoting the offending value or
    counting the words over the limit -- rather than just the bare flag
    name. Returns {flag_name: detail_sentence}, so callers can filter/join
    as needed (e.g. compute_final_decision() drops ignored flags first)."""
    raw_text = item.get("text") or ""
    text = raw_text.strip()
    raw_answer = item.get("answer") or ""
    answer = raw_answer.strip()
    raw_level = item.get("level") or ""
    level = raw_level.strip().upper()
    itype = (item.get("type") or "").strip().lower()
    skill = (item.get("skill") or "").strip().lower()

    details = {}
    for f in flags:
        if f == "answer_whitespace":
            details[f] = (
                "In the 'answer' field: %s (raw value: %r)."
                % (_ws_location(raw_answer), raw_answer)
            )
        elif f == "text_whitespace":
            details[f] = (
                "In the 'text' field: %s (raw value: %r)."
                % (_ws_location(raw_text), raw_text)
            )
        elif f == "missing_text":
            details[f] = "In the 'text' field: no question/prompt text was provided (field is empty)."
        elif f == "invalid_level":
            details[f] = (
                "In the 'level' field: value is %r, which is not a recognized CEFR level (expected one of %s)."
                % (raw_level, ", ".join(CEFR_LEVELS))
            )
        elif f == "no_blank_marker":
            details[f] = (
                "In the 'text' field: item type is %r (fill-in-the-blank) but no blank marker "
                "('___' or '____') appears anywhere in the text: %r."
                % (itype, text)
            )
        elif f == "missing_options":
            details[f] = (
                "In the 'options' field: item type is %r (multiple choice) but no options are listed."
                % itype
            )
        elif f == "missing_audio_or_transcript":
            details[f] = (
                "Neither the 'audio' field nor the 'transcript' field is present, "
                "but skill %r requires one." % skill
            )
        elif f == "missing_end_punctuation":
            tail = text[-12:] if text else ""
            details[f] = (
                "In the 'text' field: does not end with terminal punctuation (., ?, or !) -- text ends with: %r."
                % tail
            )
        elif f == "answer_too_long_for_level":
            word_count = len(answer.split())
            details[f] = (
                "In the 'answer' field: %d word(s) (%r), which exceeds the 4-word maximum for level %s."
                % (word_count, answer, level)
            )
        elif f == "weak_passage":
            word_count = len(text.split())
            details[f] = (
                "In the 'text' field: only %d word(s), below the %d-word minimum expected for a "
                "substantive passage/prompt -- the item's context reads as thin or underdeveloped "
                "rather than giving the learner enough to work with (text: %r)."
                % (word_count, WEAK_PASSAGE_MIN_WORDS, text)
            )
        elif f == "poor_construction":
            problem = _construction_problem(text)
            details[f] = (
                "In the 'text' field: sentence construction is malformed -- %s (text: %r)."
                % (problem or "structural issue detected", text)
            )
        else:
            details[f] = f.replace("_", " ") + "."
    return details


# ----------------------------------------------------------------------
# 2. deepeval + Gemini judge model calls
# ----------------------------------------------------------------------

CEFR_DESCRIPTORS = """A1 - Beginner: Understands/uses very familiar everyday expressions and basic
     phrases (name, age, simple facts). Present tense, short simple sentences,
     high-frequency vocabulary only.
A2 - Elementary: Understands sentences about immediate relevance (personal info,
     shopping, local area). Simple past/future, common connectors (and, but,
     because), everyday vocabulary.
B1 - Intermediate: Understands main points of clear standard input on familiar
     matters (work, school, leisure). Can handle most travel situations, describe
     experiences/opinions. Wider range of tenses, some complex sentences.
B2 - Upper-Intermediate: Understands main ideas of complex text on concrete/
     abstract topics, including technical discussion in own field. Can interact
     with fluency, produce detailed text on a range of subjects. Complex
     grammar, idiomatic expressions, nuanced vocabulary.
C1 - Advanced: Understands wide range of demanding, longer texts and implicit
     meaning. Fluent, spontaneous expression without much searching for words.
     Flexible/effective language use for social, academic, professional purposes.
C2 - Proficient: Understands virtually everything read/heard with ease.
     Summarizes/reconstructs information from different sources coherently.
     Near-native precision, subtle shades of meaning even in complex situations.
"""

CEFR_PROMPT = """You are a CEFR (Common European Framework of Reference) proficiency
classifier for a language-learning platform. Classify the difficulty level of
this item using the official CEFR level descriptors below as your criteria.
Base your decision on: (1) vocabulary difficulty/frequency, (2) grammatical
structures used.

CEFR level descriptors (use these as your classification criteria):
{descriptors}

Skill: {skill}
Type: {type}
Item text: {text}
Expected answer: {answer}
Options: {options}

Respond with ONLY a JSON object, nothing else, in this exact shape:
{{"level": "<A1, A2, B1, B2, C1, or C2>", "reason": "<one short sentence citing which criteria drove the decision>"}}
"""

def _build_metric_prompt(skill: str) -> str:
    """Builds a metric-scoring prompt tailored to the metrics configured for
    this skill, using .format(...) placeholders for the item fields."""
    config = _skill_config(skill)
    metric_lines = []
    json_fields = []
    has_grammar = any(m["key"] == "grammar_errors" for m in config["metrics"])
    has_construction = any(m["key"] == "construction_score" for m in config["metrics"])
    has_passage = any(m["key"] == "passage_strength_score" for m in config["metrics"])
    for m in config["metrics"]:
        if m["scale"] == "pct":
            target = "target: 100, flag below %d" % m["threshold"]
            json_fields.append('"%s": <0-100>' % m["key"])
        elif m["scale"] == "count":
            target = "target: 0, flag above %d" % m["threshold"]
            json_fields.append('"%s": <integer count>' % m["key"])
        else:  # five
            target = "target: 4-5, flag below %d" % m["threshold"]
            json_fields.append('"%s": <1-5>' % m["key"])
        metric_lines.append("- %s (%s): %s" % (m["label"].capitalize(), target, m["definition"]))
    if has_grammar:
        json_fields.append(
            '"grammar_error_details": [<one short string per grammar/spelling/'
            'punctuation error that actually changes or obscures the meaning, each '
            'quoting the exact problematic word(s)/phrase from the text, saying what '
            'is wrong, AND explaining how it changes/misconveys the meaning, e.g. '
            '"\'I no go there\' -- missing auxiliary verb, could be read as a command '
            'rather than a statement, should be \'I do not go there\'"; only include '
            'errors that meet this bar, NOT minor/cosmetic issues; empty list if '
            'grammar_errors is 0>]'
        )
    if has_construction:
        json_fields.append(
            '"construction_issue_details": [<one short string per sentence-construction '
            'problem found (fragments, run-ons, garbled/malformed phrasing, broken sentence '
            'structure -- NOT spelling/grammar-in-the-word-sense, which is covered separately), '
            'each quoting the exact problematic phrase and explaining specifically what is '
            'structurally wrong with it; empty list if construction_score is 4 or 5>]'
        )
    if has_passage:
        json_fields.append(
            '"passage_weakness_details": [<one short string per specific way the passage/'
            'context is thin, vague, or underdeveloped (e.g. missing concrete detail a question '
            'relies on, too short to support the task, lacks the specificity needed to answer), '
            'or empty list if there is no passage/context to evaluate or passage_strength_score '
            'is 4 or 5>]'
        )
    json_fields.append('"reason": "<one short sentence>"')

    return (
        "You are a strict QC reviewer for a language-learning platform, "
        "reviewing a %s item.\nEvaluate this item against the following metrics.\n\n"
        "Metric definitions and targets:\n%s\n\n"
        "Item:\n"
        "Skill: {skill} | Type: {type} | Labeled CEFR level: {labeled_level} | Predicted CEFR level: {predicted_level}\n"
        "Text: {text}\n"
        "Answer: {answer}\n"
        "Options: {options}\n"
        "Rule-based flags already detected: {metric_flags}\n\n"
        "Respond with ONLY a JSON object, nothing else, in this exact shape:\n"
        "{{%s}}\n"
    ) % (skill or "generic", "\n".join(metric_lines), ", ".join(json_fields))

REFORM_PROMPT = """You are an expert item writer for a language-learning platform.
The item below FAILED quality control. Rewrite it so it passes, while
preserving the original skill, question type, and topic/subject matter as
closely as possible. Do not change the labeled CEFR level target — write the
item so it is appropriate for that level.

CEFR level descriptors (write the item to fit this level):
{descriptors}

Original item:
Skill: {skill} | Type: {type} | Labeled CEFR level: {labeled_level}
Text: {text}
Answer: {answer}
Options: {options}

QC failure reason(s): {qc_reason}

Rewriting rules:
- If "missing_answer" is among the failure reasons, supply a correct answer
  consistent with the text.
- If grammar errors were found, fix them.
- If clarity or completeness was flagged, add whatever minimal context is
  needed so the question can be answered on its own, without changing what
  is being tested.
- If accuracy was flagged, correct the question and/or answer so they are
  factually consistent with each other.
- If construction was flagged (fragments, run-ons, garbled/malformed
  phrasing, repeated words, unbalanced quotes/brackets), rewrite the
  sentence(s) so they are structurally sound and read naturally.
- If passage strength was flagged (thin/vague/underdeveloped context),
  expand the passage/context with enough concrete, specific detail to fully
  support the question, without changing what is being tested.
- For fill_up items, keep exactly one blank shown as ___ per answer expected
  (if multiple answers are expected, keep multiple ___ in the text, matching
  the order of answers).
- For mcq items, keep exactly 4 options, with the same letter for the
  correct one as the "answer" field.
- Keep the item roughly the same length and register as the original.
- Do not simply describe what you would change — output the finished,
  corrected item.

Respond with ONLY a JSON object, nothing else, in this exact shape:
{{"text": "<rewritten question text>", "answer": "<rewritten answer>", "options": [<list of 4 strings, or empty list if not mcq>], "change_summary": "<one short sentence on what you changed and why>"}}
"""


def classify_level(item: dict):
    """CEFR classification call, via the shared cloud judge model
    (_judge_call(), deepeval + Gemini). Returns (level, reason). Uses a
    JSON-object parse + retry (same reliability pattern as score_metrics),
    since the prompt asks for JSON-only but the provider gives no hard
    JSON-mode guarantee."""
    prompt = CEFR_PROMPT.format(
        descriptors=CEFR_DESCRIPTORS,
        skill=item.get("skill", ""),
        type=item.get("type", ""),
        text=item.get("text", ""),
        answer=item.get("answer", ""),
        options=item.get("options", ""),
    )

    parse_attempts = 2
    last_raw = ""
    last_reason = ""
    for attempt in range(parse_attempts):
        try:
            raw = _judge_call(prompt)
        except RuntimeError as e:
            return "ERROR", "judge model unreachable: %s" % e

        last_raw = raw
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            continue  # malformed response, try again

        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            continue  # malformed JSON, try again

        reason = parsed.get("reason", "") or raw[:200]
        last_reason = reason

        level = (parsed.get("level") or "").strip().upper()
        if level not in CEFR_LEVELS:
            # "level" field was missing/empty/unrecognizable — search the
            # ENTIRE raw response (not just the level field, which may be
            # blank) for a standalone CEFR code before giving up, since the
            # model sometimes puts the code in "reason" instead, or phrases
            # the level oddly (e.g. "B2 - Upper-Intermediate").
            salvage = re.search(r"\b([ABC][12])\b", raw)
            level = salvage.group(1) if salvage else None

        if level in CEFR_LEVELS:
            return level, reason
        # Invalid/unsalvageable level this attempt — use the remaining
        # retries instead of giving up immediately.

    return "UNKNOWN", (
        "Model did not return a recognizable CEFR level after %d attempt(s). "
        "Last reason given: '%s' | Raw output: %s"
        % (parse_attempts, last_reason, last_raw[:150])
    )


def score_metrics(item: dict, predicted_level: str, flags: list) -> dict:
    """Rubric-based scores, using whichever metrics are configured for this
    item's skill (see SKILL_METRIC_CONFIG)."""
    skill = item.get("skill", "")
    config = _skill_config(skill)
    prompt = _build_metric_prompt(skill).format(
        skill=skill,
        type=item.get("type", ""),
        labeled_level=item.get("level", ""),
        predicted_level=predicted_level,
        text=item.get("text", ""),
        answer=item.get("answer", ""),
        options=item.get("options", ""),
        metric_flags=", ".join(flags) if flags else "none",
    )

    def _fallback_for(key):
        # -1 is the "could not be determined" sentinel, only meaningful for
        # grammar_errors (an integer count where -1 isn't a valid count).
        return -1 if key == "grammar_errors" else 0

    default_fail = {m["key"]: _fallback_for(m["key"]) for m in config["metrics"]}
    default_fail["reason"] = ""
    default_fail["grammar_error_details"] = []
    default_fail["construction_issue_details"] = []
    default_fail["passage_weakness_details"] = []

    parse_attempts = 2  # try once more if the model's JSON is malformed
    last_raw = ""
    for attempt in range(parse_attempts):
        try:
            raw = _judge_call(prompt)
        except RuntimeError as e:
            default_fail["reason"] = "judge model unreachable: %s" % e
            return default_fail

        last_raw = raw
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            continue  # try again
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            continue  # try again

        def _num(v, cast=int, fallback=0):
            try:
                return cast(v)
            except (TypeError, ValueError):
                return fallback

        result = {}
        missing_fields = []
        for m in config["metrics"]:
            key = m["key"]
            if key not in parsed:
                missing_fields.append(key)
            result[key] = _num(parsed.get(key, _fallback_for(key)), fallback=_fallback_for(key))

        reason = parsed.get("reason", "")
        if missing_fields and not reason:
            reason = "%s field(s) missing from model output" % ", ".join(missing_fields)
        result["reason"] = reason

        for detail_key in ("grammar_error_details", "construction_issue_details", "passage_weakness_details"):
            raw_details = parsed.get(detail_key)
            result[detail_key] = (
                [str(d).strip() for d in raw_details if str(d).strip()]
                if isinstance(raw_details, list) else []
            )
        return result

    # Exhausted retries without getting parseable JSON
    default_fail["reason"] = "Could not parse judge model output after %d attempt(s): %s" % (
        parse_attempts, last_raw[:150]
    )
    return default_fail


def reform_item(item: dict, qc_reason: str) -> dict:
    """Ask the judge model to rewrite a flagged item so it passes QC. Returns a dict
    with the (possibly) new text/answer/options plus a change_summary, or
    the original item unchanged with an error note if the call/parse fails."""
    prompt = REFORM_PROMPT.format(
        descriptors=CEFR_DESCRIPTORS,
        skill=item.get("skill", ""),
        type=item.get("type", ""),
        labeled_level=item.get("level", ""),
        text=item.get("text", ""),
        answer=item.get("answer", ""),
        options=item.get("options", ""),
        qc_reason=qc_reason,
    )
    try:
        raw = _judge_call(prompt)
    except RuntimeError as e:
        return {
            "text": item.get("text", ""), "answer": item.get("answer", ""),
            "options": item.get("options", []),
            "change_summary": "REFORM FAILED (judge model unreachable): %s" % e,
        }

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return {
            "text": item.get("text", ""), "answer": item.get("answer", ""),
            "options": item.get("options", []),
            "change_summary": "REFORM FAILED (could not parse output): %s" % raw[:150],
        }
    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return {
            "text": item.get("text", ""), "answer": item.get("answer", ""),
            "options": item.get("options", []),
            "change_summary": "REFORM FAILED (invalid JSON): %s" % raw[:150],
        }

    return {
        "text": parsed.get("text", item.get("text", "")),
        "answer": parsed.get("answer", item.get("answer", "")),
        "options": parsed.get("options", item.get("options", [])),
        "change_summary": parsed.get("change_summary", ""),
    }


def _level_within_range(labeled: str, predicted: str, tolerance: int) -> bool:
    if labeled not in CEFR_LEVELS or predicted not in CEFR_LEVELS:
        return False
    return abs(CEFR_LEVELS.index(labeled) - CEFR_LEVELS.index(predicted)) <= tolerance


def qc_verdict(level_match: bool, scores: dict, flags: list, skill: str = "",
                metric_flag_detail_map: dict = None):
    """metric_flag_detail_map, if provided (the {flag_name: detail_sentence}
    dict from metric_flag_details()), is used to expand rule flags into full
    what+where sentences directly in the returned reason string -- so
    qc_reason (and therefore every column/report derived from it) is
    self-contained and detailed for every rejected item, not just
    final_decision_reason. Falls back to bare flag keywords if no map is
    given."""
    config = _skill_config(skill)
    reasons = []
    level_reason_text = "predicted CEFR level outside defined range of labeled level"
    if not level_match:
        reasons.append(level_reason_text)

    for m in config["metrics"]:
        key, label, threshold = m["key"], m["label"], m["threshold"]
        val = scores.get(key)

        if key == "grammar_errors":
            if val is None or val < 0:
                reasons.append("grammar score could not be determined")
            elif val > threshold:
                details = scores.get("grammar_error_details") or []
                reasons.append("%d grammar error(s) found (target: %d)" % (val, threshold))
                if details:
                    for d in details:
                        reasons.append("grammar error: %s" % d)
                else:
                    reasons.append(
                        "grammar error detail not provided by judge model -- "
                        "see item text/answer manually"
                    )
            continue

        if key in ("construction_score", "passage_strength_score"):
            detail_key = (
                "construction_issue_details" if key == "construction_score"
                else "passage_weakness_details"
            )
            if val is None:
                reasons.append("%s could not be determined" % label)
            elif val < threshold:
                reasons.append("%s below target (%s/5)" % (label, val))
                details = scores.get(detail_key) or []
                if details:
                    noun = "construction issue" if key == "construction_score" else "passage weakness"
                    for d in details:
                        reasons.append("%s: %s" % (noun, d))
                else:
                    reasons.append(
                        "%s detail not provided by judge model -- see item text manually" % label
                    )
            continue

        if val is None:
            reasons.append("%s could not be determined" % label)
        elif m["compare"] == "min" and val < threshold:
            if m["scale"] == "pct":
                reasons.append("%s below target (%s%% < %s%%)" % (label, val, threshold))
            else:  # five
                reasons.append("%s below target (%s/5)" % (label, val))

    if flags:
        if metric_flag_detail_map:
            detail_sentences = [
                metric_flag_detail_map.get(f, f.replace("_", " ") + ".")
                for f in flags
            ]
            reasons.append("rule flags: " + " ".join(detail_sentences))
        else:
            reasons.append("rule flags: %s" % ", ".join(flags))

    if reasons:
        level_only = reasons == [level_reason_text]
        return "FLAG", "; ".join(reasons), level_only
    return "PASS", "meets all QC targets", False


# ----------------------------------------------------------------------
# 2b. Fairness, bias & sensitivity gate (separate from content QC above)
# ----------------------------------------------------------------------
# qc_verdict() above only checks content accuracy/grammar/clarity/
# completeness/vocabulary-fit and CEFR level -- it never asks whether the
# item is FAIR to give a speaker: whether it quietly assumes specialist or
# cultural knowledge, leans on a cultural/gender stereotype, or asks the
# speaker to produce sensitive/distressing content out loud. This is a
# fully independent second gate with its own flags list, its own judge-model
# call, and its own PASS/FLAG label, so a FAIRNESS_FLAG never gets confused
# with a content FLAG -- an item can pass one and fail the other. Required
# for the speaking skill in particular, since a speaking prompt (unlike a
# reading/writing prompt) asks the learner to personally produce spoken
# content on the topic given, so a fairness problem in the prompt lands
# directly on the speaker.

FAIRNESS_PROMPT = """You are a strict fairness/bias/accessibility reviewer for a
language-learning platform, checking a single ITEM for problems that would
make it unfair or inappropriate for a general, international adult audience.

CALIBRATION -- read carefully before scoring: the overwhelming majority of
well-written educational items about ordinary topics (work, hobbies, daily
life, travel, opinions) have NO fairness problems. Using professional/
technical vocabulary, describing a specific job, or asking about a
particular country is NOT by itself cultural bias, and a person or pronoun
appearing in a role is NOT by itself gender bias. Only flag a dimension true
when there is specific, citable evidence of an actual stereotype, an unfair
knowledge requirement, or a genuinely inappropriate topic. If you are
unsure, prefer false.

Dimensions to judge:
- requires_specialist_or_cultural_knowledge (true/false): true ONLY if
  answering/speaking requires specialist/technical knowledge or knowledge
  specific to one culture/region/religion that can't be reasonably expected
  of a general international audience.
- cultural_bias (true/false): true ONLY if the item asserts one culture's
  norms as objectively "correct" or default, or relies on an ethnic/
  national stereotype. An item simply being set in, or written from the
  perspective of, one culture is NOT bias.
- gender_bias (true/false): true ONLY if the item relies on an explicit
  gender stereotype (e.g. assuming a role can only be done by one gender)
  or unnecessarily genders a role with no basis in the item. A person in
  the item having a stated gender is NOT by itself bias.
- sensitive_content (true/false): true ONLY if the item asks the speaker to
  discuss or produce content that would be genuinely distressing, political
  controversy, graphic violence, self-harm, explicit content, or hate
  speech for a general audience. Routine professional/personal topics are
  NOT sensitive.

Item:
Skill: {skill} | Type: {type} | Labeled CEFR level: {level}
Text: {text}
Answer/model response: {answer}
Options: {options}

Respond with ONLY a JSON object, nothing else, in this exact shape:
{{"requires_specialist_or_cultural_knowledge": <true or false>, "cultural_bias": <true or false>, "gender_bias": <true or false>, "sensitive_content": <true or false>, "reason": "<one short sentence citing specific evidence for any dimension flagged true, or 'no issues found'>"}}
"""


def evaluate_fairness(item: dict) -> dict:
    """FAIRNESS/BIAS/SENSITIVITY gate: one structured-JSON judge-model call
    per item, via the shared _judge_call(). Returns a dict of booleans (one
    per dimension judged) plus a reason, or an "error": True result if the
    call/parse failed after retries -- callers must check "error" rather
    than inferring failure from an empty reason, since a timeout failure
    still fills in a reason string."""
    prompt = FAIRNESS_PROMPT.format(
        skill=item.get("skill", ""),
        type=item.get("type", ""),
        level=(item.get("level") or "").strip().upper(),
        text=item.get("text", ""),
        answer=item.get("answer", ""),
        options=item.get("options", ""),
    )

    result = {
        "requires_specialist_or_cultural_knowledge": None, "cultural_bias": None,
        "gender_bias": None, "sensitive_content": None,
        "reason": "", "error": False,
    }

    parse_attempts = 2
    last_raw = ""
    for attempt in range(parse_attempts):
        try:
            raw = _judge_call(prompt)
        except RuntimeError as e:
            result["reason"] = "fairness check failed: judge model unreachable: %s" % e
            result["error"] = True
            return result

        last_raw = raw
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            continue  # try again
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            continue  # try again

        if any(k not in parsed for k in ("cultural_bias", "gender_bias", "sensitive_content")):
            continue  # incomplete JSON, try again

        result["requires_specialist_or_cultural_knowledge"] = bool(
            parsed.get("requires_specialist_or_cultural_knowledge", False)
        )
        result["cultural_bias"] = bool(parsed.get("cultural_bias", False))
        result["gender_bias"] = bool(parsed.get("gender_bias", False))
        result["sensitive_content"] = bool(parsed.get("sensitive_content", False))
        result["reason"] = parsed.get("reason", "")
        return result

    result["reason"] = "fairness check failed: could not parse judge model output after %d attempt(s): %s" % (
        parse_attempts, last_raw[:150]
    )
    result["error"] = True
    return result


def fairness_flags(item: dict, llm_result: dict) -> list:
    """LLM-derived flags for FAIRNESS/BIAS/SENSITIVITY -- separate from
    metric_flags() (formatting/structure) and qc_verdict()'s content
    rubric. Purely about whether the item is fair and appropriate for a
    general audience to be asked to speak about."""
    flags = []

    if llm_result.get("requires_specialist_or_cultural_knowledge") is True:
        flags.append("requires_specialist_knowledge")
    if llm_result.get("cultural_bias") is True:
        flags.append("cultural_bias")
    if llm_result.get("gender_bias") is True:
        flags.append("gender_bias")
    if llm_result.get("sensitive_content") is True:
        flags.append("sensitive_content_review_needed")

    if llm_result.get("error"):
        flags.append("fairness_check_error")

    return flags


def fairness_qc_gate(flags: list, llm_reason: str):
    """Independent PASS/FLAG gate for FAIRNESS/BIAS/SENSITIVITY only.
    Deliberately separate from qc_verdict() (content) so the two dimensions
    are reported, and can fail, independently of each other."""
    if flags == ["fairness_check_error"]:
        # A failed call is not a real fairness judgment, so don't let it
        # read as FAIRNESS_FLAG (auto-reject) or silently fall through to
        # FAIRNESS_PASS.
        return "FAIRNESS_ERROR", (llm_reason or "check error")
    if flags:
        # Each flagged dimension gets its own descriptive sentence (what
        # kind of fairness problem this is), plus the judge model's own
        # cited evidence for *why*.
        real_flags = [f for f in flags if f != "fairness_check_error"]
        parts = [FAIRNESS_FLAG_DESCRIPTIONS.get(f, f.replace("_", " ")) for f in real_flags]
        if llm_reason and llm_reason.strip().lower() not in ("", "no issues found"):
            parts.append("evidence: %s" % llm_reason.strip())
        return "FAIRNESS_FLAG", "; ".join(parts)
    return "FAIRNESS_PASS", "pass"


# ----------------------------------------------------------------------
# 3. I/O helpers
# ----------------------------------------------------------------------

# The rest of the script reads item["text"], item["type"], item["level"],
# etc. exactly. Real-world spreadsheets often use different column names
# for the same thing (e.g. "question_text" instead of "text"), which would
# otherwise silently look like missing data (-> false "missing_text" flags,
# blank CEFR classification, etc). This maps common variants onto the
# canonical field name the script uses internally, WITHOUT removing or
# renaming your original columns in the output — both will be present.
FIELD_ALIASES = {
    "text": ["text", "question_text", "prompt", "question", "item_text"],
    "type": ["type", "question_type", "item_type", "qtype"],
    "level": ["level", "cefr_level", "labeled_level", "target_level"],
    "answer": ["answer", "expected_answer", "correct_answer", "key"],
    "skill": ["skill", "skill_type"],
    "options": ["options", "choices"],
}


def _flatten_field_value(value):
    """Coerces a field value into a plain string, unwrapping nested dicts
    (e.g. {"text": "actual question"}) or JSON-blob strings that sometimes
    show up when data has been round-tripped through a spreadsheet/export
    pipeline. Falls back to a reasonable string representation rather than
    ever handing a dict/list downstream, since every check in this script
    assumes text/answer/level/type are plain strings."""
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("text", "value", "content", "question", "answer"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner
        # No recognizable inner text key — fall back to any string-ish values
        parts = [str(v) for v in value.values() if isinstance(v, (str, int, float))]
        return " ".join(parts)
    if isinstance(value, list):
        return " ".join(_flatten_field_value(v) if isinstance(v, dict) else str(v) for v in value)
    if isinstance(value, str):
        stripped = value.strip()
        # Handle the case where the "text" is literally a JSON string like
        # '{"text": "actual question"}' rather than a real nested dict.
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    return _flatten_field_value(parsed)
            except json.JSONDecodeError:
                pass
        return value
    return str(value)


def _normalize_item(item: dict) -> dict:
    """Fills in canonical fields (text/type/level/answer/skill/options) from
    known alias column names if the canonical field is missing or blank,
    and flattens any nested-dict/JSON-string values into plain strings.
    Original columns are left untouched, so nothing is lost in the output."""
    item = dict(item)
    for canonical, aliases in FIELD_ALIASES.items():
        if canonical == "options":
            # options must stay a list — just pull from an alias if the
            # canonical key is missing/empty; never flatten it to a string.
            if not item.get("options"):
                for alias in aliases:
                    if item.get(alias):
                        item["options"] = item[alias]
                        break
            continue

        value = item.get(canonical)
        if value in (None, "", {}, []):
            for alias in aliases:
                if item.get(alias):
                    value = item[alias]
                    break
        item[canonical] = _flatten_field_value(value)
    return item


def load_items(path: str) -> list:
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else [data]
    else:  # CSV
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            items = []
            for row in reader:
                if "options" in row and row["options"]:
                    row["options"] = row["options"].split("|")
                items.append(row)

    return [_normalize_item(it) for it in items]


# Columns dropped from the final output. "question"/"answer" are dropped
# because "question" duplicates "text" and speaking prompts don't have a
# learner-facing answer field; completeness_score/vocabulary_level_fit are
# dropped because they're n/a for speaking (only reading uses them).
# grammar_errors and level are kept: grammar_errors IS scored for speaking
# items, and level (the original labeled level) is useful alongside
# predicted_level/level_match in the output.
OUTPUT_EXCLUDE_COLUMNS = {
    "question", "answer", "type",
    "completeness_score", "vocabulary_level_fit",
}


def _strip_excluded_columns(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in OUTPUT_EXCLUDE_COLUMNS}


def save_results(rows: list, path: str) -> None:
    rows = [_strip_excluded_columns(r) for r in rows]
    if path.lower().endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
    else:  # CSV
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            for row in rows:
                row = dict(row)
                if isinstance(row.get("options"), list):
                    row["options"] = "|".join(row["options"])
                writer.writerow(row)


# ----------------------------------------------------------------------
# 3a. Checkpointing (resumable progress for long runs)
# ----------------------------------------------------------------------
# A run over a large item set can take a long time since every item makes
# one or more judge-model calls. These helpers periodically persist
# progress to a checkpoint file so an interrupted/crashed run can pick up
# where it left off with --resume, instead of starting over from item 1.

def _default_checkpoint_path(output_path: str) -> str:
    base, _ext = os.path.splitext(output_path)
    return base + ".checkpoint.json"


def save_checkpoint(path: str, total_items: int, results: list) -> None:
    """Writes current progress to `path`. Written atomically (temp file +
    rename) so a crash mid-write can never leave a corrupt/partial
    checkpoint behind."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"total_items": total_items, "results": results}, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def load_checkpoint(path: str, total_items: int):
    """Loads a checkpoint to resume from, if one exists and matches the
    current input (same item count). Returns the saved results list, or
    None if there's nothing usable to resume from."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        print("Checkpoint at %s could not be read -- starting over." % path, file=sys.stderr)
        return None
    if data.get("total_items") != total_items:
        print(
            "Checkpoint at %s doesn't match the current input (%s items saved vs %d now) -- "
            "starting over." % (path, data.get("total_items", "?"), total_items),
            file=sys.stderr,
        )
        return None
    return data.get("results")


def clear_checkpoint(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


# ----------------------------------------------------------------------
# 3b. HTML report (reviewer-facing UI, alongside the CSV/JSON output)
# ----------------------------------------------------------------------
# The CSV/JSON above is the machine-readable output; this is a self-
# contained, single-file HTML report a reviewer can open straight in a
# browser -- no server, no external CDN, no build step -- to page through
# every item, see its SELECT/REVIEW/REJECT verdict at a glance, filter down
# to just the ones that need a human look, and search by text/skill.

HTML_REPORT_COLUMNS = [
    # (row key, column header)
    ("question_id", "Question ID"),
    ("skill", "Skill"),
    ("text", "Text"),
    ("level", "Labeled Level"),
    ("predicted_level", "Predicted Level"),
    ("level_change_reason", "Level Reason"),
    ("qc_label", "Content QC"),
    ("qc_reason", "Content QC Reason"),
    ("fairness_qc_label", "Fairness QC"),
    ("fairness_qc_reason", "Fairness QC Reason"),
    ("metric_flags", "Rule Flags"),
    ("final_decision", "Final Decision"),
    ("final_decision_reason", "Final Decision Reason"),
    ("time_taken_sec", "Time Taken (s)"),
]


def _html_escape(value) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;").replace("'", "&#39;")
    )


def save_html_report(rows: list, path: str) -> None:
    """Writes a single-file HTML report of QC results for reviewers, with
    a SELECT/REVIEW/REJECT summary, filter buttons, a text search box, and
    sortable columns. Rows are embedded as a JSON blob and rendered client-
    side with vanilla JS (text-only insertion, never innerHTML on item
    content) so no item text/reason can break out of its table cell."""
    counts = {"SELECT": 0, "REVIEW": 0, "REJECT": 0}
    for row in rows:
        d = (row.get("final_decision") or "").upper()
        if d in counts:
            counts[d] += 1

    table_rows = []
    for row in rows:
        table_rows.append({key: row.get(key, "") for key, _ in HTML_REPORT_COLUMNS})
    data_json = json.dumps(table_rows, ensure_ascii=False)
    # Guard against a literal "</script>" inside item text/reasons breaking
    # out of the embedded JSON <script> block.
    data_json = data_json.replace("</", "<\\/")

    columns_json = json.dumps([{"key": k, "label": lbl} for k, lbl in HTML_REPORT_COLUMNS])

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QC Review Queue</title>
<style>
  :root{
    --paper:#FAF9F5; --ink:#1B2430; --ink-soft:#5B6572; --line:#E4E1D8;
    --select:#2F6F62; --select-bg:#E7F1EE;
    --review:#B5791F; --review-bg:#FBF0DD;
    --reject:#AE3B2C; --reject-bg:#FBE9E6;
    --card:#FFFFFF; --accent:#2F6F62;
  }
  *{box-sizing:border-box;}
  body{
    margin:0; background:var(--paper); color:var(--ink);
    font-family:-apple-system,"Inter",Segoe UI,Helvetica,Arial,sans-serif;
    font-size:14px; line-height:1.5;
  }
  code, .mono, .badge, td.num{ font-family:"IBM Plex Mono","SF Mono",Consolas,monospace; }
  header{
    padding:28px 32px 20px; border-bottom:1px solid var(--line);
    display:flex; align-items:baseline; justify-content:space-between; flex-wrap:wrap; gap:12px;
  }
  header h1{ font-size:19px; margin:0; letter-spacing:-0.01em; }
  header .sub{ color:var(--ink-soft); font-size:12.5px; margin-top:4px;}
  .stats{ display:flex; gap:10px; padding:18px 32px; flex-wrap:wrap; }
  .stat{
    background:var(--card); border:1px solid var(--line); border-radius:8px;
    padding:12px 16px; min-width:108px;
  }
  .stat .n{ font-size:22px; font-weight:600; font-family:"IBM Plex Mono",monospace; }
  .stat .l{ font-size:11px; color:var(--ink-soft); text-transform:uppercase; letter-spacing:.04em; margin-top:2px;}
  .stat.select .n{ color:var(--select);} .stat.review .n{ color:var(--review);} .stat.reject .n{ color:var(--reject);}
  .toolbar{
    display:flex; gap:10px; align-items:center; padding:0 32px 16px; flex-wrap:wrap;
  }
  .toolbar input[type=text]{
    flex:1; min-width:200px; padding:8px 12px; border:1px solid var(--line); border-radius:6px;
    font-size:13px; background:var(--card);
  }
  .chip{
    padding:6px 12px; border-radius:999px; border:1px solid var(--line); background:var(--card);
    font-size:12.5px; cursor:pointer; color:var(--ink-soft); user-select:none;
  }
  .chip.active{ border-color:transparent; color:#fff; }
  .chip[data-val="select"].active{ background:var(--select); }
  .chip[data-val="review"].active{ background:var(--review); }
  .chip[data-val="reject"].active{ background:var(--reject); }
  .chip[data-val="all"].active{ background:var(--ink); }
  select{ padding:7px 10px; border:1px solid var(--line); border-radius:6px; background:var(--card); font-size:12.5px; color:var(--ink);}
  main{ padding:0 32px 40px; }
  table{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
  thead th{
    text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--ink-soft);
    padding:10px 14px; border-bottom:1px solid var(--line); background:#F3F1EA; white-space:nowrap;
  }
  tbody td{ padding:11px 14px; border-bottom:1px solid var(--line); vertical-align:top; }
  tbody tr:last-child td{ border-bottom:none; }
  tbody tr{ cursor:pointer; }
  tbody tr:hover{ background:#F6F5F0; }
  .q-preview{ max-width:360px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .reason-preview{ max-width:320px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--ink-soft); }
  .badge{
    display:inline-block; padding:3px 9px; border-radius:5px; font-size:11px; font-weight:600;
    text-transform:uppercase; letter-spacing:.03em;
  }
  .badge.select{ background:var(--select-bg); color:var(--select); }
  .badge.review{ background:var(--review-bg); color:var(--review); }
  .badge.reject{ background:var(--reject-bg); color:var(--reject); }
  .badge.flag{ background:var(--reject-bg); color:var(--reject); }
  .badge.pass{ background:var(--select-bg); color:var(--select); }
  .badge.skipped, .badge.error{ background:var(--review-bg); color:var(--review); }
  .level{ font-size:12px; }
  .level .arrow{ color:var(--ink-soft); margin:0 3px; }
  .level .changed{ color:var(--reject); font-weight:600; }
  .empty{ text-align:center; padding:60px 20px; color:var(--ink-soft); }
  /* detail drawer */
  .backdrop{
    display:none; position:fixed; inset:0; background:rgba(27,36,48,.45); z-index:10;
    align-items:flex-start; justify-content:center; padding:40px 20px; overflow:auto;
  }
  .backdrop.open{ display:flex; }
  .drawer{
    background:var(--paper); border-radius:10px; max-width:720px; width:100%;
    padding:28px 30px 30px; box-shadow:0 20px 60px rgba(0,0,0,.25);
  }
  .drawer h2{ font-size:15px; margin:0 0 2px; }
  .drawer .drawer-sub{ font-size:12px; color:var(--ink-soft); margin-bottom:18px; }
  .drawer-close{
    float:right; background:none; border:1px solid var(--line); border-radius:6px; width:28px; height:28px;
    cursor:pointer; font-size:14px; color:var(--ink-soft);
  }
  .block{ margin-bottom:16px; }
  .block .k{ font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--ink-soft); margin-bottom:5px; }
  .block .v{ font-size:13.5px; white-space:pre-wrap; }
  .gates{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:16px; }
  .gate{ border:1px solid var(--line); border-radius:8px; padding:10px 12px; background:var(--card); }
  .gate .k{ font-size:10.5px; text-transform:uppercase; color:var(--ink-soft); letter-spacing:.03em; margin-bottom:6px;}
  .flags{ font-size:12px; color:var(--ink-soft); margin-top:6px; }
  footer{ padding:16px 32px 32px; color:var(--ink-soft); font-size:11.5px; }
</style>
</head>
<body>

<header>
  <div>
    <h1>QC Review Queue</h1>
    <div class="sub">Content, level and fairness QC &mdash; final call on whether an item goes to a student</div>
  </div>
  <div class="sub" id="generated-sub"></div>
</header>

<div class="stats" id="stats"></div>

<div class="toolbar">
  <input type="text" id="search" placeholder="Search text, reasons...">
  <div class="chip active" data-val="all">All</div>
  <div class="chip" data-val="select">Select</div>
  <div class="chip" data-val="review">Review</div>
  <div class="chip" data-val="reject">Reject</div>
  <select id="skill-filter"><option value="">All skills</option></select>
  <select id="level-filter"><option value="">All levels</option></select>
</div>

<main>
  <table>
    <thead>
      <tr>
        <th>Question ID</th><th>Skill</th><th>Text</th><th>Level</th>
        <th>Decision</th><th>Reason</th>
      </tr>
    </thead>
    <tbody id="rows"></tbody>
  </table>
  <div class="empty" id="empty" style="display:none;">No items match these filters.</div>
</main>

<footer>Click any row for the full QC breakdown (content / fairness gates, rule flags, level reasoning).</footer>

<div class="backdrop" id="backdrop">
  <div class="drawer" id="drawer"></div>
</div>

<script>
const DATA = __DATA_JSON__;

const els = {
  stats: document.getElementById('stats'),
  rows: document.getElementById('rows'),
  empty: document.getElementById('empty'),
  search: document.getElementById('search'),
  skillFilter: document.getElementById('skill-filter'),
  levelFilter: document.getElementById('level-filter'),
  backdrop: document.getElementById('backdrop'),
  drawer: document.getElementById('drawer'),
  genSub: document.getElementById('generated-sub'),
};

let activeDecision = 'all';

function esc(s){ return (s===undefined||s===null) ? '' : String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

function decisionOf(row){ return (row.final_decision || '').toLowerCase(); }
function badgeClass(value){
  const v = (value || '').toUpperCase();
  if (v.includes('SELECT')) return 'select';
  if (v.includes('REVIEW')) return 'review';
  if (v.includes('REJECT')) return 'reject';
  if (v.includes('FLAG')) return 'flag';
  if (v.includes('PASS')) return 'pass';
  if (v.includes('SKIP')) return 'skipped';
  if (v.includes('ERROR')) return 'error';
  return '';
}

function buildStats(){
  const counts = {select:0, review:0, reject:0};
  DATA.forEach(r => { const d = decisionOf(r); if (counts[d] !== undefined) counts[d]++; });
  const total = DATA.length;
  els.stats.innerHTML = `
    <div class="stat"><div class="n">${total}</div><div class="l">Total items</div></div>
    <div class="stat select"><div class="n">${counts.select}</div><div class="l">Select</div></div>
    <div class="stat review"><div class="n">${counts.review}</div><div class="l">Review</div></div>
    <div class="stat reject"><div class="n">${counts.reject}</div><div class="l">Reject</div></div>
  `;
  els.genSub.textContent = `Generated __GENERATED_AT__ &middot; ${total} items reviewed`;
}

function buildFilterOptions(){
  const skills = [...new Set(DATA.map(r => r.skill).filter(Boolean))].sort();
  const levels = [...new Set(DATA.map(r => r.predicted_level || r.level).filter(Boolean))].sort();
  skills.forEach(s => els.skillFilter.insertAdjacentHTML('beforeend', `<option value="${esc(s)}">${esc(s)}</option>`));
  levels.forEach(l => els.levelFilter.insertAdjacentHTML('beforeend', `<option value="${esc(l)}">${esc(l)}</option>`));
}

function levelCell(row){
  const labeled = row.level || '—';
  const predicted = row.predicted_level || '—';
  const changed = labeled !== '—' && predicted !== '—' && labeled !== predicted;
  return `<span class="level mono">${esc(labeled)}<span class="arrow">&rarr;</span><span class="${changed ? 'changed' : ''}">${esc(predicted)}</span></span>`;
}

function textPreview(row){
  return row.text || '(no text)';
}

function matchesFilters(row){
  if (activeDecision !== 'all' && decisionOf(row) !== activeDecision) return false;
  if (els.skillFilter.value && row.skill !== els.skillFilter.value) return false;
  if (els.levelFilter.value && (row.predicted_level || row.level) !== els.levelFilter.value) return false;
  const q = els.search.value.trim().toLowerCase();
  if (q){
    const hay = [textPreview(row), row.final_decision_reason, row.qc_reason,
                 row.fairness_qc_reason, row.metric_flags].join(' ').toLowerCase();
    if (!hay.includes(q)) return false;
  }
  return true;
}

function render(){
  const filtered = DATA.filter(matchesFilters);
  els.empty.style.display = filtered.length ? 'none' : 'block';
  els.rows.innerHTML = filtered.map((row) => {
    const idx = DATA.indexOf(row);
    const decision = decisionOf(row) || 'review';
    return `
      <tr data-idx="${idx}">
        <td class="mono">${esc(row.question_id || idx)}</td>
        <td>${esc(row.skill)}</td>
        <td class="q-preview" title="${esc(textPreview(row))}">${esc(textPreview(row))}</td>
        <td>${levelCell(row)}</td>
        <td><span class="badge ${decision}">${esc(decision)}</span></td>
        <td class="reason-preview" title="${esc(row.final_decision_reason)}">${esc(row.final_decision_reason)}</td>
      </tr>`;
  }).join('');
  [...els.rows.querySelectorAll('tr')].forEach(tr => {
    tr.addEventListener('click', () => openDrawer(parseInt(tr.dataset.idx, 10)));
  });
}

function gateBlock(label, value, reason){
  const cls = badgeClass(value) || 'review';
  return `<div class="gate">
    <div class="k">${label}</div>
    <span class="badge ${cls}">${esc(value || 'n/a')}</span>
    <div class="flags">${esc(reason || '')}</div>
  </div>`;
}

function openDrawer(idx){
  const row = DATA[idx];
  const decision = decisionOf(row) || 'review';
  els.drawer.innerHTML = `
    <button class="drawer-close" onclick="closeDrawer()">&times;</button>
    <h2>${esc(row.question_id || idx)} &mdash; <span class="badge ${decision}">${esc(decision)}</span></h2>
    <div class="drawer-sub">${esc(row.skill)} &middot; labeled ${esc(row.level||'—')} &rarr; predicted ${esc(row.predicted_level||'—')}</div>

    <div class="block"><div class="k">Text</div><div class="v">${esc(textPreview(row))}</div></div>

    <div class="gates">
      ${gateBlock('Content QC', row.qc_label, row.qc_reason)}
      ${gateBlock('Fairness', row.fairness_qc_label, row.fairness_qc_reason)}
    </div>

    <div class="block"><div class="k">Level reason</div><div class="v">${esc(row.level_change_reason)}</div></div>
    <div class="block"><div class="k">Rule-based flags</div><div class="v mono">${esc(row.metric_flags)}</div></div>
    ${row.time_taken_sec ? `<div class="block"><div class="k">Time taken</div><div class="v mono">${esc(row.time_taken_sec)}s</div></div>` : ''}
    <div class="block"><div class="k">Final decision reason</div><div class="v"><strong>${esc(row.final_decision_reason)}</strong></div></div>
  `;
  els.backdrop.classList.add('open');
}
function closeDrawer(){ els.backdrop.classList.remove('open'); }
els.backdrop.addEventListener('click', e => { if (e.target === els.backdrop) closeDrawer(); });

[...document.querySelectorAll('.chip')].forEach(chip => {
  chip.addEventListener('click', () => {
    document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
    chip.classList.add('active');
    activeDecision = chip.dataset.val;
    render();
  });
});
els.search.addEventListener('input', render);
els.skillFilter.addEventListener('change', render);
els.levelFilter.addEventListener('change', render);

buildStats();
buildFilterOptions();
render();
</script>
</body>
</html>
"""

    html = (
        html
        .replace("__GENERATED_AT__", _html_escape(time.strftime("%Y-%m-%d %H:%M:%S")))
        .replace("__DATA_JSON__", data_json)
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def run_full_qc(item: dict, flags: list, level_tolerance: int,
                 metric_flag_detail_map: dict = None) -> dict:
    """Runs CEFR classification + rubric scoring + verdict for one item.
    Returns a dict of all the derived QC columns (used for both the initial
    pass and for re-checking items after --reform). Metric columns not
    configured for this item's skill are set to 'n/a' so output stays
    consistent across rows with different skills.

    metric_flag_detail_map is the {flag_name: detail_sentence} dict from
    metric_flag_details(item, flags) -- pass it in if the caller already
    computed it (main() does, to reuse for compute_final_decision), or
    leave it out and it's computed here from item/flags directly."""
    if metric_flag_detail_map is None:
        metric_flag_detail_map = metric_flag_details(item, flags)

    predicted_level, level_reason = classify_level(item)
    labeled_level = (item.get("level") or "").strip().upper()
    level_match = _level_within_range(labeled_level, predicted_level, level_tolerance)

    if predicted_level not in ("ERROR", "UNKNOWN") and labeled_level == predicted_level:
        # No change in CEFR level -- report a clean "pass" instead of
        # echoing the classifier's reasoning, since there's nothing to
        # explain when the labeled and predicted levels already agree.
        level_change_reason = "pass"
    elif labeled_level and predicted_level not in ("ERROR", "UNKNOWN") and labeled_level != predicted_level:
        level_change_reason = "Changed from %s to %s: %s" % (labeled_level, predicted_level, level_reason)
    else:
        level_change_reason = level_reason

    skill = item.get("skill", "")
    scores = score_metrics(item, predicted_level, flags)
    label, reason, level_only = qc_verdict(level_match, scores, flags, skill,
                                            metric_flag_detail_map=metric_flag_detail_map)

    out = {
        "predicted_level": predicted_level,
        "level_match": level_match,
        "level_change_reason": level_change_reason,
    }
    for key in ALL_METRIC_KEYS:
        out[key] = scores[key] if key in scores else "n/a"
    out["qc_label"] = label
    out["qc_reason"] = reason
    # True only when the content FLAG is caused solely by the predicted
    # CEFR level falling outside the labeled level's range, with no other
    # content defects. Used by compute_final_decision() to route level-only
    # mismatches to REVIEW instead of an auto-REJECT.
    out["qc_level_only"] = level_only
    return out


RULE_FLAG_DESCRIPTIONS = {
    "answer_whitespace": "the answer field has leading or trailing whitespace that needs to be trimmed",
    "text_whitespace": "the item text has leading or trailing whitespace that needs to be trimmed",
    "missing_text": "the item is missing its question/prompt text entirely",
    "invalid_level": "the labeled CEFR level is not one of the recognized values (A1-C2)",
    "no_blank_marker": "the item is typed as fill-in-the-blank but contains no blank marker ('___' or '____') in the text",
    "missing_options": "the item is typed as multiple-choice but has no answer options listed",
    "missing_audio_or_transcript": "this item's skill requires an audio clip or transcript, but neither is present",
    "missing_end_punctuation": "the item text does not end with terminal punctuation (., ?, or !)",
    "answer_too_long_for_level": "the answer is longer than expected for an A1/A2-level item (more than 4 words)",
    "weak_passage": "the item's text is too short/thin to serve as a substantive passage or prompt",
    "poor_construction": "the item's text has malformed sentence construction (e.g. a repeated word or unbalanced quotes/brackets)",
}

FAIRNESS_FLAG_DESCRIPTIONS = {
    "requires_specialist_knowledge": "the item assumes specialist or cultural knowledge that a general test-taker can't be expected to have",
    "cultural_bias": "the item shows a cultural bias that could disadvantage test-takers from other backgrounds",
    "gender_bias": "the item shows a gender bias in its framing, roles, or assumptions",
    "sensitive_content_review_needed": "the item touches on sensitive content that needs human review before being shown to a student",
    "fairness_check_error": "the fairness check itself could not be completed (judge model call failed)",
}


def _cap(s: str) -> str:
    """Capitalizes just the first character, leaving the rest of the
    string (which may contain its own capitalized words, e.g. 'CEFR')
    untouched -- unlike str.capitalize(), which forces everything else
    to lowercase."""
    return s[0].upper() + s[1:] if s else s


def _elaborate_reason_sentences(raw_reason: str) -> str:
    """Expands a ';'-joined terse reason string (as produced by
    qc_verdict()) into a properly punctuated, capitalized run of full
    sentences, so final_decision_reason is self-contained and doesn't
    require cross-referencing qc_reason separately."""
    if not raw_reason:
        return ""
    parts = [p.strip() for p in raw_reason.split(";") if p.strip()]
    sentences = [_cap(p if p.endswith(".") else p + ".") for p in parts]
    return " ".join(sentences)


def _elaborate_flag_list(flags: list, descriptions: dict) -> str:
    """Expands a list of short flag keywords (rule flags or fairness
    flags) into full descriptive sentences using the given lookup table,
    falling back to a de-slugified version of the flag name for anything
    not in the table (so a newly added flag never silently disappears
    from the explanation)."""
    if not flags:
        return ""
    sentences = [
        _cap(descriptions.get(f, f.replace("_", " ")) + ".")
        for f in flags
    ]
    return " ".join(sentences)


def compute_final_decision(qc_label: str, fairness_qc_label: str, metric_flags: str,
                            qc_reason: str = "", fairness_qc_reason: str = "",
                            qc_level_only: bool = False,
                            metric_flag_detail_map: dict = None) -> tuple:
    """Rolls the two independent gates (content QC, fairness) up into a
    single actionable verdict -- can this item be given to a student as-is
    -- plus a reason string for that verdict.

      - REJECT: either gate actively FLAGged the item -- except a content
                FLAG caused solely by the CEFR level (qc_level_only=True),
                which is downgraded to REVIEW (see below).
      - SELECT: every gate that ran came back PASS (or, if no LLM gate ran
                at all -- e.g. --skip-llm / skill filtered out -- the
                rule-based metric_flags checks came back clean). Safe to
                give to a student.
      - REVIEW: no gate flagged it (or only a level-only content FLAG did),
                but coverage is incomplete (a gate SKIPPED or ERRORed, or
                rule flags fired with no LLM gate to confirm) -- needs a
                human glance before it's given to a student, rather than an
                auto select/reject.

    metric_flag_detail_map, if provided, is the {flag_name: detail_sentence}
    dict from metric_flag_details() -- when present, the "rules: ..." part
    of final_decision_reason uses those full what+where sentences instead
    of the bare flag keywords, so the reason is self-contained.

    The content/fairness portions of the reason (qc_reason/fairness_qc_reason)
    are run through _elaborate_reason_sentences() so each terse, ';'-joined
    fragment becomes its own capitalized sentence. This matters most for
    grammar: qc_verdict() now emits one "grammar error: ..." reason per
    error found (each already quoting the offending phrase and explaining
    why it's wrong, per the judge-model prompt), so final_decision_reason
    ends up as a per-error list rather than a single "N grammar errors"
    count.

    Returns (final_decision, final_decision_reason).

    NOTE: qc_label comes back bare ("FLAG"/"PASS") from qc_verdict(), but
    fairness_qc_label comes back prefixed ("FAIRNESS_FLAG"/"FAIRNESS_PASS",
    or "FAIRNESS_ERROR" when the judge model call itself failed). Matching
    must account for both forms."""
    gates = [
        ("content", qc_label, qc_reason),
        ("fairness", fairness_qc_label, fairness_qc_reason),
    ]
    labels = [qc_label, fairness_qc_label]

    ignored_rule_flags = {"missing_end_punctuation"}
    effective_flags = [
        f for f in (metric_flags.split(", ") if metric_flags not in ("", "none") else [])
        if f not in ignored_rule_flags
    ]
    rule_flags_present = bool(effective_flags)

    combined_parts = []
    for name, label, reason in gates:
        if reason and (label.endswith("FLAG") or label.endswith("ERROR")):
            combined_parts.append("%s: %s" % (name, _elaborate_reason_sentences(reason)))
    # Only append a standalone "rules: ..." block when the content gate
    # never ran (qc_label == "", e.g. --skip-llm or a skill filtered out of
    # this run) -- in that case qc_reason is blank and rule flags are the
    # only signal available. When the content gate DID run, qc_verdict()
    # already folds each rule flag's full detail sentence into qc_reason
    # itself (via the same metric_flag_detail_map), so it's already covered
    # by the "content: ..." part above; appending it again here would just
    # duplicate the same sentences.
    if rule_flags_present and qc_label == "":
        if metric_flag_detail_map:
            detail_sentences = [
                metric_flag_detail_map.get(f, f.replace("_", " ") + ".")
                for f in effective_flags
            ]
            combined_parts.append("rules: " + " ".join(detail_sentences))
        else:
            combined_parts.append("rules: %s" % ", ".join(effective_flags))
    combined_reason = "; ".join(combined_parts)

    # A content FLAG caused solely by the predicted CEFR level falling
    # outside the labeled level's range doesn't count as a hard-reject
    # trigger -- level mismatches are a judgment call worth a human look,
    # not a content defect. Fairness FLAGs are unaffected by this.
    content_is_hard_reject = qc_label == "FLAG" and not qc_level_only
    fairness_is_hard_reject = (fairness_qc_label or "").endswith("FLAG")

    if content_is_hard_reject or fairness_is_hard_reject:
        return "REJECT", "REJECT -- " + (combined_reason or "gate flagged this item")

    if qc_label == "FLAG" and qc_level_only:
        return "REVIEW", "REVIEW -- " + (combined_reason or "CEFR level needs review")

    if all(l == "" for l in labels):
        if rule_flags_present:
            return "REVIEW", "REVIEW -- " + combined_reason
        return "SELECT", "SELECT -- rule checks clean"

    if any(l for l in labels) and all(l.endswith("PASS") for l in labels if l):
        return "SELECT", "SELECT -- passed"

    return "REVIEW", "REVIEW -- " + (combined_reason or "incomplete coverage")


# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="QC-flag items on Accuracy/Grammar/Clarity/Completeness/CEFR Level/Fairness using deepeval + Gemini as the judge model")
    ap.add_argument("--input", required=True, help="Path to input .json or .csv file")
    ap.add_argument("--output", required=True, help="Path to output .json or .csv file")
    ap.add_argument("--level-tolerance", type=int, default=LEVEL_TOLERANCE,
                     help="How many CEFR tiers off is still considered 'within range' (default: 0 = exact match)")
    ap.add_argument("--skip-llm", action="store_true", help="Only run rule-based checks (no judge model call)")
    ap.add_argument("--reform", action="store_true",
                     help="For items that FLAG, ask the judge model to rewrite them and re-run QC on the rewrite (up to --reform-attempts times)")
    ap.add_argument("--reform-attempts", type=int, default=2,
                     help="Max reform+re-check cycles per flagged item (default: 2)")
    ap.add_argument("--skill-filter", default="reading,writing,speaking",
                     help="Only run LLM-based QC (CEFR classification, scoring, reform) on items "
                          "whose skill is in this comma-separated list (case-insensitive). Other "
                          "skills still get rule-based checks but are marked SKIPPED for the LLM "
                          "steps. Use 'all' to disable filtering. (default: reading,writing,speaking)")
    ap.add_argument("--html-output", default=None,
                     help="Path to write a self-contained HTML review report (SELECT/REVIEW/REJECT "
                          "summary, filters, search). Defaults to --output with its extension "
                          "replaced by .html. Pass 'none' to skip writing an HTML report.")
    ap.add_argument("--timing", action="store_true",
                     help="Time how long each item takes to QC and record it in a 'time_taken_sec' column")
    ap.add_argument("--checkpoint-every", type=int, default=10,
                     help="Save resumable progress, and rewrite the CSV/JSON output and the HTML "
                          "report together, every N items. Set to 0 to disable checkpointing. (default: 10)")
    ap.add_argument("--checkpoint-file", default=None,
                     help="Path to the checkpoint file. Defaults to --output with '.checkpoint.json' appended.")
    ap.add_argument("--resume", action="store_true",
                     help="Resume from the checkpoint file if one exists, instead of starting over")
    ap.add_argument("--workers", type=int, default=1,
                     help="Number of items to QC concurrently (each makes its own judge-model "
                          "calls over the network, so this is I/O-bound and safe to parallelize). "
                          "Results are still written/checkpointed in original item order. "
                          "Start around 4-8 and raise it if the Gemini API isn't rate-limiting you; "
                          "lower it if you start seeing retries/timeouts. Default: 1 (sequential, "
                          "same behavior as before).")
    args = ap.parse_args()

    items = load_items(args.input)
    if not items:
        print("No items found in input file.", file=sys.stderr)
        sys.exit(1)
    total = len(items)

    html_path = None
    if (args.html_output or "").strip().lower() != "none":
        html_path = args.html_output
        if not html_path:
            base, _ext = os.path.splitext(args.output)
            html_path = base + ".html"

    checkpoint_path = args.checkpoint_file or _default_checkpoint_path(args.output)
    checkpointing_enabled = args.checkpoint_every > 0

    results = []
    start_index = 0
    if args.resume and checkpointing_enabled:
        loaded = load_checkpoint(checkpoint_path, total)
        if loaded is not None:
            results = loaded
            start_index = len(results)
            print("Resuming from checkpoint: %d/%d items already done." % (start_index, total))

    def _checkpoint_if_due():
        """Saves resumable progress and rewrites the CSV/JSON output plus
        the HTML report together, so the two stay in sync at every
        checkpoint rather than only at the very end of the run."""
        if not checkpointing_enabled or len(results) % args.checkpoint_every != 0:
            return
        save_checkpoint(checkpoint_path, total, results)
        save_results(results, args.output)
        if html_path:
            save_html_report(results, html_path)
        print("    [checkpoint] %d/%d done -> %s%s" % (
            len(results), total, args.output, (" + " + html_path) if html_path else ""
        ))

    def _process_item(i, item, args, total):
        item_start_time = time.time() if args.timing else None
        flags = metric_flags(item)
        flag_detail_map = metric_flag_details(item, flags)
        out = dict(item)
        out["metric_flags"] = ", ".join(flags) if flags else "none"

        item_skill = (item.get("skill", "") or "").strip().lower()
        filter_list = [s.strip().lower() for s in args.skill_filter.split(",")]
        skill_ok = ("all" in filter_list) or (item_skill in filter_list)

        if args.skip_llm or not skill_ok:
            blank_metrics = {key: "" for key in ALL_METRIC_KEYS}
            out.update({
                "predicted_level": "", "level_match": "",
                "level_change_reason": "",
                **blank_metrics,
                "qc_label": "" if args.skip_llm else "SKIPPED",
                "qc_reason": "" if args.skip_llm else (
                    "LLM QC not applicable: skill is '%s', filter is '%s'"
                    % (item.get("skill", ""), args.skill_filter)
                ),
                "fairness_flags": "",
                "fairness_qc_label": "" if args.skip_llm else "SKIPPED",
                "fairness_qc_reason": "" if args.skip_llm else (
                    "Fairness QC not applicable: skill is '%s', filter is '%s'"
                    % (item.get("skill", ""), args.skill_filter)
                ),
            })
            out["final_decision"], out["final_decision_reason"] = compute_final_decision(
                out["qc_label"], out["fairness_qc_label"], out["metric_flags"],
                out["qc_reason"], out["fairness_qc_reason"],
                metric_flag_detail_map=flag_detail_map,
            )
            if args.timing:
                out["time_taken_sec"] = round(time.time() - item_start_time, 3)
            if args.skip_llm:
                print("[%d/%d] rule flags only: %s" % (i, total, out["metric_flags"]))
            else:
                print("[%d/%d] SKIPPED (skill='%s' not in filter '%s')" % (
                    i, total, item.get("skill", ""), args.skill_filter))
            return out

        qc = run_full_qc(item, flags, args.level_tolerance, metric_flag_detail_map=flag_detail_map)
        out.update(qc)

        # Fairness gate runs independently of the content QC above -- its
        # own call, its own flags, its own label. An item can PASS content
        # QC and still get FAIRNESS_FLAG, or vice versa.
        fairness_result = evaluate_fairness(item)
        f_flags = fairness_flags(item, fairness_result)
        f_label, f_reason = fairness_qc_gate(f_flags, fairness_result.get("reason", ""))
        out["fairness_flags"] = ", ".join(f_flags) if f_flags else "none"
        out["fairness_qc_label"] = f_label
        out["fairness_qc_reason"] = f_reason

        out["final_decision"], out["final_decision_reason"] = compute_final_decision(
            out["qc_label"], out["fairness_qc_label"], out["metric_flags"],
            out["qc_reason"], out["fairness_qc_reason"],
            qc_level_only=out.get("qc_level_only", False),
            metric_flag_detail_map=flag_detail_map,
        )

        print("[%d/%d] %s (predicted=%s, labeled=%s) -> %s" % (
            i, total, qc["qc_label"], qc["predicted_level"],
            (item.get("level") or "").strip().upper() or "?", qc["qc_reason"]
        ))
        print("    fairness: %s -> %s" % (f_label, f_reason))

        # ---- Optional: ask the judge model to rewrite the item until it passes ----
        if args.reform and qc["qc_label"] == "FLAG":
            out["original_text"] = item.get("text", "")
            out["original_answer"] = item.get("answer", "")
            out["reform_change_summary"] = ""

            working_item = dict(item)
            for attempt in range(1, args.reform_attempts + 1):
                print("    reforming (attempt %d/%d)..." % (attempt, args.reform_attempts))
                rewrite = reform_item(working_item, qc["qc_reason"])

                if rewrite["change_summary"].startswith("REFORM FAILED"):
                    out["reform_change_summary"] = rewrite["change_summary"]
                    print("    %s" % rewrite["change_summary"])
                    break

                working_item["text"] = rewrite["text"]
                working_item["answer"] = rewrite["answer"]
                if rewrite.get("options"):
                    working_item["options"] = rewrite["options"]
                out["reform_change_summary"] = rewrite["change_summary"]

                new_flags = metric_flags(working_item)
                flag_detail_map = metric_flag_details(working_item, new_flags)
                new_qc = run_full_qc(working_item, new_flags, args.level_tolerance, metric_flag_detail_map=flag_detail_map)

                if new_qc["qc_label"] == "PASS":
                    print("    reform succeeded on attempt %d" % attempt)
                    out["text"] = working_item["text"]
                    out["answer"] = working_item["answer"]
                    out["options"] = working_item.get("options", out.get("options"))
                    out["metric_flags"] = ", ".join(new_flags) if new_flags else "none"
                    out.update(new_qc)
                    break
                else:
                    print("    still flagged: %s" % new_qc["qc_reason"])
                    out["text"] = working_item["text"]
                    out["answer"] = working_item["answer"]
                    out["options"] = working_item.get("options", out.get("options"))
                    out["metric_flags"] = ", ".join(new_flags) if new_flags else "none"
                    out.update(new_qc)

            # Reform only re-runs content QC, not the fairness gate --
            # recompute the roll-up since out["qc_label"] may have changed.
            out["final_decision"], out["final_decision_reason"] = compute_final_decision(
                out["qc_label"], out["fairness_qc_label"], out["metric_flags"],
                out["qc_reason"], out["fairness_qc_reason"],
                qc_level_only=out.get("qc_level_only", False),
                metric_flag_detail_map=flag_detail_map,
            )

        if args.timing:
            out["time_taken_sec"] = round(time.time() - item_start_time, 3)
        return out

    remaining_indices = list(range(start_index + 1, total + 1))
    remaining_items = items[start_index:]

    if args.workers and args.workers > 1:
        print("Running with %d concurrent workers..." % args.workers)
        worker_fn = functools.partial(_process_item, args=args, total=total)
        # executor.map yields results strictly in input order (it still runs
        # items concurrently under the hood), so checkpointing here still
        # only ever sees a complete, in-order prefix of results -- same
        # resume guarantee as the sequential path below.
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            for out in executor.map(worker_fn, remaining_indices, remaining_items):
                results.append(out)
                _checkpoint_if_due()
    else:
        for i, item in zip(remaining_indices, remaining_items):
            out = _process_item(i, item, args, total)
            results.append(out)
            _checkpoint_if_due()

    save_results(results, args.output)
    print("\nDone. Wrote %d results to %s" % (len(results), args.output))

    if html_path:
        save_html_report(results, html_path)
        print("Wrote HTML report to %s" % html_path)

    if checkpointing_enabled:
        clear_checkpoint(checkpoint_path)


if __name__ == "__main__":
    main()