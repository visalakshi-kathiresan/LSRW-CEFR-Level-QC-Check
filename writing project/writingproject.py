import argparse
import csv
import json
import os
import re
import sys
import time

from deepeval.models import GeminiModel
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# ----------------------------------------------------------------------
# deepeval judge model + GEval metric cache
# ----------------------------------------------------------------------
# Every LLM call in this file -- score_metrics(), classify_level(), and
# reform_item() -- now goes through the same deepeval-wrapped judge model
# below (_JUDGE_MODEL), the hosted gemma-4-26b-a4b-it model via the Gemini
# API (Google AI Studio). score_metrics() uses it through deepeval's GEval
# metric class (_get_geval_metric()); classify_level() and reform_item()
# use it directly through deepeval's DeepEvalBaseLLM.generate() interface
# (_judge_call()), since classification/rewrite tasks don't fit GEval's
# "score this against a criteria" shape but still benefit from going
# through the same deepeval model wrapper (retries, provider abstraction)
# instead of a hand-rolled HTTP call to a separate local backend. There is
# no local/Ollama path left anywhere in this file. Requires GOOGLE_API_KEY
# to be set in the environment (or pass api_key= directly below) and
# `pip install google-genai`. GEval metrics are stateless once
# constructed, so they're built once per (name, criteria) pair and reused
# across every item/call.
_JUDGE_MODEL = GeminiModel(
    model="gemma-4-26b-a4b-it",
    temperature=0.2,
)

_GEVAL_METRIC_CACHE: dict = {}


def _get_geval_metric(name: str, criteria: str) -> GEval:
    if name not in _GEVAL_METRIC_CACHE:
        _GEVAL_METRIC_CACHE[name] = GEval(
            name=name,
            criteria=criteria,
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=_JUDGE_MODEL,
        )
    return _GEVAL_METRIC_CACHE[name]


def _judge_call(prompt: str, retries: int = 2) -> str:
    """Calls _JUDGE_MODEL directly through deepeval's DeepEvalBaseLLM.
    generate() interface -- the same judge model and same deepeval model
    wrapper that _get_geval_metric() hands to GEval above, just invoked
    without the GEval scoring machinery, for the two classification/
    generation tasks (classify_level, reform_item) that need a raw text/
    JSON response rather than a 0-1 criteria score. Replaces the old
    direct-to-Ollama HTTP call; retry/backoff behavior mirrors what that
    call used to do."""
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
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("Could not reach judge model (Gemini via deepeval): %s" % last_err)


def _measure_geval(geval: GEval, test_case: LLMTestCase, retries: int = 2):
    """Runs geval.measure() with the same retry/backoff pattern as
    _judge_call(), since GEval's own measure() call doesn't retry on
    transient provider errors. Returns (score, reason, is_error). On
    repeated failure, is_error=True and reason is a short one-line marker
    (not the raw exception/JSON dump) so a transient API failure doesn't
    get displayed as if it were the judge's actual reasoning."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            geval.measure(test_case)
            score = geval.score if geval.score is not None else 0.0
            return score, (geval.reason or ""), False
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    short = str(last_err).splitlines()[0][:120]
    return 0.0, "[scoring error after %d attempt(s): %s]" % (retries + 1, short), True


CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


SKILL_METRIC_CONFIG = {
    "reading": {
        "metrics": [
        ],
    },
    "writing": {
        "metrics": [
        ],
    },
    # Fallback used for any skill not explicitly listed above (falls back to
    # the reading config, since reading is the primary skill in scope).
    "_default": {
        "metrics": [
        ],
    },
}

# Severity of each deterministic rule flag (see metric_flags()) when it
# feeds into qc_verdict(): "reject" is reserved for the item actively
# conveying WRONG information to the student. None of these rule flags
# assert that the item is wrong -- they're formatting/structural issues
# (missing options, no blank marker, stray whitespace, etc.) that are
# usable-but-imperfect, not incorrect, so they route to REVIEW for a human
# to fix rather than auto-REJECT. Any future flag not listed here also
# defaults to "review" (see qc_verdict()).
#
# Fairness flags (see evaluate_fairness()) are "reject", not "review":
# unlike the formatting-only rule flags above, a genuine fairness issue
# (cultural bias, gender bias, an assumption of specialist/cultural
# knowledge, or sensitive content) makes the item actively unfit to show a
# student -- it isn't something a human reviewer should merely be nudged
# to look at, it should stop the item automatically. fairness_check_failed
# (the judge-model-unreachable/unparseable case) stays "review" since
# that's an infra failure, not an actual fairness judgment on the item.
FLAG_SEVERITY = {
    "missing_text": "review",
    "invalid_level": "review",
    "missing_options": "review",
    "no_blank_marker": "review",
    "answer_whitespace": "review",
    "missing_end_punctuation": "review",
    "answer_too_long_for_level": "review",
    "requires_specialist_or_cultural_knowledge": "reject",
    "cultural_bias": "reject",
    "gender_bias": "reject",
    "sensitive_content": "reject",
    "fairness_check_failed": "review",
}

LEVEL_TOLERANCE = 0          # 0 = exact CEFR match required, 1 = allow one tier off (e.g. A2<->B1)

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
    text = (item.get("text") or "").strip()
    answer = (item.get("answer") or "")
    level = (item.get("level") or "").strip().upper()
    itype = (item.get("type") or "").strip().lower()
    skill = (item.get("skill") or "").strip().lower()

    if answer != answer.strip():
        flags.append("answer_whitespace")
    if not text:
        flags.append("missing_text")
    if level not in CEFR_LEVELS:
        flags.append("invalid_level")
    if itype == "fill_up" and "___" not in text and "____" not in text:
        flags.append("no_blank_marker")
    if itype in ("mcq", "multiple_choice") and not item.get("options"):
        flags.append("missing_options")
    if text and not re.search(r"[.?!]\s*$", text.replace("___", "").replace("____", "")):
        flags.append("missing_end_punctuation")
    if level in ("A1", "A2") and len(answer.split()) > 4:
        flags.append("answer_too_long_for_level")

    return flags


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
    name. Returns {flag_name: detail_sentence}; qc_verdict() uses these (if
    given) so qc_reason is self-contained and detailed instead of just
    listing bare keywords like 'rule flag: no_blank_marker'."""
    raw_answer = item.get("answer") or ""
    answer = raw_answer.strip()
    raw_level = item.get("level") or ""
    level = raw_level.strip().upper()
    text = (item.get("text") or "").strip()
    itype = (item.get("type") or "").strip().lower()

    details = {}
    for f in flags:
        if f == "answer_whitespace":
            details[f] = (
                "In the 'answer' field: %s (raw value: %r)."
                % (_ws_location(raw_answer), raw_answer)
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
        else:
            details[f] = f.replace("_", " ") + "."
    return details


# ----------------------------------------------------------------------
# 2. Gemma (via deepeval + Google Gemini API) calls
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
- If completeness was flagged, add whatever minimal context is
  needed so the question can be answered on its own, without changing what
  is being tested.
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
    """CEFR classification call. Returns (level, reason). Asks the judge
    model for a JSON object (parsed defensively below) + retries, instead
    of a free-text LEVEL:/REASON: template, since small models like Gemma
    often drift from an exact text format but stay more consistent when
    asked to emit JSON directly in the prompt."""
    prompt = CEFR_PROMPT.format(
        descriptors=CEFR_DESCRIPTORS,
        skill=item.get("skill", ""),
        type=(item.get("type") or "").strip() or "unspecified / free-response",
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
            return "ERROR", "Judge model unreachable: %s" % e

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


FAIRNESS_PROMPT = """You are a fairness reviewer for a language-learning platform whose
students are international students from many countries and cultural backgrounds. Review
the item below for fairness issues that would disadvantage or alienate a general
international student. Check specifically for:

1. requires_specialist_or_cultural_knowledge - the item assumes knowledge (trivia, local
   customs, idioms, specific historical/geographic facts) that a general international
   student could not be expected to know, beyond the language skill actually being tested.
2. cultural_bias - the item asserts one culture's norms, holidays, food, institutions, etc.
   as the default/universal case, or leans on a stereotype about a culture or nationality.
3. gender_bias - the item relies on a gender stereotype, or genders/ungenders a role with
   no reason to do so.
4. sensitive_content - the item asks the student to read or engage with distressing,
   violent, political, or otherwise inappropriate content.

Item:
Skill: {skill} | Type: {type} | Level: {level}
Text: {text}
Answer: {answer}
Options: {options}

Respond with ONLY a JSON object, nothing else, in this exact shape:
{{"fairness_flags": [<zero or more of: "requires_specialist_or_cultural_knowledge", "cultural_bias", "gender_bias", "sensitive_content">], "reasons": {{"<flag>": "<one short sentence on specifically what in the item triggered this flag>"}}}}
If there are no issues, return {{"fairness_flags": [], "reasons": {{}}}}.
"""

FAIRNESS_FLAG_NAMES = (
    "requires_specialist_or_cultural_knowledge",
    "cultural_bias",
    "gender_bias",
    "sensitive_content",
)


def evaluate_fairness(item: dict):
    """Judge-model fairness check for a writing item. Folded into the same
    rule-flag pipeline as metric_flags()/metric_flag_details() (see
    FLAG_SEVERITY and run_full_qc()) rather than being its own separate
    gate/column -- a fairness issue just becomes another flag that
    qc_verdict() rolls into the existing qc_label/qc_reason, the same way a
    rule flag like missing_options does.

    Checks the four categories described in FAIRNESS_PROMPT. Returns
    (flags, reasons):
      - flags: list of flag names (subset of FAIRNESS_FLAG_NAMES) that
        applied to this item.
      - reasons: {flag_name: detail_sentence}, in the same shape
        metric_flag_details() returns, so qc_verdict() can expand each
        flag into a full sentence in qc_reason instead of just the bare
        keyword.

    All four fairness flags are "reject" severity (see FLAG_SEVERITY) --
    a real fairness problem should auto-reject the item, not just route it
    to REVIEW. If the judge model call itself fails or returns something
    unparseable, that's reported as a single 'fairness_check_failed' flag
    (severity: review) so the failure is visible to a human instead of the
    item silently passing fairness with no check having actually run."""
    prompt = FAIRNESS_PROMPT.format(
        skill=item.get("skill", ""),
        type=(item.get("type") or "").strip() or "unspecified / free-response",
        level=item.get("level", ""),
        text=item.get("text", ""),
        answer=item.get("answer", ""),
        options=item.get("options", ""),
    )

    try:
        raw = _judge_call(prompt)
    except RuntimeError as e:
        return ["fairness_check_failed"], {
            "fairness_check_failed": "Fairness judge model unreachable: %s" % e
        }

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return ["fairness_check_failed"], {
            "fairness_check_failed": "Fairness judge returned no parseable JSON: %s" % raw[:150]
        }

    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return ["fairness_check_failed"], {
            "fairness_check_failed": "Fairness judge returned malformed JSON: %s" % raw[:150]
        }

    raw_flags = parsed.get("fairness_flags") or []
    raw_reasons = parsed.get("reasons") or {}
    if not isinstance(raw_flags, list) or not isinstance(raw_reasons, dict):
        return ["fairness_check_failed"], {
            "fairness_check_failed": "Fairness judge returned unexpected JSON shape: %s" % raw[:150]
        }

    flags = [f for f in raw_flags if f in FAIRNESS_FLAG_NAMES]
    reasons = {}
    for f in flags:
        detail = (raw_reasons.get(f) or "").strip()
        reasons[f] = (
            "Fairness (%s): %s" % (f.replace("_", " "), detail)
            if detail else
            "Fairness (%s): flagged by judge model, no reason given." % f.replace("_", " ")
        )
    return flags, reasons


def score_metrics(item: dict, predicted_level: str, flags: list) -> dict:
    """Rubric-based scores, using whichever metrics are configured for this
    item's skill (see SKILL_METRIC_CONFIG). Each metric is scored
    independently as its own deepeval GEval criteria (0-1 score + reason),
    rather than asking the model to fill in one hand-built multi-metric
    JSON blob -- GEval metrics are cached per (skill, key) in
    _GEVAL_METRIC_CACHE so the criteria prompt is only built once.

    Returns a dict with, for every configured metric: result[key] (the
    numeric score) and result[key + "_reason"] (that metric's own short
    reason, not concatenated with any other metric's). Keeping reasons
    per-metric instead of one joined string is what "run_full_qc" surfaces
    to the output columns -- one column per metric, each independently
    readable, instead of one long paragraph mixing multiple metrics."""
    skill = item.get("skill", "")
    config = _skill_config(skill)

    # NOTE on input/actual_output: GEval's default framing assumes
    # "actual_output" is a system's response to "input". Here there is no
    # such input->output relationship -- the item's text IS the artifact
    # being judged, not a response to anything.
    #
    # An earlier version of this code tried to route around that by telling
    # the judge, inside `input`, to "evaluate the item text below directly."
    # That backfired: GEval's template still treats `actual_output` as "the
    # response produced for input," so an instruction like "evaluate the
    # item below" inside `input` reads to the model as the task that
    # `actual_output` was supposed to carry out. Since `actual_output` is
    # just the raw item text (not an evaluation), the judge concluded the
    # "response" had failed to perform the task -- producing confused
    # "Actual Output fails to perform the task described in the input"
    # reasoning instead of an actual metric judgment.
    #
    # Fix: keep `input` as plain descriptive metadata only -- no
    # instruction to "evaluate" anything. The evaluation instruction
    # belongs in `criteria` (built below from use_definition), which GEval
    # already receives as the actual scoring rubric.
    test_case = LLMTestCase(
        input=(
            "%s-skill item, labeled CEFR level %s (type: %s). "
            "Predicted CEFR level: %s | Rule-based flags already detected: %s"
        ) % (
            skill, item.get("level", ""), item.get("type", ""), predicted_level,
            ", ".join(flags) if flags else "none",
        ),
        actual_output="Text: %s\nAnswer: %s\nOptions: %s" % (
            item.get("text", ""), item.get("answer", ""), item.get("options", ""),
        ),
    )

    has_answer = bool((item.get("answer") or "").strip())

    result = {}
    for m in config["metrics"]:
        key, label, scale, threshold = m["key"], m["label"], m["scale"], m["threshold"]

        # Some metrics define a separate
        # "definition_no_answer" criteria to use when the item has no
        # answer/rubric key to validate, so the judge isn't scored down for
        # a check that can't apply to this item. Falls back to "definition"
        # when no such variant is configured, or when the item has an answer.
        use_definition = m["definition"]
        variant_suffix = ""
        if not has_answer and "definition_no_answer" in m:
            use_definition = m["definition_no_answer"]
            variant_suffix = "__noanswer"

        criteria = (
            "%s\nScore 1.0 if the item fully meets this criteria with no issues, "
            "0.0 if it completely fails it, and proportionally in between."
        ) % use_definition
        geval = _get_geval_metric("%s__%s%s" % (skill or "generic", key, variant_suffix), criteria)

        raw_score, metric_reason, is_error = _measure_geval(geval, test_case)

        if scale == "pct":
            value = round(raw_score * 100)
        elif scale == "count":
            # Count-style metrics (flag above N) don't map naturally onto
            # GEval's 0-1 "meets criteria" score; approximate the count by
            # scaling the "how much it fails" fraction against the
            # threshold. Not currently exercised by SKILL_METRIC_CONFIG.
            value = round((1 - raw_score) * threshold * 2)
        else:  # five
            value = max(1, min(5, round(raw_score * 4) + 1))

        result[key] = value
        # Keep each reason to one short sentence instead of GEval's full
        # multi-sentence paragraph, so the output cell/column stays neat and
        # scannable. Errors are already short (see _measure_geval) and are
        # left as-is. First cut at the first sentence boundary; if that
        # sentence is itself long, hard-truncate at a word boundary.
        clean_reason = " ".join((metric_reason or "").split())
        if not is_error and clean_reason:
            sentence_end = re.search(r"[.!?](\s|$)", clean_reason)
            if sentence_end:
                clean_reason = clean_reason[:sentence_end.end()].strip()
            if len(clean_reason) > 160:
                clean_reason = clean_reason[:160].rsplit(" ", 1)[0] + "\u2026"
        result[key + "_reason"] = clean_reason
        result[key + "_reason_is_error"] = is_error

    return result


def reform_item(item: dict, qc_reason: str) -> dict:
    """Ask Gemma to rewrite a flagged item so it passes QC. Returns a dict
    with the (possibly) new text/answer/options plus a change_summary, or
    the original item unchanged with an error note if the call/parse fails."""
    prompt = REFORM_PROMPT.format(
        descriptors=CEFR_DESCRIPTORS,
        skill=item.get("skill", ""),
        type=(item.get("type") or "").strip() or "unspecified / free-response",
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
    """Three-tier verdict: SELECT (no issues), REVIEW (usable but needs a
    human look), REJECT (the item conveys WRONG information, or any other
    "reject"-severity flag; see SKILL_METRIC_CONFIG/FLAG_SEVERITY). REJECT
    takes priority over REVIEW if both kinds of
    issues are present. A reason is always attached whenever the verdict
    isn't SELECT -- reject_reasons/review_reasons are collected separately
    so the REJECT-vs-REVIEW decision is never based on an empty reason
    string.

    metric_flag_detail_map, if provided (the {flag_name: detail_sentence}
    dict from metric_flag_details()), is used to expand each rule flag into
    a full what+where sentence directly in the returned reason string, so
    qc_reason is self-contained and explains exactly why the item was
    flagged instead of just naming the flag. Falls back to the bare flag
    keyword if no map is given."""
    config = _skill_config(skill)
    reject_reasons = []
    review_reasons = []

    for f in flags:
        severity = FLAG_SEVERITY.get(f, "review")
        target = reject_reasons if severity == "reject" else review_reasons
        detail = (metric_flag_detail_map or {}).get(f)
        target.append(detail if detail else "rule flag: %s" % f)

    if not level_match:
        review_reasons.append("predicted CEFR level outside defined range of labeled level")

    for m in config["metrics"]:
        key, label, threshold = m["key"], m["label"], m["threshold"]
        severity = m.get("severity", "review")
        val = scores.get(key)

        if val is None:
            review_reasons.append("%s could not be determined" % label)
            continue
        if m["compare"] == "min" and val < threshold:
            if m["scale"] == "pct":
                msg = "%s below target (%s%% < %s%%)" % (label, val, threshold)
            else:  # five
                msg = "%s below target (%s/5)" % (label, val)
            # Fold in the judge model's own reasoning for this specific
            # metric (captured per-metric in score_metrics()), so the
            # verdict explains WHY it scored low, not just the number.
            judge_reason = (scores.get(key + "_reason") or "").strip()
            if judge_reason:
                msg += " -- %s" % judge_reason
            (reject_reasons if severity == "reject" else review_reasons).append(msg)

    if reject_reasons:
        return "REJECT", "; ".join(reject_reasons + review_reasons)
    if review_reasons:
        return "REVIEW", "; ".join(review_reasons)
    return "SELECT", "meets all QC targets"


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


def save_results(rows: list, path: str) -> None:
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
                if isinstance(row.get("original_options"), list):
                    row["original_options"] = "|".join(row["original_options"])
                writer.writerow(row)


def run_full_qc(item: dict, flags: list, level_tolerance: int,
                 metric_flag_detail_map: dict = None) -> dict:
    """Runs CEFR classification + rubric scoring + verdict for one item.
    Returns a dict of all the derived QC columns (used for both the initial
    pass and for re-checking items after --reform). Metric columns not
    configured for this item's skill are set to 'n/a' so output stays
    consistent across rows with different skills.

    metric_flag_detail_map is the {flag_name: detail_sentence} dict from
    metric_flag_details(item, flags) -- pass it in if the caller already
    computed it (main() does, to avoid recomputing it), or leave it out and
    it's computed here from item/flags directly.

    Also runs the fairness check (evaluate_fairness()) and folds any
    fairness flags straight into `flags`/`metric_flag_detail_map` before
    the verdict is computed, so a fairness issue shows up in the existing
    qc_label/qc_reason columns (and can drive an auto-REJECT, since
    fairness flags are "reject" severity -- see FLAG_SEVERITY) rather than
    needing its own separate gate/column."""
    if metric_flag_detail_map is None:
        metric_flag_detail_map = metric_flag_details(item, flags)

    fairness_flags, fairness_reasons = evaluate_fairness(item)
    if fairness_flags:
        flags = list(flags) + fairness_flags
        metric_flag_detail_map = dict(metric_flag_detail_map)
        metric_flag_detail_map.update(fairness_reasons)

    predicted_level, level_reason = classify_level(item)
    labeled_level = (item.get("level") or "").strip().upper()
    level_match = _level_within_range(labeled_level, predicted_level, level_tolerance)

    if labeled_level and predicted_level not in ("ERROR", "UNKNOWN") and labeled_level != predicted_level:
        level_change_reason = "Changed from %s to %s: %s" % (labeled_level, predicted_level, level_reason)
    else:
        level_change_reason = level_reason

    skill = item.get("skill", "")
    scores = score_metrics(item, predicted_level, flags)
    label, reason = qc_verdict(level_match, scores, flags, skill,
                                metric_flag_detail_map=metric_flag_detail_map)

    out = {
        "predicted_level": predicted_level,
        "level_match": level_match,
        "level_change_reason": level_change_reason,
    }
    for key in ALL_METRIC_KEYS:
        out[key] = scores[key] if key in scores else "n/a"
        # One reason column per metric. If a metric's score came from a
        # failed API call rather than a real judgment, prefix it clearly so
        # it reads as "scoring failed" at a glance instead of looking like
        # a genuine low-quality verdict on the item itself.
        reason_text = scores.get(key + "_reason", "n/a" if key not in scores else "")
        if scores.get(key + "_reason_is_error"):
            reason_text = "[SCORING ERROR -- not a content judgment] " + reason_text
        out[key + "_reason"] = reason_text
    out["qc_label"] = label
    out["qc_reason"] = reason
    return out


# Columns dropped from the final output: "type" (not needed downstream),
# "question" (duplicate of canonical "text"), and "answer" (not needed
# downstream). Still fully used internally for QC/scoring/reform — only
# excluded from the saved result.
OUTPUT_DROP_COLUMNS = ["type", "question", "answer", "level"]


def _strip_output_columns(out: dict) -> dict:
    for col in OUTPUT_DROP_COLUMNS:
        out.pop(col, None)
    return out


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
    ("qc_label", "QC"),
    ("qc_reason", "QC Reason"),
    ("metric_flags", "Rule Flags"),
]

# Appended to HTML_REPORT_COLUMNS only when --timing is passed (see main());
# kept separate so the report doesn't grow an always-empty "Time (s)" column
# on normal runs where nobody asked for it.
TIMING_COLUMN = ("elapsed_sec", "Time (s)")


def _html_escape(value) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;").replace("'", "&#39;")
    )


def save_html_report(rows: list, path: str, columns: list = None) -> None:
    """Writes a single-file HTML report of QC results for reviewers, with
    a SELECT/REVIEW/REJECT summary, filter buttons, a text search box, and
    a detail drawer per item. Rows are embedded as a JSON blob and rendered
    client-side with vanilla JS (text-only insertion, never innerHTML on
    item content) so no item text/reason can break out of its table cell.

    columns defaults to HTML_REPORT_COLUMNS; callers pass a longer list
    (e.g. HTML_REPORT_COLUMNS + [TIMING_COLUMN] when --timing is on) to add
    columns without mutating the module-level default."""
    if columns is None:
        columns = HTML_REPORT_COLUMNS
    counts = {"SELECT": 0, "REVIEW": 0, "REJECT": 0}
    for row in rows:
        d = (row.get("qc_label") or "").upper()
        if d in counts:
            counts[d] += 1

    table_rows = []
    for row in rows:
        table_rows.append({key: row.get(key, "") for key, _ in columns})
    data_json = json.dumps(table_rows, ensure_ascii=False)
    # Guard against a literal "</script>" inside item text/reasons breaking
    # out of the embedded JSON <script> block.
    data_json = data_json.replace("</", "<\\/")

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
  footer{ padding:16px 32px 32px; color:var(--ink-soft); font-size:11.5px; }
</style>
</head>
<body>

<header>
  <div>
    <h1>QC Review Queue</h1>
    <div class="sub">Content and level QC &mdash; final call on whether an item goes to a student</div>
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

<footer>Click any row for the full QC breakdown (rule flags, level reasoning).</footer>

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

function decisionOf(row){ return (row.qc_label || '').toLowerCase(); }
function badgeClass(value){
  const v = (value || '').toUpperCase();
  if (v.includes('SELECT')) return 'select';
  if (v.includes('REVIEW')) return 'review';
  if (v.includes('REJECT')) return 'reject';
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
    const hay = [textPreview(row), row.qc_reason, row.level_change_reason, row.metric_flags].join(' ').toLowerCase();
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
        <td class="reason-preview" title="${esc(row.qc_reason)}">${esc(row.qc_reason)}</td>
      </tr>`;
  }).join('');
  [...els.rows.querySelectorAll('tr')].forEach(tr => {
    tr.addEventListener('click', () => openDrawer(parseInt(tr.dataset.idx, 10)));
  });
}

function openDrawer(idx){
  const row = DATA[idx];
  const decision = decisionOf(row) || 'review';
  els.drawer.innerHTML = `
    <button class="drawer-close" onclick="closeDrawer()">&times;</button>
    <h2>${esc(row.question_id || idx)} &mdash; <span class="badge ${decision}">${esc(decision)}</span></h2>
    <div class="drawer-sub">${esc(row.skill)} &middot; labeled ${esc(row.level||'—')} &rarr; predicted ${esc(row.predicted_level||'—')}</div>

    <div class="block"><div class="k">Text</div><div class="v">${esc(textPreview(row))}</div></div>
    <div class="block"><div class="k">QC reason</div><div class="v"><strong>${esc(row.qc_reason)}</strong></div></div>
    <div class="block"><div class="k">Level reason</div><div class="v">${esc(row.level_change_reason)}</div></div>
    <div class="block"><div class="k">Rule-based flags</div><div class="v mono">${esc(row.metric_flags)}</div></div>
    ${row.elapsed_sec !== undefined && row.elapsed_sec !== '' ? `<div class="block"><div class="k">Processing time</div><div class="v mono">${esc(row.elapsed_sec)}s</div></div>` : ''}
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


# ----------------------------------------------------------------------
# 3c. Checkpointing (resume after an interrupted run)
# ----------------------------------------------------------------------
# Each item can take several LLM calls (classify + N metrics + optional
# reform cycles), so a long batch is expensive to redo. If the process is
# killed -- OOM, network drop, terminal closed, ctrl-C -- everything not
# yet written to --output was previously lost and had to be recomputed
# from scratch. To avoid that, every completed item is appended to a
# JSONL checkpoint file immediately (flushed + fsynced), and re-running
# the same command reloads it and skips items already done.

import hashlib


def _item_key(item: dict, idx: int) -> str:
    """Stable identifier for one input row, used to match it up against a
    checkpoint entry on resume. Prefers question_id (present on this
    dataset); falls back to a content hash + row index for inputs without
    one, so re-running against the *same* input file still resumes
    correctly even if it has no id column."""
    qid = (item.get("question_id") or "").strip()
    if qid:
        return qid
    digest = hashlib.sha1(
        json.dumps(item, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return "row%d:%s" % (idx, digest)


def _load_checkpoint(path: str) -> dict:
    """Loads a JSONL checkpoint file of already-completed rows, keyed by
    item key. Returns {} if the file doesn't exist yet. Tolerates a
    truncated/corrupt last line -- e.g. the process was killed mid-write --
    by skipping just that one line rather than failing the whole load, so
    at worst you redo the single item that was in flight."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("_checkpoint_key")
            if key:
                done[key] = row
    return done


def _append_checkpoint(path: str, key: str, out_row: dict, html_row: dict) -> None:
    """Appends one completed item's result to the checkpoint file and
    flushes + fsyncs immediately, so this item's work survives even if the
    process is killed while working on the next one."""
    payload = {"_checkpoint_key": key, "out": out_row, "html": html_row}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="QC-flag items on Grammar/Completeness/CEFR Level using Gemma via the Google Gemini API")
    ap.add_argument("--input", required=True, help="Path to input .json or .csv file")
    ap.add_argument("--output", required=True, help="Path to output .json or .csv file")
    ap.add_argument("--level-tolerance", type=int, default=LEVEL_TOLERANCE,
                     help="How many CEFR tiers off is still considered 'within range' (default: 0 = exact match)")
    ap.add_argument("--skip-llm", action="store_true", help="Only run rule-based checks (no Gemini API call)")
    ap.add_argument("--timing", action="store_true",
                     help="Measure and record wall-clock time spent per item (rule checks, LLM "
                          "calls, and any --reform attempts combined). Adds an 'elapsed_sec' "
                          "column to the output/HTML report, prints each item's time alongside "
                          "its result, and prints a total/average summary at the end. Off by "
                          "default since it adds a time.time() call per item and a bit of noise "
                          "to the console output.")
    ap.add_argument("--reform", action="store_true",
                     help="For items that come back REVIEW or REJECT, ask Gemma to rewrite them and "
                          "re-run QC on the rewrite (up to --reform-attempts times)")
    ap.add_argument("--reform-attempts", type=int, default=2,
                     help="Max reform+re-check cycles per flagged item (default: 2)")
    ap.add_argument("--html-output", default=None,
                     help="Path to write a reviewer-facing HTML report (defaults to "
                          "<output basename>.html; pass 'none' to skip writing one)")
    ap.add_argument("--skill-filter", default="reading,writing",
                     help="Only run LLM-based QC (CEFR classification, scoring, reform) on items "
                          "whose skill is in this comma-separated list (case-insensitive). Other "
                          "skills still get rule-based checks but are marked SKIPPED for the LLM "
                          "steps. Use 'all' to disable filtering. (default: reading,writing)")
    ap.add_argument("--checkpoint", default=None,
                     help="Path to a checkpoint file that records each item's result the moment "
                          "it's computed (default: <output>.checkpoint.jsonl). If this run gets "
                          "killed partway through -- crash, network drop, closed terminal -- "
                          "re-running the exact same command reloads this file and skips every "
                          "item already done, instead of re-paying for their LLM calls.")
    ap.add_argument("--fresh-start", action="store_true",
                     help="Ignore any existing checkpoint file for this run and reprocess every "
                          "item from scratch (the checkpoint file is overwritten).")
    ap.add_argument("--save-every", type=int, default=5,
                     help="Re-write --output and the HTML report after every N completed items "
                          "(default: 5), not just at the very end. This is what makes the report "
                          "actually readable if the run is killed partway -- without it, only the "
                          "raw checkpoint file survives an interruption, not a viewable report. "
                          "Set to 0 to only write the report once, at the end.")
    args = ap.parse_args()

    items = load_items(args.input)
    if not items:
        print("No items found in input file.", file=sys.stderr)
        sys.exit(1)

    checkpoint_path = args.checkpoint or (args.output + ".checkpoint.jsonl")
    if args.fresh_start and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    checkpoint_done = _load_checkpoint(checkpoint_path)
    if checkpoint_done:
        print("Resuming from checkpoint: %d/%d item(s) already done -- %s" % (
            len(checkpoint_done), len(items), checkpoint_path))

    write_html = (args.html_output or "").strip().lower() != "none"
    html_path = args.html_output
    if write_html and not html_path:
        base, _ext = os.path.splitext(args.output)
        html_path = base + ".html"

    # Only add the "Time (s)" column to the HTML report when --timing is
    # on, so a normal run's report doesn't grow an always-empty column.
    report_columns = HTML_REPORT_COLUMNS + [TIMING_COLUMN] if args.timing else HTML_REPORT_COLUMNS

    def _flush_reports(partial: bool) -> None:
        """Re-writes --output (and the HTML report, unless disabled) from
        whatever's in `results`/`html_rows` so far. Called periodically
        during the run (see --save-every) as well as at the end, so a run
        that's killed partway still leaves a real, openable report behind
        -- not just the raw checkpoint log."""
        save_results(results, args.output)
        if write_html:
            save_html_report(html_rows, html_path, columns=report_columns)
        if partial:
            print("    (partial report saved: %d/%d items so far)" % (len(results), len(items)))

    results = []
    html_rows = []
    item_timings = []  # elapsed seconds per item actually processed this run (--timing only)
    for i, item in enumerate(items, 1):
        item_key = _item_key(item, i)
        cached = checkpoint_done.get(item_key)
        if cached is not None:
            html_rows.append(cached["html"])
            results.append(cached["out"])
            print("[%d/%d] resumed from checkpoint" % (i, len(items)))
            continue

        item_start = time.time() if args.timing else None

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
            })
            timing_suffix = ""
            if args.timing:
                elapsed = time.time() - item_start
                item_timings.append(elapsed)
                out["elapsed_sec"] = round(elapsed, 3)
                timing_suffix = " [%.2fs]" % elapsed
            html_row = dict(out)
            result_row = _strip_output_columns(out)
            html_rows.append(html_row)
            results.append(result_row)
            _append_checkpoint(checkpoint_path, item_key, result_row, html_row)
            if args.skip_llm:
                print("[%d/%d] rule flags only: %s%s" % (i, len(items), out["metric_flags"], timing_suffix))
            else:
                print("[%d/%d] SKIPPED (skill='%s' not in filter '%s')%s" % (
                    i, len(items), item.get("skill", ""), args.skill_filter, timing_suffix))
            if args.save_every and i % args.save_every == 0:
                _flush_reports(partial=True)
            continue

        qc = run_full_qc(item, flags, args.level_tolerance, metric_flag_detail_map=flag_detail_map)
        out.update(qc)

        print("[%d/%d] %s (predicted=%s, labeled=%s) -> %s" % (
            i, len(items), qc["qc_label"], qc["predicted_level"],
            (item.get("level") or "").strip().upper() or "?", qc["qc_reason"]
        ))

        # ---- Optional: ask Gemma to rewrite items that need review/reject work ----
        if args.reform and qc["qc_label"] in ("REVIEW", "REJECT"):
            out["original_text"] = item.get("text", "")
            out["original_answer"] = item.get("answer", "")
            out["original_options"] = item.get("options", [])
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
                new_flag_detail_map = metric_flag_details(working_item, new_flags)
                new_qc = run_full_qc(working_item, new_flags, args.level_tolerance,
                                      metric_flag_detail_map=new_flag_detail_map)

                if new_qc["qc_label"] == "SELECT":
                    print("    reform succeeded on attempt %d" % attempt)
                    out["text"] = working_item["text"]
                    out["answer"] = working_item["answer"]
                    out["options"] = working_item.get("options", out.get("options"))
                    out["metric_flags"] = ", ".join(new_flags) if new_flags else "none"
                    out.update(new_qc)
                    break
                else:
                    print("    still %s: %s" % (new_qc["qc_label"], new_qc["qc_reason"]))
                    out["text"] = working_item["text"]
                    out["answer"] = working_item["answer"]
                    out["options"] = working_item.get("options", out.get("options"))
                    out["metric_flags"] = ", ".join(new_flags) if new_flags else "none"
                    out.update(new_qc)

        if args.timing:
            elapsed = time.time() - item_start
            item_timings.append(elapsed)
            out["elapsed_sec"] = round(elapsed, 3)
            print("    [%.2fs]" % elapsed)

        html_row = dict(out)
        result_row = _strip_output_columns(out)
        html_rows.append(html_row)
        results.append(result_row)
        _append_checkpoint(checkpoint_path, item_key, result_row, html_row)
        if args.save_every and i % args.save_every == 0:
            _flush_reports(partial=True)

    _flush_reports(partial=False)
    print("\nDone. Wrote %d results to %s" % (len(results), args.output))
    if write_html:
        print("Wrote HTML report to %s" % html_path)
    if args.timing and item_timings:
        total = sum(item_timings)
        avg = total / len(item_timings)
        print("Timing: %d item(s) processed this run, %.2fs total, %.2fs avg/item (min %.2fs, max %.2fs)" % (
            len(item_timings), total, avg, min(item_timings), max(item_timings)))
    print("(checkpoint saved at %s -- safe to delete, or leave it: a re-run with the "
          "same --output will just skip straight through via the checkpoint)" % checkpoint_path)


if __name__ == "__main__":
    main()