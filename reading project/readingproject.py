import argparse
import concurrent.futures
import csv
import json
import os
import re
import sys
import threading
import time
from collections import OrderedDict

# Raise deepeval's per-call timeout (default ~90s) before the judge model is
# constructed below. We were seeing "call timed out after 88.5s" on a
# meaningful fraction of calls with gemma-4-26b-a4b-it, and every timeout
# silently reads downstream as a worst-case score (see score_metrics()),
# not as "we don't know" -- so a slow-but-fine response was indistinguishable
# from a genuinely bad item. Only set if the caller hasn't already overridden it.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "180")

from deepeval.models import GeminiModel
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# ----------------------------------------------------------------------
# deepeval judge model + GEval metric cache
# ----------------------------------------------------------------------
_JUDGE_MODEL = GeminiModel(
    model="gemma-4-26b-a4b-it",
    temperature=0.2,
)


# ----------------------------------------------------------------------
# rate limiting + 429 backoff
# ----------------------------------------------------------------------
class _RateLimiter:
    """Approximate sliding-window token-bucket limiter, safe to share across
    the --workers thread pool."""

    def __init__(self, tokens_per_minute: int = 16000, safety_margin: float = 0.85):
        self.budget = int(tokens_per_minute * safety_margin)
        self.window_seconds = 60.0
        self._lock = threading.Lock()
        self._events = []  # list of (timestamp, tokens_estimate)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def acquire(self, text: str) -> None:
        tokens_needed = min(self._estimate_tokens(text), self.budget)
        while True:
            with self._lock:
                now = time.time()
                cutoff = now - self.window_seconds
                self._events = [(t, n) for t, n in self._events if t > cutoff]
                used = sum(n for _, n in self._events)
                if used + tokens_needed <= self.budget:
                    self._events.append((now, tokens_needed))
                    return
                oldest = self._events[0][0] if self._events else now
                wait = max(0.5, (oldest + self.window_seconds) - now)
            time.sleep(min(wait, 5.0))


_RATE_LIMITER = _RateLimiter()

_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s?", re.IGNORECASE)


def _parse_retry_delay(err_text: str, default: float) -> float:
    m = _RETRY_DELAY_RE.search(err_text or "")
    if m:
        try:
            return max(default, float(m.group(1)) + 1.0)
        except ValueError:
            pass
    return default


_original_judge_generate = _JUDGE_MODEL.generate


def _rate_limited_generate(prompt, *args, **kwargs):
    prompt_text = prompt if isinstance(prompt, str) else str(prompt)
    last_err = None
    for attempt in range(6):
        _RATE_LIMITER.acquire(prompt_text)
        try:
            return _original_judge_generate(prompt, *args, **kwargs)
        except Exception as e:
            last_err = e
            err_text = str(e)
            if "RESOURCE_EXHAUSTED" in err_text or " 429" in err_text or "'code': 429" in err_text:
                delay = _parse_retry_delay(err_text, default=15.0 * (attempt + 1))
                time.sleep(min(delay, 90.0))
            else:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("Judge model unreachable after retries: %s" % last_err)


_JUDGE_MODEL.generate = _rate_limited_generate

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


def _judge_call(prompt: str, retries: int = 1) -> str:
    last_err = None
    for attempt in range(retries + 1):
        try:
            raw = _JUDGE_MODEL.generate(prompt)
            if isinstance(raw, tuple):
                raw = raw[0]
            return str(raw).strip()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2.0)
    raise RuntimeError("Could not reach judge model (Gemini via deepeval): %s" % last_err)


def _item_test_case(item: dict, predicted_level: str = "") -> LLMTestCase:
    passage = _truncate_passage(item.get("content", ""))
    question = item.get("question_text") or item.get("question") or item.get("text", "")
    return LLMTestCase(
        input=(
            "Skill: %s | Type: %s | Labeled CEFR level: %s | Predicted CEFR level: %s\n"
            "Passage: %s"
        ) % (
            item.get("skill", ""), item.get("type", ""),
            (item.get("level") or "").strip().upper() or "(none)",
            predicted_level or "(none)",
            passage or "(none provided)",
        ),
        actual_output="Question: %s\nAnswer: %s\nOptions: %s" % (
            question, item.get("answer", ""), item.get("options", "")
        ),
    )


CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
TRUE_FALSE_OPTIONS = ["True", "False", "Not Given"]
_TRUE_FALSE_OPTIONS_NORM = {o.lower() for o in TRUE_FALSE_OPTIONS}

_TRUE_FALSE_ACCEPTED_OPTION_SETS_NORM = [
    {"true", "false", "not given"},
    {"true", "false"},
    {"yes", "no", "not given"},
    {"yes", "no"},
]
_TRUE_FALSE_ANSWER_NORM = {
    v for option_set in _TRUE_FALSE_ACCEPTED_OPTION_SETS_NORM for v in option_set
}

SKILL_METRIC_CONFIG = {
    "reading": {
        "metrics": [
            {"key": "accuracy_pct", "label": "accuracy", "scale": "pct", "compare": "min", "threshold": 90,
             "definition": "Is the question and its expected answer factually/grammatically correct together?"},
            {"key": "grammar_errors", "label": "grammar errors", "scale": "count", "compare": "max", "threshold": 0,
             "definition": "Count of grammar/spelling errors in the question text and answer. 0 = no errors found."},
            {"key": "clarity_score", "label": "clarity", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "How easy is the question to understand, 5 = completely unambiguous, 1 = very confusing."},
            {"key": "completeness_score", "label": "completeness", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "Is all necessary information/context provided to answer the question, with no missing context."},
        ],
    },
    "_default": {
        "metrics": [
            {"key": "accuracy_pct", "label": "accuracy", "scale": "pct", "compare": "min", "threshold": 90,
             "definition": "Is the question and its expected answer factually/grammatically correct together?"},
            {"key": "grammar_errors", "label": "grammar errors", "scale": "count", "compare": "max", "threshold": 0,
             "definition": "Count of grammar/spelling errors in the question text and answer. 0 = no errors found."},
            {"key": "clarity_score", "label": "clarity", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "How easy is the question to understand, 5 = completely unambiguous, 1 = very confusing."},
            {"key": "completeness_score", "label": "completeness", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "Is all necessary information/context provided to answer the question, with no missing context."},
        ],
    },
}

LEVEL_TOLERANCE = 0         

ALL_METRIC_KEYS = []
for _cfg in SKILL_METRIC_CONFIG.values():
    for _m in _cfg["metrics"]:
        if _m["key"] not in ALL_METRIC_KEYS:
            ALL_METRIC_KEYS.append(_m["key"])


def _skill_config(skill: str) -> dict:
    key = (skill or "").strip().lower()
    return SKILL_METRIC_CONFIG.get(key, SKILL_METRIC_CONFIG["_default"])

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
    "what", "who", "whom", "whose", "which", "where", "when", "why", "how",
    "in", "on", "at", "of", "for", "to", "from", "with", "by", "about",
    "and", "or", "but", "not", "this", "that", "these", "those", "it",
    "its", "as", "be", "been", "being", "has", "have", "had", "can",
    "could", "will", "would", "should", "may", "might", "does",
}


def _content_words(text: str) -> set:
    words = re.findall(r"[a-zA-Z']+", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _answer_text(answer) -> str:
    if isinstance(answer, (list, dict)):
        return _flatten_answer(answer) or ""
    return str(answer or "")


def _resolve_answer_text(item: dict) -> str:
    answer = item.get("answer")
    raw = _answer_text(answer)
    options = item.get("options") or []
    itype = (item.get("question_type") or item.get("type") or "").strip().lower()

    if itype in ("mcq", "multiple_choice") and options and raw:
        letter = raw.strip().upper().rstrip(".)")
        if len(letter) == 1 and letter.isalpha():
            idx = ord(letter) - ord("A")
            if 0 <= idx < len(options):
                option_text = str(options[idx])
                option_text = re.sub(r"^[A-Za-z][\.\)]\s*", "", option_text)
                return option_text
    return raw


FULL_REDUNDANCY_SIM_THRESHOLD = 0.75     
PARTIAL_REDUNDANCY_SIM_THRESHOLD = 0.35  
PARTIAL_ANSWER_SIM_FLOOR = 0.15          
REASONING_WORDS = {
    "infer", "inferred", "imply", "implies", "suggest", "suggests", "why",
    "explain", "evaluate", "compare", "contrast", "analyze", "opinion",
    "meaning", "purpose", "tone", "attitude",
}


def detect_redundant_and_trivial(doc_questions: list) -> dict:
    flags_by_id = {q.get("sub_question_id") or "": [] for q in doc_questions}
    ids = list(flags_by_id.keys())

    for i in range(len(doc_questions)):
        for j in range(i + 1, len(doc_questions)):
            qi, qj = doc_questions[i], doc_questions[j]
            words_i = _content_words(qi.get("question") or qi.get("text", ""))
            words_j = _content_words(qj.get("question") or qj.get("text", ""))
            sim = _jaccard(words_i, words_j)

            ans_i = _content_words(_resolve_answer_text(qi))
            ans_j = _content_words(_resolve_answer_text(qj))
            answers_equal = bool(ans_i) and bool(ans_j) and ans_i == ans_j
            answers_subset = (
                bool(ans_i) and bool(ans_j) and not answers_equal
                and (ans_i.issubset(ans_j) or ans_j.issubset(ans_i))
            )
            answers_overlap = bool(ans_i & ans_j) if (ans_i and ans_j) else False
            mutual_swap = bool(ans_i & words_j) and bool(ans_j & words_i)

            is_full = mutual_swap or answers_equal or (sim >= FULL_REDUNDANCY_SIM_THRESHOLD)
            is_partial = (not is_full) and (
                sim >= PARTIAL_REDUNDANCY_SIM_THRESHOLD
                or answers_subset
                or (answers_overlap and sim >= PARTIAL_ANSWER_SIM_FLOOR)
            )

            if is_full:
                flags_by_id[ids[i]].append("fully_redundant_with:%s" % ids[j])
                flags_by_id[ids[j]].append("fully_redundant_with:%s" % ids[i])
            elif is_partial:
                flags_by_id[ids[i]].append("partially_redundant_with:%s" % ids[j])
                flags_by_id[ids[j]].append("partially_redundant_with:%s" % ids[i])

    for i, q in enumerate(doc_questions):
        question_text = q.get("question") or q.get("text", "")
        answer_text = _answer_text(q.get("answer"))
        q_words = _content_words(question_text)
        a_words = _content_words(answer_text)

        if not a_words or not q_words:
            continue

        has_reasoning_language = bool(q_words & REASONING_WORDS)
        answer_len_ok = len(a_words) <= 2  
        answer_is_copy_of_question = a_words.issubset(q_words)

        if answer_len_ok and answer_is_copy_of_question and not has_reasoning_language:
            flags_by_id[ids[i]].append("trivial_lookup")

    return flags_by_id


def metric_flags(item: dict) -> list:
    flags = []
    text = (item.get("text") or "").strip()
    answer = (item.get("answer") or "")
    level = (item.get("level") or "").strip().upper()
    itype = (item.get("type") or "").strip().lower()

    if answer != answer.strip():
        flags.append("answer_whitespace")
    if text != text.strip():
        flags.append("text_whitespace")
    if not text:
        flags.append("missing_text")
    if not answer.strip():
        flags.append("missing_answer")
    if level not in CEFR_LEVELS:
        flags.append("invalid_level")
    if itype == "fill_up" and "___" not in text and "____" not in text:
        flags.append("no_blank_marker")
    if itype in ("mcq", "multiple_choice") and not item.get("options"):
        flags.append("missing_options")
    if itype == "true_false":
        options = item.get("options") or []
        options_norm = {str(o).strip().lower() for o in options}
        if not options:
            flags.append("missing_options")
        elif options_norm not in _TRUE_FALSE_ACCEPTED_OPTION_SETS_NORM:
            flags.append("invalid_true_false_options")
        if answer.strip().lower() not in _TRUE_FALSE_ANSWER_NORM:
            flags.append("invalid_true_false_answer")
    if text and not re.search(r"[.?!]\s*$", text.replace("___", "").replace("____", "")):
        flags.append("missing_end_punctuation")
    if re.search(r"[ \t]{2,}", text):
        flags.append("double_spacing")
    if level in ("A1", "A2") and len(answer.split()) > 4:
        flags.append("answer_too_long_for_level")

    return flags


CEFR_DESCRIPTORS = """A1 - Beginner: Understands/uses very familiar everyday expressions and basic phrases.
A2 - Elementary: Understands sentences about immediate relevance.
B1 - Intermediate: Understands main points of clear standard input.
B2 - Upper-Intermediate: Understands main ideas of complex text.
C1 - Advanced: Understands wide range of demanding, longer texts.
C2 - Proficient: Understands virtually everything read/heard with ease.
"""

CEFR_PROMPT = """You are a CEFR proficiency classifier. Classify the difficulty level of the QUESTION below:
{descriptors}
Background passage: {passage}
Skill: {skill} | Type: {type} | QUESTION to rate: {question} | Expected answer: {answer} | Options: {options}
Respond with ONLY a JSON object: {{"level": "<A1, A2, B1, B2, C1, C2>", "reason": "<one sentence>"}}
"""

COGNITIVE_SKILLS = ["recall", "comprehension", "inference", "vocabulary", "analysis", "evaluation"]

COGNITIVE_SKILL_DESCRIPTORS = """recall - Verbatim lookup.
comprehension - Connecting stated facts.
inference - Unstated conclusion.
vocabulary - Meaning of a word.
analysis - Structure or relationships.
evaluation - Author stance or bias.
"""

COGNITIVE_SKILL_PROMPT = """Classify cognitive skill for:
{descriptors}
Passage: {passage} | Type: {type} | QUESTION: {question} | Expected answer: {answer} | Options: {options} | Labeled: {labeled_skill}
Respond with ONLY a JSON object: {{"predicted_cognitive_skill": "<skill>", "matches_label": <true/false>, "reason": "<sentence>"}}
"""

CONSTRUCTION_PROMPT = """Judge ITEM construction quality:
Item: Skill: {skill} | Type: {type} | Predicted CEFR level: {predicted_level}
Passage: {passage} | QUESTION: {question} | Expected answer: {answer} | Options: {options} | Is MCQ: {is_mcq}
Respond with ONLY a JSON object: {{"distractor_plausibility": <1-5/null>, "single_correct_answer": <true/false/null>, "passage_relevance": <1-5>, "answerable_without_passage": <true/false>, "sensitive_content": <true/false>, "reason": "<sentence>"}}
"""

FAIRNESS_PROMPT = """Check fairness and sensitivity for:
Item: Skill: {skill} | Type: {type} | Predicted CEFR level: {predicted_level}
Passage: {passage} | QUESTION: {question} | Expected answer: {answer} | Options: {options}
Respond with ONLY a JSON object: {{"requires_specialist_or_cultural_knowledge": <true/false>, "cultural_bias": <true/false>, "gender_bias": <true/false>, "sensitive_content": <true/false>, "reason": "<sentence>"}}
"""


def _truncate_passage(passage: str, max_chars: int = 400) -> str:
    passage = (passage or "").strip()
    if len(passage) <= max_chars:
        return passage
    snippet = passage[:max_chars]
    cut = max(snippet.rfind(". "), snippet.rfind("! "), snippet.rfind("? "))
    if cut > max_chars * 0.5:
        snippet = snippet[:cut + 1]
    return snippet.strip() + " [...]"


REFORM_PROMPT = """Rewrite this failed item so it passes QC:
Original: Skill: {skill} | Type: {type} | Labeled CEFR level: {labeled_level} | Text: {text} | Answer: {answer} | Options: {options}
Failure reason: {qc_reason}
Respond with ONLY a JSON object: {{"text": "<rewritten text>", "answer": "<rewritten answer>", "options": [<list>], "change_summary": "<sentence>"}}
"""


def classify_level(item: dict):
    question = item.get("question_text") or item.get("question") or item.get("text", "")
    passage = _truncate_passage(item.get("content", ""))

    prompt = CEFR_PROMPT.format(
        descriptors=CEFR_DESCRIPTORS, skill=item.get("skill", ""), type=item.get("type", ""),
        passage=passage or "(none provided)", question=question, answer=item.get("answer", ""), options=item.get("options", ""),
    )

    for _ in range(2):
        try:
            raw = _judge_call(prompt)
        except RuntimeError as e:
            return "ERROR", "judge model unreachable: %s" % e

        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            continue
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            continue

        level = (parsed.get("level") or "").strip().upper()
        if level not in CEFR_LEVELS:
            salvage = re.search(r"\b([ABC][12])\b", level) or re.search(r"\b([ABC][12])\b", raw)
            level = salvage.group(1) if salvage else "UNKNOWN"
        return level, parsed.get("reason", "") or raw[:200]

    return "UNKNOWN", "Could not parse judge model output"


def classify_cognitive_skill(item: dict):
    labeled_skill = (item.get("cognitive_skill") or "").strip().lower()
    if not labeled_skill:
        return "", None, "no cognitive_skill label on item to check against"

    question = item.get("question_text") or item.get("question") or item.get("text", "")
    passage = _truncate_passage(item.get("content", ""))

    prompt = COGNITIVE_SKILL_PROMPT.format(
        descriptors=COGNITIVE_SKILL_DESCRIPTORS, passage=passage or "(none provided)",
        type=item.get("type", ""), question=question, answer=item.get("answer", ""),
        options=item.get("options", ""), labeled_skill=labeled_skill,
    )

    for _ in range(2):
        try:
            raw = _judge_call(prompt)
        except RuntimeError as e:
            return "ERROR", None, "judge model unreachable: %s" % e

        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            continue
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            continue

        predicted = (parsed.get("predicted_cognitive_skill") or "").strip().lower()
        if predicted not in COGNITIVE_SKILLS:
            salvage = None
            for skill_name in COGNITIVE_SKILLS:
                if re.search(r"\b%s\b" % re.escape(skill_name), raw.lower()):
                    salvage = skill_name
                    break
            predicted = salvage or "UNKNOWN"

        matches = (predicted == labeled_skill) if predicted != "UNKNOWN" else None
        return predicted, matches, parsed.get("reason", "") or raw[:200]

    return "UNKNOWN", None, "Could not parse judge model output"


def _parse_judge_json(prompt: str, required_keys: tuple, parse_attempts: int = 2):
    last_raw = ""
    for _ in range(parse_attempts):
        try:
            raw = _judge_call(prompt)
        except RuntimeError as e:
            return None, "judge model unreachable: %s" % e

        last_raw = raw
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            continue
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            continue
        if any(k not in parsed for k in required_keys):
            continue
        return parsed, raw

    return None, "Could not parse judge model output: %s" % last_raw[:150]


def classify_construction(item: dict, predicted_level: str = "") -> dict:
    itype = (item.get("type") or "").strip().lower()
    is_mcq = itype in ("mcq", "multiple_choice")
    passage = _truncate_passage(item.get("content", ""))
    question = item.get("question_text") or item.get("question") or item.get("text", "")

    prompt = CONSTRUCTION_PROMPT.format(
        skill=item.get("skill", ""), type=item.get("type", ""),
        predicted_level=predicted_level or "(none)", passage=passage or "(none provided)",
        question=question, answer=item.get("answer", ""), options=item.get("options", ""), is_mcq=is_mcq,
    )

    result = {
        "distractor_plausibility": 5 if not is_mcq else 0,
        "single_correct_answer": True if not is_mcq else None,
        "passage_relevance": 0, "answerable_without_passage": None,
        "sensitive_content": None, "reason": "", "error": False,
    }

    parsed, raw_or_err = _parse_judge_json(
        prompt, required_keys=("passage_relevance", "answerable_without_passage", "sensitive_content"),
    )
    if parsed is None:
        result["reason"] = "construction check failed: %s" % raw_or_err
        result["error"] = True
        return result

    if is_mcq:
        dp = parsed.get("distractor_plausibility")
        result["distractor_plausibility"] = max(1, min(5, int(dp))) if isinstance(dp, (int, float)) else 3
        result["single_correct_answer"] = bool(parsed.get("single_correct_answer", True))

    pr = parsed.get("passage_relevance")
    result["passage_relevance"] = max(1, min(5, int(pr))) if isinstance(pr, (int, float)) else 3
    result["answerable_without_passage"] = bool(parsed.get("answerable_without_passage", False))
    result["sensitive_content"] = bool(parsed.get("sensitive_content", False))
    result["reason"] = parsed.get("reason", "")
    return result


def classify_fairness(item: dict, predicted_level: str = "") -> dict:
    passage = _truncate_passage(item.get("content", ""))
    question = item.get("question_text") or item.get("question") or item.get("text", "")

    prompt = FAIRNESS_PROMPT.format(
        skill=item.get("skill", ""), type=item.get("type", ""),
        predicted_level=predicted_level or "(none)", passage=passage or "(none provided)",
        question=question, answer=item.get("answer", ""), options=item.get("options", ""),
    )

    result = {
        "requires_specialist_or_cultural_knowledge": None, "cultural_bias": None,
        "gender_bias": None, "sensitive_content": None, "reason": "", "error": False,
    }

    parsed, raw_or_err = _parse_judge_json(
        prompt, required_keys=("cultural_bias", "gender_bias", "sensitive_content"),
    )
    if parsed is None:
        result["reason"] = "fairness check failed: %s" % raw_or_err
        result["error"] = True
        return result

    result["requires_specialist_or_cultural_knowledge"] = bool(parsed.get("requires_specialist_or_cultural_knowledge", False))
    result["cultural_bias"] = bool(parsed.get("cultural_bias", False))
    result["gender_bias"] = bool(parsed.get("gender_bias", False))
    result["sensitive_content"] = bool(parsed.get("sensitive_content", False))
    result["reason"] = parsed.get("reason", "")
    return result


def score_metrics(item: dict, predicted_level: str, flags: list) -> dict:
    skill = item.get("skill", "")
    config = _skill_config(skill)
    test_case = _item_test_case(item, predicted_level)

    result = {}
    reasons = []

    for m in config["metrics"]:
        key, scale = m["key"], m["scale"]

        if key == "grammar_errors":
            try:
                metric = _get_geval_metric("grammar_cleanliness", "Check for grammar/spelling errors.")
                metric.measure(test_case)
                result[key] = 0 if metric.score >= 0.9 else 1
                reasons.append(metric.reason)
            except Exception as e:
                result[key] = -1
                reasons.append("grammar check failed: %s" % e)
            continue

        try:
            metric = _get_geval_metric(m["label"], m["definition"])
            metric.measure(test_case)
        except Exception as e:
            result[key] = -1
            reasons.append("%s check failed: %s" % (m["label"], e))
            continue

        if scale == "pct":
            result[key] = round(metric.score * 100)
        elif scale == "five":
            result[key] = max(1, min(5, round(metric.score * 5)))
        else:
            result[key] = round(metric.score * 100)
        reasons.append(metric.reason)

    result["reason"] = " | ".join(r for r in reasons if r)
    return result


def reform_item(item: dict, qc_reason: str) -> dict:
    prompt = REFORM_PROMPT.format(
        skill=item.get("skill", ""), type=item.get("type", ""), labeled_level=item.get("level", ""),
        text=item.get("text", ""), answer=item.get("answer", ""), options=item.get("options", ""), qc_reason=qc_reason,
    )
    try:
        raw = _judge_call(prompt)
    except RuntimeError as e:
        return {"text": item.get("text", ""), "answer": item.get("answer", ""), "options": item.get("options", []), "change_summary": "REFORM FAILED: %s" % e}

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return {"text": item.get("text", ""), "answer": item.get("answer", ""), "options": item.get("options", []), "change_summary": "REFORM FAILED: Parse error"}
    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return {"text": item.get("text", ""), "answer": item.get("answer", ""), "options": item.get("options", []), "change_summary": "REFORM FAILED: Invalid JSON"}

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


def qc_verdict(level_match: bool, scores: dict, flags: list, skill: str = ""):
    config = _skill_config(skill)
    reasons = []

    for m in config["metrics"]:
        key, label, threshold = m["key"], m["label"], m["threshold"]
        val = scores.get(key)

        if key == "grammar_errors":
            if val is None or val < 0:
                reasons.append("grammar score could not be determined")
            elif val > threshold:
                reasons.append("%d grammar error(s) found" % val)
            continue

        if val is None or val < 0:
            reasons.append("%s could not be determined" % label)
        elif m["compare"] == "min" and val < threshold:
            reasons.append("%s below target (%s < %s)" % (label, val, threshold))

    if flags:
        reasons.append("rule flags: %s" % ", ".join(flags))

    if reasons:
        return "FLAG", "; ".join(reasons)
    return "PASS", "meets all QC targets"


CONSTRUCTION_THRESHOLDS = {"distractor_plausibility_min": 3, "passage_relevance_min": 3}
CONSTRUCTION_THRESHOLDS_BY_LEVEL = {
    "A1": {"distractor_plausibility_min": 2, "passage_relevance_min": 2},
    "A2": {"distractor_plausibility_min": 2, "passage_relevance_min": 3},
    "B1": {"distractor_plausibility_min": 3, "passage_relevance_min": 3},
    "B2": {"distractor_plausibility_min": 3, "passage_relevance_min": 4},
    "C1": {"distractor_plausibility_min": 4, "passage_relevance_min": 4},
    "C2": {"distractor_plausibility_min": 4, "passage_relevance_min": 4},
}


def _construction_thresholds_for_level(level: str) -> dict:
    return CONSTRUCTION_THRESHOLDS_BY_LEVEL.get((level or "").strip().upper(), CONSTRUCTION_THRESHOLDS)


def evaluate_construction(item: dict, predicted_level: str = "") -> dict:
    return classify_construction(item, predicted_level)


def construction_flags(item: dict, llm_result: dict, level: str = "") -> list:
    flags = []
    itype = (item.get("type") or "").strip().lower()
    is_mcq = itype in ("mcq", "multiple_choice")
    thresholds = _construction_thresholds_for_level(level)

    dp = llm_result.get("distractor_plausibility")
    if is_mcq:
        if dp is not None and 0 < dp < thresholds["distractor_plausibility_min"]:
            flags.append("weak_distractors")
        if llm_result.get("single_correct_answer") is False:
            flags.append("multiple_defensible_answers")

    if llm_result.get("sensitive_content") is True:
        flags.append("sensitive_content_review_needed")
    if llm_result.get("error"):
        flags.append("construction_check_error")

    return flags


def construction_qc_gate(flags: list, llm_reason: str):
    if flags == ["construction_check_error"]:
        return "CONSTRUCTION_ERROR", "check error"
    if flags:
        return "CONSTRUCTION_FLAG", ", ".join(flags)
    return "CONSTRUCTION_PASS", llm_reason or "meets all construction QC targets"


def evaluate_fairness(item: dict, predicted_level: str = "") -> dict:
    return classify_fairness(item, predicted_level)


def fairness_flags(item: dict, llm_result: dict) -> list:
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
    if flags == ["fairness_check_error"]:
        return "FAIRNESS_ERROR", "check error"
    if flags:
        return "FAIRNESS_FLAG", ", ".join(flags)
    return "FAIRNESS_PASS", "pass"


def answer_letter_bias_summary(results: list) -> str:
    counts = {}
    total = 0
    for r in results:
        itype = (r.get("type") or r.get("question_type") or "").strip().lower()
        if itype not in ("mcq", "multiple_choice"):
            continue
        letter = str(r.get("answer", "")).strip().upper().rstrip(".)")
        if len(letter) == 1 and letter.isalpha():
            counts[letter] = counts.get(letter, 0) + 1
            total += 1
    if total < 8:
        return "Answer-key bias check skipped."
    lines = ["Answer-key letter distribution across %d MCQ item(s):" % total]
    for letter in sorted(counts):
        lines.append("  %s: %d (%.0f%%)" % (letter, counts[letter], 100 * counts[letter] / total))
    return "\n".join(lines)


FIELD_ALIASES = {
    "text": ["text", "question_text", "prompt", "question", "item_text"],
    "type": ["type", "question_type", "item_type", "qtype"],
    "level": ["level", "cefr_level", "labeled_level", "target_level"],
    "answer": ["answer", "expected_answer", "correct_answer", "key"],
    "skill": ["skill", "skill_type", "category"],
    "options": ["options", "choices"],
}


def _flatten_field_value(value):
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("text", "value", "content", "question", "answer"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner
        return " ".join([str(v) for v in value.values() if isinstance(v, (str, int, float))])
    if isinstance(value, list):
        return " ".join(_flatten_field_value(v) if isinstance(v, dict) else str(v) for v in value)
    return str(value).strip()


def _normalize_item(item: dict) -> dict:
    item = dict(item)
    for canonical, aliases in FIELD_ALIASES.items():
        if canonical == "options":
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

    if item.get("question") != item.get("text", ""):
        item["question_text"] = item.get("text", "")

    for alias in ("content", "passage", "reading_passage"):
        if item.get(alias):
            passage = _flatten_field_value(item[alias])
            if not item.get("content"):
                item["content"] = passage
            if passage and item.get("text") and passage not in item["text"]:
                item["text"] = "Passage: %s\n\nQuestion: %s" % (passage, item["text"])
            break

    itype = (item.get("type") or "").strip().lower()
    if itype == "true_false" and not item.get("options"):
        item["options"] = ["True", "False", "Not Given"]

    return item


def _flatten_answer(ans):
    if isinstance(ans, list):
        return "; ".join(str(a) for a in ans)
    if isinstance(ans, dict):
        return "; ".join("%s->%s" % (k, v) for k, v in ans.items())
    return ans


def _expand_nested_document(doc: dict) -> list:
    base = {
        "question_id": doc.get("content_id", ""),
        "skill": doc.get("category", doc.get("skill", "")),
        "cefr_level": doc.get("cefr_level", ""),
        "content": doc.get("content", ""),
    }
    expanded = []
    for q in doc.get("questions", []):
        row = dict(base)
        row["sub_question_id"] = q.get("sub_question_id", "")
        row["question_type"] = q.get("question_type", "")
        if q.get("cognitive_skill"):
            row["cognitive_skill"] = q["cognitive_skill"]
        row["question"] = q.get("question", "")
        row["options"] = q.get("options", [])
        row["answer"] = _flatten_answer(q.get("answer"))
        expanded.append(row)
    return expanded


def load_items(path: str) -> list:
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_items = data if isinstance(data, list) else [data]
        items = []
        for it in raw_items:
            if isinstance(it, dict) and isinstance(it.get("questions"), list):
                items.extend(_expand_nested_document(it))
            else:
                items.append(it)
    else:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            items = []
            for row in reader:
                if "options" in row and row["options"]:
                    row["options"] = row["options"].split("|")
                items.append(row)

    return [_normalize_item(it) for it in items]


def compute_final_decision(qc_label: str, construction_qc_label: str,
                            fairness_qc_label: str, metric_flags: str,
                            qc_reason: str = "", construction_qc_reason: str = "",
                            fairness_qc_reason: str = "", level_match=True) -> tuple:
    gates = [
        ("content", qc_label, qc_reason),
        ("construction", construction_qc_label, construction_qc_reason),
        ("fairness", fairness_qc_label, fairness_qc_reason),
    ]
    labels = [qc_label, construction_qc_label, fairness_qc_label]

    ignored_rule_flags = {"missing_end_punctuation"}
    effective_flags = [
        f for f in (metric_flags.split(", ") if metric_flags not in ("", "none") else [])
        if f not in ignored_rule_flags
    ]
    rule_flags_present = bool(effective_flags)

    combined_parts = []
    for name, label, reason in gates:
        if reason and (label.endswith("FLAG") or label.endswith("ERROR")):
            combined_parts.append("%s: %s" % (name, reason))
    if rule_flags_present:
        combined_parts.append("rules: %s" % ", ".join(effective_flags))
    combined_reason = "; ".join(combined_parts)

    if any(l.endswith("FLAG") for l in labels if l):
        return "REJECT", "REJECT -- " + (combined_reason or "gate flagged")

    if all(l == "" for l in labels):
        if rule_flags_present:
            return "REVIEW", "REVIEW -- " + combined_reason
        return "USE", "USE -- rule checks clean"

    if any(l for l in labels) and all(l.endswith("PASS") for l in labels if l):
        if level_match is False:
            return "REVIEW", "REVIEW -- predicted CEFR level did not match labeled level"
        return "USE", "USE -- passed"

    return "REVIEW", "REVIEW -- " + (combined_reason or "incomplete coverage")


PREFERRED_COLUMN_ORDER = [
    "question_id", "sub_question_id", "skill", "question_type",
    "content", "question", "options", "answer",
    "cefr_level", "cognitive_skill",
    "predicted_level", "level_change_reason", "qc_label", "metric_flags",
    "predicted_cognitive_skill", "cognitive_skill_reason",
    "accuracy_pct", "grammar_errors", "clarity_score", "completeness_score",
    "construction_flags", "fairness_flags", "fairness_qc_reason",
    "final_decision", "final_decision_reason",
]

COMPACT_COLUMN_ORDER = [
    "question_id", "sub_question_id", "skill", "question_type",
    "reform_change_summary", "final_decision", "final_decision_reason",
]


def _ordered_fieldnames(rows: list) -> list:
    seen = []
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.append(key)
    ordered = [c for c in PREFERRED_COLUMN_ORDER if c in seen]
    ordered += [c for c in seen if c not in ordered]
    return ordered


HTML_REPORT_COLUMNS = [
    ("question_id", "Question ID"),
    ("sub_question_id", "Sub Question ID"),
    ("skill", "Skill"),
    ("text", "Text"),
    ("level", "Labeled Level"),
    ("predicted_level", "Predicted Level"),
    ("level_change_reason", "Level Reason"),
    ("qc_label", "Content QC"),
    ("qc_reason", "Content QC Reason"),
    ("construction_qc_label", "Construction QC"),
    ("construction_qc_reason", "Construction QC Reason"),
    ("construction_flags", "Construction Flags"),
    ("fairness_qc_label", "Fairness QC"),
    ("fairness_qc_reason", "Fairness QC Reason"),
    ("fairness_flags", "Fairness Flags"),
    ("metric_flags", "Rule Flags"),
    ("final_decision", "Final Decision"),
    ("final_decision_reason", "Final Decision Reason"),
]


def _html_escape(value) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;").replace("'", "&#39;")
    )


_HTML_REPORT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>QC Review Queue</title>
<style>
  :root {
    --good-bg: #e6f4ea; --good-fg: #1e7b34;
    --warn-bg: #fdf3d6; --warn-fg: #92700c;
    --bad-bg:  #fbe4e2; --bad-fg:  #b3261e;
    --neutral-bg: #ececec; --neutral-fg: #555;
  }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; padding: 24px 28px; background: #f7f7f5; color: #222; }
  h1 { margin: 0 0 2px 0; font-size: 26px; }
  .subtitle { color: #666; font-size: 14px; margin: 0; }
  .meta { color: #888; font-size: 13px; text-align: right; }
  .top-row { display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px; }
  .stats { display: flex; gap: 14px; margin: 18px 0 16px 0; flex-wrap: wrap; }
  .stat-card { background: #fff; border: 1px solid #e2e2e2; border-radius: 10px; padding: 14px 22px; min-width: 100px; }
  .stat-card .num { font-size: 26px; font-weight: 700; }
  .stat-card .label { font-size: 11px; letter-spacing: .04em; color: #888; text-transform: uppercase; margin-top: 2px; }
  .stat-card.select .num { color: var(--good-fg); }
  .stat-card.review .num { color: var(--warn-fg); }
  .stat-card.reject .num { color: var(--bad-fg); }
  .controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }
  #search { flex: 1; min-width: 220px; padding: 9px 12px; border: 1px solid #d8d8d8; border-radius: 8px; font-size: 14px; }
  .chip-group { display: flex; gap: 6px; }
  .chip { padding: 8px 14px; border-radius: 20px; border: 1px solid #d8d8d8; background: #fff; font-size: 13px; cursor: pointer; color: #444; }
  .chip.active { background: #222; color: #fff; border-color: #222; }
  select.filter-select { padding: 8px 10px; border-radius: 8px; border: 1px solid #d8d8d8; background: #fff; font-size: 13px; }
  .shown-count { font-size: 12px; color: #888; white-space: nowrap; }
  .table-wrap { background: #fff; border: 1px solid #e2e2e2; border-radius: 10px; overflow: auto; max-height: 72vh; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #eee; vertical-align: top; }
  th { position: sticky; top: 0; background: #fafafa; font-size: 11px; text-transform: uppercase; letter-spacing: .03em; color: #777; z-index: 1; }
  tbody tr { cursor: pointer; }
  tbody tr:hover { background: #f7f9fc; }
  .text-cell { max-width: 320px; }
  .reason-cell { max-width: 260px; color: #555; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; letter-spacing: .02em; }
  .badge-good { background: var(--good-bg); color: var(--good-fg); }
  .badge-warn { background: var(--warn-bg); color: var(--warn-fg); }
  .badge-bad { background: var(--bad-bg); color: var(--bad-fg); }
  .badge-neutral { background: var(--neutral-bg); color: var(--neutral-fg); }
  .overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.4); align-items: flex-start; justify-content: center; padding: 5vh 20px; z-index: 10; overflow: auto; }
  .overlay.open { display: flex; }
  .modal { background: #fff; border-radius: 12px; max-width: 720px; width: 100%; padding: 26px 28px; position: relative; }
  .modal .close-btn { position: absolute; top: 16px; right: 18px; border: none; background: none; font-size: 20px; cursor: pointer; color: #888; }
  .modal h2 { margin: 0 0 4px 0; font-size: 18px; display: flex; align-items: center; gap: 10px; }
  .modal .sub { color: #888; font-size: 13px; margin: 0 0 16px 0; }
  .modal .section-label { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #999; margin: 16px 0 4px 0; }
  .modal .text-block { white-space: pre-line; font-size: 14px; line-height: 1.5; }
  .box-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 8px; }
  .box { border: 1px solid #eee; border-radius: 10px; padding: 12px 14px; background: #fbfbfb; }
  .box .section-label { margin-top: 0; }
  .box .reason { font-size: 13px; color: #444; margin-top: 6px; }
  .final-reason { font-weight: 700; font-size: 14px; }
</style>
</head>
<body>
<div class="top-row">
  <div>
    <h1>QC Review Queue</h1>
    <p class="subtitle">Content, level and fairness QC &mdash; final call on whether an item goes to a student</p>
  </div>
  <div class="meta" id="meta-info"></div>
</div>

<div class="stats" id="stats"></div>

<div class="controls">
  <input id="search" type="text" placeholder="Search text, skill, reasons...">
  <div class="chip-group" id="decision-chips">
    <div class="chip active" data-filter="all">All</div>
    <div class="chip" data-filter="select">Select</div>
    <div class="chip" data-filter="review">Review</div>
    <div class="chip" data-filter="reject">Reject</div>
  </div>
  <select id="skill-filter" class="filter-select"><option value="all">All skills</option></select>
  <select id="level-filter" class="filter-select"><option value="all">All levels</option></select>
  <span class="shown-count" id="shown-count"></span>
</div>

<div class="table-wrap">
<table>
  <thead>
    <tr>
      <th>Question ID</th><th>Skill</th><th>Text</th>
      <th>Labeled Level</th><th>Predicted Level</th><th>Level Reason</th>
      <th>Content QC</th><th>Content QC Reason</th>
      <th>Fairness QC</th><th>Fairness QC Reason</th>
      <th>Rule Flags</th><th>Final Decision</th><th>Final Decision Reason</th>
    </tr>
  </thead>
  <tbody id="rows"></tbody>
</table>
</div>

<div class="overlay" id="overlay">
  <div class="modal">
    <button class="close-btn" id="close-btn">&times;</button>
    <h2><span id="m-id"></span><span id="m-decision-badge"></span></h2>
    <p class="sub" id="m-sub"></p>
    <div class="section-label">Text</div>
    <div class="text-block" id="m-text"></div>
    <div class="box-grid">
      <div class="box">
        <div class="section-label">Content QC</div>
        <span id="m-content-badge"></span>
        <div class="reason" id="m-content-reason"></div>
      </div>
      <div class="box">
        <div class="section-label">Fairness</div>
        <span id="m-fairness-badge"></span>
        <div class="reason" id="m-fairness-reason"></div>
      </div>
      <div class="box">
        <div class="section-label">Construction QC</div>
        <span id="m-construction-badge"></span>
        <div class="reason" id="m-construction-reason"></div>
      </div>
    </div>
    <div class="section-label">Level Reason</div>
    <div id="m-level-reason"></div>
    <div class="section-label">Rule-Based Flags</div>
    <div id="m-rule-flags"></div>
    <div class="section-label">Final Decision Reason</div>
    <div class="final-reason" id="m-final-reason"></div>
  </div>
</div>

<script>
  const data = __DATA_JSON__;

  function badgeClass(val) {
    const v = (val || "").toString().toUpperCase();
    if (["SELECT", "PASS", "FAIRNESS_PASS"].includes(v)) return "badge-good";
    if (v === "REVIEW") return "badge-warn";
    if (["REJECT", "FLAG"].includes(v)) return "badge-bad";
    return "badge-neutral";
  }
  function badgeHtml(val) {
    const text = (val || "n/a").toString();
    return '<span class="badge ' + badgeClass(val) + '">' + escapeHtml(text) + '</span>';
  }
  function escapeHtml(s) {
    return (s === null || s === undefined ? "" : String(s))
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function decisionBucket(val) {
    const v = (val || "").toString().toUpperCase();
    if (v === "SELECT") return "select";
    if (v === "REVIEW") return "review";
    if (v === "REJECT") return "reject";
    return "other";
  }

  // ---- meta / stat cards ----
  document.getElementById("meta-info").textContent =
    "Generated " + "__GENERATED__" + " \u00b7 " + data.length + " items reviewed";

  const counts = { select: 0, review: 0, reject: 0 };
  data.forEach(r => { const b = decisionBucket(r.final_decision); if (counts[b] !== undefined) counts[b]++; });
  const statsEl = document.getElementById("stats");
  statsEl.innerHTML =
    '<div class="stat-card"><div class="num">' + data.length + '</div><div class="label">Total items</div></div>' +
    '<div class="stat-card select"><div class="num">' + counts.select + '</div><div class="label">Select</div></div>' +
    '<div class="stat-card review"><div class="num">' + counts.review + '</div><div class="label">Review</div></div>' +
    '<div class="stat-card reject"><div class="num">' + counts.reject + '</div><div class="label">Reject</div></div>';

  // ---- filter dropdowns, populated from data ----
  const skillSel = document.getElementById("skill-filter");
  const levelSel = document.getElementById("level-filter");
  [...new Set(data.map(r => r.skill).filter(Boolean))].sort().forEach(s => {
    const opt = document.createElement("option"); opt.value = s; opt.textContent = s; skillSel.appendChild(opt);
  });
  [...new Set(data.map(r => r.predicted_level).filter(Boolean))].sort().forEach(l => {
    const opt = document.createElement("option"); opt.value = l; opt.textContent = l; levelSel.appendChild(opt);
  });

  let currentDecisionFilter = "all";
  let currentSkill = "all";
  let currentLevel = "all";
  let currentSearch = "";

  const tbody = document.getElementById("rows");
  const shownCountEl = document.getElementById("shown-count");

  function rowMatches(r) {
    if (currentDecisionFilter !== "all" && decisionBucket(r.final_decision) !== currentDecisionFilter) return false;
    if (currentSkill !== "all" && r.skill !== currentSkill) return false;
    if (currentLevel !== "all" && r.predicted_level !== currentLevel) return false;
    if (currentSearch) {
      const hay = [r.text, r.skill, r.qc_reason, r.fairness_qc_reason, r.final_decision_reason]
        .join(" ").toLowerCase();
      if (!hay.includes(currentSearch)) return false;
    }
    return true;
  }

  function render() {
    tbody.innerHTML = "";
    const filtered = data.filter(rowMatches);
    filtered.forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" + escapeHtml(r.question_id) + "</td>" +
        "<td>" + escapeHtml(r.skill) + "</td>" +
        "<td class='text-cell'>" + escapeHtml(r.text) + "</td>" +
        "<td>" + escapeHtml(r.level) + "</td>" +
        "<td>" + escapeHtml(r.predicted_level) + "</td>" +
        "<td class='reason-cell'>" + escapeHtml(r.level_change_reason) + "</td>" +
        "<td>" + badgeHtml(r.qc_label) + "</td>" +
        "<td class='reason-cell'>" + escapeHtml(r.qc_reason) + "</td>" +
        "<td>" + badgeHtml(r.fairness_qc_label) + "</td>" +
        "<td class='reason-cell'>" + escapeHtml(r.fairness_qc_reason) + "</td>" +
        "<td>" + escapeHtml(r.metric_flags) + "</td>" +
        "<td>" + badgeHtml(r.final_decision) + "</td>" +
        "<td class='reason-cell'>" + escapeHtml(r.final_decision_reason) + "</td>";
      tr.addEventListener("click", () => openModal(r));
      tbody.appendChild(tr);
    });
    shownCountEl.textContent = filtered.length + " of " + data.length + " shown";
  }

  // ---- modal ----
  const overlay = document.getElementById("overlay");
  function openModal(r) {
    document.getElementById("m-id").textContent = r.sub_question_id || r.question_id || "";
    document.getElementById("m-decision-badge").innerHTML = badgeHtml(r.final_decision);
    document.getElementById("m-sub").textContent =
      (r.skill || "") + " \u00b7 labeled " + (r.level || "n/a") + " \u2192 predicted " + (r.predicted_level || "n/a");
    document.getElementById("m-text").textContent = r.text || "";
    document.getElementById("m-content-badge").innerHTML = badgeHtml(r.qc_label);
    document.getElementById("m-content-reason").textContent = r.qc_reason || "";
    document.getElementById("m-fairness-badge").innerHTML = badgeHtml(r.fairness_qc_label);
    document.getElementById("m-fairness-reason").textContent = r.fairness_qc_reason || "";
    document.getElementById("m-construction-badge").innerHTML = badgeHtml(r.construction_qc_label);
    document.getElementById("m-construction-reason").textContent = r.construction_qc_reason || "";
    document.getElementById("m-level-reason").textContent = r.level_change_reason || "";
    document.getElementById("m-rule-flags").textContent = r.metric_flags || "none";
    document.getElementById("m-final-reason").textContent = r.final_decision_reason || "";
    overlay.classList.add("open");
  }
  document.getElementById("close-btn").addEventListener("click", () => overlay.classList.remove("open"));
  overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.classList.remove("open"); });

  // ---- controls ----
  document.querySelectorAll("#decision-chips .chip").forEach(chip => {
    chip.addEventListener("click", () => {
      document.querySelectorAll("#decision-chips .chip").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      currentDecisionFilter = chip.dataset.filter;
      render();
    });
  });
  skillSel.addEventListener("change", () => { currentSkill = skillSel.value; render(); });
  levelSel.addEventListener("change", () => { currentLevel = levelSel.value; render(); });
  document.getElementById("search").addEventListener("input", (e) => {
    currentSearch = e.target.value.trim().toLowerCase();
    render();
  });

  render();
</script>
</body>
</html>"""


def save_html_report(rows: list, path: str) -> None:
    # Ensure parent directory exists
    parent_dir = os.path.dirname(os.path.abspath(path))
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    table_rows = []
    for row in rows:
        table_rows.append({key: row.get(key, "") for key, _ in HTML_REPORT_COLUMNS})
    data_json = json.dumps(table_rows, ensure_ascii=False).replace("</", "<\\/")

    html = (
        _HTML_REPORT_TEMPLATE
        .replace("__DATA_JSON__", data_json)
        .replace("__GENERATED__", time.strftime("%Y-%m-%d %H:%M:%S"))
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
        f.flush()




def save_results(rows: list, path: str, compact: bool = False) -> None:
    # Ensure parent directory exists
    parent_dir = os.path.dirname(os.path.abspath(path))
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    for row in rows:
        if not row.get("cefr_level") and row.get("level"):
            row["cefr_level"] = row["level"]
        if not row.get("question_type") and row.get("type"):
            row["question_type"] = row["type"]

    rows = [{k: v for k, v in row.items() if k not in (
        "text", "level", "type", "construction_qc_label", "fairness_qc_label",
        "qc_reason", "construction_qc_reason", "level_match",
    )} for row in rows]

    if compact:
        rows = [{k: v for k, v in row.items() if k in COMPACT_COLUMN_ORDER} for row in rows]

    if path.lower().endswith(".json"):
        fieldnames = _ordered_fieldnames(rows)
        ordered_rows = [{k: row.get(k, "") for k in fieldnames if k in row} for row in rows]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ordered_rows, f, indent=2, ensure_ascii=False)
    else:
        fieldnames = _ordered_fieldnames(rows)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            for row in rows:
                row = dict(row)
                if isinstance(row.get("options"), list):
                    row["options"] = "|".join(row["options"])
                writer.writerow(row)


def run_full_qc(item: dict, flags: list, level_tolerance: int) -> dict:
    predicted_level, level_reason = classify_level(item)
    predicted_cognitive_skill, cognitive_skill_match, cognitive_skill_reason = classify_cognitive_skill(item)

    labeled_level = (item.get("level") or "").strip().upper()
    level_match = _level_within_range(labeled_level, predicted_level, level_tolerance)

    if predicted_level in ("ERROR", "UNKNOWN"):
        level_change_reason = "Could not determine level: %s" % level_reason
    elif not labeled_level:
        level_change_reason = "No labeled level -- predicted %s: %s" % (predicted_level, level_reason)
    elif labeled_level != predicted_level:
        level_change_reason = "Changed from %s to %s: %s" % (labeled_level, predicted_level, level_reason)
    else:
        level_change_reason = "No change -- confirmed %s: %s" % (labeled_level, level_reason)

    skill = item.get("skill", "")
    scores = score_metrics(item, predicted_level, flags)
    label, reason = qc_verdict(level_match, scores, flags, skill)

    out = {
        "predicted_level": predicted_level, "level_match": level_match,
        "level_change_reason": level_change_reason,
    }
    for key in ALL_METRIC_KEYS:
        out[key] = scores[key] if key in scores else "n/a"

    out["predicted_cognitive_skill"] = predicted_cognitive_skill
    out["cognitive_skill_reason"] = cognitive_skill_reason or "pass"
    out["qc_label"] = label
    out["qc_reason"] = reason
    return out


def main():
    ap = argparse.ArgumentParser(description="QC-flag items on Accuracy/Grammar/Clarity/Completeness/CEFR Level")
    ap.add_argument("--input", required=True, help="Path to input .json or .csv file")
    ap.add_argument("--output", required=True, help="Path to output .json or .csv file")
    ap.add_argument("--level-tolerance", type=int, default=LEVEL_TOLERANCE)
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--reform", action="store_true")
    ap.add_argument("--reform-attempts", type=int, default=2)
    ap.add_argument("--skill-filter", default="reading")
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--html-output", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force-recalculate", action="store_true")
    ap.add_argument("--live-interval", type=float, default=10.0,
                     help="Seconds between live CSV/HTML writes while processing runs. 0 disables live updates.")
    args = ap.parse_args()

    # Create target output directory if missing
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = args.checkpoint or (args.output + ".checkpoint.jsonl")

    items = load_items(args.input)
    if not items:
        print("No items found in input file.", file=sys.stderr)
        sys.exit(1)

    groups = OrderedDict()
    for idx, item in enumerate(items):
        if not item.get("sub_question_id"):
            item["sub_question_id"] = "idx_%d" % idx
        groups.setdefault(item.get("question_id", idx), []).append(item)

    redundancy_flags = {}
    for group_items in groups.values():
        redundancy_flags.update(detect_redundant_and_trivial(group_items))

    checkpoint_cache = {}
    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("sub_question_id"):
                        checkpoint_cache[rec["sub_question_id"]] = rec
                except json.JSONDecodeError:
                    continue

    checkpoint_lock = threading.Lock()
    checkpoint_file = open(checkpoint_path, "a", encoding="utf-8")

    def save_checkpoint(rec: dict) -> None:
        with checkpoint_lock:
            checkpoint_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            checkpoint_file.flush()

    print_lock = threading.Lock()

    def _process_item_inner(numbered_item):
        i, item = numbered_item
        log = []
        item_redundancy_flags = redundancy_flags.get(item.get("sub_question_id"), [])
        flags = metric_flags(item) + item_redundancy_flags
        out = dict(item)
        out["metric_flags"] = ", ".join(flags) if flags else "none"

        item_skill = (item.get("skill", "") or "").strip().lower()
        filter_list = [s.strip().lower() for s in args.skill_filter.split(",")]
        skill_ok = ("all" in filter_list) or (item_skill in filter_list)

        if args.skip_llm or not skill_ok:
            blank_metrics = {key: "" for key in ALL_METRIC_KEYS}
            out.update({
                "predicted_level": "", "level_match": "", "level_change_reason": "", **blank_metrics,
                "predicted_cognitive_skill": "", "cognitive_skill_reason": "",
                "qc_label": "SKIPPED", "qc_reason": "Skipped or non-LLM",
                "construction_flags": "", "construction_qc_label": "SKIPPED", "construction_qc_reason": "",
                "fairness_flags": "", "fairness_qc_label": "SKIPPED", "fairness_qc_reason": "",
            })
            out["final_decision"], out["final_decision_reason"] = compute_final_decision(
                out["qc_label"], out["construction_qc_label"], out["fairness_qc_label"], out["metric_flags"]
            )
            return out

        qc = run_full_qc(item, flags, args.level_tolerance)
        out.update(qc)

        effective_level = qc["predicted_level"] if qc["predicted_level"] in CEFR_LEVELS else (
            (item.get("level") or item.get("cefr_level") or "").strip().upper()
        )
        construction_result = evaluate_construction(item, predicted_level=effective_level)
        fairness_result = evaluate_fairness(item, predicted_level=effective_level)

        c_flags = construction_flags(item, construction_result, level=effective_level)
        c_label, c_reason = construction_qc_gate(c_flags, construction_result.get("reason", ""))
        out["construction_flags"] = ", ".join(c_flags) if c_flags else "none"
        out["construction_qc_label"] = c_label
        out["construction_qc_reason"] = c_reason

        f_flags = fairness_flags(item, fairness_result)
        f_label, f_reason = fairness_qc_gate(f_flags, fairness_result.get("reason", ""))
        out["fairness_flags"] = ", ".join(f_flags) if f_flags else "none"
        out["fairness_qc_label"] = f_label
        out["fairness_qc_reason"] = f_reason

        out["final_decision"], out["final_decision_reason"] = compute_final_decision(
            out["qc_label"], out["construction_qc_label"], out["fairness_qc_label"],
            out["metric_flags"], out["qc_reason"], out["construction_qc_reason"],
            out["fairness_qc_reason"], level_match=out.get("level_match", True),
        )

        log.append("[%d/%d] %s -> %s" % (i, len(items), qc["qc_label"], qc["qc_reason"]))
        with print_lock:
            for line in log:
                print(line)
        return out

    def process_item(numbered_item):
        i, item = numbered_item
        key = item.get("sub_question_id")
        cached = checkpoint_cache.get(key)
        if cached is not None and not args.force_recalculate:
            with print_lock:
                print("[%d/%d] reusing cached checkpoint result" % (i, len(items)))
            return cached
        out = _process_item_inner(numbered_item)
        save_checkpoint(out)
        return out

    # Resolve the HTML output path up front so both the live-update thread
    # and the final save use the same destination.
    html_path = None
    if (args.html_output or "").strip().lower() != "none":
        html_path = args.html_output
        if not html_path:
            base, _ = os.path.splitext(args.output)
            html_path = base + ".html"

    # results_slots is filled in-place by worker threads as each item
    # finishes, so the live-saver thread (or anything else) can read a
    # partial, always-up-to-date snapshot while the pool is still running.
    results_slots = [None] * len(items)
    results_lock = threading.Lock()
    last_saved_count = 0

    def live_snapshot():
        with results_lock:
            snapshot = [r for r in results_slots if r is not None]
        return snapshot

    def do_live_save(force: bool = False):
        nonlocal last_saved_count
        snapshot = live_snapshot()
        if not snapshot:
            return
        if not force and len(snapshot) == last_saved_count:
            return  # nothing new since the last write
        save_results(snapshot, args.output, compact=args.compact)
        if html_path:
            save_html_report(snapshot, html_path)
        last_saved_count = len(snapshot)
        with print_lock:
            print("  (live update: %d/%d results written to %s%s)" % (
                len(snapshot), len(items), os.path.abspath(args.output),
                " and " + os.path.abspath(html_path) if html_path else "",
            ))

    stop_live_saving = threading.Event()

    def live_saver():
        # Wakes up every --live-interval seconds and flushes whatever has
        # completed so far to CSV/JSON + HTML. wait() returns True as soon
        # as stop_live_saving is set, which ends the loop promptly.
        while not stop_live_saving.wait(args.live_interval):
            do_live_save()

    live_thread = None
    if args.live_interval and args.live_interval > 0:
        live_thread = threading.Thread(target=live_saver, daemon=True)
        live_thread.start()

    def process_and_store(numbered_item):
        i, _item = numbered_item
        out = process_item(numbered_item)
        with results_lock:
            results_slots[i - 1] = out
        return out

    results = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            results = list(pool.map(process_and_store, enumerate(items, 1)))
    finally:
        # Guarantee checkpoint file closure
        checkpoint_file.close()

        # Stop the live-update thread now that processing has finished
        # (or was interrupted/cancelled).
        stop_live_saving.set()
        if live_thread is not None:
            live_thread.join(timeout=args.live_interval + 5)

        # ALWAYS save results & HTML report on completion or if cancelled,
        # using whatever made it into results_slots even on early exit.
        final_rows = results if results else live_snapshot()
        if final_rows:
            save_results(final_rows, args.output, compact=args.compact)
            print("\nWrote %d results to %s" % (len(final_rows), os.path.abspath(args.output)))

            if html_path:
                save_html_report(final_rows, html_path)
                print("Wrote HTML report to %s" % os.path.abspath(html_path))


if __name__ == "__main__":
    main()