import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Raise deepeval's per-call timeout (default ~90s) before the judge model is
# constructed below. Slow-but-fine responses from the judge model were
# otherwise timing out and silently reading downstream as a worst-case
# score (see classify_and_score()), not as "we don't know" -- so a slow response
# was indistinguishable from a genuinely bad item. Only set if the caller
# hasn't already overridden it.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "180")

from deepeval.models import GeminiModel

# ----------------------------------------------------------------------
# deepeval judge model
# ----------------------------------------------------------------------
# Every LLM call in this file -- classify_and_score(), evaluate_construction(),
# classify_fairness(), and reform_item() -- goes through the same deepeval-wrapped judge model below
# (_JUDGE_MODEL), the hosted gemma-4-26b-a4b-it model via the Gemini API
# (Google AI Studio), through deepeval's DeepEvalBaseLLM.generate()
# interface (_judge_call()). There is no local/Ollama path left anywhere in
# this file. Requires GOOGLE_API_KEY to be set in the environment (or pass
# api_key= directly below) and `pip install google-genai`.
_JUDGE_MODEL = GeminiModel(
    model="gemma-4-26b-a4b-it",
    temperature=0.2,
)

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

SKILL_METRIC_CONFIG = {
    "listening": {
        "check_audio_transcript": True,
        "metrics": [
            {"key": "accuracy_pct", "label": "accuracy", "scale": "pct", "compare": "min", "threshold": 90,
             "definition": "Is the question and its expected answer factually/grammatically correct together, consistent with the transcript?"},
            {"key": "grammar_errors", "label": "grammar errors", "scale": "count", "compare": "max", "threshold": 0,
             "definition": "Count of grammar/spelling errors in the question text and answer. 0 = no errors found."},
            {"key": "clarity_score", "label": "clarity", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "How easy is the question to understand, 5 = completely unambiguous, 1 = very confusing."},
            {"key": "completeness_score", "label": "completeness", "scale": "five", "compare": "min", "threshold": 4,
             "definition": "Is all necessary information/context provided to answer the question, with no missing context."},
        ],
    },
    "_default": {
        "check_audio_transcript": False,
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

# Rule flags in this set are surfaced as-is in the metric_flags column, but
# are too minor/noisy to drive an item to FLAG on their own -- shared by
# qc_verdict() (content_qc_label) and compute_final_decision() (rule flag
# reporting) so the two agree on what counts as "real" content problem.
_IGNORED_MINOR_FLAGS = {"missing_end_punctuation"}


ALL_METRIC_KEYS = []
for _cfg in SKILL_METRIC_CONFIG.values():
    for _m in _cfg["metrics"]:
        if _m["key"] not in ALL_METRIC_KEYS:
            ALL_METRIC_KEYS.append(_m["key"])


def _skill_config(skill: str) -> dict:
    key = (skill or "").strip().lower()
    return SKILL_METRIC_CONFIG.get(key, SKILL_METRIC_CONFIG["_default"])
 

# ----------------------------------------------------------------------
# 1b. Cross-question redundancy / triviality checks
# ----------------------------------------------------------------------
# Two failure modes this catches, neither of which the per-item rule checks
# or the CEFR classifier can see, because both only ever look at ONE
# question in isolation:
#
#   1. REDUNDANT PAIRS: two questions on the same passage that ask about the
#      same fact in different words, e.g. "Where does Gokul work?" and
#      "Who works at Google?" -- these aren't testing two different
#      comprehension skills, they're the same probe twice.
#
#   2. TRIVIAL LOOKUP: a question whose answer is a single proper
#      noun/number that also appears verbatim in the question itself (or is
#      the *only* content word that differs between question and answer),
#      meaning it can be answered by string-matching the passage rather than
#      by actually understanding it -- too easy regardless of the passage's
#      labeled CEFR level.

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
    """Returns the actual answer CONTENT for a question, resolving MCQ
    answer letters (e.g. "A") to the text of the matching option (e.g.
    "Google"). Without this, an MCQ whose answer field is just a bare
    letter is invisible to redundancy/triviality comparisons -- two
    questions asking the same fact, one as MCQ and one as short_answer,
    would otherwise look completely unrelated."""
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
                # Strip a leading "A. " / "A) " label if the option text
                # repeats it, so we compare on the substance, not the letter.
                option_text = re.sub(r"^[A-Za-z][\.\)]\s*", "", option_text)
                return option_text
    return raw


FULL_REDUNDANCY_SIM_THRESHOLD = 0.75     # near-identical question wording
PARTIAL_REDUNDANCY_SIM_THRESHOLD = 0.35  # moderate overlap, still worth a look
PARTIAL_ANSWER_SIM_FLOOR = 0.15          # min wording overlap to count answer-only overlap as partial
REASONING_WORDS = {
    "infer", "inferred", "imply", "implies", "suggest", "suggests", "why",
    "explain", "evaluate", "compare", "contrast", "analyze", "opinion",
    "meaning", "purpose", "tone", "attitude",
}


def detect_redundant_and_trivial(doc_questions: list) -> dict:
    """doc_questions: list of item dicts that all share the same passage
    (same question_id). Returns {sub_question_id: [flags]}.

    Redundancy is split into two tiers rather than one blanket flag:

      - FULLY redundant: the two questions test the exact same fact and add
        zero discriminating value over each other. Triggered by either the
        subject/object swap signature (e.g. "Where does Gokul work?" /
        "Who works at Google?") or near-identical question wording with the
        same answer (e.g. "What is Gokul's job?" / "What is Gokul's
        occupation?"). One of the pair should probably just be cut.

      - PARTIALLY redundant: the two questions overlap meaningfully (similar
        wording, or the answers share content) but aren't simply the same
        probe restated -- e.g. one asks what Gokul does every morning
        (answer mentions the metro) and another asks specifically how he
        commutes (answer: by metro). These aren't necessarily wrong, but are
        worth a human glance since they're testing overlapping ground.
    """
    flags_by_id = {q.get("sub_question_id", i): [] for i, q in enumerate(doc_questions)}
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

            # Subject/object swap: e.g. "Where does Gokul work?" (answer:
            # Google) vs "Who works at Google?" (answer: Gokul). Wording
            # overlap between the two questions is low, so Jaccard alone
            # misses this -- but each answer is a keyword sitting right in
            # the OTHER question, which is the signature of "same fact,
            # inverted". This is always a FULL duplicate.
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

    # --- Trivial lookup: answer is a bare proper noun/number already
    #     sitting in the question, with no reasoning language required ---
    for i, q in enumerate(doc_questions):
        question_text = q.get("question") or q.get("text", "")
        answer_text = _answer_text(q.get("answer"))
        q_words = _content_words(question_text)
        a_words = _content_words(answer_text)

        if not a_words or not q_words:
            continue

        has_reasoning_language = bool(q_words & REASONING_WORDS)
        answer_len_ok = len(a_words) <= 2  # short factual answer
        # every content word in the answer also literally appears in the
        # question -- i.e. the "answer" is just copy-pasted from the prompt
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
    skill = (item.get("skill") or "").strip().lower()

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
    # Manual override: set "skip_audio_check": true (or "no_audio_needed": true)
    # on an individual item in your input file to suppress this flag for that
    # question specifically, e.g. when the passage genuinely has no separate
    # audio/transcript by design. Everything else still gets checked normally.
    # _truthy() is used instead of bool() because CSV input gives string
    # values -- bool("false") is True in Python, which would silently
    # suppress the flag on rows where the column is literally the text
    # "false", "0", "no", or empty.
    def _truthy(val):
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() in ("1", "true", "yes", "y")

    _skip_audio_check = _truthy(item.get("skip_audio_check")) or _truthy(item.get("no_audio_needed"))
    if (_skill_config(skill).get("check_audio_transcript") and not item.get("audio")
            and not item.get("transcript") and not _skip_audio_check):
        flags.append("missing_audio_or_transcript")
    if text and not re.search(r"[.?!]\s*$", text.replace("___", "").replace("____", "")):
        flags.append("missing_end_punctuation")
    if re.search(r"[ \t]{2,}", text):
        flags.append("double_spacing")
    return flags


# ----------------------------------------------------------------------
# 1c. Human-readable explanations for rule-based flags
# ----------------------------------------------------------------------
# metric_flags()/detect_redundant_and_trivial() intentionally emit short
# machine-friendly codes (e.g. "answer_whitespace"). Those codes are fine
# for filtering/grouping, but on their own they don't tell a reviewer WHAT
# went wrong or WHERE to look. describe_rule_flags() expands each code into
# a plain-English sentence naming the specific field/location involved, so
# that every place a flag list gets turned into a reason string (content QC
# reason, final_decision_result) reads as an actual explanation rather than
# a bare code.
RULE_FLAG_DESCRIPTIONS = {
    "answer_whitespace": "the 'answer' field has leading/trailing whitespace",
    "text_whitespace": "the question/passage 'text' field has leading/trailing whitespace",
    "missing_text": "the question/passage 'text' field is empty",
    "missing_answer": "the 'answer' field is empty",
    "invalid_level": "the labeled 'level' field is missing or not one of A1/A2/B1/B2/C1/C2",
    "no_blank_marker": "type is 'fill_up' but the text has no ___ blank marker",
    "missing_options": "type is MCQ but the 'options' field has no answer choices",
    "missing_audio_or_transcript": "a listening item is missing both 'audio' and 'transcript'",
    "missing_end_punctuation": "the question text does not end in . ? or !",
    "double_spacing": "the question/passage text contains a double (or larger) space",
    "trivial_lookup": "the answer is a bare word/number copied straight out of the question, so it can be found by string-matching rather than by understanding the passage",
}


def _describe_one_rule_flag(flag: str) -> str:
    if flag.startswith("fully_redundant_with:"):
        other_id = flag.split(":", 1)[1]
        return "this question fully duplicates question '%s' on the same passage (same fact, different wording)" % other_id
    if flag.startswith("partially_redundant_with:"):
        other_id = flag.split(":", 1)[1]
        return "this question overlaps significantly with question '%s' on the same passage (similar wording/answer)" % other_id
    return RULE_FLAG_DESCRIPTIONS.get(flag, flag.replace("_", " "))


def describe_rule_flags(flags: list) -> str:
    """Turns a list of rule-based flag codes into a single semicolon-joined
    string of "code (explanation)" entries, e.g.
    "missing_options (type is MCQ but the 'options' field has no answer
    choices)". Returns "" for an empty/falsy flag list."""
    if not flags:
        return ""
    return "; ".join("%s (%s)" % (f, _describe_one_rule_flag(f)) for f in flags)


# ----------------------------------------------------------------------
# 2. Judge model (Gemini via deepeval) calls
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

COGNITIVE_SKILLS = ["recall", "comprehension", "vocabulary", "inference"]

COGNITIVE_SKILL_DESCRIPTORS = """recall - The answer is stated verbatim (or with only trivial rewording) in
     the passage/transcript. No synthesis or reasoning is needed, just
     locating and copying the fact.
comprehension - The answer requires understanding and lightly restating or
     connecting information from the passage/transcript (e.g. summarizing a
     stated role, purpose, or relationship), but is NOT a word-for-word
     lift and does NOT require the reader to supply a conclusion the text
     itself doesn't already state.
vocabulary - The question asks what a specific word or phrase means, or
     tests understanding of a term's meaning in context.
inference - The answer is NOT explicitly stated in the passage/transcript.
     The reader must draw a conclusion, motive, cause, or implication that
     the text only suggests indirectly. If the question's own wording
     already states the conclusion (e.g. "...implies X" where X is the
     literal implication) and the "answer" blank is just a quoted phrase
     from the text, it is NOT a true inference item -- it's recall or
     comprehension wearing an inference label.
"""

COGNITIVE_SKILL_PROMPT = """You are a reading/listening item-quality reviewer for a
language-learning platform. Classify which COGNITIVE SKILL the QUESTION below
actually requires a learner to use in order to answer it correctly, using the
background material (reading passage or listening transcript) as ground truth
for what is and isn't explicitly stated.

Cognitive skill definitions (use these as your classification criteria):
{descriptors}

Background material -- passage or transcript:
{background}

Skill: {skill}
Type: {type}
QUESTION to classify: {question}
Expected answer: {answer}
Options: {options}

Judge strictly by whether the expected answer is explicitly stated in the
background material (-> recall or comprehension depending on how much
restating/connecting is required), asks about word meaning (-> vocabulary),
or requires the reader to work out something the text never says outright
(-> inference). A question stem that spells out the implication for the
reader, or an answer that is a verbatim/near-verbatim quote from the
background material, is NOT inference even if the item is labeled that way.

Respond with ONLY a JSON object, nothing else, in this exact shape:
{{"cognitive_skill": "<one of: recall, comprehension, vocabulary, inference>", "reason": "<one short sentence explaining why, referencing whether/where the answer appears in the background material>"}}
"""


FAIRNESS_PROMPT = """You are a strict fairness/bias/accessibility reviewer for a
language-learning reading/listening platform, checking a single ITEM for
problems that would make it unfair or inappropriate for a general,
international adult audience.

CALIBRATION -- read carefully before scoring: the overwhelming majority of
well-written educational items about ordinary topics (work, science,
technical processes, hobbies, daily life) have NO fairness problems. Using
professional/technical vocabulary, describing a specific job, or setting
background material in a particular country is NOT by itself cultural
bias, and a person or pronoun appearing in a role is NOT by itself gender
bias. Only flag a dimension true when there is specific, citable evidence
of an actual stereotype, an unfair knowledge requirement, or a genuinely
inappropriate topic. If you are unsure, prefer false.

Dimensions to judge:
- requires_specialist_or_cultural_knowledge (true/false): true ONLY if
  answering requires specialist/technical knowledge or knowledge specific
  to one culture/region/religion that is NOT given in the background
  material itself and can't be reasonably expected of a general
  international audience. If the background material itself supplies
  everything needed to answer, this is false, even if the topic is
  technical.
- cultural_bias (true/false): true ONLY if the text asserts one culture's
  norms as objectively "correct" or default, or relies on an ethnic/
  national stereotype. Background material simply being set in, or
  written/spoken from the perspective of, one culture is NOT bias.
- gender_bias (true/false): true ONLY if the text relies on an explicit
  gender stereotype (e.g. assuming a role can only be done by one gender)
  or unnecessarily genders a role with no basis in the background
  material. A person in the background material having a stated gender is
  NOT by itself bias.
- sensitive_content (true/false): true ONLY if the text touches on content
  that would be genuinely distressing, political controversy, graphic
  violence, self-harm, explicit content, or hate speech for a general
  audience. Routine professional/technical subject matter is NOT
  sensitive.

Background material -- passage or transcript:
{background}

Skill: {skill} | Type: {type} | Predicted CEFR level: {predicted_level}
QUESTION: {question}
Expected answer: {answer}
Options: {options}

Respond with ONLY a JSON object, nothing else, in this exact shape:
{{"requires_specialist_or_cultural_knowledge": <true or false>, "cultural_bias": <true or false>, "gender_bias": <true or false>, "sensitive_content": <true or false>, "reason": "<one short sentence citing specific evidence for any dimension flagged true, or 'no issues found'>"}}
"""


CONSTRUCTION_PROMPT = """You are a strict item-construction QC reviewer for a
language-learning platform. You are NOT re-grading whether the labeled
answer is correct -- that is checked separately. Your ONLY job is to judge
how WELL-CONSTRUCTED this question is, independent of whether its content
is accurate.

Background material -- passage or transcript:
{background}

Skill: {skill} | Type: {type} | Labeled CEFR level: {labeled_level} | Predicted CEFR level: {predicted_level}
QUESTION: {question}
Expected answer: {answer}
Options: {options}

Judge the following, based ONLY on how the question is built. Use the
labeled/predicted CEFR level as context for what "well-constructed" means at
that level (distractor subtlety and background-grounding expectations scale
with level -- see the per-level thresholds already applied downstream):

1. distractor_plausibility (MCQ only, 1-5): For the WRONG options, how
   plausible/tempting are they to someone who hasn't read/listened
   carefully? 5 = all distractors are reasonable, on-topic, similar
   length/form to the correct answer. 1 = distractors are absurd, off-topic,
   or an obvious giveaway (e.g. much shorter/longer than the others). If not
   MCQ, return 5.

2. single_correct_answer (MCQ only, true/false): Could more than one of the
   listed options be defensibly argued as correct given the background
   material? true = only one option is correct. false = more than one
   option could be defended as correct. If not MCQ, return true.

3. background_relevance (1-5): Does the question actually require the
   background material to answer meaningfully, and does it connect clearly
   to something stated or implied there? 5 = fully grounded in the
   background material, 1 = the question doesn't meaningfully relate to it
   at all.

4. sensitive_content (true/false): Does the question or background material
   touch on content that could be inappropriate, distressing, or unsuitable
   for a general language-learning audience (e.g. graphic violence,
   self-harm, explicit content, hate speech)? true = yes, flag for human
   review.

Respond with ONLY a JSON object, nothing else, in this exact shape:
{{"distractor_plausibility": <1-5>, "single_correct_answer": <true/false>, "background_relevance": <1-5>, "sensitive_content": <true/false>, "reason": "<one short sentence covering the most important issue found, or 'no construction issues found'>"}}
"""

CONSTRUCTION_THRESHOLDS = {
    "distractor_plausibility_min": 3,
    "background_relevance_min": 3,
}

# Per-CEFR-level construction thresholds. A flat threshold treats a
# beginner item's construction quality the same as an advanced one's, which
# is backwards: at A1/A2, obviously-wrong distractors are CORRECT design (a
# beginner shouldn't be tripped up by a subtle wrong answer), so the bar for
# "plausible enough" distractors should be lower. At C1/C2, weak/obvious
# distractors defeat the point of the item -- an advanced learner needs
# genuinely tempting wrong options to be meaningfully tested, so the bar
# should be higher. background_relevance follows the same logic: a simple
# A1 item can lean on a very short, direct link to the background, while a
# C1/C2 item is expected to require tighter, more substantive grounding.
CONSTRUCTION_THRESHOLDS_BY_LEVEL = {
    "A1": {"distractor_plausibility_min": 2, "background_relevance_min": 2},
    "A2": {"distractor_plausibility_min": 2, "background_relevance_min": 3},
    "B1": {"distractor_plausibility_min": 3, "background_relevance_min": 3},
    "B2": {"distractor_plausibility_min": 3, "background_relevance_min": 4},
    "C1": {"distractor_plausibility_min": 4, "background_relevance_min": 4},
    "C2": {"distractor_plausibility_min": 4, "background_relevance_min": 4},
}


def _construction_thresholds_for_level(level: str) -> dict:
    """Looks up the level-specific threshold pair, falling back to the flat
    CONSTRUCTION_THRESHOLDS default when the level is missing, unrecognized,
    or came back as ERROR/UNKNOWN from classify_level()."""
    level = (level or "").strip().upper()
    return CONSTRUCTION_THRESHOLDS_BY_LEVEL.get(level, CONSTRUCTION_THRESHOLDS)


def _truncate_background(background: str, max_chars: int = 400) -> str:
    """Keeps only a short lead-in of the background material (reading
    passage or listening transcript) for CEFR classification context,
    instead of the full text. The full text is what was swamping the
    classifier and dragging every question in a document to the
    background material's own (usually higher) level regardless of how
    simple the individual question was -- a short lead-in still orients
    the model on topic/register without dominating the prompt. Cuts at
    the nearest sentence boundary so it doesn't end mid-word."""
    background = (background or "").strip()
    if len(background) <= max_chars:
        return background
    snippet = background[:max_chars]
    cut = max(snippet.rfind(". "), snippet.rfind("! "), snippet.rfind("? "))
    if cut > max_chars * 0.5:
        snippet = snippet[:cut + 1]
    return snippet.strip() + " [...]"

def _build_combined_prompt(skill: str) -> str:
    """Builds the single combined judge-model prompt used by
    classify_and_score() for CEFR level + cognitive skill + QC metrics in
    one call. Built dynamically per skill so the metric fields match
    SKILL_METRIC_CONFIG.

    Uses a {{...}} -> literal-brace-after-.format() trick: %s placeholders
    are substituted immediately (skill name, metric lines, json fields),
    while {descriptors}, {cog_descriptors}, {skill}, {type}, etc. stay as
    .format() placeholders for classify_and_score() to fill in per-item."""
    config = _skill_config(skill)
    metric_lines = []
    json_fields = []
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

    return (
        "You are a combined CEFR-level classifier, cognitive-skill "
        "classifier, and QC metric scorer for a language-learning platform, "
        "reviewing a single %s item. Produce THREE independent judgments "
        "about the QUESTION below in one response -- rate each on its own "
        "criteria, don't let one judgment bias another.\n\n"
        "=== PART 1: CEFR LEVEL ===\n"
        "Classify the difficulty of the QUESTION itself (not the background "
        "passage/transcript) using these official descriptors:\n"
        "{descriptors}\n"
        "Background material is often written/spoken in a noticeably "
        "higher, more literary or fast-paced register than the question -- "
        "do NOT let its vocabulary or complexity drive your rating. Base "
        "the level on the QUESTION's own vocabulary difficulty, grammatical "
        "structures, and how much inference it demands. Do NOT answer the "
        "question or evaluate whether the expected answer is correct -- "
        "that is a separate, already-solved task. The level field must be "
        "exactly one of A1, A2, B1, B2, C1, C2 -- never an answer-option "
        "letter and never a level name written out in words.\n\n"
        "=== PART 2: COGNITIVE SKILL ===\n"
        "Classify which cognitive skill the QUESTION requires a learner to "
        "use to answer it correctly, using the background material as "
        "ground truth for what is/isn't explicitly stated:\n"
        "{cog_descriptors}\n"
        "Judge strictly by whether the expected answer is explicitly stated "
        "in the background material (-> recall or comprehension, depending "
        "on how much restating/connecting is required), asks about word "
        "meaning (-> vocabulary), or requires working out something the "
        "text never says outright (-> inference). A question stem that "
        "spells out the implication for the reader, or an answer that is a "
        "verbatim/near-verbatim quote from the background material, is NOT "
        "inference even if the item is labeled that way.\n\n"
        "=== PART 3: QC METRICS ===\n"
        "Evaluate the item against these metrics:\n%s\n\n"
        "Item:\n"
        "Skill: {skill} | Type: {type} | Labeled CEFR level: {labeled_level}\n"
        "Background material -- passage or transcript (context only -- do "
        "NOT base the CEFR rating on this):\n{background}\n"
        "QUESTION: {question}\n"
        "Expected answer: {answer}\n"
        "Options: {options}\n"
        "Rule-based flags already detected: {metric_flags}\n\n"
        "Respond with ONLY a JSON object, nothing else, in this exact shape:\n"
        '{{"level": "<one of: A1, A2, B1, B2, C1, C2>", '
        '"level_reason": "<one short sentence citing specific words/structures in the QUESTION itself>", '
        '"cognitive_skill": "<one of: recall, comprehension, vocabulary, inference>", '
        '"cognitive_skill_reason": "<one short sentence explaining why>", '
        '%s}}\n'
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


import random

# ----------------------------------------------------------------------
# Rate-limit-aware judge-model calling
# ----------------------------------------------------------------------
# The free tier of the Gemini API enforces a per-minute quota (tokens AND/
# or requests, depending on plan). With --workers > 1, every thread was
# independently retrying on a flat 1.5s*(attempt+1) backoff regardless of
# what the API actually asked for, so on a real 429 all N worker threads
# would hammer the same exhausted quota roughly simultaneously, each
# burning its own retry budget and frequently giving up entirely
# (ERROR/UNKNOWN) well before the quota window had actually reset. The
# pieces below fix that:
#   1. _extract_retry_delay() reads the server-suggested wait time Gemini
#      already includes in a 429's error details (e.g. {'retryDelay': '8s'})
#      instead of guessing.
#   2. _RATE_LIMIT_UNTIL is a timestamp shared across ALL worker threads:
#      the first thread to hit a 429 sets it, and every other thread
#      checks it before firing its own call, so the whole pool backs off
#      together instead of independently re-discovering the same outage.
#   3. _pace_calls() is an OPTIONAL proactive throttle (off by default) --
#      set JUDGE_MAX_CALLS_PER_MINUTE to cap how many calls start per
#      minute across the whole pool, so you can stay under a known RPM
#      limit and avoid triggering 429s in the first place rather than
#      just reacting to them faster.
_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_UNTIL = 0.0

_MAX_CALLS_PER_MINUTE = int(os.environ.get("JUDGE_MAX_CALLS_PER_MINUTE", "0") or "0")
_CALL_TIMES_LOCK = threading.Lock()
_CALL_TIMES = []  # sliding 60s window of call start times; only used if pacing enabled

_RETRY_DELAY_RE = re.compile(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'")


def _extract_retry_delay(err) -> float:
    """Pulls the server-suggested wait time out of a 429 RESOURCE_EXHAUSTED
    error's message. Returns None if the error doesn't include one (i.e.
    it's not a rate-limit error, or came from a provider with a different
    error shape) so the caller can fall back to a flat backoff."""
    match = _RETRY_DELAY_RE.search(str(err))
    return float(match.group(1)) if match else None


def _pace_calls():
    """Blocks the calling thread until it's safe to start another judge-
    model call, based on JUDGE_MAX_CALLS_PER_MINUTE (a sliding 60s window
    shared across all worker threads). No-op if pacing is disabled
    (the default) -- this only proactively slows things down when you've
    told it your account's actual per-minute limit; it does nothing to
    help you discover that limit."""
    if _MAX_CALLS_PER_MINUTE <= 0:
        return
    while True:
        now = time.time()
        with _CALL_TIMES_LOCK:
            cutoff = now - 60
            while _CALL_TIMES and _CALL_TIMES[0] < cutoff:
                _CALL_TIMES.pop(0)
            if len(_CALL_TIMES) < _MAX_CALLS_PER_MINUTE:
                _CALL_TIMES.append(now)
                return
            wait_for = _CALL_TIMES[0] + 60 - now
        time.sleep(max(wait_for, 0.05))


def _judge_call(prompt: str, retries: int = 4) -> str:
    """Calls _JUDGE_MODEL through deepeval's DeepEvalBaseLLM.generate()
    interface. retries raised from 2 to 4: on a real quota outage the old
    2-retry budget often expired before the quota window even reset,
    turning a temporary 429 into a permanent ERROR/UNKNOWN for that item.
    Backoff behavior: on a rate-limit error, wait exactly what the API
    asked for (plus jitter) and tell every other worker thread to do the
    same via _RATE_LIMIT_UNTIL; on any other error, fall back to the old
    flat 1.5s*(attempt+1) backoff. All prompts in this file already
    instruct the model to respond with ONLY a JSON object, so every caller
    still parses JSON out of the returned text -- Gemini via deepeval
    doesn't offer Ollama's hard `format: json` constraint, so the
    regex-extract-then-json.loads + retry pattern downstream now carries
    a bit more of the reliability burden than it used to."""
    global _RATE_LIMIT_UNTIL
    last_err = None
    for attempt in range(retries + 1):
        with _RATE_LIMIT_LOCK:
            wait_for = _RATE_LIMIT_UNTIL - time.time()
        if wait_for > 0:
            time.sleep(wait_for)

        _pace_calls()

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
            retry_delay = _extract_retry_delay(e)
            if retry_delay is not None:
                with _RATE_LIMIT_LOCK:
                    _RATE_LIMIT_UNTIL = max(_RATE_LIMIT_UNTIL, time.time() + retry_delay)
                time.sleep(retry_delay + random.uniform(0.1, 1.0))
            else:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("Could not reach judge model (Gemini via deepeval): %s" % last_err)


def classify_fairness(item: dict, predicted_level: str = "") -> dict:
    """FAIRNESS/BIAS/SENSITIVITY gate, judging FOUR independent dimensions
    (specialist/cultural-knowledge requirement, cultural bias, gender bias,
    sensitive content) in a single structured-JSON judge call, rather than
    one blended "fair"/"unfair" verdict -- so fairness_flags() can report
    exactly which dimension(s) tripped instead of one opaque label. See
    FAIRNESS_PROMPT for the calibration guidance this relies on to avoid
    over-flagging routine technical/professional content. Same fallback/
    error-flagging shape as evaluate_construction(), so a failed or
    unparseable call reads as an explicit error rather than a silent PASS.

    `predicted_level` should be the item's predicted_level from
    classify_level(), so this gate has the same CEFR context as the other
    gates."""
    question = item.get("question_text") or item.get("question") or item.get("text", "")
    background = _truncate_background(item.get("content", ""))

    prompt = FAIRNESS_PROMPT.format(
        skill=item.get("skill", ""),
        type=item.get("type", ""),
        predicted_level=predicted_level or "(none)",
        background=background or "(none provided)",
        question=question,
        answer=item.get("answer", ""),
        options=item.get("options", ""),
    )

    fallback = {
        "requires_specialist_or_cultural_knowledge": None, "cultural_bias": None,
        "gender_bias": None, "sensitive_content": None,
        "reason": "", "error": False,
    }

    def _bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes")
        return None

    parse_attempts = 2
    last_raw = ""
    for attempt in range(parse_attempts):
        try:
            raw = _judge_call(prompt)
        except RuntimeError as e:
            fallback["reason"] = "judge model unreachable: %s" % e
            fallback["error"] = True
            return fallback

        last_raw = raw
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            continue  # try again
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            continue  # try again

        required = ("cultural_bias", "gender_bias", "sensitive_content")
        if any(k not in parsed for k in required):
            continue  # try again -- can't score dimensions without these

        return {
            "requires_specialist_or_cultural_knowledge": _bool(
                parsed.get("requires_specialist_or_cultural_knowledge")
            ),
            "cultural_bias": _bool(parsed.get("cultural_bias")),
            "gender_bias": _bool(parsed.get("gender_bias")),
            "sensitive_content": _bool(parsed.get("sensitive_content")),
            "reason": parsed.get("reason", ""),
            "error": False,
        }

    print("    [classify_fairness] could not parse a valid fairness verdict after %d attempt(s). Raw judge model output was:\n    %s" % (
        parse_attempts, last_raw[:300]
    ))
    fallback["reason"] = "Could not parse judge model output after %d attempt(s): %s" % (
        parse_attempts, last_raw[:150]
    )
    fallback["error"] = True
    return fallback


def fairness_flags(item: dict, llm_result: dict) -> list:
    """LLM-derived flags for FAIRNESS/BIAS/SENSITIVITY -- separate from
    metric_flags() (formatting/structure), run_full_qc()'s content rubric,
    and construction_flags() (how the question is built). Purely about
    whether the item is fair and appropriate for a general audience."""
    flags = []

    if llm_result.get("requires_specialist_or_cultural_knowledge") is True:
        flags.append("requires_specialist_knowledge")
    if llm_result.get("cultural_bias") is True:
        flags.append("cultural_bias")
    if llm_result.get("gender_bias") is True:
        flags.append("gender_bias")
    if llm_result.get("sensitive_content") is True:
        flags.append("sensitive_content_review_needed")

    # Same fix as construction_flags(): rely on the explicit error flag
    # classify_fairness() sets on failure instead of inferring failure from
    # an empty reason string, since a timeout failure still fills in
    # `reason` with an error message and was silently reading as PASS.
    if llm_result.get("error"):
        flags.append("fairness_check_error")

    return flags


def evaluate_construction(item: dict, predicted_level: str = "") -> dict:
    """Single judge-model call judging item-CONSTRUCTION quality (distractors,
    single-correct-answer, background grounding, sensitive content). Fully
    independent of score_metrics()/classify_level()/classify_cognitive_skill
    -- separate prompt, separate call, separate JSON shape. On failure,
    returns conservative zero/None values plus an error reason rather than
    raising, matching the pattern used elsewhere in this file.

    `predicted_level` should be the item's predicted_level from
    classify_level() (already computed by run_full_qc() before this is
    called), passed through so the LLM judging construction quality has the
    same labeled/predicted CEFR context that score_metrics() and
    construction_flags()'s thresholds already use."""
    question = item.get("question_text") or item.get("question") or item.get("text", "")
    background = _truncate_background(item.get("content", ""))
    labeled_level = (item.get("level") or item.get("cefr_level") or "").strip().upper()

    prompt = CONSTRUCTION_PROMPT.format(
        skill=item.get("skill", ""),
        background=background or "(none provided)",
        type=item.get("type", ""),
        labeled_level=labeled_level or "(none)",
        predicted_level=predicted_level or "(none)",
        question=question,
        answer=item.get("answer", ""),
        options=item.get("options", ""),
    )

    fallback = {
        "distractor_plausibility": 0, "single_correct_answer": None,
        "background_relevance": 0,
        "sensitive_content": None, "reason": "", "error": False,
    }

    def _int(v, fallback_v=0):
        try:
            return int(v)
        except (TypeError, ValueError):
            return fallback_v

    def _bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes")
        return None

    parse_attempts = 2
    last_raw = ""
    for attempt in range(parse_attempts):
        try:
            raw = _judge_call(prompt)
        except RuntimeError as e:
            fallback["reason"] = "judge model unreachable: %s" % e
            fallback["error"] = True
            return fallback

        last_raw = raw
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            continue
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            continue

        return {
            "distractor_plausibility": _int(parsed.get("distractor_plausibility"), 0),
            "single_correct_answer": _bool(parsed.get("single_correct_answer")),
            "background_relevance": _int(parsed.get("background_relevance"), 0),
            "sensitive_content": _bool(parsed.get("sensitive_content")),
            "reason": parsed.get("reason", ""),
        }

    fallback["reason"] = "Could not parse judge model output after %d attempt(s): %s" % (
        parse_attempts, last_raw[:150]
    )
    fallback["error"] = True
    return fallback


def construction_flags(item: dict, llm_result: dict, level: str = "") -> list:
    """LLM-derived flags for QUESTION CONSTRUCTION quality -- separate from
    metric_flags() (formatting/structure) and separate from run_full_qc()'s
    content rubric. Purely about how well the question is built, never
    about whether its content is correct.

    `level` should be the item's effective CEFR level (predicted_level from
    classify_level() if available, else the labeled level) -- thresholds
    are looked up per-level via _construction_thresholds_for_level() rather
    than using one flat cutoff for every item regardless of difficulty."""
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

    # Detect a failed call (judge model unreachable/timed out, or
    # unparseable output after retries) via the explicit error flag set by
    # evaluate_construction() on failure, rather than inferring it from an
    # empty reason string -- a timeout failure still populates `reason`
    # with an error message, so that heuristic silently missed real
    # failures and let them read as a clean PASS.
    if llm_result.get("error"):
        flags.append("construction_check_error")

    return flags


def construction_qc_gate(flags: list, llm_reason: str):
    """Independent PASS/FLAG gate for CONSTRUCTION quality only. Deliberately
    separate from qc_verdict() (content accuracy/grammar/clarity/
    completeness/level) so the two dimensions are reported, and can fail,
    independently of each other."""
    if flags == ["construction_check_error"]:
        # The judge model call itself failed (unreachable/timeout/
        # unparseable output) -- this is not a real construction judgment, so it must
        # not read as CONSTRUCTION_FLAG. Surfacing it as its own label
        # keeps it distinguishable from both a clean pass and a real flag.
        return "CONSTRUCTION_ERROR", llm_reason or "construction check failed"
    if flags:
        extra = " | %s" % llm_reason if llm_reason and llm_reason != "no construction issues found" else ""
        return "CONSTRUCTION_FLAG", "; ".join(flags) + extra
    return "CONSTRUCTION_PASS", llm_reason or "meets all construction QC targets"


def classify_and_score(item: dict, flags: list) -> dict:
    """Combines CEFR-level classification, cognitive-skill classification,
    and QC metric scoring into a single judge-model call (via
    _build_combined_prompt()) instead of three sequential ones. Returns a
    dict with predicted_level/level_reason/predicted_cognitive_skill/
    cognitive_skill_reason/metric-key/reason fields for run_full_qc() to
    consume."""
    skill = item.get("skill", "")
    config = _skill_config(skill)
    question = item.get("question_text") or item.get("question") or item.get("text", "")
    background = _truncate_background(item.get("content", ""))

    prompt = _build_combined_prompt(skill).format(
        descriptors=CEFR_DESCRIPTORS,
        cog_descriptors=COGNITIVE_SKILL_DESCRIPTORS,
        skill=skill,
        type=item.get("type", ""),
        labeled_level=item.get("level", ""),
        background=background or "(none provided)",
        question=question,
        answer=item.get("answer", ""),
        options=item.get("options", ""),
        metric_flags=", ".join(flags) if flags else "none",
    )

    def _fallback_for(key):
        return -1 if key == "grammar_errors" else 0

    fail = {
        "predicted_level": "ERROR",
        "level_reason": "",
        "predicted_cognitive_skill": "ERROR",
        "cognitive_skill_reason": "",
        "reason": "",
    }
    for m in config["metrics"]:
        fail[m["key"]] = _fallback_for(m["key"])

    def _num(v, cast=int, fallback=0):
        try:
            return cast(v)
        except (TypeError, ValueError):
            return fallback

    parse_attempts = 2
    last_raw = ""
    for attempt in range(parse_attempts):
        try:
            raw = _judge_call(prompt)
        except RuntimeError as e:
            fail["level_reason"] = "judge model unreachable: %s" % e
            fail["cognitive_skill_reason"] = fail["level_reason"]
            fail["reason"] = fail["level_reason"]
            return fail

        last_raw = raw
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            continue  # try again
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            continue  # try again

        # -- level (same salvage behavior as classify_level()) --
        level = (parsed.get("level") or "").strip().upper()
        if level not in CEFR_LEVELS:
            salvage = re.search(r"\b([ABC][12])\b", level)
            if not salvage:
                salvage = re.search(r"\b([ABC][12])\b", raw)
            level = salvage.group(1) if salvage else "UNKNOWN"

        # -- cognitive skill (same salvage behavior as classify_cognitive_skill()) --
        cog_skill = (parsed.get("cognitive_skill") or "").strip().lower()
        if cog_skill not in COGNITIVE_SKILLS:
            salvage = next((s for s in COGNITIVE_SKILLS if s in cog_skill), None)
            if not salvage:
                salvage = next((s for s in COGNITIVE_SKILLS if s in raw.lower()), None)
            cog_skill = salvage if salvage else "UNKNOWN"

        # -- metrics (same missing-field handling as score_metrics()) --
        missing_fields = []
        metric_reason = ""
        result = {
            "predicted_level": level,
            "level_reason": parsed.get("level_reason", "") or raw[:200],
            "predicted_cognitive_skill": cog_skill,
            "cognitive_skill_reason": parsed.get("cognitive_skill_reason", "") or raw[:200],
        }
        for m in config["metrics"]:
            key = m["key"]
            if key not in parsed:
                missing_fields.append(key)
            result[key] = _num(parsed.get(key, _fallback_for(key)), fallback=_fallback_for(key))

        metric_reason = parsed.get("reason", "")
        if missing_fields and not metric_reason:
            metric_reason = "%s field(s) missing from model output" % ", ".join(missing_fields)
        result["reason"] = metric_reason
        return result

    print("    [classify_and_score] could not parse a valid judge response after %d attempt(s). Raw judge model output was:\n    %s" % (
        parse_attempts, last_raw[:300]
    ))
    err = "Could not parse judge model output after %d attempt(s): %s" % (parse_attempts, last_raw[:150])
    fail["level_reason"] = err
    fail["cognitive_skill_reason"] = err
    fail["reason"] = err
    return fail


def reform_item(item: dict, qc_reason: str) -> dict:
    """Ask the judge model to rewrite a flagged item so it passes QC.
    Returns a dict with the (possibly) new text/answer/options plus a
    change_summary, or the original item unchanged with an error note if
    the call/parse fails."""
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


def _cognitive_skill_match(labeled: str, predicted: str) -> bool:
    """Cognitive skill has no ordinal scale like CEFR levels do (recall and
    inference aren't "close" to each other the way B1 and B2 are), so this
    is an exact match rather than a tolerance-based comparison. An
    unrecognized labeled/predicted value never counts as a match."""
    labeled = (labeled or "").strip().lower()
    predicted = (predicted or "").strip().lower()
    if labeled not in COGNITIVE_SKILLS or predicted not in COGNITIVE_SKILLS:
        return False
    return labeled == predicted


def qc_verdict(level_match: bool, scores: dict, flags: list, skill: str = ""):
    """The cognitive-skill check is judged and reported independently (see
    predicted_cognitive_skill/cognitive_skill_reason in run_full_qc) but is
    deliberately never folded into this verdict or its reason -- a
    cognitive-skill mismatch alone never FLAGs an item."""
    config = _skill_config(skill)
    reasons = []
    if not level_match:
        reasons.append("predicted CEFR level outside defined range of labeled level")

    for m in config["metrics"]:
        key, label, threshold = m["key"], m["label"], m["threshold"]
        val = scores.get(key)

        if key == "grammar_errors":
            if val is None or val < 0:
                reasons.append("grammar score could not be determined")
            elif val > threshold:
                reasons.append("%d grammar error(s) found (target: %d)" % (val, threshold))
            continue

        if val is None:
            reasons.append("%s could not be determined" % label)
        elif m["compare"] == "min" and val < threshold:
            if m["scale"] == "pct":
                reasons.append("%s below target (%s%% < %s%%)" % (label, val, threshold))
            else:  # five
                reasons.append("%s below target (%s/5)" % (label, val))

    effective_flags = [f for f in flags if f not in _IGNORED_MINOR_FLAGS]
    if effective_flags:
        reasons.append("rule flags: %s" % describe_rule_flags(effective_flags))

    # Captured BEFORE judge_detail is appended below, so it reflects whether
    # the level mismatch is the only substantive problem -- not whether the
    # final reason STRING happens to be exactly the bare level-mismatch
    # phrase. compute_final_decision() uses this (rather than a fragile
    # string == comparison against content_qc_reason) to decide whether to
    # downgrade a level-only flag from reject to review, so an appended
    # judge detail no longer defeats that downgrade.
    level_only = (not level_match) and reasons == ["predicted CEFR level outside defined range of labeled level"]

    # The generic per-metric strings above say WHICH target was missed, but
    # not WHAT the actual problem is (e.g. which word/sentence has the
    # grammar error). The judge model is asked for that specific detail via
    # the "reason" field of the same prompt -- surface it here rather than
    # discarding it, same as construction/fairness reasons already do.
    judge_detail = (scores.get("reason") or "").strip()
    if reasons and judge_detail:
        reasons.append("detail: %s" % judge_detail)

    if reasons:
        return "FLAG", "; ".join(reasons), level_only
    return "PASS", "meets all QC targets", False


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
    "skill": ["skill", "skill_type", "category"],
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

    # Snapshot the clean question-only text under a stable output name,
    # BEFORE the passage/content gets folded into "text" below. "text" ends
    # up being passage+question combined (used internally for LLM prompts,
    # and dropped from final output); "question_text" stays just the
    # question, as a plain string regardless of whether the source data
    # called it "text", "question_text", "prompt", or "question".
    if item.get("question") != item.get("text", ""):
        item["question_text"] = item.get("text", "")

    # Reading items carry the passage in "content"/"passage"; listening items
    # carry the audio transcript in "transcript"/"audio_transcript" — both are
    # background material distinct from the actual question in "text". Fold
    # it in ahead of the question so the LLM (CEFR classification + rubric
    # scoring) sees full context instead of just the bare question — without
    # this, short/ambiguous questions like "What can be inferred...?" give
    # judge model nothing to actually classify, which is what was producing
    # UNKNOWN predicted levels.
    background_label = "Passage"
    for alias in ("content", "passage", "reading_passage", "transcript", "audio_transcript"):
        if item.get(alias):
            background = _flatten_field_value(item[alias])
            if alias in ("transcript", "audio_transcript"):
                background_label = "Transcript"
            # Stash the raw background text under a stable canonical key
            # ("content") regardless of whether the source called it
            # "content", "passage", or "transcript" -- so downstream code
            # (classify_level's light-touch background excerpt) always
            # knows where to look, no matter the skill.
            if not item.get("content"):
                item["content"] = background
            if background and item.get("text") and background not in item["text"]:
                item["text"] = "%s: %s\n\nQuestion: %s" % (background_label, background, item["text"])
            break

    return item


def _flatten_answer(ans):
    """Turns list/dict answer shapes into a single readable string, since
    every downstream check (missing_answer, accuracy scoring, etc.) expects
    answer to be a plain string.
      - fill_up items with multiple blanks: answer is a list -> join with '; '
      - match items: answer is a {statement_number: option_letter} dict
        -> render as '1->C; 2->E; 3->D'
    """
    if isinstance(ans, list):
        return "; ".join(str(a) for a in ans)
    if isinstance(ans, dict):
        return "; ".join("%s->%s" % (k, v) for k, v in ans.items())
    return ans


def _expand_nested_document(doc: dict) -> list:
    """Some exports group multiple sub-questions under one shared reading
    passage or listening transcript, e.g.:
        {"content_id": ..., "content": "<passage>", "cefr_level": "C1",
         "category": "reading", "questions": [{...}, {...}, ...]}
    or, for listening:
        {"content_id": ..., "transcript": "<transcript>", "audio_url": "...",
         "cefr_level": "B1", "category": "listening", "questions": [...]}
    Every downstream check in this script (metric_flags, classify_level,
    score_metrics, etc.) is written to expect ONE question per row. This
    expands a document like that into one flat item dict per sub-question,
    all sharing the parent passage/transcript/level/skill, so the rest of
    the pipeline doesn't need to know this nested shape exists."""
    background = doc.get("content") or doc.get("transcript") or doc.get("audio_transcript") or ""
    base = {
        "question_id": doc.get("content_id", ""),
        "skill": doc.get("category", doc.get("skill", "")),
        "cefr_level": doc.get("cefr_level", ""),
        "content": background,
    }
    if doc.get("transcript") or doc.get("audio_transcript"):
        # Also carry the original field name through to output, so a
        # listening row shows a "transcript" column (not just a
        # passage-flavored "content" column) -- "content" stays populated
        # too since that's the canonical key classify_level/score_metrics
        # read from internally, regardless of skill.
        base["transcript"] = doc.get("transcript") or doc.get("audio_transcript")
    if doc.get("audio_url") or doc.get("audio"):
        base["audio"] = doc.get("audio_url") or doc.get("audio")
    # Manual override to suppress missing_audio_or_transcript: honor it if
    # set at the shared-passage level (applies to every question under it)
    # so a whole listening doc can be exempted at once.
    if doc.get("skip_audio_check") or doc.get("no_audio_needed"):
        base["skip_audio_check"] = True
    expanded = []
    for q in doc.get("questions", []):
        row = dict(base)
        row["sub_question_id"] = q.get("sub_question_id", "")
        row["question_type"] = q.get("question_type", "")
        if q.get("cognitive_skill"):
            row["cognitive_skill"] = q["cognitive_skill"]
        # Also honor the override set on an individual sub-question, so a
        # single question can be exempted without affecting its siblings
        # under the same passage/transcript.
        if q.get("skip_audio_check") or q.get("no_audio_needed"):
            row["skip_audio_check"] = True

        if (q.get("question_type") or "").strip().lower() == "match":
            # "match" items have a distinct shape: a title, a list of
            # statements to match, and a list of paragraph-label options,
            # with the answer as a {statement_number: option_letter} map —
            # there's no single "question" string, so build one.
            statements = q.get("questions") or []
            numbered = "; ".join(
                "%d) %s" % (n, s) for n, s in enumerate(statements, 1)
            )
            row["question"] = "%s: %s" % (
                q.get("title", "Match the following"), numbered
            )
        else:
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
                # Nested passage-document shape (see _expand_nested_document).
                items.extend(_expand_nested_document(it))
            else:
                items.append(it)
    else:  # CSV
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            items = []
            for row in reader:
                if "options" in row and row["options"]:
                    row["options"] = row["options"].split("|")
                items.append(row)

    return [_normalize_item(it) for it in items]


PREFERRED_COLUMN_ORDER = [
    # ---- GIVEN: the item as authored/labeled, nothing derived ----
    # Identity
    "question_id", "sub_question_id", "skill", "question_type",
    # Content, as authored
    "content", "transcript", "audio", "question", "question_text", "options", "answer",
    # Labels supplied with the item
    "cefr_level", "cognitive_skill",
    # Pre-reform originals (only populated when --reform changes an item)
    "original_text", "original_answer",

    # ---- GENERATED: everything produced by the QC pipeline ----
    # Rule-based checks (no LLM call)
    "metric_flags",
    # Level check
    "predicted_level", "level_match", "level_change_reason",
    # Cognitive-skill check -- reported here, fully independent of the
    # content QC gate and final_decision
    "predicted_cognitive_skill", "cognitive_skill_reason",
    # Content QC gate scores + bare verdict (feeds final_decision below)
    "accuracy_pct", "grammar_errors", "clarity_score", "completeness_score",
    "content_qc_label", "content_qc_reason",
    # Construction gate (independent of content QC) -- also feeds final_decision
    "construction_flags", "construction_qc_label", "construction_qc_reason",
    # Fairness/bias gate (independent) -- also feeds final_decision
    "fairness_flags", "fairness_qc_label", "fairness_qc_reason",
    # Reform trail summary (only populated when --reform is used)
    "reform_change_summary",
    # Overall verdict -- roll-up of the three QC gates above. The reason
    # column reports only the CEFR level-change reason (or "pass"), placed
    # last since it's the final read-out over everything else in the row.
    "final_decision", "final_decision_result",
]


def _ordered_fieldnames(rows: list) -> list:
    """Puts columns in a stable, human-friendly order: IDs first, then the
    item's own content, then QC-derived columns. Any column not in
    PREFERRED_COLUMN_ORDER (e.g. a passthrough field unique to one input
    format) is appended at the end in first-seen order, so nothing is ever
    silently dropped."""
    seen = []
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.append(key)
    ordered = [c for c in PREFERRED_COLUMN_ORDER if c in seen]
    ordered += [c for c in seen if c not in ordered]
    return ordered


def save_results(rows: list, path: str) -> None:
    # "text" is an internal working column: the passage/transcript+question
    # combo built by _normalize_item() to feed the judge model. "level" and
    # "type" are
    # meant to just be normalized copies of "cefr_level"/"question_type" --
    # but for flat-shaped input that only ever supplied "level"/"type" (the
    # original column names this script used before nested-document support
    # was added), "cefr_level"/"question_type" are never populated at all.
    # Backfill them here before dropping "level"/"type", so that data isn't
    # silently lost from rows shaped that way.
    for row in rows:
        if not row.get("cefr_level") and row.get("level"):
            row["cefr_level"] = row["level"]
        if not row.get("question_type") and row.get("type"):
            row["question_type"] = row["type"]
    rows = [{k: v for k, v in row.items() if k not in ("text", "level", "type")} for row in rows]

    if path.lower().endswith(".json"):
        fieldnames = _ordered_fieldnames(rows)
        ordered_rows = [{k: row.get(k, "") for k in fieldnames if k in row} for row in rows]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ordered_rows, f, indent=2, ensure_ascii=False)
    else:  # CSV
        fieldnames = _ordered_fieldnames(rows)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            for row in rows:
                row = dict(row)
                if isinstance(row.get("options"), list):
                    row["options"] = "|".join(row["options"])
                writer.writerow(row)


_HTML_REPORT_TEMPLATE = r"""<!DOCTYPE html>
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
  .gates{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-bottom:16px; }
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
    <div class="sub">Content, question/answer and cognitive-skill QC &mdash; final call on whether an item goes to a student</div>
  </div>
  <div class="sub" id="generated-sub"></div>
</header>

<div class="stats" id="stats"></div>

<div class="toolbar">
  <input type="text" id="search" placeholder="Search question, answer, reason...">
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
        <th>S.No</th><th>ID</th><th>Sub-Q ID</th><th>Skill</th><th>Type</th><th>Question</th><th>CEFR (labeled &rarr; predicted)</th>
        <th>Cognitive skill (labeled &rarr; predicted)</th><th>Decision</th><th>Reason</th>
      </tr>
    </thead>
    <tbody id="rows"></tbody>
  </table>
  <div class="empty" id="empty" style="display:none;">No items match these filters.</div>
</main>

<footer>Click any row for the full QC breakdown (content / construction / fairness gates, rule flags, reform trail).</footer>

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
  els.genSub.textContent = `${total} items reviewed`;
}

function buildFilterOptions(){
  const skills = [...new Set(DATA.map(r => r.skill).filter(Boolean))].sort();
  const levels = [...new Set(DATA.map(r => r.predicted_level || r.cefr_level).filter(Boolean))].sort();
  skills.forEach(s => els.skillFilter.insertAdjacentHTML('beforeend', `<option value="${esc(s)}">${esc(s)}</option>`));
  levels.forEach(l => els.levelFilter.insertAdjacentHTML('beforeend', `<option value="${esc(l)}">${esc(l)}</option>`));
}

function levelCell(row){
  const labeled = row.cefr_level || '—';
  const predicted = row.predicted_level || '—';
  const changed = labeled !== '—' && predicted !== '—' && labeled !== predicted;
  return `<span class="level mono">${esc(labeled)}<span class="arrow">&rarr;</span><span class="${changed ? 'changed' : ''}">${esc(predicted)}</span></span>`;
}

function cognitiveCell(row){
  const labeled = row.cognitive_skill || '—';
  const predicted = row.predicted_cognitive_skill || '—';
  if (labeled === '—' && predicted === '—') return '—';
  const changed = labeled !== '—' && predicted !== '—' && labeled !== predicted;
  return `<span class="level">${esc(labeled)}<span class="arrow">&rarr;</span><span class="${changed ? 'changed' : ''}">${esc(predicted)}</span></span>`;
}

function questionPreview(row){
  return row.question || row.question_text || row.text || '(no question text)';
}

function matchesFilters(row){
  if (activeDecision !== 'all' && decisionOf(row) !== activeDecision) return false;
  if (els.skillFilter.value && row.skill !== els.skillFilter.value) return false;
  if (els.levelFilter.value && (row.predicted_level || row.cefr_level) !== els.levelFilter.value) return false;
  const q = els.search.value.trim().toLowerCase();
  if (q){
    const hay = [row.question_id, row.sub_question_id, questionPreview(row), row.answer, row.final_decision_result,
                 row.content_qc_reason, row.construction_qc_reason, row.fairness_qc_reason,
                 row.metric_flags].join(' ').toLowerCase();
    if (!hay.includes(q)) return false;
  }
  return true;
}

function render(){
  const filtered = DATA.filter(matchesFilters);
  els.empty.style.display = filtered.length ? 'none' : 'block';
  els.rows.innerHTML = filtered.map((row, i) => {
    const idx = DATA.indexOf(row);
    const decision = decisionOf(row) || 'review';
    return `
      <tr data-idx="${idx}">
        <td class="mono">${i + 1}</td>
        <td class="mono">${esc(row.question_id || idx)}</td>
        <td class="mono">${esc(row.sub_question_id || '—')}</td>
        <td>${esc(row.skill)}</td>
        <td>${esc(row.question_type)}</td>
        <td class="q-preview" title="${esc(questionPreview(row))}">${esc(questionPreview(row))}</td>
        <td>${levelCell(row)}</td>
        <td>${cognitiveCell(row)}</td>
        <td><span class="badge ${decision}">${esc(decision)}</span></td>
        <td class="reason-preview" title="${esc(row.final_decision_result)}">${esc(row.final_decision_result)}</td>
      </tr>`;
  }).join('');
  [...els.rows.querySelectorAll('tr')].forEach(tr => {
    tr.addEventListener('click', () => openDrawer(parseInt(tr.dataset.idx, 10)));
  });
}

function gateBlock(label, value, reason){
  const cls = (value || '').toLowerCase().includes('flag') ? 'reject' :
              (value || '').toLowerCase().includes('pass') ? 'select' : 'review';
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
    <h2>${esc(row.question_id || idx)}${row.sub_question_id ? ' / ' + esc(row.sub_question_id) : ''} &mdash; <span class="badge ${decision}">${esc(decision)}</span></h2>
    <div class="drawer-sub">${esc(row.skill)} / ${esc(row.question_type)} &middot; CEFR ${esc(row.cefr_level||'—')} &rarr; ${esc(row.predicted_level||'—')} &middot; cognitive skill ${esc(row.cognitive_skill||'—')} &rarr; ${esc(row.predicted_cognitive_skill||'—')}</div>

    <div class="block"><div class="k">Question</div><div class="v">${esc(questionPreview(row))}</div></div>
    <div class="block"><div class="k">Expected answer</div><div class="v">${esc(row.answer)}</div></div>
    ${row.options ? `<div class="block"><div class="k">Options</div><div class="v">${esc(Array.isArray(row.options) ? row.options.join(' | ') : row.options)}</div></div>` : ''}

    <div class="gates">
      ${gateBlock('Content QC', row.content_qc_label, row.content_qc_reason)}
      ${gateBlock('Construction', row.construction_qc_label, row.construction_qc_reason)}
      ${gateBlock('Fairness', row.fairness_qc_label, row.fairness_qc_reason)}
    </div>

    <div class="block"><div class="k">CEFR level reason</div><div class="v">${esc(row.level_change_reason)}</div></div>
    <div class="block"><div class="k">Cognitive skill reason</div><div class="v">${esc(row.cognitive_skill_reason)}</div></div>
    <div class="block"><div class="k">Rule-based flags</div><div class="v mono">${esc(row.metric_flags)}</div></div>
    <div class="block"><div class="k">Final decision reason</div><div class="v"><strong>${esc(row.final_decision_result)}</strong></div></div>
    ${row.reform_change_summary ? `<div class="block"><div class="k">Reform trail</div><div class="v">${esc(row.reform_change_summary)}</div></div>` : ''}
    ${row.elapsed_sec !== undefined && row.elapsed_sec !== null ? `<div class="block"><div class="k">Processing time</div><div class="v mono">${esc(row.elapsed_sec)}s</div></div>` : ''}
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


def generate_html_report(rows: list, path: str) -> None:
    """Writes a self-contained, single-file HTML review dashboard alongside
    the CSV/JSON output. Embeds the full result set as JSON directly in the
    page (no server, no upload step -- open the file in a browser). Gives
    reviewers a filterable/searchable queue (by decision, skill, CEFR level,
    free text) with a click-through detail view per item showing every QC
    gate (content/construction/fairness), the CEFR level-change reason, the
    cognitive-skill reason, and the rolled-up final_decision/
    final_decision_result -- everything a human needs to confirm or
    overrule the pipeline's select/review/reject call without touching the
    raw CSV."""
    clean_rows = [{k: v for k, v in row.items() if k != "text"} for row in rows]
    data_json = json.dumps(clean_rows, ensure_ascii=False)
    # Guard against a literal "</script>" inside any question/answer text
    # breaking out of the embedded <script> block.
    data_json = data_json.replace("</script>", "<\\/script>")
    html = _HTML_REPORT_TEMPLATE.replace("__DATA_JSON__", data_json)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def run_full_qc(item: dict, flags: list, level_tolerance: int) -> dict:
    """Runs CEFR classification + rubric scoring + verdict for one item.
    Returns a dict of all the derived QC columns (used for both the initial
    pass and for re-checking items after --reform). Metric columns not
    configured for this item's skill are set to 'n/a' so output stays
    consistent across rows with different skills.

    Deliberately CONTENT-only (level, cognitive skill, rubric scores) --
    construction and fairness are separate independent gates (see
    evaluate_construction/construction_qc_gate and
    classify_fairness/fairness_qc_gate) run once per item in main(), NOT
    re-run inside this function. That matters for --reform: the reform loop
    re-calls run_full_qc on each rewrite to re-check content QC, but a
    rewrite changing text/answer wording doesn't need construction/fairness
    re-checked on every attempt, so those calls are kept out of the reform
    loop entirely."""
    # Single combined judge call replaces what used to be 3 sequential calls
    # (classify_level + classify_cognitive_skill + score_metrics) -- see
    # classify_and_score()/_build_combined_prompt() for the merged prompt.
    judged = classify_and_score(item, flags)

    predicted_level = judged["predicted_level"]
    level_reason = judged["level_reason"]
    labeled_level = (item.get("level") or "").strip().upper()
    level_match = _level_within_range(labeled_level, predicted_level, level_tolerance)

    if labeled_level and predicted_level not in ("ERROR", "UNKNOWN") and labeled_level != predicted_level:
        level_change_reason = "Changed from %s to %s: %s" % (labeled_level, predicted_level, level_reason)
    elif level_match:
        level_change_reason = "PASS"
    else:
        level_change_reason = level_reason

    predicted_cognitive_skill = judged["predicted_cognitive_skill"]
    cognitive_skill_reason_raw = judged["cognitive_skill_reason"]
    labeled_cognitive_skill = (item.get("cognitive_skill") or "").strip().lower()
    cognitive_skill_match = _cognitive_skill_match(labeled_cognitive_skill, predicted_cognitive_skill)

    if not labeled_cognitive_skill:
        cognitive_skill_reason = ""
    elif predicted_cognitive_skill in ("ERROR", "UNKNOWN"):
        cognitive_skill_reason = (
            "could not be determined -- %s" % cognitive_skill_reason_raw
        )
    elif cognitive_skill_match:
        cognitive_skill_reason = "PASS"
    else:
        cognitive_skill_reason = (
            "mismatch: labeled '%s' but question actually tests '%s' -- %s"
            % (labeled_cognitive_skill, predicted_cognitive_skill, cognitive_skill_reason_raw)
        )

    skill = item.get("skill", "")
    scores = judged  # judged already carries the metric keys + "reason"
    content_qc_label, content_qc_reason, content_flagged_on_level_only = qc_verdict(level_match, scores, flags, skill)

    out = {
        "predicted_level": predicted_level,
        "level_match": level_match,
        "level_change_reason": level_change_reason,
        "predicted_cognitive_skill": predicted_cognitive_skill,
        "cognitive_skill_reason": cognitive_skill_reason,
    }
    for key in ALL_METRIC_KEYS:
        out[key] = scores[key] if key in scores else "n/a"
    out["content_qc_label"] = content_qc_label
    out["content_qc_reason"] = content_qc_reason
    out["content_flagged_on_level_only"] = content_flagged_on_level_only
    return out


def fairness_qc_gate(flags: list, llm_reason: str):
    """Independent PASS/FLAG gate for FAIRNESS/BIAS only -- same bare-label
    pattern as construction_qc_gate(), so both feed compute_final_decision()
    uniformly."""
    if flags == ["fairness_check_error"]:
        return "FAIRNESS_ERROR", llm_reason or "fairness check failed"
    if flags:
        extra = " | %s" % llm_reason if llm_reason and llm_reason not in ("", "none") else ""
        return "FAIRNESS_FLAG", "; ".join(flags) + extra
    return "FAIRNESS_PASS", llm_reason or "meets all fairness QC targets"


def compute_final_decision(content_qc_label: str, construction_qc_label: str,
                            fairness_qc_label: str, metric_flags: str,
                            content_qc_reason: str = "", construction_qc_reason: str = "",
                            fairness_qc_reason: str = "", level_change_reason: str = "",
                            content_flagged_on_level_only: bool = False) -> tuple:
    """Rolls the three independent gates (content QC, construction,
    fairness) up into a single actionable verdict: REJECT / USE / REVIEW.
    "missing_end_punctuation" is excluded from the rule-based flags that
    feed this decision -- it's too minor/noisy to affect the final call,
    though it still shows up as-is in the metric_flags column itself.

    final_decision_result reports, in priority order:
      - the CEFR level-change reason (why the predicted level differs from
        the labeled level), whenever the level did in fact change, and/or
        a plain-English breakdown of every rule-based flag that fired
        (metric_flags checks + cross-question redundancy/triviality
        checks) -- if BOTH are present, they're joined with " | ".
      - otherwise, if content QC, construction, or fairness flagged/errored
        on its own grounds (e.g. weak distractors, bias, an accuracy/
        grammar/clarity/completeness miss) with no CEFR-change/rule-flag
        story to tell, that gate's own reason is surfaced here instead
        (prefixed "content:"/"construction:"/"fairness:") -- so a reject/
        review decision is never left explaining itself as "pass".
      - "pass" only when nothing above applies, i.e. every gate that ran
        came back clean.

    - reject: at least one gate actively FLAGged the item.
    - select: every gate that ran came back PASS (or, if no LLM gates ran
              at all -- e.g. --skip-llm / skill filtered out -- the
              rule-based metric_flags checks came back clean).
    - review: no gate flagged it, but coverage is incomplete (some gates
              SKIPPED or ERRORed, or rule flags fired with no LLM gate to
              confirm) -- needs a human glance rather than an auto
              select/reject. Also covers the case where content QC's ONLY
              flag reason is a CEFR level mismatch (predicted level outside
              the labeled level's range) with construction/fairness both
              clean -- that's downgraded from reject to review rather than
              auto-rejected, since a level disagreement alone isn't
              necessarily a bad item.

    NOTE: content_qc_label comes back bare ("FLAG"/"PASS") from
    qc_verdict(), while construction_qc_label/fairness_qc_label come back
    prefixed ("CONSTRUCTION_FLAG"/"CONSTRUCTION_PASS",
    "FAIRNESS_FLAG"/"FAIRNESS_PASS", plus "CONSTRUCTION_ERROR"/
    "FAIRNESS_ERROR" when the judge model call itself failed) -- matching on
    endswith("FLAG")/endswith("PASS") handles both forms.
    """
    labels = [content_qc_label, construction_qc_label, fairness_qc_label]

    effective_flags = [
        f for f in (metric_flags.split(", ") if metric_flags not in ("", "none") else [])
        if f not in _IGNORED_MINOR_FLAGS
    ]
    rule_flags_present = bool(effective_flags)

    # The reason column reports ONLY the CEFR level-change story: "pass"
    # whenever the predicted level did not move off the labeled level,
    # regardless of what the overall decision ends up being (select/review/
    # reject) -- that overall roll-up reasoning lives in the individual
    # content_qc_reason/construction_qc_reason/fairness_qc_reason columns.
    level_changed = level_change_reason.startswith("Changed from ")
    rule_flag_detail = describe_rule_flags(effective_flags) if rule_flags_present else ""

    if level_changed and rule_flag_detail:
        non_pass_reason = "%s | rule flags: %s" % (level_change_reason, rule_flag_detail)
    elif level_changed:
        non_pass_reason = level_change_reason
    elif rule_flag_detail:
        non_pass_reason = "rule flags: %s" % rule_flag_detail
    else:
        # No CEFR-level story and no rule flags -- but content QC,
        # construction, or fairness may still have flagged/errored on their
        # own grounds (e.g. weak distractors, bias, or a rubric miss like
        # accuracy/grammar/clarity/completeness). Surface that gate's own
        # reason here instead of "pass", so the reason column never claims
        # "pass" next to a reject/review decision that a gate actually
        # drove.
        gate_reasons = []
        if content_qc_label == "FLAG" and content_qc_reason and not content_flagged_on_level_only:
            gate_reasons.append("content: %s" % content_qc_reason)
        if (construction_qc_label or "").endswith(("FLAG", "ERROR")) and construction_qc_reason:
            gate_reasons.append("construction: %s" % construction_qc_reason)
        if (fairness_qc_label or "").endswith(("FLAG", "ERROR")) and fairness_qc_reason:
            gate_reasons.append("fairness: %s" % fairness_qc_reason)
        non_pass_reason = " | ".join(gate_reasons) if gate_reasons else "pass"

    if any(l.endswith("FLAG") for l in labels if l):
        # If content QC's ONLY reason for flagging is the CEFR level being
        # outside the labeled level's range -- and construction/fairness
        # didn't independently flag the item for their own reasons -- this
        # is a level-mismatch case, not a real quality/construction/
        # fairness problem. Downgrade it to "review" (needs a human look)
        # rather than an outright "reject".
        other_gates_flagged = (
            (construction_qc_label or "").endswith("FLAG")
            or (fairness_qc_label or "").endswith("FLAG")
        )
        if content_flagged_on_level_only and not other_gates_flagged:
            return "review", non_pass_reason
        return "reject", non_pass_reason

    if all(l == "" for l in labels):
        if rule_flags_present:
            return "review", non_pass_reason
        return "select", "pass"

    if any(l for l in labels) and all(l.endswith("PASS") for l in labels if l):
        return "select", "pass"

    return "review", non_pass_reason




# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="QC-flag items on Accuracy/Grammar/Clarity/Completeness/CEFR Level using a Gemini judge model (gemma-4-26b-a4b-it) via deepeval")
    ap.add_argument("--input", required=True, help="Path to input .json or .csv file")
    ap.add_argument("--output", required=True, help="Path to output .json or .csv file")
    ap.add_argument("--level-tolerance", type=int, default=LEVEL_TOLERANCE,
                     help="How many CEFR tiers off is still considered 'within range' (default: 0 = exact match)")
    ap.add_argument("--skip-llm", action="store_true", help="Only run rule-based checks (no judge model call)")
    ap.add_argument("--reform", action="store_true",
                     help="For items that FLAG, ask the judge model to rewrite them and re-run QC on the rewrite (up to --reform-attempts times)")
    ap.add_argument("--reform-attempts", type=int, default=2,
                     help="Max reform+re-check cycles per flagged item (default: 2)")
    ap.add_argument("--skill-filter", default="listening",
                     help="Only run LLM-based QC (CEFR classification, scoring, reform) on items "
                          "whose skill is in this comma-separated list (case-insensitive). Other "
                          "skills still get rule-based checks but are marked SKIPPED for the LLM "
                          "steps. Use 'all' to disable filtering. (default: listening)")
    ap.add_argument("--html-report", default=None,
                     help="Path to write a self-contained HTML review dashboard (filterable/"
                          "searchable select/review/reject queue with full QC detail per item). "
                          "Defaults to the --output path with its extension replaced by .html.")
    ap.add_argument("--no-html-report", action="store_true",
                     help="Skip writing the HTML dashboard entirely.")
    ap.add_argument("--workers", type=int, default=10,
                     help="Number of items to process concurrently via a thread pool "
                          "(default: 10). Each item makes several sequential judge-model "
                          "API calls, but items are independent of each other, so this is "
                          "the main lever for total wall-clock time. Start conservative "
                          "(8-10) and raise it if you aren't hitting rate limits/timeouts; "
                          "set to 1 to fall back to the old strictly-sequential behavior.")
    ap.add_argument("--checkpoint", default=None,
                     help="Path to a JSONL checkpoint file that every completed item is "
                          "appended to as soon as it finishes. If this file already exists "
                          "on startup, items whose identity (question_id/sub_question_id) "
                          "is already in it are skipped and reused instead of re-run, so an "
                          "interrupted run can be resumed by re-running the same command. "
                          "Defaults to '<output>.checkpoint.jsonl'.")
    ap.add_argument("--no-checkpoint", action="store_true",
                     help="Disable checkpointing entirely (no resume, no per-item durability).")
    ap.add_argument("--save-every", type=int, default=10,
                     help="Re-write --output and the HTML report from whatever's completed "
                          "so far every N finished items, so a killed/crashed run still "
                          "leaves a usable partial report on disk (default: 10). Set to 0 "
                          "to only save once at the very end.")
    ap.add_argument("--timing", action="store_true",
                     help="Track and report how long processing takes: adds an "
                          "'elapsed_sec' column per item, and logs running elapsed time / "
                          "average seconds-per-item / ETA at each periodic save.")
    args = ap.parse_args()

    items = load_items(args.input)
    if not items:
        print("No items found in input file.", file=sys.stderr)
        sys.exit(1)

    # Group items sharing the same passage (question_id) and run the
    # cross-question redundancy/triviality check on each group, since this
    # is invisible to every other check which only looks at one item at a
    # time.
    groups = OrderedDict()
    for idx, item in enumerate(items):
        groups.setdefault(item.get("question_id", idx), []).append(item)

    redundancy_flags = {}  # sub_question_id -> [flags]
    for group_items in groups.values():
        redundancy_flags.update(detect_redundant_and_trivial(group_items))

    # Resolved once, up front, so both the periodic partial saves and the
    # final save write the same HTML path.
    html_path = args.html_report or (os.path.splitext(args.output)[0] + ".html")

    # ------------------------------------------------------------------
    # Checkpointing: every completed row is appended to a JSONL file as
    # soon as it finishes, keyed by the item's own identity (not its list
    # position), so re-running the same command after a crash/interrupt/
    # Ctrl-C skips whatever's already done instead of re-spending judge
    # calls on it. Disabled entirely with --no-checkpoint.
    # ------------------------------------------------------------------
    def _item_key(item, idx):
        qid = item.get("question_id", "") or item.get("content_id", "")
        sqid = item.get("sub_question_id", "")
        if qid or sqid:
            return "%s::%s" % (qid, sqid)
        return "__idx%d" % idx  # no stable id on this item -- fall back to position

    checkpoint_path = None if args.no_checkpoint else (
        args.checkpoint or (args.output + ".checkpoint.jsonl")
    )
    checkpoint_rows = {}  # item key -> previously-completed row
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a truncated last line from a hard kill
                key = row.get("_checkpoint_key")
                if key:
                    checkpoint_rows[key] = row

    _checkpoint_lock = threading.Lock()

    def _write_checkpoint(row):
        if not checkpoint_path:
            return
        with _checkpoint_lock:
            with open(checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Guards the console so progress lines from different worker threads
    # don't get interleaved mid-line. Doesn't affect correctness of the
    # actual QC work -- only cosmetic ordering of stdout.
    _print_lock = threading.Lock()

    def _log(msg):
        with _print_lock:
            print(msg)

    def process_item(i, item, total):
        """Runs the full QC pipeline for ONE item and returns its output
        row. Pulled out of the old for-loop body so it can be called from
        worker threads: every item here is independent of every other
        item (results aren't touched, only local variables), which is
        what makes concurrent execution safe."""
        flags = metric_flags(item)
        flags.extend(redundancy_flags.get(item.get("sub_question_id", ""), []))
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
                "predicted_cognitive_skill": "",
                "cognitive_skill_reason": "",
                "content_qc_label": "" if args.skip_llm else "SKIPPED",
                "content_qc_reason": "" if args.skip_llm else (
                    "LLM QC not applicable: skill is '%s', filter is '%s'"
                    % (item.get("skill", ""), args.skill_filter)
                ),
                "construction_flags": "",
                "construction_qc_label": "" if args.skip_llm else "SKIPPED",
                "construction_qc_reason": "" if args.skip_llm else (
                    "Construction QC not applicable: skill is '%s', filter is '%s'"
                    % (item.get("skill", ""), args.skill_filter)
                ),
                "fairness_flags": "",
                "fairness_qc_label": "" if args.skip_llm else "SKIPPED",
                "fairness_qc_reason": "" if args.skip_llm else (
                    "Fairness QC not applicable: skill is '%s', filter is '%s'"
                    % (item.get("skill", ""), args.skill_filter)
                ),
                **blank_metrics,
            })
            out["final_decision"], out["final_decision_result"] = compute_final_decision(
                out["content_qc_label"], out["construction_qc_label"], out["fairness_qc_label"],
                out["metric_flags"], out["content_qc_reason"], out["construction_qc_reason"],
                out["fairness_qc_reason"],
            )
            if args.skip_llm:
                _log("[%d/%d] rule flags only: %s" % (i, total, out["metric_flags"]))
            else:
                _log("[%d/%d] SKIPPED (skill='%s' not in filter '%s')" % (
                    i, total, item.get("skill", ""), args.skill_filter))
            return out

        qc = run_full_qc(item, flags, args.level_tolerance)
        out.update(qc)

        # Construction and fairness are independent gates, run once per
        # item here (not inside run_full_qc -- see its docstring). Both use
        # the item's effective CEFR level: prefer the judge model's own
        # predicted_level (just computed above) since that reflects the
        # question's real difficulty; fall back to the labeled level only
        # if the prediction failed (ERROR/UNKNOWN/blank).
        effective_level = qc["predicted_level"] if qc["predicted_level"] in CEFR_LEVELS else (
            (item.get("level") or item.get("cefr_level") or "").strip().upper()
        )

        construction_result = evaluate_construction(item, predicted_level=effective_level)
        c_flags = construction_flags(item, construction_result, level=effective_level)
        c_label, c_reason = construction_qc_gate(c_flags, construction_result.get("reason", ""))
        out["construction_flags"] = ", ".join(c_flags) if c_flags else "none"
        out["construction_qc_label"] = c_label
        out["construction_qc_reason"] = c_reason

        fairness_result = classify_fairness(item, predicted_level=effective_level)
        f_flags = fairness_flags(item, fairness_result)
        f_label, f_reason = fairness_qc_gate(f_flags, fairness_result.get("reason", ""))
        out["fairness_flags"] = ", ".join(f_flags) if f_flags else "none"
        out["fairness_qc_label"] = f_label
        out["fairness_qc_reason"] = f_reason

        out["final_decision"], out["final_decision_result"] = compute_final_decision(
            out["content_qc_label"], out["construction_qc_label"], out["fairness_qc_label"],
            out["metric_flags"], out["content_qc_reason"], out["construction_qc_reason"],
            out["fairness_qc_reason"], out["level_change_reason"],
            out["content_flagged_on_level_only"],
        )

        _log("[%d/%d] content=%s construction=%s fairness=%s (predicted=%s, labeled=%s) -> %s: %s" % (
            i, total, qc["content_qc_label"], c_label, f_label, qc["predicted_level"],
            (item.get("level") or "").strip().upper() or "?",
            out["final_decision"], out["final_decision_result"]
        ))

        if args.reform and qc["content_qc_label"] == "FLAG":
            out["original_text"] = item.get("text", "")
            out["original_answer"] = item.get("answer", "")
            out["reform_change_summary"] = ""

            working_item = dict(item)
            for attempt in range(1, args.reform_attempts + 1):
                _log("    [%d/%d] reforming (attempt %d/%d)..." % (i, total, attempt, args.reform_attempts))
                rewrite = reform_item(working_item, qc["content_qc_reason"])

                if rewrite["change_summary"].startswith("REFORM FAILED"):
                    out["reform_change_summary"] = rewrite["change_summary"]
                    _log("    [%d/%d] %s" % (i, total, rewrite["change_summary"]))
                    break

                working_item["text"] = rewrite["text"]
                working_item["answer"] = rewrite["answer"]
                if rewrite.get("options"):
                    working_item["options"] = rewrite["options"]
                out["reform_change_summary"] = rewrite["change_summary"]

                new_flags = metric_flags(working_item)
                new_qc = run_full_qc(working_item, new_flags, args.level_tolerance)
                qc = new_qc

                if new_qc["content_qc_label"] == "PASS":
                    _log("    [%d/%d] reform succeeded on attempt %d" % (i, total, attempt))
                    out["text"] = working_item["text"]
                    out["answer"] = working_item["answer"]
                    out["options"] = working_item.get("options", out.get("options"))
                    out["metric_flags"] = ", ".join(new_flags) if new_flags else "none"
                    out.update(new_qc)
                    break
                else:
                    _log("    [%d/%d] still flagged: %s" % (i, total, new_qc["content_qc_reason"]))
                    out["text"] = working_item["text"]
                    out["answer"] = working_item["answer"]
                    out["options"] = working_item.get("options", out.get("options"))
                    out["metric_flags"] = ", ".join(new_flags) if new_flags else "none"
                    out.update(new_qc)

            # Reform only re-runs content QC, not construction/fairness --
            # recompute the roll-up since out["content_qc_label"] may have
            # changed.
            out["final_decision"], out["final_decision_result"] = compute_final_decision(
                out["content_qc_label"], out["construction_qc_label"], out["fairness_qc_label"],
                out["metric_flags"], out["content_qc_reason"], out["construction_qc_reason"],
                out["fairness_qc_reason"], out["level_change_reason"],
                out["content_flagged_on_level_only"],
            )

        return out

    _run_start = time.time()

    def _run_item(i, item, total, key):
        """Wraps process_item with the two cross-cutting concerns that
        apply to every item regardless of skill/filter/reform path:
        optional wall-clock timing and checkpoint durability. Kept
        outside process_item itself so those don't have to be threaded
        through its several internal return points."""
        t0 = time.time() if args.timing else None
        out = process_item(i, item, total)
        if args.timing:
            out["elapsed_sec"] = round(time.time() - t0, 2)
        out["_checkpoint_key"] = key
        _write_checkpoint(out)
        return out

    # results[k] must line up with items[k] regardless of which order
    # worker threads finish in, so we pre-size the list and write each
    # result to its own index rather than appending in completion order.
    total = len(items)
    results = [None] * total
    workers = max(1, args.workers)

    # Reuse whatever's already in the checkpoint (from a prior interrupted
    # run) instead of re-processing it; only truly-new items get submitted.
    key_by_index = [_item_key(item, idx) for idx, item in enumerate(items)]
    pending_indices = []
    for idx, key in enumerate(key_by_index):
        if key in checkpoint_rows:
            results[idx] = checkpoint_rows[key]
        else:
            pending_indices.append(idx)

    already_done = total - len(pending_indices)
    if already_done:
        print("Resuming from checkpoint: %d/%d items already completed, %d remaining"
              % (already_done, total, len(pending_indices)))

    def _clean(rows):
        # "_checkpoint_key" is bookkeeping for resume matching, not
        # something that belongs in the user-facing output/report.
        return [{k: v for k, v in row.items() if k != "_checkpoint_key"} for row in rows]

    def _flush():
        done_rows = _clean([r for r in results if r is not None])
        if not done_rows:
            return
        save_results(done_rows, args.output)
        if not args.no_html_report:
            generate_html_report(done_rows, html_path)
        if args.timing:
            elapsed = time.time() - _run_start
            n = len(done_rows)
            avg = elapsed / n if n else 0.0
            remaining = max(total - n, 0) * avg
            print("    [flush] %d/%d done -- elapsed %.1fs, avg %.2fs/item, ETA %.1fs"
                  % (n, total, elapsed, avg, remaining))

    completed = already_done
    try:
        if workers == 1:
            # Old strictly-sequential path, kept as a fallback (e.g. for
            # debugging, or if the API can't tolerate concurrent requests).
            for idx in pending_indices:
                results[idx] = _run_item(idx + 1, items[idx], total, key_by_index[idx])
                completed += 1
                if args.save_every and completed % args.save_every == 0:
                    _flush()
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_index = {
                    executor.submit(_run_item, idx + 1, items[idx], total, key_by_index[idx]): idx
                    for idx in pending_indices
                }
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    results[idx] = future.result()
                    completed += 1
                    if args.save_every and completed % args.save_every == 0:
                        _flush()
    except KeyboardInterrupt:
        print("\nInterrupted -- flushing %d/%d completed items before exiting..."
              % (completed, total))
        _flush()
        if checkpoint_path:
            print("Partial results saved to %s / %s. Re-run the same command to "
                  "resume from %s." % (args.output, html_path, checkpoint_path))
        else:
            print("Partial results saved to %s / %s. (--no-checkpoint was set, so a "
                  "re-run will start over from scratch.)" % (args.output, html_path))
        sys.exit(130)

    _flush()
    print("\nDone. Wrote %d results to %s" % (completed, args.output))
    if not args.no_html_report:
        print("Wrote review dashboard to %s" % html_path)
    if args.timing:
        print("Total elapsed: %.1fs" % (time.time() - _run_start))


if __name__ == "__main__":
    main()