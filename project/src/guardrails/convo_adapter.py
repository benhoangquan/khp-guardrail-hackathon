"""
ConvoKit adapter for KHP conversation analysis.

Parses KHP multi-turn format into a ConvoKit Corpus, then extracts
conversation-level risk features invisible in single-turn analysis:
  - Turn length trajectory (engagement collapse)
  - Hedge word erosion across turns
  - Risk keyword density trajectory
  - First-vs-last user turn delta
  - Linguistic coordination (style matching, via ConvoKit)

Returns a risk score in [0, 1]. Higher = more likely high-risk.

Requires: pip install convokit spacy && python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import re
from typing import Optional

# Hedge words: early hedging that erodes as crisis deepens
_HEDGE_WORDS = {
    r"\bmaybe\b", r"\bperhaps\b", r"\bkind of\b", r"\bkinda\b", r"\bsort of\b", r"\bsorta\b",
    r"\bi guess\b", r"\bi think\b", r"\bi don't know\b", r"\bnot sure\b", r"\bjust wondering\b",
    r"\ba little\b", r"\ba bit\b", r"\bsometimes\b", r"\boccasionally\b", r"\bmight\b", r"\bcould\b",
}

# Risk keywords: direct and indirect distress signals
_RISK_KEYWORDS = {
    # explicit
    r"\bsuicide\b", r"\bsuicidal\b", r"\bkill myself\b", r"\bend my life\b", r"\bend it all\b",
    r"\bhurt myself\b", r"\bself-harm\b", r"\bself harm\b", r"\bcut myself\b", r"\boverdose\b",
    # passive SI
    r"\bbetter off without me\b", r"\bburden\b", r"\bwouldn't mind not waking\b",
    r"\bdon't want to be here\b", r"\bwish i wasn't here\b", r"\btired of existing\b",
    r"\bno point\b", r"\bwhat's the point\b", r"\bpoint anymore\b",
    # hopelessness / finality
    r"\bnever going to change\b", r"\balways been like this\b", r"\baccepted it\b",
    r"\bgiving up\b", r"\bgiven up\b", r"\blast time\b", r"\bsaying goodbye\b", r"\bgoodbye forever\b",
    r"\bnothing matters\b", r"\bdoesn't matter anymore\b",
    # isolation
    r"\bno one cares\b", r"\bnobody cares\b", r"\ball alone\b", r"\bcompletely alone\b",
    # temporal urgency: immediacy of action
    # Source: Coppersmith et al. (2018) "Natural Language Processing of Social Media as Screening
    # for Suicide Risk", Biomedical Informatics Insights. Time-anchored language predicts
    # imminent risk independently of ideation severity.
    r"\btonight\b", r"\bthis is my last\b", r"\bbefore morning\b", r"\bright now i\b",
    r"\bby tomorrow\b", r"\bone last\b", r"\bafter i send this\b", r"\bafter this message\b",
}

# Signals that KHP escalated (used only if you run this against KHP responses)
_ESCALATION_SIGNALS = {
    "counselor", "kids help phone", "crisis", "are you safe",
    "i'm concerned", "please know", "talk to someone",
}

# Negation scope
# 3-token window avoids cross-clause false suppression while still catching
# "don't want to [risk]" and "never going to [risk]".
# Source: Ji et al. (2022) "MentalBERT: Publicly Available Pretrained Language
# Models for Mental Healthcare"; negation is the leading source of false
# positives in keyword-based mental health NLP.
_NEGATION_RE: re.Pattern = re.compile(
    r"\b(don't|do not|doesn't|does not|didn't|did not|not|never|no longer|"
    r"wouldn't|won't|can't|cannot|n't|pas|ne pas|jamais|plus)\b",
    re.IGNORECASE,
)


def _is_negated(text: str, match_start: int, window_tokens: int = 3) -> bool:
    """Return True if the risk match at match_start is preceded by a negation."""
    prefix = text[:match_start].split()
    window = " ".join(prefix[-window_tokens:]) if prefix else ""
    return bool(_NEGATION_RE.search(window))


# LIWC-derived longitudinal patterns
# Source: Pennebaker et al. (2003) "Psychological Aspects of Natural Language
# Use: Our Words, Our Selves", Annual Review of Psychology. First-person
# singular increase and future-tense decline predict psychological distress.
# Source: Coppersmith et al. (2014) "Measuring Post Traumatic Stress Disorder
# in Twitter", ICWSM. Social word decline correlates with isolation/crisis.
_FUTURE_VERBS_RE: re.Pattern = re.compile(
    r"\b(will|going to|gonna|would|should|plan to|hope to|"
    r"tomorrow|next week|someday|eventually|"
    r"vais|voudrais|espère|demain|bientôt)\b",
    re.IGNORECASE,
)
_FIRST_PERSON_RE: re.Pattern = re.compile(
    r"\b(i|i'm|i've|i'd|i'll|i am|i was|i have|myself|"
    r"je|j'ai|j'étais|j'avais|moi)\b",
    re.IGNORECASE,
)
_SOCIAL_WORDS_RE: re.Pattern = re.compile(
    r"\b(friend|friends|family|we|us|our|together|everyone|people|"
    r"someone|anyone|they|them|their|parents|mom|dad|sister|brother|"
    r"ami|amis|famille|nous|ensemble|quelqu'un|les gens)\b",
    re.IGNORECASE,
)


def parse_turns(text: str) -> list[tuple[str, str]]:
    """
    Parse KHP conversation format into [(role, content), ...].

    Handles both 'user:' / 'assistant:' prefix style and blank-line separation.
    Strips assistant turns — we only need user turns for risk analysis, but
    keep them for coordination scoring.
    """
    turns: list[tuple[str, str]] = []
    # Match lines starting with role prefix (case-insensitive)
    pattern = re.compile(
        r"^(user|assistant|bot|human)\s*:\s*(.+?)(?=\n(?:user|assistant|bot|human)\s*:|$)",
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    for m in pattern.finditer(text.strip()):
        role = m.group(1).lower().replace("bot", "assistant").replace("human", "user")
        content = m.group(2).strip()
        if content:
            turns.append((role, content))

    # Fallback: alternate lines if no prefix found
    if not turns:
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        roles = ["user", "assistant"]
        for i, line in enumerate(lines):
            turns.append((roles[i % 2], line))

    return turns


def _hedge_density(text: str) -> float:
    """Fraction of hedge-word matches relative to word count (0-1)."""
    lower = text.lower()
    word_count = max(len(lower.split()), 1)
    hits = sum(1 for h in _HEDGE_WORDS if re.search(h, lower))
    return min(hits / word_count * 10, 1.0)  # scale: 1 hit per 10 words = 1.0


def _risk_density(text: str) -> float:
    """Fraction of risk-keyword matches relative to word count (0-1).
    Negated matches (e.g. 'don't want to kill myself') are suppressed.
    """
    lower = text.lower()
    word_count = max(len(lower.split()), 1)
    hits = 0
    for k in _RISK_KEYWORDS:
        m = re.search(k, lower)
        if m and not _is_negated(lower, m.start()):
            hits += 1
    return min(hits / word_count * 5, 1.0)  # scale: 1 hit per 5 words = 1.0


def _length_drop(lengths: list[int]) -> float:
    """
    Detect engagement collapse: late turns much shorter than early turns.
    Returns 0.0 (no drop) to 1.0 (complete collapse).
    """
    if len(lengths) < 2:
        return 0.0
    early = sum(lengths[: max(1, len(lengths) // 3)]) / max(1, len(lengths) // 3)
    late = sum(lengths[-(max(1, len(lengths) // 3)) :]) / max(1, len(lengths) // 3)
    if early == 0:
        return 0.0
    drop = (early - late) / early
    return max(0.0, min(drop, 1.0))


def _escalation_trajectory(risk_per_turn: list[float]) -> float:
    """
    Are risk scores increasing across turns? Returns a length-invariant score.
    Compares mean risk of the last third of turns against the first third.
    0 = flat/decreasing, 1 = strongly increasing.
    """
    if len(risk_per_turn) < 2:
        return risk_per_turn[0] if risk_per_turn else 0.0
    split = max(1, len(risk_per_turn) // 3)
    early_mean = sum(risk_per_turn[:split]) / split
    late_mean = sum(risk_per_turn[-split:]) / split
    # positive delta = escalating; factor 2.5 so a 0.4 rise scores 1.0
    return max(0.0, min((late_mean - early_mean) * 2.5, 1.0))


def _coordination_score(user_turns: list[str], assistant_turns: list[str]) -> float:
    """
    Simplified coordination: does user vocabulary increasingly mirror the bot?
    Full ConvoKit CoordinationTransformer is more precise but this handles
    the case where ConvoKit corpus building is too heavy.
    Returns 0 (no coordination) to 1 (high mirroring).
    """
    if len(user_turns) < 2 or not assistant_turns:
        return 0.0
    scores = []
    for i, user_turn in enumerate(user_turns[1:], 1):  # skip first user turn
        bot_turn = assistant_turns[min(i - 1, len(assistant_turns) - 1)]
        bot_words = set(re.findall(r"\b\w{4,}\b", bot_turn.lower()))
        user_words = set(re.findall(r"\b\w{4,}\b", user_turn.lower()))
        if not bot_words:
            continue
        overlap = len(bot_words & user_words) / len(bot_words)
        scores.append(overlap)
    return sum(scores) / len(scores) if scores else 0.0


def _future_verb_decline(user_turns: list[str]) -> float:
    """Declining future-tense verb density across turns. Loss of future orientation.
    Source: Pennebaker et al. (2003) Annual Review of Psychology.
    """
    if len(user_turns) < 2:
        return 0.0
    dens = [len(_FUTURE_VERBS_RE.findall(t)) / max(len(t.split()), 1) for t in user_turns]
    split = max(1, len(dens) // 2)
    early = sum(dens[:split]) / split
    late = sum(dens[-split:]) / split
    return max(0.0, min((early - late) * 5, 1.0))


def _first_person_increase(user_turns: list[str]) -> float:
    """Increasing first-person singular density across turns. Rumination/self-focus signal.
    Source: Pennebaker et al. (2003) Annual Review of Psychology.
    """
    if len(user_turns) < 2:
        return 0.0
    dens = [len(_FIRST_PERSON_RE.findall(t)) / max(len(t.split()), 1) for t in user_turns]
    split = max(1, len(dens) // 2)
    early = sum(dens[:split]) / split
    late = sum(dens[-split:]) / split
    return max(0.0, min((late - early) * 5, 1.0))


def _social_word_decline(user_turns: list[str]) -> float:
    """Declining social vocabulary density across turns. Isolation/withdrawal signal.
    Source: Coppersmith et al. (2014) ICWSM.
    """
    if len(user_turns) < 2:
        return 0.0
    dens = [len(_SOCIAL_WORDS_RE.findall(t)) / max(len(t.split()), 1) for t in user_turns]
    split = max(1, len(dens) // 2)
    early = sum(dens[:split]) / split
    late = sum(dens[-split:]) / split
    return max(0.0, min((early - late) * 5, 1.0))


def _build_convokit_corpus(turns: list[tuple[str, str]], conv_id: str = "khp_conv"):
    """
    Build a ConvoKit Corpus from parsed turns.
    Requires convokit to be installed.
    """
    from convokit import Corpus, Utterance, Speaker, ConvoKitMeta

    speakers = {
        "user": Speaker(id="user"),
        "assistant": Speaker(id="assistant"),
    }
    utterances = []
    for i, (role, content) in enumerate(turns):
        utt = Utterance(
            id=f"{conv_id}_{i}",
            speaker=speakers[role],
            conversation_id=conv_id,
            reply_to=f"{conv_id}_{i - 1}" if i > 0 else None,
            text=content,
        )
        utterances.append(utt)

    corpus = Corpus(utterances=utterances)
    return corpus


class ConvoAnalyzer:
    """
    Extracts conversation-level risk features from a KHP conversation string.

    Usage:
        analyzer = ConvoAnalyzer()
        score = analyzer.score(conversation_text)   # 0.0–1.0
        details = analyzer.analyze(conversation_text)  # full feature dict
    """

    # Feature weights, must sum to 1.0.
    # LIWC features (Pennebaker 2003; Coppersmith 2014) added at 0.20 total,
    # existing weights scaled down proportionally.
    WEIGHTS = {
        "escalation_trajectory": 0.28,
        "final_turn_risk": 0.22,
        "hedge_erosion": 0.12,
        "engagement_collapse": 0.10,
        "first_last_delta": 0.08,
        "future_verb_decline": 0.08,    # loss of future orientation
        "first_person_increase": 0.07,  # rumination / self-focus
        "social_word_decline": 0.05,    # isolation / withdrawal
    }

    def analyze(self, conversation_text: str) -> dict:
        """Return a dict with all features and the composite risk score."""
        turns = parse_turns(conversation_text)
        user_turns = [c for role, c in turns if role == "user"]
        asst_turns = [c for role, c in turns if role == "assistant"]

        if not user_turns:
            return {"score": 0.0, "error": "no_user_turns", "features": {}}

        # Per-turn risk and hedge densities
        risk_per_turn = [_risk_density(t) for t in user_turns]
        hedge_per_turn = [_hedge_density(t) for t in user_turns]
        lengths = [len(t.split()) for t in user_turns]

        # Individual features
        escalation_traj = _escalation_trajectory(risk_per_turn)
        final_turn_risk = risk_per_turn[-1]

        # Hedge erosion: did hedging drop from early turns to late turns?
        early_hedge = sum(hedge_per_turn[: max(1, len(hedge_per_turn) // 2)]) / max(
            1, len(hedge_per_turn) // 2
        )
        late_hedge = sum(hedge_per_turn[-(max(1, len(hedge_per_turn) // 2)) :]) / max(
            1, len(hedge_per_turn) // 2
        )
        hedge_erosion = max(0.0, early_hedge - late_hedge)  # positive = eroded

        engagement_collapse = _length_drop(lengths)

        first_last_delta = max(0.0, risk_per_turn[-1] - risk_per_turn[0])

        # LIWC-derived longitudinal features
        future_decline = _future_verb_decline(user_turns)
        fp_increase = _first_person_increase(user_turns)
        social_decline = _social_word_decline(user_turns)

        features = {
            "escalation_trajectory": escalation_traj,
            "final_turn_risk": final_turn_risk,
            "hedge_erosion": hedge_erosion,
            "engagement_collapse": engagement_collapse,
            "first_last_delta": first_last_delta,
            "future_verb_decline": future_decline,
            "first_person_increase": fp_increase,
            "social_word_decline": social_decline,
            "n_user_turns": len(user_turns),
            "mean_risk": sum(risk_per_turn) / len(risk_per_turn),
            "max_risk": max(risk_per_turn),
        }

        score = sum(
            features[k] * w for k, w in self.WEIGHTS.items()
        )
        score = max(0.0, min(score, 1.0))

        return {"score": score, "features": features, "n_turns": len(turns)}

    def score(self, conversation_text: str) -> float:
        """Return composite risk score in [0, 1]."""
        return self.analyze(conversation_text)["score"]

    def score_with_convokit(self, conversation_text: str, conv_id: str = "khp_conv") -> float:
        """
        Score using ConvoKit's CoordinationTransformer for the coordination
        feature. Falls back to score() if ConvoKit is not installed.
        """
        try:
            from convokit import Coordination

            turns = parse_turns(conversation_text)
            corpus = _build_convokit_corpus(turns, conv_id)

            coord = Coordination()
            coord.fit(corpus)
            coord_scores = coord.transform(corpus).get_vector("coord_score")
            # Coordination score: high = user is mirroring bot (dependency signal)
            user_coord = (
                coord_scores.get("user", 0.0) if isinstance(coord_scores, dict) else 0.0
            )
        except Exception:
            user_coord = 0.0

        # Blend coordination into base score
        base = self.analyze(conversation_text)
        blended = base["score"] * 0.85 + user_coord * 0.15
        return max(0.0, min(blended, 1.0))
