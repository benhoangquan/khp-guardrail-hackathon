"""
fr_patch.py — French language support for the Cassandre guardrail stack.

Importing this module does three things:

  1. FR keyword patch (lines 14-103)
     Extends convo_adapter._RISK_KEYWORDS / _HEDGE_WORDS and
     ddm_model._RISK_TERMS / _HEDGE_TERMS at import time with French
     equivalents.  Neither source file is modified on disk.

  2. _detect_lang() + _FR_STOPWORDS (lines 32-55)
     Stopword-fraction heuristic: if ≥ 15 % of tokens are French function
     words, classify as "fr".  No external NLP library required.

  3. BILINGUAL_PROMPT (lines 110-162)
     Same JSON schema as the existing few-shot prompt but with a bilingual
     header and two French few-shot examples (one high-risk, one low-risk).
     Pass it directly via LLMJudgeGuardrail(user_input_prompt=BILINGUAL_PROMPT)
     so the shared template file is untouched.

Usage:
    import src.guardrails.fr_patch  # side-effect: patches both modules

    from src.guardrails.fr_patch import BILINGUAL_PROMPT, _detect_lang
    lang = _detect_lang(text)          # "fr" or "en"
"""

from __future__ import annotations

import re

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LANGUAGE DETECTION (lines 32-55 per spec)
# ─────────────────────────────────────────────────────────────────────────────

# ~50 high-frequency French function words (determiners, prepositions,
# pronouns, conjunctions, auxiliary verbs).  Chosen to minimise overlap with
# common English words.
_FR_STOPWORDS: frozenset[str] = frozenset({
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "me", "te", "se", "lui", "leur", "y", "en",
    "le", "la", "les", "un", "une", "des", "du", "de", "au", "aux",
    "et", "ou", "mais", "donc", "or", "ni", "car",
    "que", "qui", "quoi", "dont", "où", "quel", "quelle",
    "est", "sont", "était", "étais", "avait", "avais", "avoir",
    "être", "faire", "dit", "fait",
    "ne", "pas", "plus", "jamais", "rien", "personne",
    "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
    "ce", "cet", "cette", "ces",
    "avec", "dans", "sur", "sous", "pour", "par", "chez",
    "tout", "toute", "tous", "toutes",
    "très", "trop", "bien", "si", "même",
})

_FR_THRESHOLD = 0.15  # ≥ 15 % stopword fraction → French


def _detect_lang(text: str) -> str:
    """
    Classify text as "fr" (French) or "en" (English/other) using a
    stopword-fraction heuristic.  No external library required.

    Args:
        text: Raw conversation text (may be multi-turn).

    Returns:
        "fr" if ≥ 15 % of tokens are French function words, else "en".
    """
    tokens = re.findall(r"\b[a-zA-ZÀ-ÿ''-]+\b", text.lower())
    if not tokens:
        return "en"
    fr_hits = sum(1 for t in tokens if t in _FR_STOPWORDS)
    return "fr" if (fr_hits / len(tokens)) >= _FR_THRESHOLD else "en"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FRENCH KEYWORD LISTS (basis for patch below)
# ─────────────────────────────────────────────────────────────────────────────

# French risk keywords — mirrors the English lists in convo_adapter / ddm_model
_FR_RISK_KEYWORDS: set[str] = {
    # ── Explicit SI ──────────────────────────────────────────────────────
    r"\bsuicide\b",                         # same spelling in French
    r"\bme suicider\b",
    r"\bme tuer\b",
    r"\bje vais me tuer\b",
    r"\bfin de ma vie\b",
    r"\bmettre fin à ma vie\b",
    r"\bmettre fin à mes jours\b",
    r"\bmettre fin à tout\b",
    r"\bm'enlever la vie\b",                # very common QC phrasing
    r"\ben finir\b",                         # "en finir avec tout/la vie"
    r"\ben finir avec\b",
    r"\bpasser à l'acte\b",                 # clinical: acting on thoughts
    r"\bme faire du mal\b",
    r"\bse faire du mal\b",
    r"\bse blesser\b",
    r"\bauto[-\s]?mutilation\b",
    r"\bme couper\b",
    r"\bme cutter\b",                        # anglicism used by QC youth
    r"\bscarification\b",
    r"\bme brûler\b",
    r"\bme mutiler\b",
    r"\bsurdose\b",
    r"\bcomprimés?\b",
    r"\bmédicaments\b",                      # means: medications
    r"\blames?\b",                           # means: razor blades
    r"\bje veux mourir\b",
    r"\benvie de mourir\b",
    r"\bje veux crever\b",                   # vulgar but common youth term
    r"\bj'veux crever\b",
    r"\bquitter ce monde\b",
    r"\bdisparaître\b",
    r"\bje veux disparaître\b",
    r"\bpartir pour de bon\b",               # euphemism: "leave for good"
    # ── Passive SI / burden ──────────────────────────────────────────────
    r"\bà quoi bon\b",
    r"\bje sers à rien\b",
    r"\bje ne sers à rien\b",
    r"\bje suis inutile\b",
    r"\bpersonne s'en fiche\b",
    r"\bpersonne ne s'en soucie\b",
    r"\bne veux plus être là\b",
    r"\bplus envie d'être là\b",
    r"\bfardeau\b",
    r"\bje suis un poids\b",                # "I'm a weight/burden"
    r"\bje suis de trop\b",                 # "I'm one too many"
    r"\bje gâche tout\b",                   # "I ruin everything"
    r"\bc'est moi le problème\b",
    r"\bplus envie de vivre\b",
    r"\bplus la force\b",
    r"\btout arrêter\b",
    r"\btout lâcher\b",
    r"\bplus la volonté\b",
    r"\bidées noires\b",                     # "dark thoughts" — very common
    r"\bpensées suicidaires\b",
    r"\bm'endormir pour toujours\b",
    r"\bje serais mieux mort\b",
    r"\bje serais mieux morte\b",
    r"\bj'aurais pas dû naître\b",
    r"\ble monde serait mieux sans moi\b",
    r"\bpersonne remarquerait\b",
    r"\bpu de raison de rester\b",           # QC: "no reason to stay"
    r"\bj'ai rien à perdre\b",
    # ── Hopelessness / finality ──────────────────────────────────────────
    r"\bça ne changera jamais\b",
    r"\bça n'a aucun sens\b",
    r"\brien ne va changer\b",
    r"\bj'en peux plus\b",
    r"\bje n'en peux plus\b",
    r"\bpu capable\b",                       # QC joual — extremely common
    r"\bchu pu capable\b",                   # QC contraction
    r"\bécœuré de toute\b",                 # QC: "sick of everything"
    r"\btanné de toute\b",                  # QC: "fed up with everything"
    r"\bau revoir pour toujours\b",
    r"\bdernier adieu\b",
    r"\blettre d'adieu\b",                  # farewell letter
    r"\bdonner mes affaires\b",             # giving away belongings
    r"\bj'abandonne\b",
    r"\btout abandonner\b",
    r"\bc'est sans espoir\b",
    r"\baucun espoir\b",
    r"\brien ne vaut la peine\b",
    r"\bplus rien n'a d'importance\b",
    r"\bsans issue\b",                       # "no way out"
    r"\bpris au piège\b",                   # "trapped"
    r"\bje vois pas le bout\b",             # "I don't see the end"
    r"\bça finira jamais\b",
    r"\bje suis brisé\b",                   # "I'm broken"
    r"\bje suis brisée\b",
    r"\bje me sens vide\b",                 # "I feel empty"
    r"\bje sens pu rien\b",                 # QC: "I don't feel anything"
    # Temporal urgency: imminent action
    # Source: Coppersmith et al. (2018) "Natural Language Processing of Social
    # Media as Screening for Suicide Risk", Biomedical Informatics Insights.
    r"\bce soir\b",
    r"\bavant demain\b",
    r"\bmaintenant\b",
    r"\bc'est la dernière fois\b",
    r"\bdernier message\b",
    r"\bavant ce soir\b",
    r"\bce sera le dernier\b",
    r"\baprès ce message\b",
    r"\bune dernière fois\b",
    r"\bce soir c'est fini\b",
    # ── Isolation ────────────────────────────────────────────────────────
    r"\btout seul\b",
    r"\btoute seule\b",
    r"\bcompletement seul\b",
    r"\bpersonne ne m'aime\b",
    r"\bpersonne ne me comprend\b",
    r"\btotalement isolé\b",
    r"\bje suis invisible\b",
    r"\bpersonne veut de moi\b",
    r"\bj'ai personne\b",
    r"\bj'ai pu personne\b",                # QC: "I have nobody left"
    r"\btout le monde s'en fout\b",
    r"\bje dérange\b",                       # "I'm a bother"
}

# French hedge words — erode across turns as crisis deepens
_FR_HEDGE_WORDS: set[str] = {
    r"\bpeut-être\b",
    r"\bprobablement\b",
    r"\bje pense que\b",
    r"\bje crois que\b",
    r"\bj'imagine\b",
    r"\bun peu\b",
    r"\bun peu comme\b",
    r"\bparfois\b",
    r"\bde temps en temps\b",
    r"\ben quelque sorte\b",
    r"\bça dépend\b",
    r"\bje ne suis pas sûr\b",
    r"\bje ne sais pas trop\b",
    r"\bj'sais pas\b",
    r"\bjsp\b",                              # texting: "je sais pas"
    r"\bpas vraiment\b",
    r"\bevéntuellement\b",
    r"\bpossiblement\b",
    r"\bgenre\b",                            # QC youth filler: "like"
    r"\bmettons\b",                          # QC: "let's say" / hedging
    r"\btsé\b",                              # QC contraction: "tu sais"
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PATCH BOTH MODULES AT IMPORT TIME (lines 14-103 per spec)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_fr_patch() -> None:
    """
    Extend the live keyword sets in convo_adapter and ddm_model with French
    entries.  Using module-level set/list mutation means the patch is visible
    to all subsequent calls to those modules' functions without modifying the
    source files.
    """
    # --- convo_adapter: set-based attributes ---
    try:
        from src.guardrails import convo_adapter as _ca
        _ca._RISK_KEYWORDS.update(_FR_RISK_KEYWORDS)
        _ca._HEDGE_WORDS.update(_FR_HEDGE_WORDS)
    except (ImportError, AttributeError) as exc:
        import warnings
        warnings.warn(f"fr_patch: could not patch convo_adapter — {exc}", stacklevel=2)

    # --- ddm_model: list-based attributes ---
    try:
        from src.guardrails import ddm_model as _dm
        # Lists are used directly in re.search loops; deduplicate to keep
        # the list lean while preserving fast iteration order.
        existing_terms = set(_dm._RISK_TERMS)
        existing_hedges = set(_dm._HEDGE_TERMS)
        _dm._RISK_TERMS.extend(
            t for t in sorted(_FR_RISK_KEYWORDS) if t not in existing_terms
        )
        _dm._HEDGE_TERMS.extend(
            h for h in sorted(_FR_HEDGE_WORDS) if h not in existing_hedges
        )
    except (ImportError, AttributeError) as exc:
        import warnings
        warnings.warn(f"fr_patch: could not patch ddm_model — {exc}", stacklevel=2)


# Run patch on import — idempotent because we deduplicate before extending.
_apply_fr_patch()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — PROMPT HEADER / FOOTER + STRUCTURED EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────
# _EXAMPLES is the single source of truth.  BILINGUAL_PROMPT (static, all 8)
# and build_prompt() (dynamic, top-k via BM25+) are both derived from it.

_PROMPT_HEADER: str = """\
You are a specialized safety classifier for a youth mental health chat service (KHP — Kids Help Phone).
Your task is to classify conversations as HIGH RISK or LOW RISK, in English OR French.

You MUST apply the Columbia Suicide Severity Rating Scale (C-SSRS) clinical framework. This is CRITICAL to avoid false positives:

## C-SSRS CLINICAL DECISION LOGIC

### HIGH RISK (high_risk: true, score > 0.55) — Requires ANY of:
1. **Active Ideation with Intent**: "I want to kill myself and I've decided to do it"
2. **Active Ideation with Plan**: Mentions a specific method, time, or location ("I have pills, tonight")
3. **Active Ideation with Means**: Has access to or is acquiring lethal means
4. **Passive Suicidal Ideation (persistent)**: Repeated or escalating "I wish I were dead", "everyone would be better off", "I want it to stop" across turns
5. **Self-harm with intent**: Explicitly cutting, burning, harming — with reason to continue
6. **Burden identification**: "I am a burden", "my family would be better off without me" with hopelessness

### LOW RISK (high_risk: false, score < 0.45) — Apply for:
- **Hyperbole / idiom**: "I want to die of embarrassment", "this is killing me", "I'm dead tired"
- **Academic / social stress**: Exam pressure, friendship conflict, family arguments — WITHOUT self-harm language
- **Past tense, resolved**: "last year I struggled, but I'm okay now"
- **Venting / catharsis**: Expressing frustration without self-destructive direction
- **Information seeking**: "how do I help a friend who is depressed?"
- **Somatic complaints alone**: "I have a headache", "I feel sick" without psychological distress

For multi-turn conversations, escalating distress across turns is a stronger signal than a single isolated mention, weight recent turns more heavily.

### SCORE GUIDANCE:
- 0.0-0.30: Clearly low risk (normal stress, no distress signals)
- 0.30-0.55: Mild distress (monitor but no immediate action needed)
- 0.55-0.75: Passive ideation / high distress — watch closely
- 0.75-1.0: Active ideation, intent, plan, or means present

---

## TWO-STEP CLASSIFICATION PROCESS

Before deciding, work through these two steps inside your reasoning field:

**Step 1 — Evidence extraction**: Quote up to 3 statements from the conversation that are most clinically concerning (or note "none found" if absent).

**Step 2 — C-SSRS mapping**: For each quoted statement, name the closest C-SSRS criterion (e.g. "passive ideation", "active ideation with plan", "burden identification", "hyperbole") — or "none" if no criterion applies.

Then state your conclusion. Only after this reasoning, set high_risk and score.

---

## EXAMPLES

"""

_PROMPT_FOOTER: str = """

---

**Return ONLY valid JSON — no explanations outside the JSON block:**
{
    "reasoning": "Step 1: [quoted statements]. Step 2: [C-SSRS mappings]. Conclusion: [final assessment]. English or French.",
    "high_risk": true/false,
    "score": 0.0-1.0
}
"""

# Each entry: content (used for BM25+ retrieval), label, lang, text (injected).
_EXAMPLES: list[dict] = [
    {
        "content": (
            "Nobody would miss me if I disappeared. "
            "I've been thinking about it for weeks. I don't see the point anymore."
        ),
        "label": "high_risk",
        "lang": "en",
        "text": (
            "**EN Example 1 — HIGH RISK** (passive ideation + burden):\n"
            'Content: "Nobody would miss me if I disappeared. '
            "I've been thinking about it for weeks. I don't see the point anymore.\"\n"
            'Output: {"reasoning": "Step 1: burden language (nobody would miss me), '
            "sustained passive ideation (thinking about it for weeks), hopelessness (don't see the point). "
            "Step 2: burden identification; passive ideation—persistent; hopelessness. "
            'Conclusion: persistent passive SI with burden language.", "high_risk": true, "score": 0.88}'
        ),
    },
    {
        "content": (
            "I literally want to die. This exam is going to kill me. Why is school so hard??"
        ),
        "label": "low_risk",
        "lang": "en",
        "text": (
            "**EN Example 2 — LOW RISK** (hyperbole with academic stress):\n"
            'Content: "I literally want to die. This exam is going to kill me. Why is school so hard??"\n'
            'Output: {"reasoning": "Step 1: hyperbolic death reference (want to die), '
            "hyperbolic kill reference (going to kill me). "
            "Step 2: hyperbole—academic context; hyperbole. "
            'Conclusion: situational exam frustration, no genuine ideation.", "high_risk": false, "score": 0.12}'
        ),
    },
    {
        "content": (
            "I've decided. I know how I'm going to do it. I just wanted to say goodbye."
        ),
        "label": "high_risk",
        "lang": "en",
        "text": (
            "**EN Example 3 — HIGH RISK** (active ideation + plan):\n"
            'Content: "I\'ve decided. I know how I\'m going to do it. I just wanted to say goodbye."\n'
            'Output: {"reasoning": "Step 1: decided to act (I\'ve decided), '
            "implicit method knowledge (I know how), farewell behavior (say goodbye). "
            "Step 2: active ideation with intent; active ideation with plan; finality/farewell. "
            'Conclusion: active SI with intent and implicit plan—highest severity.", "high_risk": true, "score": 0.97}'
        ),
    },
    {
        "content": (
            "I'm really sad today. My best friend and I had a fight and I don't know what to do."
        ),
        "label": "low_risk",
        "lang": "en",
        "text": (
            "**EN Example 4 — LOW RISK** (general sadness):\n"
            'Content: "I\'m really sad today. My best friend and I had a fight and I don\'t know what to do."\n'
            'Output: {"reasoning": "Step 1: transient sadness from friend conflict. '
            "Step 2: none—interpersonal stress, no SI language. "
            'Conclusion: temporary distress without risk indicators.", "high_risk": false, "score": 0.18}'
        ),
    },
    {
        "content": (
            "je sers vraiment à rien. Ma famille serait mieux sans moi. "
            "J'ai l'impression que ça ne changera jamais, peu importe ce que je fais."
        ),
        "label": "high_risk",
        "lang": "fr",
        "text": (
            "**FR Exemple 5 — RISQUE ÉLEVÉ** (idéation passive persistante + fardeau):\n"
            'Contenu: "je sers vraiment à rien. Ma famille serait mieux sans moi. '
            "J'ai l'impression que ça ne changera jamais, peu importe ce que je fais.\"\n"
            'Sortie: {"reasoning": "Etape 1: sentiment d\'inutilite (je sers a rien), '
            "fardeau (famille serait mieux sans moi), desespoir (ca ne changera jamais). "
            "Etape 2: identification comme fardeau; burden identification; desespoir persistant. "
            'Conclusion: ideation passive avec desespoir profond.", "high_risk": true, "score": 0.85}'
        ),
    },
    {
        "content": (
            "je veux mourir tellement j'ai honte de mon exam. "
            "C'était terrible, je veux disparaître de la classe."
        ),
        "label": "low_risk",
        "lang": "fr",
        "text": (
            "**FR Exemple 6 — FAIBLE RISQUE** (frustration scolaire, hyperbole):\n"
            'Contenu: "je veux mourir tellement j\'ai honte de mon exam. '
            "C'était terrible, je veux disparaître de la classe.\"\n"
            'Sortie: {"reasoning": "Etape 1: je veux mourir (contexte scolaire), je veux disparaitre. '
            "Etape 2: hyperbole academique; hyperbole situationnelle. "
            'Conclusion: expressions hyperboliques liees a la honte scolaire.", "high_risk": false, "score": 0.14}'
        ),
    },
    {
        "content": (
            "j'ai plus la force de continuer. J'ai pas dormi depuis 3 jours, j'ai rien mangé. "
            "Je regarde les médicaments dans l'armoire et je me demande..."
        ),
        "label": "high_risk",
        "lang": "fr",
        "text": (
            "**FR Exemple 7 — RISQUE ÉLEVÉ** (idéation active + accès aux moyens):\n"
            'Contenu: "j\'ai plus la force de continuer. J\'ai pas dormi depuis 3 jours. '
            "Je regarde les médicaments dans l'armoire et je me demande...\"\n"
            'Sortie: {"reasoning": "Etape 1: plus la force de continuer, '
            "regarde les medicaments dans l'armoire, je me demande... "
            "Etape 2: ideation passive avec epuisement; acces aux moyens letaux; pensee incomplete suggestive. "
            'Conclusion: ideation active avec acces aux moyens.", "high_risk": true, "score": 0.93}'
        ),
    },
    {
        "content": (
            "mon ami m'a dit qu'il pensait parfois à se faire du mal. Comment je peux l'aider?"
        ),
        "label": "low_risk",
        "lang": "fr",
        "text": (
            "**FR Exemple 8 — FAIBLE RISQUE** (recherche d'aide pour un ami):\n"
            'Contenu: "mon ami m\'a dit qu\'il pensait parfois à se faire du mal. Comment je peux l\'aider?"\n'
            'Sortie: {"reasoning": "Etape 1: il pensait a se faire du mal (tiers, pas l\'auteur). '
            "Etape 2: aucun critere pour l'auteur—comportement prosocial. "
            'Conclusion: risque direct nul pour la personne qui ecrit.", "high_risk": false, "score": 0.25}'
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — BM25+ FEW-SHOT RETRIEVER (TF-IDF fallback)
# ─────────────────────────────────────────────────────────────────────────────

class _FewShotRetriever:
    """
    Dynamic few-shot selector using BM25Plus (Robertson & Zaragoza 2009)
    with a TF-IDF cosine-similarity fallback when rank_bm25 is unavailable.

    BM25+ is preferred over plain TF-IDF because it normalises for document
    length — important here since conversations range from 1 to 20+ turns —
    and its lower-bound fix ensures every matching term contributes a non-zero
    score.  Both backends are O(n_examples) per query (~8 docs → negligible).

    Diversity guarantee: the retrieved set always contains at least one
    HIGH RISK and one LOW RISK example when k ≥ 2.  This prevents the LLM
    from seeing a homogeneous shot set that could anchor classification.

    References:
      - Robertson & Zaragoza (2009) "The Probabilistic Relevance Framework:
        BM25 and Beyond", Foundations and Trends in IR 3(4).
      - Lv & Zhai (2011) "Lower-bounding term frequency normalization"
        (BM25+), CIKM '11 — fixes zero lower-bound issue in BM25.
    """

    def __init__(self, examples: list[dict]) -> None:
        self._examples = examples
        self._corpus = [self._tokenize(ex["content"]) for ex in examples]
        self._backend = self._build_index()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b[a-zA-ZÀ-ÿ''-]+\b", text.lower())

    def _build_index(self) -> str:
        try:
            from rank_bm25 import BM25Plus  # type: ignore
            self._index = BM25Plus(self._corpus)
            return "bm25+"
        except ImportError:
            pass
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
            texts = [ex["content"] for ex in self._examples]
            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=2000).fit(texts)
            self._tfidf_matrix = self._tfidf.transform(texts)
            self._cosine_similarity = cosine_similarity
            return "tfidf"
        except ImportError:
            return "fallback"

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Return k examples most similar to query, diversity-guaranteed."""
        n = len(self._examples)
        k = min(k, n)

        if self._backend == "bm25+":
            scores = self._index.get_scores(self._tokenize(query))
            indices = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
        elif self._backend == "tfidf":
            qvec = self._tfidf.transform([query])
            sims = self._cosine_similarity(qvec, self._tfidf_matrix)[0]
            indices = sorted(range(n), key=lambda i: float(sims[i]), reverse=True)[:k]
        else:
            # Fallback: interleave high/low examples for guaranteed diversity
            hr = [i for i, ex in enumerate(self._examples) if ex["label"] == "high_risk"]
            lr = [i for i, ex in enumerate(self._examples) if ex["label"] == "low_risk"]
            indices = [x for pair in zip(hr, lr) for x in pair][:k]

        selected = [self._examples[i] for i in indices]

        # Diversity guarantee: at least 1 high_risk AND 1 low_risk when k >= 2
        if k >= 2:
            labels = {ex["label"] for ex in selected}
            if len(labels) < 2:
                missing = "low_risk" if "high_risk" in labels else "high_risk"
                for ex in self._examples:
                    if ex["label"] == missing and ex not in selected:
                        selected[-1] = ex
                        break

        return selected


# Module-level singleton — index built once at import, zero per-query overhead
_retriever = _FewShotRetriever(_EXAMPLES)


def build_prompt(query: str, k: int = 3) -> str:
    """
    Build a dynamic few-shot system prompt by retrieving the k most relevant
    examples for the given query using BM25+ (TF-IDF fallback).

    The returned prompt is ready to use as LLMJudgeGuardrail.user_input_prompt.

    Args:
        query: Conversation text being classified (used for retrieval only).
        k:     Number of few-shot examples to inject (default 3).
    """
    examples = _retriever.retrieve(query, k=k)
    return _PROMPT_HEADER + "\n\n".join(ex["text"] for ex in examples) + _PROMPT_FOOTER


# Static prompt (all 8 examples) — stable baseline / used when query is
# unavailable at init time (e.g. guardrail constructed before first call).
BILINGUAL_PROMPT: str = _PROMPT_HEADER + "\n\n".join(ex["text"] for ex in _EXAMPLES) + _PROMPT_FOOTER

