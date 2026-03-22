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
# SECTION 4 — BILINGUAL PROMPT (lines 110-162 per spec)
# ─────────────────────────────────────────────────────────────────────────────

BILINGUAL_PROMPT: str = """\
You are a specialized safety classifier for a youth mental health chat service (KHP — Kids Help Phone).
Your task is to classify conversations as HIGH RISK or LOW RISK, in English OR French.

**Safety is our absolute priority: always err on the side of caution.**
A missed high-risk case can cost a life. A false alarm costs a counsellor call. When in doubt, flag HIGH RISK.

## HIGH RISK (high_risk: true) — Flag if ANY of the following are present, even vaguely:

**Suicidal / self-harm signals:**
- Any expression of wanting to die, disappear, or not exist — even indirect ("what's the point", "I can't do this anymore", "en finir")
- Passive ideation: "everyone would be better off without me", "nobody would miss me", "je sers à rien"
- Active ideation with plan, method, timeline, or means ("I have pills", "je regarde les médicaments")
- Finality language: "saying goodbye", "last time", "dernier adieu", giving away belongings
- Self-harm: cutting, burning, hitting self — current or intended
- Fear of own thoughts or impulses: "I'm scared of what I might do"

**Hopelessness / entrapment signals:**
- Deep hopelessness with no way out: "it'll never change", "ça ne changera jamais", "sans issue"
- Feeling trapped, worthless, or a burden: "je suis un fardeau", "I ruin everything"
- Emotional numbness or shutdown: "I don't feel anything anymore", "je sens pu rien"
- Withdrawal + isolation framing: "I'm better off alone", "no one would notice"

**Crisis signals:**
- Active abuse, violence, or immediate physical danger
- Substance use to cope with intent to overdose
- Any incomplete or trailing sentence suggesting the worst ("I've been thinking...", "je me demande si...")

## LOW RISK (high_risk: false) — ONLY when clearly and obviously safe:
- **Unambiguous hyperbole in a light context**: "this exam is literally killing me lol" — only safe if the surrounding tone is clearly playful with no other distress signals
- **Pure info-seeking**: "how do I support a depressed friend?" with no personal distress language
- **Clearly resolved past event**: "last year was hard but I'm doing much better now"
- **Routine frustration only**: normal school/friend stress with zero self-destructive language

**When borderline: always choose HIGH RISK.**

## SCORE GUIDANCE (calibrated for high recall):
- 0.0–0.20: Clearly safe, zero distress signals
- 0.20–0.45: Mild stress, no risk language
- 0.45–0.65: Distress present, ambiguous signals — lean HIGH RISK
- 0.65–0.85: Clear passive ideation, burden language, hopelessness
- 0.85–1.0: Active ideation, plan, means, finality, or immediate danger

---

## EXAMPLES

**EN 1 — HIGH RISK** (vague despair, giving up):
Content: "I'm just giving up on everything. Saying goodbye feels right. I can't stop thinking about it."
Output: {"high_risk": true, "score": 0.88, "reasoning": "Finality language ('saying goodbye'), giving up, intrusive thoughts. Classic passive SI with finality. Flag immediately."}

**EN 2 — HIGH RISK** (subtle, dark thoughts):
Content: "I've been thinking about some really dark stuff. I'm scared because I can't stop these thoughts."
Output: {"high_risk": true, "score": 0.82, "reasoning": "Fear of own thoughts, persistent dark ideation. Even vague, this meets high-risk threshold — err on caution."}

**EN 3 — HIGH RISK** (burden + isolation, single turn):
Content: "Nobody would miss me if I was gone. I'm just a burden to everyone around me."
Output: {"high_risk": true, "score": 0.84, "reasoning": "Burden language + passive SI ('nobody would miss me', 'gone'). Does not need to be persistent — single expression is enough to flag."}

**EN 4 — HIGH RISK** (active plan):
Content: "I've decided. I know how I'm going to do it. I just wanted to say goodbye."
Output: {"high_risk": true, "score": 0.97, "reasoning": "Active ideation with implicit plan and finality language. Highest severity."}

**EN 5 — LOW RISK** (obvious hyperbole, playful tone):
Content: "I literally want to die lol this exam is going to kill me. Why is school so hard??"
Output: {"high_risk": false, "score": 0.12, "reasoning": "Clear hyperbole in a light, joking tone ('lol'). Pure academic stress, zero burden or ideation language."}

**EN 6 — LOW RISK** (general sadness, no risk language):
Content: "I'm really sad today. My best friend and I had a fight and I don't know what to do."
Output: {"high_risk": false, "score": 0.18, "reasoning": "Interpersonal conflict, temporary sadness. No self-harm or suicidal language anywhere."}

**FR 7 — RISQUE ÉLEVÉ** (idéation passive + fardeau, QC joual):
Contenu: "chu pu capable de toute. Je sers à rien, ma famille serait ben mieux sans moi. J'ai des idées noires depuis un boutte."
Sortie: {"high_risk": true, "score": 0.89, "reasoning": "Langage de fardeau ('ma famille serait mieux sans moi'), idées noires persistantes, épuisement total ('pu capable'). Même sans plan explicite, signaux suffisants pour signaler immédiatement."}

**FR 8 — RISQUE ÉLEVÉ** (moyens létaux, phrase incomplète):
Contenu: "j'ai plus la force de continuer. J'ai pas dormi depuis 3 jours, j'ai rien mangé. Je regarde les médicaments dans l'armoire et je me demande..."
Sortie: {"high_risk": true, "score": 0.95, "reasoning": "Accès aux moyens létaux (médicaments), phrase incomplète suggestive, détresse physique sévère. Signaux de crise imminente."}

**FR 9 — RISQUE ÉLEVÉ** (désespoir, finitude, ton plat):
Contenu: "tanné de toute. Ça changera jamais. J'ai même pu envie de me lever le matin. C'est correct de même."
Sortie: {"high_risk": true, "score": 0.78, "reasoning": "Résignation totale ('c'est correct de même' = finality acceptance), anhédonie, désespoir profond. Ton plat avec acceptation de la fin = signal élevé même sans mots explicites."}

**FR 10 — FAIBLE RISQUE** (hyperbole scolaire, contexte clair):
Contenu: "je veux mourir tellement j'ai honte de mon exam. C'était terrible, franchement."
Sortie: {"high_risk": false, "score": 0.13, "reasoning": "Hyperbole scolaire claire, contexte situationnel unique, aucun langage de fardeau ou d'idéation persistante."}

**FR 11 — FAIBLE RISQUE** (aide pour un tiers):
Contenu: "mon ami m'a dit qu'il pensait parfois à se faire du mal. Comment je peux l'aider?"
Sortie: {"high_risk": false, "score": 0.22, "reasoning": "Personne cherche de l'aide pour un ami, pas pour elle-même. Démarche prosociale, faible risque direct."}

---

**Return ONLY valid JSON:**
{
    "high_risk": true/false,
    "score": 0.0-1.0,
    "reasoning": "Brief explanation citing the specific signal. English or French."
}
"""

