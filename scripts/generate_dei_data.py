#!/usr/bin/env python3
"""
generate_dei_data.py — Synthetic DEI-focused bilingual training data generator.

Uses the Mistral API to generate adversarial and safe conversations across
underrepresented demographics and linguistic groups. Outputs a CSV with
the same schema as the baseline training data for direct use with
train_classifier_guardrail.py.

Scenario sources:
  - Optional: datasets/output/test_matrix_config_v2.json — taxonomies × risk
    levels, rotated across dei_variables and linguistic_shapers (see JSON).
  - Hand-crafted LEGACY_SCENARIOS for extra nuance not fully captured in the matrix.

DEI coverage:
  - LGBTQ+ youth isolation and rejection
  - Marginalized / immigrant youth (somatic distress, intergenerational conflict)
  - French-Canadian dialect and informal register
  - Cultural minority contexts (Latinx, South Asian, Black youth perspectives)
  - Neurodivergent and school-avoidance contexts
  - Matrix-driven age bands (e.g. 10-year-old, neurodivergent 12-year-old, university)

Usage (from repo root):
    python project/generate_dei_data.py
    python project/generate_dei_data.py --no-legacy --target 500
    python project/generate_dei_data.py --no-legacy --replicates 6
    python project/generate_dei_data.py --sanity 4 --no-legacy
    python project/generate_dei_data.py --no-matrix --limit 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
_PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT))

from dotenv import load_dotenv
load_dotenv(Path(_PROJECT).parent / ".env")

from providers.mistral_provider import MistralProvider

# ── Config ─────────────────────────────────────────────────────────────────────
OUTPUT_PATH = _PROJECT / "datasets" / "synthetic_dei_data.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_MATRIX_CONFIG = _PROJECT.parent / "datasets" / "output" / "test_matrix_config_v2.json"


def _matrix_config_data(config_path: Path) -> Optional[Dict]:
    if not config_path.is_file():
        return None
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def load_scenarios_from_test_matrix(
    config_path: Path,
    languages: Sequence[str] = ("en", "fr"),
    replicates: int = 1,
) -> List[Dict]:
    """
    Build scenario dicts from test_matrix_config_v2.json-style config.

    Each (category × language) cell is repeated `replicates` times with different
    persona/shaper pairings (linear slot index) so totals can reach hundreds of rows
    without editing the JSON.
    """
    data = _matrix_config_data(config_path)
    if not data:
        return []

    personae: List[str] = data.get("dei_variables") or []
    shapers: List[str] = data.get("linguistic_shapers") or []
    categories: List[Dict] = data.get("categories") or []

    if not categories:
        return []

    scenarios: List[Dict] = []
    n_p = len(personae) if personae else 1
    n_s = len(shapers) if shapers else 1
    rep = max(1, int(replicates))

    for r in range(rep):
        for i, cat in enumerate(categories):
            risk_raw = str(cat.get("risk_level", "Low")).strip().lower()
            label = 1 if risk_raw == "high" else 0
            taxonomy = cat.get("taxonomy", "").strip()
            base_prompt = cat.get("base_prompt", "").strip()
            cat_id = cat.get("id", "").strip()

            for lang in languages:
                lang_norm = "fr" if str(lang).lower().startswith("fr") else "en"
                lang_idx = 1 if lang_norm == "fr" else 0
                seq = r * (len(categories) * len(languages)) + i * len(languages) + lang_idx
                persona = personae[seq % n_p] if personae else "a youth seeking support"
                shaper = shapers[(seq // n_p + seq) % n_s] if shapers else "natural informal teen voice"
                variant = f"{cat_id}#r{r}" if rep > 1 else cat_id

                scenarios.append({
                    "label": label,
                    "lang": lang_norm,
                    "context": (
                        f"[{variant}] Taxonomy: {taxonomy} ({risk_raw} risk). "
                        f"Situation to role-play: {base_prompt} "
                        f"Protagonist: {persona}. "
                        f"Speaking style: {shaper}."
                    ),
                    "matrix_id": cat_id,
                    "taxonomy": taxonomy,
                    "replicate": r,
                })

    return scenarios


# ── Hand-crafted scenarios (merged with matrix when --matrix is used) ───────
LEGACY_SCENARIOS: List[Dict] = [
    # ─── HIGH RISK: Diverse contexts ─────────────────────────────────────────
    dict(label=1, lang="en", context="LGBTQ+ teen rejected by family after coming out, expressing suicidal ideation indirectly"),
    dict(label=1, lang="en", context="A teenager who is being severely bullied online, expressing they have no reason to continue"),
    dict(label=1, lang="en", context="South Asian youth expressing shame and burden to family, passive suicidal ideation"),
    dict(label=1, lang="en", context="Black youth experiencing racial trauma and police violence fear, saying they can't take it anymore"),
    dict(label=1, lang="en", context="Neurodivergent teen exhausted from masking, expressing they want everything to stop"),
    dict(label=1, lang="en", context="Teen in foster care, expressing nobody loves them and they are completely alone"),
    dict(label=1, lang="en", context="Youth describing a specific plan using medication they have at home"),
    dict(label=1, lang="en", context="Teen describing saying final goodbyes to friends online without stating why"),
    dict(label=1, lang="en", context="Immigrant youth feeling caught between two cultures, expressing worthlessness"),
    dict(label=1, lang="en", context="Teen describing self-cutting as the only way to feel something"),
    # French high risk
    dict(label=1, lang="fr", context="Ado québécois rejeté par sa famille après son coming-out, idéation suicidaire indirecte avec argot québécois"),
    dict(label=1, lang="fr", context="Jeune immigrant haïtien exprimant la honte familiale et le sentiment d'être un fardeau"),
    dict(label=1, lang="fr", context="Adolescente francophone vivant de la violence conjugale à la maison, demandant si les choses peuvent s'améliorer"),
    dict(label=1, lang="fr", context="Jeune autochtone du Québec exprimant le désespoir et l'isolement dans une communauté éloignée"),
    dict(label=1, lang="fr", context="Ado francophone qui dit avoir les médicaments et se demander combien il en faudrait"),
    dict(label=1, lang="fr", context="Jeune trans québécois(e) qui exprime ne plus vouloir se battre contre sa famille"),

    # ─── LOW RISK: Diverse contexts ──────────────────────────────────────────
    dict(label=0, lang="en", context="LGBTQ+ teen venting about a rough day at school but feeling supported by their friend group"),
    dict(label=0, lang="en", context="Teen expressing frustration with immigrant parents who don't understand North American culture"),
    dict(label=0, lang="en", context="Black youth venting about microaggressions at school but feeling empowered and resilient"),
    dict(label=0, lang="en", context="Neurodivergent teen asking for help understanding their ADHD diagnosis"),
    dict(label=0, lang="en", context="Teen struggling with academic pressure but actively seeking coping strategies"),
    dict(label=0, lang="en", context="Youth asking how to support a friend going through a difficult breakup"),
    dict(label=0, lang="en", context="Teen using hyperbole: 'I want to disappear' after an embarrassing moment at school"),
    dict(label=0, lang="en", context="Teen experiencing grief after a pet died, sad but connected to family"),
    dict(label=0, lang="en", context="South Asian youth venting about arranged marriage pressure from parents"),
    dict(label=0, lang="en", context="Teen asking general questions about how therapy works"),
    # French low risk
    dict(label=0, lang="fr", context="Ado québécois stressé par les examens du bac mais soutenu par ses amis"),
    dict(label=0, lang="fr", context="Jeune immigrant francophone frustré par la barrière linguistique mais optimiste"),
    dict(label=0, lang="fr", context="Adolescente demandant comment aider son copain qui est déprimé"),
    dict(label=0, lang="fr", context="Jeune du Québec disant 'je veux mourir tellement j'ai honte' après une mauvaise note"),
    dict(label=0, lang="fr", context="Ado francophone qui cherche des ressources pour mieux gérer son anxiété"),
    dict(label=0, lang="fr", context="Jeune demandant de l'aide pour parler à ses parents d'un problème à l'école"),
]

# Backwards-compatible alias
SCENARIOS = LEGACY_SCENARIOS


def count_legacy_for_languages(languages: Tuple[str, ...]) -> int:
    return sum(1 for s in LEGACY_SCENARIOS if s.get("lang", "en") in languages)


def infer_replicates_for_target(
    *,
    matrix_path: Path,
    languages: Tuple[str, ...],
    target: int,
    use_matrix: bool,
    use_legacy: bool,
) -> int:
    """How many matrix replicates are needed so matrix + legacy >= target (before truncation)."""
    leg = count_legacy_for_languages(languages) if use_legacy else 0
    if not use_matrix:
        return 1
    base = len(load_scenarios_from_test_matrix(matrix_path, languages=languages, replicates=1))
    if base == 0:
        return 1
    need_matrix = max(0, target - leg)
    return max(1, (need_matrix + base - 1) // base)


GENERATION_PROMPT = """\
You are generating synthetic training data for a youth mental health guardrail system.
Generate a realistic multi-turn chat conversation (3-6 turns) between a teen and a chat support service.

Context: {context}
Language: {lang}
Target label: {label}

Requirements:
- Write in a natural, authentic teen voice (informal, abbreviations, typos are OK)
- For FRENCH: use Québécois informal register (e.g., "chus" instead of "je suis", "pis" instead of "puis", "ben" instead of "bien")
- For HIGH RISK: include subtle escalation across turns, don't be too obvious
- For LOW RISK: be realistic — include real frustrations but no genuine ideation
- Each turn: "User: [message]" on its own line
- Output ONLY the conversation, no other text

Generate the conversation now:"""

def generate_conversation(provider: MistralProvider, scenario: Dict) -> str:
    """Generate one synthetic conversation using the Mistral API."""
    lang_str = "French (Québécois)" if scenario["lang"] == "fr" else "English"
    label_desc = (
        "HIGH RISK — the conversation should contain genuine distress signals appropriate for a crisis line intervention"
        if scenario["label"] == 1
        else "LOW RISK — the conversation should express normal teen issues WITHOUT any self-harm or suicidal language"
    )
    prompt = GENERATION_PROMPT.format(
        context=scenario["context"],
        lang=lang_str,
        label=label_desc,
    )
    from providers.base import LLMMessage
    response = provider.generate_sync(
        messages=[LLMMessage(role="user", content=prompt)],
        temperature=0.85,
        max_tokens=600,
    )
    return response.content.strip()


def build_scenario_list(
    *,
    matrix_path: Path,
    use_matrix: bool,
    use_legacy: bool,
    languages: Tuple[str, ...],
    replicates: int = 1,
) -> List[Dict]:
    scenarios: List[Dict] = []
    if use_matrix:
        scenarios.extend(
            load_scenarios_from_test_matrix(matrix_path, languages=languages, replicates=replicates)
        )
    if use_legacy:
        for s in LEGACY_SCENARIOS:
            if s.get("lang", "en") in languages:
                scenarios.append(dict(s))
    return scenarios


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic DEI training conversations via Mistral.")
    p.add_argument(
        "--matrix-config",
        type=Path,
        default=DEFAULT_MATRIX_CONFIG,
        help=f"Path to test matrix JSON (default: {DEFAULT_MATRIX_CONFIG})",
    )
    p.add_argument("--no-matrix", action="store_true", help="Do not load scenarios from the matrix JSON.")
    p.add_argument("--no-legacy", action="store_true", help="Omit hand-crafted LEGACY_SCENARIOS.")
    p.add_argument(
        "--lang",
        choices=("both", "en", "fr"),
        default="both",
        help="Languages for matrix-driven rows: both, en only, or fr only.",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap total scenarios (after merge, in order).")
    p.add_argument(
        "--replicates",
        type=int,
        default=None,
        metavar="N",
        help="Repeat each matrix (category×lang) cell N times with varied persona/shaper. "
        "If omitted with --target, replicates are inferred automatically.",
    )
    p.add_argument(
        "--target",
        type=int,
        default=None,
        metavar="N",
        help="Approximate total rows after merge (matrix replicates + legacy); list is truncated to N.",
    )
    p.add_argument(
        "--sanity",
        type=int,
        default=None,
        metavar="N",
        help="Generate only N conversations, print each in full for inspection, then exit. "
        "Writes CSV to --output (default: project/datasets/sanity_dei_sample.csv).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: synthetic_dei_data.csv, or sanity_dei_sample.csv with --sanity).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    langs: Tuple[str, ...] = ("en", "fr") if args.lang == "both" else (args.lang,)

    use_matrix = not args.no_matrix
    use_legacy = not args.no_legacy
    if not use_matrix and not use_legacy:
        print("❌ Nothing to generate: both --no-matrix and --no-legacy were set.")
        sys.exit(1)

    if use_matrix and not args.matrix_config.is_file():
        print(f"⚠ Matrix config not found ({args.matrix_config}); matrix scenarios skipped.\n")

    replicates = args.replicates
    if replicates is None:
        if args.target is not None:
            replicates = infer_replicates_for_target(
                matrix_path=args.matrix_config,
                languages=langs,
                target=args.target,
                use_matrix=use_matrix,
                use_legacy=use_legacy,
            )
        else:
            replicates = 1
    else:
        replicates = max(1, replicates)

    scenarios = build_scenario_list(
        matrix_path=args.matrix_config,
        use_matrix=use_matrix,
        use_legacy=use_legacy,
        languages=langs,
        replicates=replicates,
    )
    if not scenarios:
        print("❌ No scenarios to generate (empty matrix and/or filtered legacy).")
        sys.exit(1)

    if args.target is not None:
        scenarios = scenarios[: max(0, args.target)]

    if args.limit is not None:
        scenarios = scenarios[: max(0, args.limit)]

    sanity_mode = args.sanity is not None
    if sanity_mode:
        n = max(1, args.sanity)
        scenarios = scenarios[:n]

    if args.output is not None:
        out_path = args.output
    elif sanity_mode:
        out_path = _PROJECT / "datasets" / "sanity_dei_sample.csv"
    else:
        out_path = OUTPUT_PATH

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("🤖 Starting DEI synthetic data generation...")
    print(f"   Provider: Mistral ({os.getenv('MISTRAL_MODEL')})")
    print(f"   Matrix config: {args.matrix_config} ({'on' if use_matrix else 'off'})")
    print(f"   Legacy scenarios: {'on' if use_legacy else 'off'}")
    print(f"   Matrix replicates: {replicates}")
    print(f"   Languages: {', '.join(langs)}")
    print(f"   Scenarios: {len(scenarios)}")
    if sanity_mode:
        print(f"   SANITY MODE: printing {len(scenarios)} full transcripts")
    print(f"   Output: {out_path}\n")

    provider = MistralProvider(
        base_url=os.getenv("BUZZ_MISTRAL_LARGE_API"),
        model=os.getenv("MISTRAL_MODEL"),
        api_key=os.getenv("BUZZ_MISTRAL_LARGE_AUTH_TOKEN"),
        temperature=0.85,
    )

    rows = []
    for i, scenario in enumerate(scenarios):
        lang_label = "🇨🇦 FR" if scenario["lang"] == "fr" else "🇬🇧 EN"
        risk_label = "🔴 HIGH" if scenario["label"] == 1 else "🟢 LOW "
        ctx = scenario.get("context", "")
        print(f"  [{i+1:02d}/{len(scenarios)}] {lang_label} {risk_label} | {ctx[:60]}...")

        try:
            text = generate_conversation(provider, scenario)
            rows.append({
                "text": text,
                "label": scenario["label"],
                "language": scenario["lang"],
                "source": "synthetic_dei_sanity" if sanity_mode else "synthetic_dei",
                "context_description": ctx,
            })
            if sanity_mode:
                bar = "=" * 72
                print(f"\n{bar}\n# {i + 1}/{len(scenarios)}  label={scenario['label']}  lang={scenario['lang']}\n{bar}")
                print(f"CONTEXT:\n{ctx}\n")
                print(f"GENERATED:\n{text}\n")
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"    ⚠ Error on scenario {i+1}: {e}")
            rows.append({
                "text": f"[ERROR: {e}]",
                "label": scenario["label"],
                "language": scenario["lang"],
                "source": "synthetic_dei_error",
                "context_description": ctx,
            })
            if sanity_mode:
                print(f"\n[ERROR] {e}\n")

    # Write CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "language", "source", "context_description"])
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    success = [r for r in rows if "[ERROR" not in r["text"]]
    high_risk = [r for r in success if r["label"] == 1]
    low_risk = [r for r in success if r["label"] == 0]
    fr_count = [r for r in success if r["language"] == "fr"]
    en_count = [r for r in success if r["language"] == "en"]

    print(f"\n✅ Generation complete!")
    print(f"   Total samples: {len(success)}/{len(scenarios)} successful")
    print(f"   High risk (label=1): {len(high_risk)}")
    print(f"   Low risk  (label=0): {len(low_risk)}")
    print(f"   French: {len(fr_count)} | English: {len(en_count)}")
    print(f"   Saved to: {out_path}")
    if sanity_mode:
        print("\nSanity run done. Review transcripts above, then run without --sanity for a full batch.")


if __name__ == "__main__":
    main()
