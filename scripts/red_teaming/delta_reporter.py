import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORT_FIGURE = "datasets/red_team_delta_report.png"

# Worst outcomes last in stack (top of bar reads as “failure headroom”); PASS anchors the base.
SEVERITY_ORDER = ["PASS", "MEDIUM", "HIGH", "CRITICAL", "FAIL", "ERROR"]
SEVERITY_COLORS = {
    "PASS": "#27ae60",
    "MEDIUM": "#f4d03f",
    "HIGH": "#e67e22",
    "CRITICAL": "#c0392b",
    "FAIL": "#7b241c",
    "ERROR": "#5d6d7e",
}

INPUT_FILES = [
    ("datasets/output/v2.1/comprehensive_stress_test_v2.1_evaluated.csv", "EN"),
    ("datasets/output/v3/comprehensive_stress_test_v3_evaluated.csv", "FR"),
]
DRIFT_EVALS = [
    ("datasets/output/v2.1/drift_test_evaluated_v2.1.json", "EN"),
    ("datasets/output/v3/drift_test_evaluated_v3.json", "FR"),
]
REPORT_FILE = "datasets/red_team_delta_report.md"


def write_delta_charts(df: pd.DataFrame, png_path: str) -> None:
    """100% stacked bars by taxonomy category + pass rate by language."""
    ct = pd.crosstab(df["category"], df["severity"])
    for s in SEVERITY_ORDER:
        if s not in ct.columns:
            ct[s] = 0
    ct = ct[[c for c in SEVERITY_ORDER if c in ct.columns]]
    row_tot = ct.sum(axis=1).replace(0, np.nan)
    pct = ct.div(row_tot, axis=0).fillna(0) * 100.0
    pct = pct.sort_values("PASS", ascending=True)
    n_cat = len(pct)
    fig_h = max(7.0, 0.22 * n_cat + 2.5)

    fig, (ax_cat, ax_lang) = plt.subplots(
        1,
        2,
        figsize=(13.5, fig_h),
        gridspec_kw={"width_ratios": [3.2, 1]},
    )
    y = np.arange(n_cat)
    left = np.zeros(n_cat)
    for col in SEVERITY_ORDER:
        if col not in pct.columns:
            continue
        vals = pct[col].to_numpy()
        ax_cat.barh(
            y,
            vals,
            left=left,
            label=col,
            color=SEVERITY_COLORS.get(col, "#95a5a6"),
            height=0.85,
        )
        left += vals
    ax_cat.set_yticks(y)
    ax_cat.set_yticklabels(pct.index, fontsize=7)
    ax_cat.invert_yaxis()
    ax_cat.set_xlim(0, 100)
    ax_cat.set_xlabel("Share of tests within category (%)")
    ax_cat.set_title("Severity mix by taxonomy category\n(sorted by pass rate, lowest at top)")
    ax_cat.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=len(SEVERITY_ORDER), fontsize=7)

    lang_pass = df.groupby("language", sort=True)["severity"].apply(lambda s: (s == "PASS").mean() * 100)
    colors = plt.cm.Set2(np.linspace(0, 1, len(lang_pass)))
    ax_lang.barh(lang_pass.index.astype(str), lang_pass.values, color=colors)
    ax_lang.set_xlim(0, 100)
    ax_lang.set_xlabel("Pass rate (%)")
    ax_lang.set_title("Pass rate by language")
    ax_lang.bar_label(ax_lang.containers[0], fmt="%.1f%%", fontsize=8, padding=2)

    fig.suptitle("Red-team delta summary", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_report():
    dfs = []
    for path, lang in INPUT_FILES:
        if not os.path.exists(path):
            print(f"⚠️  Skipping {path} (not found)")
            continue
        chunk = pd.read_csv(path)
        if "language" not in chunk.columns:
            chunk["language"] = lang
        dfs.append(chunk)

    if not dfs:
        print("❌ Error: no evaluated CSV files found. Run auto_judge_v2 first.")
        return

    df = pd.concat(dfs, ignore_index=True)
    
    report = []
    report.append("# 🚩 Red-Team Safety Audit: Delta Report\n")
    
    # 1. Executive Summary
    total = len(df)
    pass_count = len(df[df['severity'] == 'PASS'])
    report.append("## 📊 Executive Summary")
    report.append(f"- **Total Tests Run:** {total}")
    report.append(f"- **Overall Pass Rate:** {round((pass_count/total)*100, 1)}%")
    report.append(f"- **Critical Misses:** {len(df[df['severity'] == 'CRITICAL'])}")
    report.append(f"- **High Severity Failures:** {len(df[df['severity'] == 'HIGH'])}\n")

    write_delta_charts(df, REPORT_FIGURE)
    report.append("### Figure: severity mix and bilingual pass rate")
    report.append(f"![]({os.path.basename(REPORT_FIGURE)})\n")

    # 1b. Bilingual Coverage
    report.append("## 🌐 Bilingual Coverage (Pass Rate by Language)")
    lang_stats = df.groupby('language')['severity'].apply(lambda x: (x == 'PASS').mean())
    lang_counts = df.groupby('language').size()
    for lang in lang_stats.index:
        rate = round(lang_stats[lang] * 100, 1)
        count = lang_counts[lang]
        report.append(f"- **{lang}**: {rate}% pass rate ({count} tests)")
    report.append("")

    # 2. Performance by Category (combined + per-language)
    report.append("## 📈 Vulnerability Heatmap (by Taxonomy)")
    report.append("### Combined")
    cat_stats = df.groupby('category')['severity'].value_counts(normalize=True).unstack().fillna(0)
    report.append(cat_stats.to_markdown())
    report.append("")
    for lang in sorted(df['language'].unique()):
        lang_df = df[df['language'] == lang]
        report.append(f"### {lang} Only")
        cat_lang = lang_df.groupby('category')['severity'].value_counts(normalize=True).unstack().fillna(0)
        report.append(cat_lang.to_markdown())
        report.append("")
    report.append("")

    # 3. Performance by DEI Context (combined + per-language)
    report.append("## 🌍 DEI Robustness (Pass Rate by Persona)")
    report.append("### Combined")
    dei_stats = df.groupby('dei_context')['severity'].apply(lambda x: (x == 'PASS').mean()).sort_values()
    report.append(dei_stats.to_markdown())
    report.append("")
    for lang in sorted(df['language'].unique()):
        lang_df = df[df['language'] == lang]
        report.append(f"### {lang} Only")
        dei_lang = lang_df.groupby('dei_context')['severity'].apply(lambda x: (x == 'PASS').mean()).sort_values()
        report.append(dei_lang.to_markdown())
        report.append("")
    report.append("")

    # 4. Critical & High Failures Table (split by language)
    report.append("## 🚨 Failure Delta Table (Critical & High)")
    failures = df[df['severity'].isin(['CRITICAL', 'HIGH'])]

    if len(failures) > 0:
        for lang in sorted(failures['language'].unique()):
            lang_failures = failures[failures['language'] == lang]
            report.append(f"### {lang} Failures ({len(lang_failures)} cases)")
            failure_table = lang_failures[['category', 'input_column', 'bot_response', 'severity', 'auto_judge_reason']]
            report.append(failure_table.to_markdown(index=False))
            report.append("")
    else:
        report.append("*No Critical or High failures detected.*")
    report.append("")

    # 5. Multi-Turn Drift Analysis
    report.append("## 🌀 Multi-Turn Drift Audit")
    for drift_path, drift_lang in DRIFT_EVALS:
        if not os.path.exists(drift_path):
            continue
        with open(drift_path, 'r') as f:
            drift_data = json.load(f)
        for session in drift_data:
            eval_res = session.get('eval', {})
            lang_label = session.get('language', drift_lang)
            report.append(f"### [{lang_label}] Persona: {session['persona_name']}")
            report.append(f"- **Verdict:** {eval_res.get('severity', 'UNKNOWN')}")
            report.append(f"- **Reason:** {eval_res.get('reason', 'n/a')}\n")
    
    # Write to file
    with open(REPORT_FILE, 'w') as f:
        f.writelines([line + "\n" for line in report])
    
    print(f"✅ Final Delta Report generated: {REPORT_FILE}")

if __name__ == "__main__":
    generate_report()
