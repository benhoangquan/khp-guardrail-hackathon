import pandas as pd
import json
import os

INPUT_FILES = [
    ("datasets/output/v2.1/comprehensive_stress_test_v2.1_evaluated.csv", "EN"),
    ("datasets/output/v3/comprehensive_stress_test_v3_evaluated.csv", "FR"),
]
DRIFT_EVALS = [
    ("datasets/output/v2.1/drift_test_evaluated_v2.1.json", "EN"),
    ("datasets/output/v3/drift_test_evaluated_v3.json", "FR"),
]
REPORT_FILE = "datasets/red_team_delta_report.md"

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
