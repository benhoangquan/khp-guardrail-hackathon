import pandas as pd
import json
import os

INPUT_FILE = "datasets/comprehensive_stress_test_v2_evaluated.csv"
DRIFT_EVAL = "datasets/drift_test_evaluated_v2.json"
REPORT_FILE = "datasets/red_team_delta_report.md"

def generate_report():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found. Run auto_judge_v2 first.")
        return

    df = pd.read_csv(INPUT_FILE)
    
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

    # 2. Performance by Category
    report.append("## 📈 Vulnerability Heatmap (by Taxonomy)")
    cat_stats = df.groupby('category')['severity'].value_counts(normalize=True).unstack().fillna(0)
    report.append(cat_stats.to_markdown())
    report.append("\n")

    # 3. Performance by DEI Context
    report.append("## 🌍 DEI Robustness (Pass Rate by Persona)")
    dei_stats = df.groupby('dei_context')['severity'].apply(lambda x: (x == 'PASS').mean()).sort_values()
    report.append(dei_stats.to_markdown())
    report.append("\n")

    # 4. Critical & High Failures Table
    report.append("## 🚨 Failure Delta Table (Critical & High)")
    failures = df[df['severity'].isin(['CRITICAL', 'HIGH'])]
    
    if len(failures) > 0:
        failure_table = failures[['category', 'input_column', 'bot_response', 'severity', 'auto_judge_reason']]
        report.append(failure_table.to_markdown(index=False))
    else:
        report.append("*No Critical or High failures detected.*")
    report.append("\n")

    # 5. Multi-Turn Drift Analysis
    report.append("## 🌀 Multi-Turn Drift Audit")
    if os.path.exists(DRIFT_EVAL):
        with open(DRIFT_EVAL, 'r') as f:
            drift_data = json.load(f)
        for session in drift_data:
            eval_res = session.get('eval', {})
            report.append(f"### Persona: {session['persona_name']}")
            report.append(f"- **Verdict:** {eval_res.get('severity', 'UNKNOWN')}")
            report.append(f"- **Reason:** {eval_res.get('reason', 'n/a')}\n")
    
    # Write to file
    with open(REPORT_FILE, 'w') as f:
        f.writelines([line + "\n" for line in report])
    
    print(f"✅ Final Delta Report generated: {REPORT_FILE}")

if __name__ == "__main__":
    generate_report()
