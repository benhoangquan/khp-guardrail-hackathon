import json
import os
import pandas as pd

DRIFT_FILE = "datasets/drift_results.json"
OUTPUT_FILE = "datasets/drift_results_labeled.json"

def triage_drift():
    if not os.path.exists(DRIFT_FILE):
        print(f"Error: {DRIFT_FILE} not found. Run scripts/run_drift_tests.py first.")
        return

    with open(DRIFT_FILE, 'r') as f:
        data = json.load(f)

    for i, session in enumerate(data):
        if 'human_label' in session:
            continue

        os.system('clear')
        print(f"=== SESSION {i+1}/{len(data)}: {session['persona_name']} ===")
        print(f"AUTO-JUDGE VERDICT: Safety: {session['eval'].get('safety_pass')}, Boundary: {session['eval'].get('boundary_pass')}")
        print(f"SUMMARY: {session['eval'].get('summary')}")
        print("-" * 50)
        
        # Display transcript
        for t in session['transcript']:
            print(f"Turn {t['turn']} | USER: {t['user']}")
            print(f"       | BOT:  {t['bot']}")
            print("-" * 20)

        print("\n" + "="*50)
        val = input("Agree with Auto-Judge? (y/n) or mark as (f)ail, (p)ass, (q)uit: ").lower()

        if val == 'q':
            break
        elif val == 'y':
            session['human_label'] = "AGREE"
        elif val == 'n':
            session['human_label'] = "DISAGREE"
        elif val == 'f':
            session['human_label'] = "FAIL"
        elif val == 'p':
            session['human_label'] = "PASS"

        # Save progress
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    print(f"\n✅ Triage complete. Results in {OUTPUT_FILE}")

if __name__ == "__main__":
    triage_drift()
