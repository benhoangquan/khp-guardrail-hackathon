# Reproduction Guide: Automated Red-Team Stress Testing

This guide explains how to run the automated safety audit pipeline to identify vulnerabilities in the KHP Virtual Assistant.

## Prerequisites
- Ensure your `.env` file contains `BUZZ_COHERE_AUTH_TOKEN` and `BUZZ_COHERE_API`.
- Set your chatbot credentials in your environment:
  ```bash
  export KHP_CHAT_USERNAME="your_username"
  export KHP_CHAT_PASSWORD="your_password"
  ```

## 1. Step-by-Step Pipeline Execution

### Phase 1: Data Generation
Generate a stratified test matrix of 100+ cases covering 23 taxonomies, DEI variables, and linguistic variants (slang, euphemisms, typos).
```bash
python scripts/generate_test_matrix_v2.py
```
**Output:** `datasets/comprehensive_stress_test_v2.csv`

### Phase 2: Execution Engine
Run the single-turn and multi-turn (15+ turns) drift tests against the KHP Bot API.
```bash
python scripts/batch_runner_v2.py
```
**Outputs:** 
- `datasets/comprehensive_stress_test_v2_output.csv` (Single-turn)
- `datasets/drift_test_logs_v2.json` (Multi-turn)

### Phase 3: Automated Judging
Evaluate all responses using a 4-tier severity model (`CRITICAL`, `HIGH`, `MEDIUM`, `PASS`).
```bash
python scripts/auto_judge_v2.py
```
**Outputs:**
- `datasets/comprehensive_stress_test_v2_evaluated.csv`
- `datasets/drift_test_evaluated_v2.json`

### Phase 4: Analytics & Reporting
Generate a summary report including a vulnerability heatmap and a failure delta table.
```bash
python scripts/delta_reporter.py
```
**Output:** `datasets/red_team_delta_report.md`

## 2. Manual Verification (Triage)
To review the failures and verify the Auto-Judge's verdicts:
```bash
python triage.py
```

## 3. Customizing Tests
To add new safety scenarios or change the evaluation strictness, modify:
- `datasets/test_matrix_config_v2.json`
