import os
import sys
import time
import json
import pandas as pd
import requests
from typing import List, Tuple, Dict, Any

# Add project root to path
sys.path.append(os.getcwd())

from project.providers.cohere_provider import CohereProvider
from project.providers.base import LLMMessage

INPUT_FILE = "datasets/output/v3/comprehensive_stress_test_v3.csv"
CONFIG_FILE = "datasets/output/v3/test_matrix_config_v3.json"
OUTPUT_FILE = "datasets/output/v3/comprehensive_stress_test_v3_output.csv"
DRIFT_LOG = "datasets/drift_test_logs_v2.json"
BASE_URL = "https://example.invalid"

# Credentials
USERNAME = os.getenv("KHP_CHAT_USERNAME", "team_055")
PASSWORD = os.getenv("KHP_CHAT_PASSWORD", "REDACTED")

def login():
    r = requests.post(f"{BASE_URL}/auth/login", json={"username": USERNAME, "password": PASSWORD}, timeout=15)
    if r.status_code != 200: raise Exception(f"Login failed: {r.text}")
    return r.json()["session_token"]

def chat_khp(token, message, thread_id=None):
    r = requests.post(f"{BASE_URL}/chat", headers={"Authorization": f"Bearer {token}"}, json={"message": message, "thread_id": thread_id}, timeout=60)
    if r.status_code != 200: raise Exception(f"Chat failed: {r.text}")
    return r.json()["response"], r.json()["thread_id"]

def run_single_turn_batch(df, token):
    results = []
    total = len(df)
    print(f"🚀 Running Single-Turn Batch ({total} cases)...")
    
    for i, row in df.iterrows():
        print(f"Processing {i+1}/{total}...", end="\r")
        try:
            resp, tid = chat_khp(token, row['input_column'])
            results.append({
                **row.to_dict(),
                "bot_response": resp,
                "thread_id": tid,
                "turn_number": 1,
                "status": "success"
            })
        except Exception as e:
            results.append({**row.to_dict(), "bot_response": "", "thread_id": "", "turn_number": 1, "status": str(e)})
        time.sleep(5)
    
    return pd.DataFrame(results)

def run_multi_turn_drift(config, token):
    print(f"\n🚀 Running Deep Drift Tests ({len(config['multi_turn_drift'])} personas)...")
    drift_results = []
    
    for drift in config["multi_turn_drift"]:
        print(f"Persona: {drift['name']} ({drift['turns']} turns)")
        p_bot = CohereProvider(system_prompt=drift['persona'])
        p_memory: List[LLMMessage] = []
        khp_thread_id = None
        transcript = []

        for turn in range(1, drift['turns'] + 1):
            print(f"  Turn {turn}/{drift['turns']}...", end="\r")
            
            # Persona generates message
            instr = "Respond naturally as the persona. Keep it under 2 sentences."
            p_memory.append(LLMMessage(role="user", content=instr))
            p_resp = p_bot.generate_sync(p_memory, temperature=0.8)
            user_msg = p_resp.content.replace('"', '').strip()
            
            p_memory.pop()
            p_memory.append(LLMMessage(role="assistant", content=user_msg))
            
            # KHP replies
            khp_reply, khp_thread_id = chat_khp(token, user_msg, khp_thread_id)
            p_memory.append(LLMMessage(role="user", content=khp_reply))
            
            transcript.append({
                "turn_number": turn,
                "user_prompt": user_msg,
                "bot_response": khp_reply,
                "thread_id": khp_thread_id,
                "matrix_coordinates": f"Drift|{drift['id']}|Turn_{turn}"
            })
            time.sleep(5)

        drift_results.append({
            "persona_id": drift['id'],
            "persona_name": drift['name'],
            "transcript": transcript
        })
    
    return drift_results

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found. Run generate_matrix_v2 first.")
        return
        
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
        
    khp_token = login()
    
    # Run single-turn
    df = pd.read_csv(INPUT_FILE)
    results_df = run_single_turn_batch(df, khp_token)
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    # Run multi-turn
    drift_data = run_multi_turn_drift(config, khp_token)
    with open(DRIFT_LOG, 'w') as f:
        json.dump(drift_data, f, indent=2)
        
    print(f"\n✅ All tests complete!")
    print(f"📁 Single-turn results: {OUTPUT_FILE}")
    print(f"📁 Multi-turn logs: {DRIFT_LOG}")

if __name__ == "__main__":
    main()
