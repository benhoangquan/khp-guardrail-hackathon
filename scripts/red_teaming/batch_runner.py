import os
import sys
import time
import pandas as pd
import requests

INPUT_FILE = "datasets/comprehensive_stress_test.csv"
OUTPUT_FILE = "datasets/comprehensive_stress_test_output.csv"
BASE_URL = "https://example.invalid"

USERNAME = os.getenv("KHP_CHAT_USERNAME", "team_055")
PASSWORD = os.getenv("KHP_CHAT_PASSWORD", "REDACTED")
REQUEST_TIMEOUT = 60

def login():
    url = f"{BASE_URL}/auth/login"
    r = requests.post(url, headers={"Content-Type": "application/json"}, json={"username": USERNAME, "password": PASSWORD}, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200: raise Exception(f"Login failed: {r.text}")
    return r.json()["session_token"]

def chat(token, message):
    url = f"{BASE_URL}/chat"
    r = requests.post(url, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, json={"message": message, "thread_id": None}, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200: raise Exception(f"Chat failed: {r.text}")
    data = r.json()
    return data["response"], data["thread_id"]

def logout(token):
    url = f"{BASE_URL}/auth/logout"
    requests.post(url, headers={"Authorization": f"Bearer {token}"}, timeout=REQUEST_TIMEOUT)

def run_one_test(prompt):
    start = time.time()
    token = None
    try:
        token = login()
        response, thread_id = chat(token, prompt)
        logout(token)
        latency = round(time.time() - start, 2)
        return response, thread_id, latency, "success"
    except Exception as e:
        latency = round(time.time() - start, 2)
        return "", "", latency, str(e)

def run_batch():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run generate_test_matrix.py first.")
        return

    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    responses, thread_ids, latencies, statuses = [], [], [], []
    total = len(df)

    print(f"🚀 Starting Batch Execution of {total} rows...")
    for i, prompt in enumerate(df['input_column'], start=1):
        print(f"Processing {i}/{total}...", end="\r")
        
        if pd.isna(prompt):
            responses.append(""); thread_ids.append(""); latencies.append(None); statuses.append("empty prompt")
            continue

        resp, tid, lat, status = run_one_test(str(prompt))
        responses.append(resp)
        thread_ids.append(tid)
        latencies.append(lat)
        statuses.append(status)
        time.sleep(1) # throttle to prevent hammering the API

    df['output_column'] = responses
    df['thread_id'] = thread_ids
    df['latency_seconds'] = latencies
    df['status'] = statuses

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Batch execution finished! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_batch()
