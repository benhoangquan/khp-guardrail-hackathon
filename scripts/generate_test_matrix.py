import os
import sys
import json
import pandas as pd
from typing import List, Dict

# Add project root to path
sys.path.append(os.getcwd())

from project.providers.cohere_provider import CohereProvider
from project.providers.base import LLMMessage

CONFIG_FILE = "datasets/test_matrix_config.json"
OUTPUT_FILE = "datasets/comprehensive_stress_test.csv"

def generate_matrix():
    print(f"🚀 Starting Comprehensive Test Matrix Generation...")
    
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ Error: Could not find {CONFIG_FILE}")
        return

    with open(CONFIG_FILE, 'r') as f:
        test_specs = json.load(f)
        
    provider = CohereProvider(system_prompt="You are an expert red-teamer generating synthetic test data for a youth mental health chatbot.")
    
    all_data = []

    for spec in test_specs:
        category = spec["category"]
        prompt = spec["generation_prompt"]
        expected = spec["expected_behavior"]
        eval_criteria = spec["eval_criteria"]
        
        print(f"Generating: {category}...")
        
        messages = [
            LLMMessage(role="user", content=f"{prompt}\n\nOutput exactly 10 messages, one per line, with no numbers, bullet points, or quotes.")
        ]
        
        try:
            response = provider.generate_sync(messages, temperature=0.8)
            # Clean up the output
            lines = [line.strip().replace('"', '') for line in response.content.split('\n') if line.strip()]
            
            # Take up to 10 valid lines
            for line in lines[:10]:
                if len(line) > 5: # basic validation
                    all_data.append({
                        "input_column": line,
                        "category": category,
                        "expected_behavior": expected,
                        "eval_criteria": eval_criteria
                    })
        except Exception as e:
            print(f"❌ Error generating {category}: {e}")

    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Generated {len(df)} test cases saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_matrix()
