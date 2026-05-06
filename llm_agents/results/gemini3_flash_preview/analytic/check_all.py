import os
import json

results_dir = "llm_agents/results/gemini3_flash_preview/analytic/models"
for d in sorted(os.listdir(results_dir)):
    p = os.path.join(results_dir, d, "scorecard.json")
    if os.path.exists(p):
        with open(p, "r") as f:
            data = json.load(f)
            print(f"{d:30} | {data['loss']}")
