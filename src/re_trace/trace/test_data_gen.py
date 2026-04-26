import json
from datasets import load_dataset

ds = load_dataset("allenai/real-toxicity-prompts", split="train[:5000]")

out_path = "data/RTP_hf_5k_prompts.jsonl"

with open(out_path, "w") as f:
    for i, row in enumerate(ds):
        prompt_text = row["prompt"]["text"]
        f.write(json.dumps({
            "index": i,
            "prompt": {
                "text": prompt_text
            }
        }) + "\n")

print(f"Saved {len(ds)} prompts to {out_path}")