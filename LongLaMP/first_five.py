import json

src = "inputs_pr_temporal_val.json"
dst = "inputs_pr_temporal_val_first5.json"

with open(src) as f:
    data = json.load(f)   # full list of dicts

# keep only first 5
subset = data[:5]

with open(dst, "w") as f:
    json.dump(subset, f, indent=2)

print(f"Wrote {len(subset)} examples to {dst}")