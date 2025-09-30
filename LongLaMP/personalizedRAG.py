import json
from datasets import load_dataset

def to_profile_list(raw_profile):
    if raw_profile is None:
        return []
    if isinstance(raw_profile, str):
        return [{"text": raw_profile}]
    if isinstance(raw_profile, list):
        out = []
        for x in raw_profile:
            if isinstance(x, str):
                out.append({"text": x})
            elif isinstance(x, dict) and "text" in x:
                out.append({"text": x["text"]})
            else:
                out.append({"text": str(x)})
        return out
    return [{"text": str(raw_profile)}]

def convert_split(hf_split):
    items = []
    for i, row in enumerate(hf_split):
        items.append({
            "id": i,                              
            "reviewerId": row.get("reviewerId", ""),
            "question": row["input"],             
            "profile": to_profile_list(row.get("profile", "")),
            "gold_output": row["output"],         
        })
    return items

def main():
    ds = load_dataset("LongLaMP/LongLaMP", "product_review_temporal")
    out = {
        "train": convert_split(ds["train"]),
        "val": convert_split(ds["val"]),
        "test": convert_split(ds["test"]),
    }
    with open("inputs_pr_temporal_val.json", "w") as f:
        json.dump(out["val"], f, ensure_ascii=False, indent=2)
    with open("inputs_pr_temporal_test.json", "w") as f:
        json.dump(out["test"], f, ensure_ascii=False, indent=2)
    print("Wrote inputs_pr_temporal_val.json and inputs_pr_temporal_test.json")

if __name__ == "__main__":
    main()
