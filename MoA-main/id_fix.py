import json

def fix_ids(input_file: str, output_file: str):
    with open(input_file, "r") as f:
        data = json.load(f)

    fixed = {}
    for k, v in data.items():
        new_list = []
        for item in v:
            item["id"] = str(k)
            new_list.append(item)
        fixed[str(k)] = new_list

    with open(output_file, "w") as f:
        json.dump(fixed, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    input_file = "pr_temporal_val_output_1.json"
    output_file = "final_pr_temporal_val_output_1.json"
    fix_ids(input_file, output_file)
