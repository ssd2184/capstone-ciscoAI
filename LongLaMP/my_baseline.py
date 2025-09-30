import argparse, json, os
from tqdm import tqdm

from data.dataset import load_dataset
from data.formetters import get_baseline_no_rag_formatter, get_baseline_rag_formatter
from utils.json_utils import str_to_json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json as _json

def detect_device_for_offload():
    """
    If MPS is available we still offload most to CPU (8B > 9GB VRAM).
    We'll let HF/accelerate do device_map='auto' with max_memory caps.
    """
    has_mps = torch.backends.mps.is_available()
    dtype = torch.float16 if has_mps else torch.float32
    return ("mps" if has_mps else "cpu"), dtype

def clip_to_ctx(text, tokenizer, max_len=8192, reserve=160):
    """
    Clip a long prompt to fit into context window, keeping the tail
    (task + most recent context often lives near the end).
    """
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    budget = max(256, max_len - reserve)
    if len(ids) > budget:
        ids = ids[-budget:]
    return tokenizer.decode(ids, skip_special_tokens=True)

def to_text(prompt):
    """
    Your formatters may return either plain strings or chat-message arrays.
    We collapse arrays into a single text block.
    """
    if isinstance(prompt, list):
        return "\n".join([m.get("content", "") for m in prompt])
    return prompt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_addr", required=True)
    ap.add_argument("--inputs_addr", required=True)
    ap.add_argument("--output_addr", required=True)

    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=128)

    ap.add_argument("--num_contexts", type=int, default=1)
    ap.add_argument("--rag", action="store_true")
    ap.add_argument("--cache_dir", default="./cache")

    ap.add_argument("--batch_size", type=int, default=2)       
    ap.add_argument("--max_model_len", type=int, default=8192) 
    ap.add_argument("--limit", type=int, default=0)         

    ap.add_argument("--mps_mem_gb", type=int, default=8, help="Cap MPS memory for HF device_map=auto")
    ap.add_argument("--cpu_mem_gb", type=int, default=24, help="Cap CPU RAM for HF device_map=auto")
    ap.add_argument("--offload_dir", default="./offload", help="Folder for CPU/disk offload")

    args = ap.parse_args()

    device, dtype = detect_device_for_offload()
    print(f"Using device={device}, dtype={dtype}")

    ds = load_dataset(args.inputs_addr, cache_dir=args.cache_dir)[:5]

    def _to_list_of_dicts(ds):
        if isinstance(ds, dict):
            keys = list(ds.keys())
            n = len(ds[keys[0]]) if keys else 0
            return [{k: ds[k][i] for k in keys} for i in range(n)]
        if isinstance(ds, list):
            if ds and isinstance(ds[0], str):
                return [_json.loads(s) for s in ds]
            return ds
        if isinstance(ds, str):
            obj = _json.loads(ds)
            return _to_list_of_dicts(obj)
        raise ValueError(f"Unsupported dataset type: {type(ds)}")

    ds = _to_list_of_dicts(ds)

    for ex in ds:
        if "question" not in ex and "input" in ex:
            ex["question"] = ex["input"]
        if "id" not in ex:
            if "reviewerId" in ex:
                ex["id"] = ex["reviewerId"]
            elif "qid" in ex:
                ex["id"] = ex["qid"]
            else:
                ex.setdefault("id", str(hash(_json.dumps(ex, sort_keys=True))))
        ex.setdefault("profile", ex.get("user_profile", ex.get("profile", "")))


    if args.limit and args.limit > 0:
        ds = ds[:args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_addr, cache_dir=args.cache_dir)
    tokenizer.padding_side = "left"  
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    os.makedirs(args.offload_dir, exist_ok=True)
    max_memory = {
        "cpu": f"{args.cpu_mem_gb}GiB"
    }
    if device == "mps":
        max_memory["mps"] = f"{args.mps_mem_gb}GiB"

    if device == "mps" and "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_addr,
        cache_dir=args.cache_dir,
        torch_dtype=dtype,
        device_map="auto",             
        max_memory=max_memory,        
        offload_folder=args.offload_dir,
        low_cpu_mem_usage=True,
    ).eval()

    formatter = (
        get_baseline_rag_formatter(tokenizer, args.num_contexts, proprietary_llm=False)
        if args.rag else
        get_baseline_no_rag_formatter(tokenizer, proprietary_llm=False)
    )

    reshaped = {"question": [], "id": [], "profile": []}
    for ex in ds:
        reshaped["question"].append(ex["question"])
        reshaped["id"].append(ex["id"])
        reshaped["profile"].append(ex["profile"])

    prompts = formatter(reshaped)
    ids = [str(i) for i in reshaped["id"]]

    outputs_dict = {}
    done_ids = set()
    if os.path.exists(args.output_addr):
        try:
            with open(args.output_addr, "r") as f:
                outputs_dict = json.load(f)
            done_ids = set(outputs_dict.keys())
            print(f"[resume] Found {len(done_ids)} completed items.")
        except Exception:
            pass

    reserve = (args.max_tokens or 128) + 32

    for start in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        sub_prompts = prompts[start:start + args.batch_size]
        sub_ids = ids[start:start + args.batch_size]

        sub = [(pmt, sid) for pmt, sid in zip(sub_prompts, sub_ids) if sid not in done_ids]
        if not sub:
            continue

        texts = []
        keep_ids = []
        for pmt, sid in sub:
            txt = to_text(pmt)
            txt = clip_to_ctx(txt, tokenizer, max_len=args.max_model_len, reserve=reserve)
            texts.append(txt)
            keep_ids.append(sid)

        batch = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        for k in batch:
            batch[k] = batch[k].to(next(model.parameters()).device)

        with torch.no_grad():
            gen = model.generate(
                **batch,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        inp_len = batch["input_ids"].shape[1]
        for i, sid in enumerate(keep_ids):
            decoded = tokenizer.decode(gen[i][inp_len:], skip_special_tokens=True)
            try:
                obj = str_to_json(decoded)
                ans = obj.get("personalized_answer", "")
            except Exception:
                obj, ans = {}, ""
            outputs_dict.setdefault(sid, []).append({
                "prompt": texts[i],
                "whole_output": obj,
                "output": ans,
                "log_prob": 0  
            })
            done_ids.add(sid)

        with open(args.output_addr, "w") as f:
            json.dump(outputs_dict, f, indent=2)

    with open(args.output_addr, "w") as f:
        json.dump(outputs_dict, f, indent=2)

if __name__ == "__main__":
    main()