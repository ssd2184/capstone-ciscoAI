
import os
import sys
import json
import argparse
import asyncio
from datetime import datetime
from typing import List, Dict, Any

import warnings

import together
from together import AsyncTogether, Together

try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

# tgp_v1_YlS5Z1ToN0cQqWGBwn2nwh4OQRNcFIXP-FxBDrQRwCs

DEFAULT_REFERENCE_MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    # "deepseek-ai/DeepSeek-V3"
]

DEFAULT_AGGREGATOR_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

AGGREGATOR_SYSTEM_PROMPT = (
    "You have been provided with a set of responses from various models to the latest user query. "
    "Your task is to synthesize these responses into a single, high-quality response. Critically evaluate the information, "
    "recognizing that some of it may be biased or incorrect. Do not simply replicate the given answers; offer a refined, "
    "accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres "
    "to the highest standards of accuracy and reliability.\n\nResponses from models:"
)

def load_dataset_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
        assert isinstance(data, list), "Dataset must be a JSON list of records."
    missing = [i for i, d in enumerate(data) if "id" not in d or "question" not in d]
    if missing:
        raise ValueError(f"Dataset items missing required fields 'id' or 'question' at indices: {missing[:10]} ...")
    return data

class MoAGenerator:
    def __init__(self, api_key: str, reference_models: List[str], aggregator_model: str, temperature: float, max_tokens: int):
        self.client = Together(api_key=api_key)
        self.async_client = AsyncTogether(api_key=api_key)
        self.reference_models = reference_models
        self.aggregator_model = aggregator_model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def _call_one(self, model: str, prompt: str) -> str:
        try:
            resp = await self.async_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content or ""
        except together.error.InvalidRequestError as e:
            print(f"[skip] {model}: {e}")
            return ""
        except Exception as e:
            print(f"[error] {model}: {e}")
            return ""

    async def propose(self, prompt: str, concurrency: int) -> List[str]:
        sem = asyncio.Semaphore(concurrency)
        async def wrapped(m):
            async with sem:
                return await self._call_one(m, prompt)
        outs = await asyncio.gather(*[wrapped(m) for m in self.reference_models])
        return [o for o in outs if o and o.strip()]

    def aggregate(self, prompt: str, proposer_outputs: List[str]) -> str:
        if not proposer_outputs:
            return ""
        block = "\n".join(f"{i+1}. {t}" for i, t in enumerate(proposer_outputs))
        stream = self.client.chat.completions.create(
            model=self.aggregator_model,
            messages=[
                {"role": "system", "content": AGGREGATOR_SYSTEM_PROMPT + "\n" + block},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        buf = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            sys.stdout.write(delta)
            sys.stdout.flush()
            buf.append(delta)
        sys.stdout.write("\n")
        return "".join(buf)

    async def close(self):
        try:
            await self.async_client.close()
        except Exception:
            pass

async def run(inputs_addr: str, output_addr: str, reference_models: List[str], aggregator_model: str, temperature: float, max_tokens: int, max_rows: int, concurrency: int):
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("Please set TOGETHER_API_KEY in your environment.")
    dataset = load_dataset_json(inputs_addr)
    if max_rows is not None:
        dataset = dataset[:max_rows]

    gen = MoAGenerator(api_key, reference_models, aggregator_model, temperature, max_tokens)
    outputs_dict: Dict[str, List[Dict[str, Any]]] = {}
    try:
        for item in dataset:
            _id = str(item["id"])
            prompt = str(item["question"])
            print(f"\n=== ID { _id } ===")
            proposers = await gen.propose(prompt, concurrency=concurrency)
            final = gen.aggregate(prompt, proposer_outputs=proposers)
            outputs_dict.setdefault(_id, []).append({
                "prompt": prompt,
                "whole_output": {"personalized_answer": final},
                "output": final,
                "log_prob": 0.0,
            })
    finally:
        await gen.close()

    with open(output_addr, "w") as f:
        json.dump(outputs_dict, f, indent=4, ensure_ascii=False)
    print(f"\nSaved outputs to: {output_addr}")

def main():
    ap = argparse.ArgumentParser(description="Generate LongLaMP-style outputs.json using Mixture-of-Agents.")
    ap.add_argument("--inputs_addr", required=True, help="Path to dataset.json (list of {id, question, narrative, rubric_aspects, ...})")
    ap.add_argument("--output_addr", required=True, help="Where to write outputs.json (dict: id -> [ {prompt, whole_output, output, log_prob} ])")
    ap.add_argument("--reference_models", nargs="*", default=DEFAULT_REFERENCE_MODELS)
    ap.add_argument("--aggregator_model", default=DEFAULT_AGGREGATOR_MODEL)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    asyncio.run(run(
        inputs_addr=args.inputs_addr,
        output_addr=args.output_addr,
        reference_models=args.reference_models,
        aggregator_model=args.aggregator_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_rows=args.max_rows,
        concurrency=args.concurrency,
    ))

if __name__ == "__main__":
    main()
