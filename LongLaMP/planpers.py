from vllm import LLM, SamplingParams
import argparse
from data.dataset import load_dataset
from data.formetters import get_planner_formatter, get_generation_with_plan_rag_formatter
import json
from utils.custom_llm import OpenAILLM
from utils.json_utils import str_to_json
import json5
from vllm.lora.request import LoRARequest
import os


def load_llm(model_addr, cache_dir):
    adapter_config_path = os.path.join(model_addr, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        return LLM(model_addr, download_dir=cache_dir, gpu_memory_utilization=0.45), None
    with open(adapter_config_path, "r") as f:
        config = json.load(f)
        base_model = config.get("base_model_name_or_path")
    llm = LLM(base_model, download_dir=cache_dir, enable_lora=True, gpu_memory_utilization=0.45)
    lora = LoRARequest("custom_lora", 1, lora_path=model_addr)
    return llm, lora

def parse_json(json_str):
    json_str = json_str.replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(json_str, strict=False)
        return obj
    except:
        pass
    try:
        obj = json5.loads(json_str)
        return obj
    except:
        print(json_str)
        raise ValueError("Invalid json object")

def prepare_prompts_planner(dataset, formater):
    reshaped_dataset = {
        "question": [],
        "id": [],
        "profile": []
    }
    for data in dataset:
        reshaped_dataset["question"].append(data["question"])
        reshaped_dataset["id"].append(data["id"])
        reshaped_dataset["profile"].append(data["profile"])
    return formater(reshaped_dataset)

def prepare_prompts_generation(dataset, plans,  formater):
    reshaped_dataset = {
        "question": [],
        "id": [],
        "profile": [],
        "plan": []
    }
    for data in dataset:
        reshaped_dataset["question"].append(data["question"])
        reshaped_dataset["id"].append(data["id"])
        reshaped_dataset["profile"].append(data["profile"])
        reshaped_dataset["plan"].append(plans[data["id"]][0]["plan"])
    return formater(reshaped_dataset)


def apply_num_generation(dataset, num_generation):
    new_dataset = []
    ids = []
    for data in dataset:
        for i in range(num_generation):
            new_dataset.append(data)
            ids.append(data["id"])
    return ids, new_dataset

def post_process_output_based_on_num_generation(output, num_generation):
    new_output = []
    temp = []
    for out in output:
        temp.append(out)
        if len(temp) == num_generation:
            new_output.append(temp)
            temp = []
    return new_output

parser = argparse.ArgumentParser()
parser.add_argument("--model_addr", type=str, required=True)
parser.add_argument("--planner_model", type=str, required=True)
parser.add_argument("--inputs_addr", type=str, required=True)
parser.add_argument("--output_addr", type=str, required=True)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--max_retries", type=int, default=100)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--max_tokens_plan", type=int, default=4096)
parser.add_argument("--num_generated_outputs", type=int, default=1)
parser.add_argument("--num_contexts", type=int, default=5)
parser.add_argument("--openai", action="store_true")
parser.add_argument("--api_key_addr", type=str, default="")
parser.add_argument("--cache_dir", default="./cache")


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_orig = load_dataset(args.inputs_addr, cache_dir = args.cache_dir)
    ids, dataset = apply_num_generation(dataset_orig, args.num_generated_outputs)
    if args.openai:
        with open(args.api_key_addr, "r") as file:
            api_key = file.read().strip()
        llm = OpenAILLM(model_name=args.model_addr, api_key=api_key)
        tokenizer = None
        is_proprietary_llm = True
    else:
        llm = LLM(model=args.model_addr, download_dir=args.cache_dir, gpu_memory_utilization=0.45)
        tokenizer = llm.get_tokenizer()
        is_proprietary_llm = False
    planner, lora_req = load_llm(args.planner_model, args.cache_dir)
    planner_tokenizer = planner.get_tokenizer()
    formater = get_planner_formatter(planner_tokenizer, args.num_contexts, False)
    prompts = prepare_prompts_planner(dataset, formater)
    plans_dict = {}
    temperature = args.temperature
    sampling_params = SamplingParams(temperature=temperature, top_p=args.top_p, max_tokens=args.max_tokens_plan, logprobs=1)
    outputs = planner.generate(prompts, sampling_params, lora_request=lora_req)
    for id, prompt, output in zip(ids, prompts,  outputs):
        if id not in plans_dict:
            plans_dict[id] = []
        text_output = output.outputs[0].text
        plans_dict[id].append(
            {
                "prompt": prompt,
                "plan": text_output,
                "log_prob": output.outputs[0].cumulative_logprob
            }
        )
    formater = get_generation_with_plan_rag_formatter(tokenizer, args.num_contexts, is_proprietary_llm)
    prompts = prepare_prompts_generation(dataset, plans_dict, formater)
    outputs_dict = {}
    temperature = args.temperature
    retries = 0
    while prompts:
        retries += 1
        wrongs = []
        sampling_params = SamplingParams(temperature=temperature, top_p=args.top_p, max_tokens=args.max_tokens, logprobs=1)
        outputs = llm.generate(prompts, sampling_params)
        for id, prompt, output in zip(ids, prompts,  outputs):
            if id not in outputs_dict:
                outputs_dict[id] = []
            try:
                text_output = output.outputs[0].text
                response_obj = str_to_json(text_output)
                outputs_dict[id].append(
                    {
                        "prompt": prompt,
                        "whole_output": response_obj,
                        "plan": plans_dict[id][0]["plan"],
                        "output": response_obj['personalized_answer'],
                        "log_prob": output.outputs[0].cumulative_logprob
                    }
                )
            except Exception as e:
                if retries > args.max_retries:
                    outputs_dict[id].append(
                        {
                            "prompt": prompt,
                            "whole_output": "",
                            "output": "",
                            "log_prob": 0
                        }
                    )
                    continue
                if temperature < 1:
                    temperature += 0.1
                print(e)
                wrongs.append((id, prompt))
        prompts = []
        ids = []
        for wrong in wrongs:
            ids.append(wrong[0])
            prompts.append(wrong[1])
    with open(args.output_addr, "w") as file:
        json.dump(outputs_dict, file, indent=4)