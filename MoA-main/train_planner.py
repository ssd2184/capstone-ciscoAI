from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import argparse
from data.formetters import get_planner_formatter
import os
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import datasets
from data.dataset import load_dataset
from peft import LoraConfig


def prepare_prompts(dataset):
    reshaped_dataset = {
        "question": [],
        "id": [],
        "profile": [],
        "rubric_aspects": []
    }
    counter = 0
    for data in dataset:
        reshaped_dataset["question"].append(data["question"])
        reshaped_dataset["id"].append(data["id"])
        reshaped_dataset["profile"].append(data["profile"])
        reshaped_dataset["rubric_aspects"].append(data["rubric_aspects"])
    return reshaped_dataset

parser = argparse.ArgumentParser()

parser.add_argument("--inputs_addr", required=True)
parser.add_argument("--cache_dir", default="./cache")
parser.add_argument("--model_addr", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--num_context", type=int, default=10)
parser.add_argument("--per_device_train_batch_size", type=int, default=64)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.00005)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--max_steps", type=int, default=5000)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--warmup_steps", type=int, default=250)
parser.add_argument("--max_seq_length", type=int, default=4096)

if __name__ == "__main__":
    args = parser.parse_args()
    dataset_orig = load_dataset(args.inputs_addr, cache_dir = args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_addr, cache_dir = args.cache_dir, use_flash_attention_2=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_addr, cache_dir = args.cache_dir)
    formatter = get_planner_formatter(tokenizer, args.num_context, train=True)
    dataset_reshaped = prepare_prompts(dataset_orig)
    dataset = formatter(dataset_reshaped)
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        do_train=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        max_seq_length=args.max_seq_length,
        save_steps=args.save_steps,
        save_only_model=True
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    trianer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config
    )
    trianer.train()