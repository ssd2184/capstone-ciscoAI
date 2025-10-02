from vllm import LLM
import argparse
import json
from evaluation.evaluator import evaluator

parser = argparse.ArgumentParser()

parser.add_argument("--evaluator_llm", type=str, required=True)
parser.add_argument("--inputs_addr", type=str, required=True)
parser.add_argument("--response_addr", type=str, required=True)
parser.add_argument("--score_addr", type=str, required=True)
parser.add_argument("--cache_dir", default="./cache")
parser.add_argument("--tensor_parallel_size", type=int, default=1)
parser.add_argument("--max_length", type=int, default=32000)


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.inputs_addr) as f:
        dataset = json.load(f)
        dataset_ids = set([data['id'] for data in dataset])
    with open(args.response_addr) as f:
        outputs = json.load(f)
        outputs_ids = set(outputs.keys())
    
    assert dataset_ids == outputs_ids, "Dataset IDs and output IDs do not match."
    assert all(len(outputs[data['id']]) > 0 for data in dataset), "All outputs must have at least one response."
    assert all(data['id'] in outputs for data in dataset), "All dataset IDs must be present in outputs."
    assert all(data['id'] in dataset_ids for data in outputs.values()), "All output IDs must be present in dataset."
    assert all('output' in response for data in dataset for response in outputs[data['id']]), "All responses must have an 'output' field."

    llm = LLM(args.evaluator_llm, download_dir=args.cache_dir, max_model_len=args.max_length, tensor_parallel_size=args.tensor_parallel_size)
    queries = [data['question'] for data in dataset]
    ids = [data['id'] for data in dataset]
    details = [data['narrative'] for data in dataset]
    responses = [str(outputs[data['id']][0]['output']) if len(outputs[data['id']]) > 0 else "" for data in dataset]
    aspects = [data['rubric_aspects'] for data in dataset]
    scores = evaluator(queries, responses, details, aspects, llm)
    for id, score in zip(ids, scores['per_question_scores']):
       score['id'] = id
    with open(args.score_addr, 'w') as f:
        json.dump(scores, f, indent=4)