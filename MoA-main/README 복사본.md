# Codes and data for the paper: [LaMP-QA: A Benchmark for Personalized Long-form Question Answering](https://arxiv.org/abs/2506.00137)

Personalization is essential for question answering systems that are user-centric. Despite its importance, personalization in answer generation has been relatively underexplored. This is mainly due to lack of resources for training and evaluating personalized question answering systems. We address this gap by introducing LaMP-QA---a benchmark designed for evaluating personalized long-form answer generation. The benchmark covers questions from three major categories: (1) Arts & Entertainment, (2) Lifestyle & Personal Development, and (3) Society & Culture, encompassing over 45 subcategories in total. To assess the quality and potential impact of the LaMP-QA benchmark for personalized question answering, we conduct comprehensive human and automatic evaluations, to compare multiple evaluation strategies for evaluating generated personalized responses and measure their alignment with human preferences. Furthermore, we benchmark a number of non-personalized and personalized approaches based on open-source and proprietary large language models (LLMs). Our results show that incorporating the personalized context provided leads to performance improvements of up to 39\%. The benchmark is publicly released to support future research in this area.

# Installing requirements

This repository uses FlashAttention 2. Please follow the [official installation guidelines](https://github.com/Dao-AILab/flash-attention) to set it up. In order to download and use this repository, you should install the following requirements:

```
pip install -r requirements.txt
```

# Downloading the dataset

The LaMP-QA dataset is available on Hugging Face Datasets: [LaMP-QA](https://huggingface.co/datasets/alireza7/LaMP-QA). To download and prepare the datasets for this repository, run the following script:

```shell
python utils/download_datasets.py \
    --dataset_save_directory /*address to the download directory*/ \
    --cache_dir /*address to the cache directory ([optional], default is ./cache)*/
```

# Evaluating the generated responses

To evaluate the generated responses to the questions in each dataset, use the following script:

```shell
python evaluate_responses.py \
    --evaluator_llm "Qwen/Qwen2.5-32B-Instruct" \
    --inputs_addr /*address to the dataset file to be evaluated (validation.json or test.json)*/ \
    --response_addr /*address to the generated responses to the questions for the dataset*/ \
    --score_addr /*address to where the output score file should be saved*/ \
```

Note: The response file must follow the specified format for the script to function correctly:

```
{
    "/*question 1 id*/" : [
        {
            "output": "/*response to the question 1*/"
        }
    ],
    "/*question 2 id*/" : [
        {
            "output": "/*response to the question 2*/"
        }
    ],
    ...,
    "/*question n id*/" : [
        {
            "output": "/*response to the question n*/"
        }
    ]
}
```

# Baselines

## No personalization

To run a non-personalized open-source LLM, use the following script:

```shell
python baselines.py \
    --model_addr /*address or name of the open model to be evaluated*/ \
    --inputs_addr /*address to the dataset file to be evaluated (validation.json or test.json)*/ \
    --output_addr /*address to where the outputs should be saved*/ \
    --temperature 0.1 \
```

For the OpenAI models, you can use the following script:

```shell
python baselines.py \
    --model_addr /*name of the OpenAI model to be evaluated*/ \
    --inputs_addr /*address to the dataset file to be evaluated (validation.json or test.json)*/ \
    --output_addr /*address to where the outputs should be saved*/ \
    --temperature 0.1 \
    --openai \
    --api_key_addr /*address to a file that contains the API key for OpenAI*/
```

## RAG-Personalization

To use RAG for personalization, we first need to sort user profiles based on their similarity to the given question. This sorting is performed once per dataset, and the results can be reused afterward. To do this, we use Contriever to rank the user profile using the following script:

```shell
python retrieval/rank_dataset.py \
    --input_dataset_addr /*address to the dataset file*/ \
    --output_dataset_addr /*address to where the dataset with sorted profile for each user should be saved*/ \
    --model_name "facebook/contriever-msmarco" \
    --batch_size 16
```

Then, we can use the following script to run RAG-Personalization for open-source LLMs:

```shell
python baselines.py \
    --model_addr /*address or name of the open model to be evaluated*/ \
    --inputs_addr /*address to the dataset file with ranked profile to be evaluated (validation_ranked.json or test_ranked.json)*/ \
    --output_addr /*address to where the outputs should be saved*/ \
    --temperature 0.1 \
    --rag \
    --num_contexts /*number of user personal contexts to be used for personalization, we use 10*/
```

For the OpenAI models, you can use the following script:

```shell
python baselines.py \
    --model_addr /*name of the OpenAI model to be evaluated*/ \
    --inputs_addr /*address to the dataset file with ranked profile to be evaluated (validation_ranked.json or test_ranked.json)*/ \
    --output_addr /*address to where the outputs should be saved*/ \
    --temperature 0.1 \
    --openai \
    --api_key_addr /*address to a file that contains the API key for OpenAI*/
    --rag \
    --num_contexts /*number of user personal contexts to be used for personalization, we use 10*/
```

## PlanPers

To run this baseline, we first need to train the planner model using the labels provided in the dataset. To perform this training step, start by running the following script:


```shell
python train_planner.py \
    --inputs_addr /*address to the dataset file with ranked profile to be trained (train_ranked.json)*/ \
    --model_addr /*the planner model name or address*/ \
    --output_dir /*address to the checkpoint directory*/ \
    --num_contexts /*number of user personal contexts to be used for personalization, we use 10*/
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 5e-5 \
    --weight_decay 0.0 \
    --max_steps 2000 \
    --save_steps 250 \
    --warmup_steps 250 \
    --max_seq_length 8192
```

After selecting the planner checkpoint, you can run this baseline with open-source LLMs using the following script:

```shell
python planpers.py \
    --model_addr /*name of the OpenAI model to be evaluated*/ \
    --planner_model /*address to the trained planner checkpoint*/ \
    --inputs_addr /*address to the dataset file with ranked profile to be evaluated (validation_ranked.json or test_ranked.json)*/ \
    --output_addr /*address to where the outputs should be saved*/ \
    --temperature 0.1 \
    --rag \
    --num_contexts /*number of user personal contexts to be used for personalization, we use 10*/
```

For the OpenAI models, you can use the following script:

```shell
python planpers.py \
    --model_addr /*name of the OpenAI model to be evaluated*/ \
    --planner_model /*address to the trained planner checkpoint*/ \
    --inputs_addr /*address to the dataset file with ranked profile to be evaluated (validation_ranked.json or test_ranked.json)*/ \
    --output_addr /*address to where the outputs should be saved*/ \
    --temperature 0.1 \
    --rag \
    --num_contexts /*number of user personal contexts to be used for personalization, we use 10*/
    --openai \
    --api_key_addr /*address to a file that contains the API key for OpenAI*/
```

# References

If you find this work helpful or use the code or dataset, please cite the following paper:

[LaMP-QA: A Benchmark for Personalized Long-form Question Answering](https://arxiv.org/abs/2506.00137)

```
@misc{salemi2025lampqabenchmarkpersonalizedlongform,
      title={LaMP-QA: A Benchmark for Personalized Long-form Question Answering}, 
      author={Alireza Salemi and Hamed Zamani},
      year={2025},
      eprint={2506.00137},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.00137}, 
}
```

# Acknowledgment

We would like to thank Cheng Li, Mingyang Zhang, Qiaozhu Mei, Weize Kong, Tao Chen, Zhuowan Li, Spurthi Amba Hombaiah, and Michael Bendersky for their valuable feedback and generous support in preparing this work and shaping the formulation of the problem. This work was supported in part by the Center for Intelligent Information Retrieval, in part by NSF grant #2143434, in part by NSF grant #2402873, and in part by Google. Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect those of the sponsor.
