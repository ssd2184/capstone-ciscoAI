import random
import json
import datasets
import copy

_RAG_SYSTEM_PROMPT = """You are a helpful assistant designed to generate personalized responses to user questions. Your task is to answer a user's question from a post in a personalized way by considering this user's past post questions and detailed descriptions of these questions.
# Your input:
    - The user's current question from a post.
    - The user's past post questions and detailed descriptions of these questions.
# Your task: Answer the user's current question in a personalized way by considering this user's past post questions and detailed descriptions of these questions, to learn about the user's preferences.
# Your output: You should generate personalized answer to the user's current question by considering this user's past post questions and detailed descriptions of these questions to learn about user's preferences. Your output should be a valid json object in ```json ``` block that contains the following fields:
    - personalized_answer: contains the personalized answer to the user's current question considering the this user's past post questions and detailed descriptions of these questions to learn about user's preferences.
"""

_RAG_WITH_PLAN_SYSTEM_PROMPT = """You are a helpful assistant designed to generate personalized responses to user questions. Your task is to answer a user's question from a post in a personalized way by considering this user's past post questions and detailed descriptions of these questions. Additionally, you are provided with the aspects that the user expects to see in the response to their question, which you can use to generate a personalized answer.

# Your input:
    - The user's current question from a post.
    - The user's past post questions and detailed descriptions of these questions.
    - The aspects that the user expects to see in the response to their question.
# Your task: Answer the user's current question in a personalized way by considering this user's past post questions and detailed descriptions of these questions, to learn about the user's preferences. Additionally, you should consider the aspects that the user expects to see in the response to their question.
# Your output: You should generate personalized answer to the user's current question by considering this user's past post questions and detailed descriptions of these questions to learn about user's preferences. Additionally, you should consider the aspects that the user expects to see in the response to their question. Your output should be a valid json object in ```json ``` block that contains the following fields: 
    - personalized_answer: contains the personalized answer to the user's current question considering the this user's past post questions and detailed descriptions of these questions to learn about user's preferences.
"""

_PLANNER_PROMPT = """You are a helpful assistant designed to generate the topics that user might expect to see in a response to their question. Your task is to extract the important aspects that the user expects to see in a response to their question considering the previous questions asked by the same user and the detailed information need they provided in the post.
# Your input:
    - The user's current question.
    - The user's past post questions and detailed descriptions of these questions.
# Your task: Extract the important aspects that the user expects to see in a response to their question considering the previous questions asked by the same user and the detailed information need they provided in the post.
# Your output: You should generate a list of aspects that are important for the user to be included in the response to their question. 
"""

_RAG_USER_PROMPT = """
# Past post questions and detailed descriptions of these questions:
{profile}
# Current post question:
{question}
"""

_RAG_WITH_PLAN_USER_PROMPT = """
# Past post questions and detailed descriptions of these questions:
{profile}
# Current post question:
{question}
# Aspects expected in the response:
{aspects}
"""

def apply_chat_template(conversation, tokenizer, tokenize=True, add_generation_prompt=False):
    try:
        return tokenizer.apply_chat_template(conversation, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
    except Exception as e:
        if e.message == "System role not supported":
            conversation_new = []
            system_prompt = conversation[0]['content']
            user_prompt = conversation[1]['content']
            new_user_prompt = system_prompt + "\n\n" + user_prompt
            conversation_new.append({"role": "user", "content": new_user_prompt, "type": "text"})
            return tokenizer.apply_chat_template(conversation_new, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
        else:
            raise e 
    
def get_baseline_no_rag_formatter(tokenizer, proprietary_llm=False):
    def formatter(data):
        texts = []
        for i in range(len(data['question'])):
            user_prompt = data['question'][i]
            conversation = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to generate personalized responses to user questions.  Your output should be a valid json object in ```json ``` block that contains the following fields:\n   - personalized_answer: contains the personalized answer to the user's current question.",
                    "type": "text"
                },
                {
                    "role": "user",
                    "content": user_prompt,
                    "type": "text"
                }
            ]
            if proprietary_llm:
                text = conversation
            else:
                text = apply_chat_template(conversation, tokenizer, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        return texts
    return formatter

def get_baseline_rag_formatter(tokenizer, num_contexts, proprietary_llm=False, train=False):
    def formatter(data):
        texts = []
        for i in range(len(data['question'])):
            user_prompt = data['question'][i]
            profile = "\n\n".join([x['text'] for x in data['profile'][i][:num_contexts]])
            conversation = [
                {
                    "role": "system",
                    "content": _RAG_SYSTEM_PROMPT,
                    "type": "text"
                },
                {
                    "role": "user",
                    "content": _RAG_USER_PROMPT.format(profile=profile, question=user_prompt),
                    "type": "text"
                }
            ]
            if proprietary_llm:
                text = conversation
            else:
                if train:
                    expeted_output = data['expected_output'][i]
                    conversation.append({
                        "role": "assistant",
                        "content": expeted_output,
                        "type": "text"
                    })
                    text = {"messages": conversation}
                else:
                    text = apply_chat_template(conversation, tokenizer, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        if train:
            dataset = datasets.Dataset.from_list(texts)
            return dataset
        return texts
    return formatter

def get_planner_formatter(tokenizer, num_contexts, train=False, proprietary_llm=False):
    def formatter(data):
        texts = []
        for i in range(len(data['question'])):
            user_prompt = data['question'][i]
            profile = "\n\n".join([x['text'] for x in data['profile'][i][:num_contexts]])
            conversation = [
                {
                    "role": "system",
                    "content": _PLANNER_PROMPT,
                    "type": "text"
                },
                {
                    "role": "user",
                    "content": _RAG_USER_PROMPT.format(profile=profile, question=user_prompt),
                    "type": "text"
                }
            ]
            if not train:
                if proprietary_llm:
                    text = conversation
                else:
                    text = apply_chat_template(conversation, tokenizer, tokenize=False, add_generation_prompt=False)
                texts.append(text)
            else:
                aspects = ""
                for aspect in data['rubric_aspects'][i]:
                    aspects += "- " + aspect['aspect'] + "\n"    
                conversation.append({
                    "role": "assistant",
                    "content": f"The user expects to see the following aspects in the response to their question:\n{aspects}",
                    "type": "text"
                })
                texts.append({"messages": conversation})
        if train:
            dataset = datasets.Dataset.from_list(texts)
            return dataset
        return texts
    return formatter

def get_generation_with_plan_rag_formatter(tokenizer, num_contexts, proprietary_llm=False):
    def formatter(data):
        texts = []
        for i in range(len(data['question'])):
            user_prompt = data['question'][i]
            profile = "\n\n".join([x['text'] for x in data['profile'][i][:num_contexts]])
            plan = data['plan'][i]
            conversation = [
                {
                    "role": "system",
                    "content": _RAG_WITH_PLAN_SYSTEM_PROMPT,
                    "type": "text"
                },
                {
                    "role": "user",
                    "content": _RAG_WITH_PLAN_USER_PROMPT.format(profile=profile, question=user_prompt, aspects=plan),
                    "type": "text"
                }
            ]
            if proprietary_llm:
                text = conversation
            else:
                text = apply_chat_template(conversation, tokenizer, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        return texts
    return formatter
