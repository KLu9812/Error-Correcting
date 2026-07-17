from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from generate_data import *
import pandas as pd
import math
import random

MODEL = "./QwQ-32B"
TAIL = "top5_comp.csv"
MAX_TOKENS = 8192
BATCH_SIZE = 25
PADDING_SIDE = "left"
NUM_QUESTIONS = 100
#"./Llama-3.3-70B-Instruct", "4o", "./Sky-T1-32B-Preview", "./Qwen3-32B", "./QwQ-32B", "gemini15pro"
PROPOSERS = ['./AceReason-Nemotron-14B5', './AceReason-Nemotron-14B3', './Qwen3-32B1', './Qwen3-32B4', './AceReason-Nemotron-14B2']

model = LLM(model=MODEL, tensor_parallel_size = 8, gpu_memory_utilization = .9, max_model_len = MAX_TOKENS, trust_remote_code = True)
sampling_params = SamplingParams(temperature = 0.7, max_tokens = MAX_TOKENS, top_p=.95)
tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side = PADDING_SIDE)
#tokenizer.chat_template = open("chat_template.txt").read()
#print(tokenizer.chat_template)
tokenizer.pad_token = tokenizer.eos_token
print("Model Loaded")

multiple_choice_role_message = '''You will solve a multiple choice question.
        Format your answer to include the following three parts:
        1. A full response
        2. Summary of the full response into a concise step by step reasoning that explains your response
        3. The single letter choice'''

multiple_choice_multi_input_role_message = {
    "role": "system",
    "content": (
        '''You are an assistant that analyzes possibly flawed solutions to multiple choice questions, and then answers the questions.
        Format your answer to include the following two parts:
        1. A full response
        2. The letter choice'''
    ),
}

multiple_choice_multi_input_prompt = '''I will give you a multiple choice question and potential solutions that may be correct or incorrect.
Your task is to analyze the reasoning of the potential solutions step by step.
If there are any errors, correct them and update your answer.
If there are no errors, answer the question matching those solutions. Your answer must be in the format of a full response, then a letter choice.'''

prompt1 = 'Divide the question into smaller, manageable parts and tackle each part individually before synthesizing the overall answer.'

prompt2 = 'Use mathematical principles and logic to solve the problem, even if it’s not a math question.'

prompt3 = 'Relate the question to a familiar concept or situation to better understand and solve it.'

prompt4 = 'Think about what the answer would be if the opposite were true, to gain a different perspective.'

prompt5 = 'Eliminate the obviously incorrect answers first and then choose the most likely correct answer.'

prompts = [prompt1, prompt2, prompt3, prompt4, prompt5]

aime_info = load_aime_info()
print("Data Loaded")

num_questions = NUM_QUESTIONS
if NUM_QUESTIONS == 0:
    num_questions = len(aime_info['questions'])

messages = []
proposer_data = {}
for proposer in PROPOSERS:
    proposer_data[proposer] = pd.read_csv(proposer[:-1] + "responses" + proposer[-1] + "_filtered.csv")
for i in range(num_questions):
    question = multiple_choice_multi_input_prompt + "\n" + "Here is the question: "
    question += aime_info['questions'][i]
    question += "\n" + "Now, here are other solutions:\n"
    solutions = []
    for proposer in PROPOSERS:
        m = proposer_data[proposer]
        solution = str(m['concise_reasoning'][i][-2000:]) + " The final answer is (" + str(m['multiple_choice_answer'][i]) + ")\n"
        solutions.append(solution)
    random.shuffle(solutions)
    for j in range(len(solutions)):
        question += "Solution " + str(j + 1) + ": "
        question += solutions[j]
    role_message = {}
    question_message = {}
    role_message['role'] = 'system'
    role_message['content'] = multiple_choice_multi_input_role_message['content']
    question_message['role'] = 'user'
    question_message['content'] = question
    tokenized_chat = tokenizer.apply_chat_template([role_message, question_message], tokenize=False, add_generation_prompt = True, return_tensors = "pt")
    messages.append(tokenized_chat)

outputs = model.generate(messages, sampling_params = sampling_params)
answers = [output.outputs[0].text for output in outputs]
full_data = {"answers": answers}
df = pd.DataFrame(full_data)
df.to_csv(MODEL + TAIL)
