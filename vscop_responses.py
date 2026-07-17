from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from generate_data import *
import pandas as pd
import math
import random
import sys


MODEL = "./aya-expanse-32b"
MAX_TOKENS = 8192
BATCH_SIZE = 25
PADDING_SIDE = "left"
NUM_QUESTIONS = 100
ITERATIONS = 20
#"./Llama-3.3-70B-Instruct", "4o", "./Sky-T1-32B-Preview", "./Qwen3-32B", "./QwQ-32B", "gemini15pro"
PROPOSERS = sys.argv[1:-1]
TAIL = sys.argv[-1]

model = LLM(model=MODEL, tensor_parallel_size = 4, gpu_memory_utilization = .9, max_model_len = MAX_TOKENS, trust_remote_code = True)
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

multiple_choice_multi_input_prompt = '''I will give you a multiple choice question and potential solutions and the accuracy that it's correct.
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

proposer_data = {}
acc_data = {}
full_data = {}
for proposer in PROPOSERS:
    proposer_data[proposer] = pd.read_csv(proposer[:-1] + "responses" + proposer[-1] + "_filtered.csv")
    num_correct = 0
    for i in range(0, 400):
        if proposer_data[proposer]['multiple_choice_answer'][i] == aime_info['correct answers'][i]:
            num_correct += 1
    acc_data[proposer] = num_correct / 400
messages = []
for iteration in range(ITERATIONS):
    for i in range(500 - num_questions, 500):
        question = multiple_choice_multi_input_prompt + "\n" + "Here is the question: "
        question += aime_info['questions'][i]
        question += "\n" + "Now, here are other solutions:\n"
        solutions = []
        for proposer in PROPOSERS:
            m = proposer_data[proposer]
            acc = acc_data[proposer]
            solution = str(m['concise_reasoning'][i][-2000:]) + " The final answer is (" + str(m['multiple_choice_answer'][i]) + ") with accuracy " + str(acc) + "\n"
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
for iteration in range(ITERATIONS):
    full_data["answers" + str(iteration)] = answers[iteration * num_questions: (iteration + 1) * num_questions]
df = pd.DataFrame(full_data)
df.to_csv(MODEL + TAIL)
