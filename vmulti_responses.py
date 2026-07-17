from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from generate_data import *
import pandas as pd
import math
import random

MODEL = "./aya-expanse-32b"
MAX_TOKENS = 8192
BATCH_SIZE = 25
PADDING_SIDE = "left"
NUM_QUESTIONS = 400
PROPOSERS = ["./Llama-3.3-70B-Instruct", "gemini15pro", "./Sky-T1-32B-Preview", "./Qwen3-32B", "./QwQ-32B"]

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

chosen_shapley = pd.read_csv('chosen_shapley.csv')
possibles = []
possible_names = []
for proposer in PROPOSERS:
    for i in range(1, 6):
        possibles.append(pd.read_csv(proposer + "responses" + str(i) + "_filtered.csv"))
        possible_names.append(proposer + str(i))
num_questions = NUM_QUESTIONS
if NUM_QUESTIONS == 0:
    num_questions = len(aime_info['questions'])

messages = []
for row in range(len(chosen_shapley)):
    for i in range(num_questions):
        if i % BATCH_SIZE == 0:
            print(i)
        question = multiple_choice_multi_input_prompt + "\n" + "Here is the question: "
        question += aime_info['questions'][i]
        question += "\n" + "Now, here are other solutions:\n"
        solutions = []
        for j in range(5):
            if not math.isnan(chosen_shapley[str(j)][row]):
                m = possibles[int(chosen_shapley[str(j)][row])]
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
full_data = {}
counter = 0
for row in range(len(chosen_shapley)):
    column_name = ''
    for m in range(5):
        if not math.isnan(chosen_shapley[str(m)][i]):
            column_name += possible_names[int(chosen_shapley[str(m)][i])] + ','
    column_name = column_name[:-1]
    full_data[column_name] = answers[counter * num_questions: (counter + 1) * num_questions]
    counter += 1
df = pd.DataFrame(full_data)
df.to_csv(MODEL + 'shapley.csv')
