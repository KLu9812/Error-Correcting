from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from generate_data import *
from ItFilter import *
import pandas as pd
import math
import random
import sys
from methods import *
from summary_comp import summarizer_comp


MODEL = "./AceReason-Nemotron-14B"
MAX_TOKENS = 8192
BATCH_SIZE = 25
PADDING_SIDE = "left"
ITERATIONS = 2
CROSSES = 5
TAIL = "scopcvgptrateresponse.csv"
#"./Llama-3.3-70B-Instruct", "4o", "./Sky-T1-32B-Preview", "./Qwen3-32B", "./QwQ-32B", "gemini20", "./aya-expanse-32b", "./AceReason-Nemotron-14B"

model = LLM(model=MODEL, tensor_parallel_size = 4, gpu_memory_utilization = .9, max_model_len = MAX_TOKENS, trust_remote_code = True)
sampling_params = SamplingParams(temperature = 0.7, max_tokens = MAX_TOKENS, top_p=.95)
tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side = PADDING_SIDE)
#tokenizer.chat_template = open("chat_template.txt").read()
#print(tokenizer.chat_template)
tokenizer.pad_token = tokenizer.eos_token
print("Model Loaded")

all_models = ["./Llama-3.3-70B-Instruct", "4o", "./Sky-T1-32B-Preview", "./Qwen3-32B", "gemini20", "./aya-expanse-32b", "./AceReason-Nemotron-14B"]
methods = [rateresponse]
#top_model, top_k_accuracy, Greedy_marginal_Shapley_fine_Q, Greedy_marginal_Shapley_DT, MOA, top_k_diversity_threshold, use_all]

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
num_questions = len(aime_info["questions"])

proposer_data = {}
for proposer in all_models:
    for i in range(1, 6):
        proposer_data[proposer + str(i)] = pd.read_csv(proposer + "responses" + str(i) + "_filtered.csv")
messages = []
proposer_choices = []
proposer_sets = []
for method in methods:
    for cross in range(CROSSES):
        question_range = range(cross * num_questions // CROSSES, (cross + 1) * num_questions // CROSSES)
        input_range = list(range(cross * num_questions // CROSSES)) + list(range((cross + 1) * num_questions // CROSSES, num_questions))
        if method == summarizer_comp:
            proposers = method(all_models, input_range, CROSSES, model, sampling_params, tokenizer, k=3)
        else:
            proposers = method(all_models, input_range, CROSSES, k = 3)
        proposers = [str(proposer) for proposer in proposers]
        proposer_choices.append(method.__name__ + str(proposers))
        proposer_sets.append(proposers)
        acc_data = {}
        for proposer in proposers:
            num_correct = 0
            for i in range(num_questions):
                if i in question_range:
                    continue
                if aime_info["correct answers"][i] == proposer_data[proposer]["multiple_choice_answer"][i]:
                    num_correct += 1
            acc_data[proposer] = num_correct / (num_questions - num_questions // CROSSES)
        for iteration in range(ITERATIONS):
            for i in question_range:
                question = multiple_choice_multi_input_prompt + "\n" + "Here is the question: "
                question += aime_info['questions'][i]
                question += "\n" + "Now, here are other solutions:\n"
                solutions = []
                for proposer in proposers:
                    acc = acc_data[proposer]
                    m = proposer_data[proposer]
                    solution = str(m['concise_reasoning'][i][-2000:])
                    if method == use_all:
                        solution = solution[-300:]
                    elif method == MOA:
                        solution = solution[-1500:]
                    solution +=" The final answer is (" + str(m['multiple_choice_answer'][i]) + ") with accuracy " + str(acc) + "\n"
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
pdf = {"proposers": proposer_choices}
pdf = pd.DataFrame(pdf)
print(pdf)
pdf.to_csv(MODEL + "proposer_choices" + TAIL)
df = {"answers": answers}
df = pd.DataFrame(df)
df['answers'] = df['answers'].apply(extract_selection)
counter = 0
for method in methods:
    for cross in range(CROSSES):
        question_range = range(cross * num_questions // CROSSES, (cross + 1) * num_questions // CROSSES)
        for iteration in range(ITERATIONS):
            for i in question_range:
                if df['answers'][counter].isalpha():
                    continue
                full_text = aime_info['questions'][i]
                m = re.search(df['answers'][counter], full_text)
                if m:
                    s_ind = m.start()
                    letter = full_text[s_ind-3]
                    if letter.isalpha():
                        df.at[counter, 'answers'] = letter
                        continue
                df.at[counter, 'answers'] = ''
                counter += 1
df.to_csv(MODEL + TAIL)
counter = 0
for method in methods:
    name = method.__name__
    num_correct = 0
    for cross in range(CROSSES):
        question_range = range(cross * num_questions // CROSSES, (cross + 1) * num_questions // CROSSES)
        for iteration in range(ITERATIONS):
            for i in question_range:
                if aime_info["correct answers"][i] == df['answers'][counter]:
                    num_correct += 1
                counter += 1
    acc = num_correct / (ITERATIONS * num_questions)
    print(name + ": " + str(acc))
