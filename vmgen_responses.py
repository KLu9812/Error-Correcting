from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from generate_data import *
import pandas as pd

MODEL = "./Llama-3.3-70B-Instruct"
MAX_TOKENS = 4096
BATCH_SIZE = 25
PADDING_SIDE = "left"
NUM_QUESTIONS = 0
#"./Llama-3.3-70B-Instruct", "4o", "./Sky-T1-32B-Preview", "./Qwen3-32B", "gemini20", "./QwQ-32B", "./aya-expanse-32b", "./AceReason-Nemotron-14B"

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

prompt1 = 'Divide the question into smaller, manageable parts and tackle each part individually before synthesizing the overall answer.'

prompt2 = 'Use mathematical principles and logic to solve the problem, even if it’s not a math question.'

prompt3 = 'Relate the question to a familiar concept or situation to better understand and solve it.'

prompt4 = 'Think about what the answer would be if the opposite were true, to gain a different perspective.'

prompt5 = 'Eliminate the obviously incorrect answers first and then choose the most likely correct answer.'

prompts = [prompt1, prompt2, prompt3, prompt4, prompt5]

mmlu_info = pd.read_csv("mmlu_info.csv")
print("Data Loaded")


for j in range(1, 6):
    messages = []
    num_questions = NUM_QUESTIONS
    if NUM_QUESTIONS == 0:
        num_questions = len(mmlu_info['questions'])
    for i in range(num_questions):
        if i % BATCH_SIZE == 0:
            print(i)
        question = mmlu_info['questions'][i]
        role_message = {}
        question_message = {}
        role_message['role'] = 'system'
        role_message['content'] = multiple_choice_role_message
        question_message['role'] = 'user'
        question_message['content'] = prompts[j - 1] + '\n' + question
        tokenized_chat = tokenizer.apply_chat_template([role_message, question_message], tokenize=False, add_generation_prompt = True, return_tensors = "pt")
        messages.append(tokenized_chat)

    outputs = model.generate(messages, sampling_params = sampling_params)
    answers = [output.outputs[0].text for output in outputs]
    df = pd.DataFrame({'answers': answers})
    df.to_csv(MODEL + 'mmluresponses' + str(j) + '.csv')
