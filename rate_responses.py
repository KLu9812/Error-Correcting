from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from generate_data import *
import pandas as pd

MODEL = "./aya-expanse-32b"
MAX_TOKENS = 8192
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

rate_message = "You will be given a question and multiple solutions to that question. Give a score ranging from 1 to 5 on the helpfulness of each solution and you must score each solution. Do not answer the question and do not explain your scores. Only provide the scores. Structure your answer so that each score is a new line, with the format Solution#: score#."

aime_info = pd.read_csv("aime_info.csv")
print("Data Loaded")
all_models = ["./Llama-3.3-70B-Instruct", "4o", "./Sky-T1-32B-Preview", "./Qwen3-32B", "gemini20", "./QwQ-32B", "./aya-expanse-32b", "./AceReason-Nemotron-14B"]
proposer_data = {}
all_proposers = []
for proposer in all_models:
    for i in range(1, 6):
        proposer_data[proposer + str(i)] = pd.read_csv(proposer + "responses" + str(i) + "_filtered.csv")
        all_proposers.append(proposer + str(i))

messages = []
num_questions = NUM_QUESTIONS
if NUM_QUESTIONS == 0:
    num_questions = len(aime_info['questions'])
for i in range(num_questions):
    question = aime_info['questions'][i]
    role_message = {}
    question_message = {}
    role_message['role'] = 'system'
    role_message['content'] = rate_message
    question_message['role'] = 'user'
    content_message = "Here is the question: "
    content_message += '\n' + question
    content_message += '\n' + "Here are the solutions you should rate:\n"
    for j in range(len(all_proposers)):
        content_message += "Solution " + str(j + 1) + ": " + proposer_data[all_proposers[j]]["concise_reasoning"][i][-300:]
    question_message['content'] = content_message
    tokenized_chat = tokenizer.apply_chat_template([role_message, question_message], tokenize=False, add_generation_prompt = True, return_tensors = "pt")
    messages.append(tokenized_chat)

outputs = model.generate(messages, sampling_params = sampling_params)
answers = [output.outputs[0].text for output in outputs]
df = pd.DataFrame({'answers': answers})
df.to_csv(MODEL + 'rateresponses.csv')
