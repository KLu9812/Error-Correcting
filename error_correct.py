# Standard library imports
import pandas as pd
import json
import numpy as np

# Third-party imports
from openai import OpenAI

#Schemas//////////////////////////////////////////////////////////////////////
digit_response_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "ResponseForm",
        "strict": "true",
        "type": "object",
        "schema": {
            "type": "object",
            "properties": {
                "full_response": {
                    "type": "string"},
                "concise_reasoning": {
                    "type": "string"},
                "digit": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 9}
                },
            "required": ["full_response", "concise_reasoning", "digit"]}}}

multiple_choice_response_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "ResponseForm",
        "strict": "true",
        "type": "object",
        "schema": {
            "type": "object",
            "properties": {
                "full_response": {
                    "type": "string"},
                "concise_reasoning": {
                    "type": "string"},
                "choice": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 1}
                },
            "required": ["full_response", "concise_reasoning", "choice"]}}}

yes_no_response_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "ResponseForm",
        "strict": "true",
        "type": "object",
        "schema": {
            "type": "object",
            "properties": {
                "full_response": {
                    "type": "string"},
                "concise_reasoning": {
                    "type": "string"},
                "answer": {
                    "type": "string",
                    "minLength": 2,
                    "maxLength": 3}
                },
            "required": ["full_response", "concise_reasoning", "answer"]}}}




#Prompts//////////////////////////////////////////////////////////////////////
digit_role_message = {
    "role": "system",
    "content": (
        '''You will solve a math question. The possible answers are integers ranging from 0 to 9, inclusive.
        Format your answer to include the following three parts:
        1. A full response
        2. Summary of the full response into a concise step by step reasoning that explains your response
        3. The single digit answer'''
    ),
}

multiple_choice_role_message = {
    "role": "system",
    "content": (
        '''You will solve a mulitple choice question.
        Format your answer to include the following three parts:
        1. A full response
        2. Summary of the full response into a concise step by step reasoning that explains your response
        3. The single letter choice'''
    ),
}

yes_no_role_message = {
    "role": "system",
    "content": (
        '''You will answer a yes or no questions.
        Format your answer to include the following three parts:
        1. A full response
        2. Summary of the full response into a concise step by step reasoning that explains your response
        3. Yes or no answer'''
    ),
}

digit_multi_input_role_message = {
    "role": "system",
    "content": (
        '''You are an assistant that analyzes possibly flawed solutions to math questions with single digit answers, and then answers the questions.
        Format your answer to include the following two parts:
        1. A full response
        2. The single digit'''
    ),
}   
    
multiple_choice_multi_input_role_message = {
    "role": "system",
    "content": (
        '''You are an assistant that analyzes possibly flawed solutions to multiple choice questions, and then answers the questions.
        Format your answer to include the following two parts:
        1. A full response
        2. The letter choice'''
    ),
}    
    
yes_no_multi_input_role_message = {
    "role": "system",
    "content": (
        '''You are an assistant that analyzes possibly flawed solutions to yes or no questions, and then answers the questions.
        Format your answer to include the following two parts:
        1. A full response
        2. The yes or no answer'''
    ),
}
    
yes_no_just_answers_role_message = {
    "role": "system",
    "content": (
        '''You are an assistant answers yes or no questions. You will also be given answers from other language models attempting to answer the same question.
        Format your answer to include the following two parts:
        1. A full response
        2. The yes or no answer'''
    ),
}

digit_multi_input_prompt = '''I will give you a math question with a single digit answer, and multiple potential solutions that may be correct or incorrect.
Your task is to analyze the reasoning of the potential solutions step by step.
If there are any errors, correct them and update your answer.
If there are no errors, answer the question matching those solutions. Your answer must be in the format of a full response, then a single digit.'''

multiple_choice_multi_input_prompt = '''I will give you a multiple choice question and multiple potential solutions that may be correct or incorrect.
Your task is to analyze the reasoning of the potential solutions step by step.
If there are any errors, correct them and update your answer.
If there are no errors, answer the question matching those solutions. Your answer must be in the format of a full response, then a letter choice.'''

yes_no_multi_input_prompt = '''I will give you a yes or no question and multiple potential solutions that may be correct or incorrect.
Your task is to analyze the reasoning of the potential solutions step by step.
If there are any errors, correct them and update your answer.
If there are no errors, answer the question matching those solutions. Your answer must be in the format of a full response, then a yes or no answer.'''

yes_no_just_answers_prompt = '''I will give you a yes or no question and multiple potential answers from other language models that may be correct or incorrect.
Your task is to answer the question using this information.
Your answer must be in the format of a full response, then a yes or no answer.'''


class ErrorCorrect:
    
    def __init__(self, **kwargs):
        # Initialize LM Studio client
        self.client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
        self.models = kwargs.get('models', None)
        self.schema = kwargs.get('schema', None)
        self.questions = kwargs.get('questions', None)
        self.correct_answers = kwargs.get('correct_answers', None)
    
    def set_models(self, models):
        self.models = models
    def set_schema(self, schema):
        self.schema = schema
    def set_questions(self, questions):
        self.questions = questions
    def set_correct_answers(self, correct_answers):
        self.correct_answers = correct_answers
    
    def set_initial_response_role(self, role):
        self.initial_response_role = role
    def save_responses(self, verbose = True):
        for model in self.models:
            answers = []
            reasonings = []
            for i in range(len(self.questions)):
                if verbose:
                    print(i)
                messages = [self.initial_response_role]
                question = self.questions[i]
                messages.append({"role": "user", "content": question})
                try:
                    response = self.client.chat.completions.create(
                        model=model, messages=messages, response_format = self.schema)
                    results = json.loads(response.choices[0].message.content)
                    answer_column = list(self.schema["json_schema"]["schema"]["properties"].keys())[2]
                    answers.append(str(results[answer_column]).lower())
                    reasonings.append(results['concise_reasoning'])
                except Exception:
                    answers.append("?")
                    reasonings.append("Unable to solve the given question.")
                full_data = {"answers": answers, "reasonings": reasonings}
                df = pd.DataFrame(full_data)
                df.to_csv(model + "responses.csv")
    
    
    def set_multi_input_prompt(self, prompt):
        self.multi_input_prompt = prompt
    def set_multi_input_role(self, role):
        self.multi_input_role = role
    def get_multi_input(self, verbose = True):
        models_reasonings = {}
        for model_b in self.models:
            models_reasonings[model_b] = pd.read_csv(model_b + "responses.csv")
            models_reasonings[model_b].fillna("", inplace = True)
        for model in self.models:
            answers = []
            for i in range(len(self.questions)):
                if verbose:
                    print(i)
                input_list = ""
                counter = 1
                for model_b in self.models:
                    if model_b == model:
                        continue
                    input_list += "Solution " + str(counter) + ": " + models_reasonings[model_b]["reasonings"][i] + ";final answer: " + models_reasonings[model_b]["answers"][i] + "...End Solution.\n"
                    counter += 1
                conditional_question = self.multi_input_prompt
                conditional_question += "Here is the question that I would like you to answer: " + self.questions[i]
                conditional_question += "Here are the potential solutions that may be correct or incorrect: "
                conditional_question += input_list
                messages = [self.multi_input_role]
                messages.append({"role": "user", "content": conditional_question})
                try:
                    response = self.client.chat.completions.create(
                        model=model, messages=messages, response_format = self.schema)
                    results = json.loads(response.choices[0].message.content)
                    answer_column = list(self.schema["json_schema"]["schema"]["properties"].keys())[2]
                    answers.append(str(results[answer_column]).lower())
                except Exception:
                    answers.append("?")
                full_data = {'answers' : answers}
                df = pd.DataFrame(full_data)
                df.to_csv(model + "multi_input.csv")
    
    def set_just_answers_prompt(self, prompt):
        self.just_answers_prompt = prompt
    def set_just_answers_role(self, role):
        self.just_answers_role = role
    def get_just_answers(self, verbose = True):
        models_reasonings = {}
        for model_b in self.models:
            models_reasonings[model_b] = pd.read_csv(model_b + "responses.csv")
            models_reasonings[model_b].fillna("", inplace = True)
        for model in self.models:
            answers = []
            for i in range(len(self.questions)):
                if verbose:
                    print(i)
                input_list = ""
                counter = 1
                for model_b in self.models:
                    if model_b == model:
                        continue
                    input_list += "Answer " + str(counter) + ": " + models_reasonings[model_b]["answers"][i] + "\n"
                    counter += 1
                conditional_question = self.just_answers_prompt
                conditional_question += "Here is the question that I would like you to answer: " + self.questions[i]
                conditional_question += "Here are the answers that may be correct or incorrect: "
                conditional_question += input_list
                messages = [self.just_answers_role]
                messages.append({"role": "user", "content": conditional_question})
                try:
                    response = self.client.chat.completions.create(
                        model=model, messages=messages, response_format = self.schema)
                    results = json.loads(response.choices[0].message.content)
                    answer_column = list(self.schema["json_schema"]["schema"]["properties"].keys())[2]
                    answers.append(str(results[answer_column]).lower())
                except Exception:
                    answers.append("?")
                full_data = {'answers' : answers}
                df = pd.DataFrame(full_data)
                df.to_csv(model + "just_answers.csv")
    
    def clean_yes_no(self):
        for model in self.models:
            data = pd.read_csv(model + "multi_input.csv")
            for i in range(len(data)):
                if data['answers'][i][:2] == 'no':
                    data.at[i, 'answers'] = "no"
            data.to_csv(model + "multi_input.csv")
            
            data = pd.read_csv(model + "responses.csv")
            for i in range(len(data)):
                if data['answers'][i][:2] == 'no':
                    data.at[i, 'answers'] = "no"
            data.to_csv(model + "responses.csv")
    
    
    def analyze(self, multi_input = True):
        full_data = {}
        for model in self.models:
            response = pd.read_csv(model + "responses.csv")
            multi = pd.read_csv(model + "multi_input.csv")
            full_data[model] = response['answers']
            full_data[model + 'multi_input'] = multi['answers']
        
        for model in self.models:
            num_correct = 0
            for answer_num in range(len(full_data[model])):
                if full_data[model][answer_num] == self.correct_answers[answer_num]:
                    num_correct += 1
            acc = num_correct / len(full_data[model])
            print("Model: " + model)
            print("Accuracy: " + str(acc))
            
            if not multi_input:
                continue
            instance_counter = np.zeros(len(self.models))
            self_correct_counter = np.zeros(len(self.models))
            original_self_correct_counter = np.zeros(len(self.models))
            for answer_num in range(len(full_data[model])):
                num_other_models_correct = 0
                for model_b in self.models:
                    if model_b == model:
                        continue
                    if full_data[model_b][answer_num] == self.correct_answers[answer_num]:
                        num_other_models_correct += 1
                instance_counter[num_other_models_correct] += 1
                if full_data[model + "multi_input"][answer_num] == self.correct_answers[answer_num]:
                    self_correct_counter[num_other_models_correct] += 1
                if full_data[model][answer_num] == self.correct_answers[answer_num]:
                    original_self_correct_counter[num_other_models_correct] += 1
            print("For Multi Inputs: ")
            for i in range(len(instance_counter)):
                output_string = ""
                for j in range(len(instance_counter) - 1 - i):
                    output_string += "I"
                for j in range(i):
                    output_string += "C"
                output_string += ": " + str(int(self_correct_counter[i])) + "/" + str(int(instance_counter[i]))
                output_string += " Original Correct: " + str(int(original_self_correct_counter[i]))
                print(output_string)


