# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def save_cladder(num_questions):
    data = pd.read_json('cladder-v1/cladder-v1-q-commonsense.json')
    
    numbers = np.random.choice(data["question_id"], num_questions)
    
    questions = []
    solutions = []
    for num in numbers:
        question = data[data["question_id"] == num]['given_info'].values[0] + " " + data[data["question_id"] == num]['question'].values[0]
        solution = str(data[data["question_id"] == num]['answer'].values[0])
        solutions.append(solution)
        questions.append(question)
    qa = pd.DataFrame({"questions": questions, "correct answers": solutions})
    
    qa.to_csv("cladder_info.csv")


def load_cladder_info():
    data = pd.read_csv("cladder_info.csv")
    return data

def load_cladder_questions():
    data = pd.read_csv("cladder_info.csv")
    return data["questions"]

def save_aime(num_questions):
    data = pd.read_csv("AIME_1983_2024/AIME_Dataset_1983_2024.csv")
    
    numbers = np.random.choice(data["ID"], num_questions, replace = False)
    
    questions = []
    solutions = []
    for num in numbers:
        question = data[data["ID"] == num]['Question'].values[0]
        question += " Which of the following choices is correct?\n"
        choices = np.random.choice(np.arange(0, 1000, 1), 4)
        correct = np.random.randint(0, 5)
        letters = ["a", "b", "c", "d", "e"]
        for i in range(5):
            question += letters[i] + ") "
            if i == 4:
                question += "None of these"
            if i == correct:
                solutions.append(letters[correct])
            if i == correct and not i == 4:
                question += str(data[data["ID"] == num]['Answer'].values[0]) + " "
            elif not i == 4:
                question += str(choices[i]) + " "
        questions.append(question)
    qa = pd.DataFrame({"questions": questions, "correct answers": solutions})
    print(qa)
    qa.to_csv("aime_info.csv")

def load_aime_info():
    data = pd.read_csv("aime_info.csv")
    return data

def load_aime_questions():
    data = pd.read_csv("aime_info.csv")
    return data["questions"]

def save_mathbench(num_questions):
    data = pd.read_parquet("hendrycks-MATH-benchmark/data/train-00000-of-00001.parquet")
    level_3s = data[data["level"] == 4]
    valid_probs = level_3s[level_3s["answer"].apply(lambda x: len(x) == 1 and x.isdigit())]
    
    numbers = np.random.choice(valid_probs["unique_id"], num_questions, replace = False)
    
    questions = []
    solutions = []
    for num in numbers:
        question = valid_probs[valid_probs["unique_id"] == num]["problem"].values[0]
        questions.append(question)
        solution = str(valid_probs[valid_probs["unique_id"] == num]["answer"].values[0])
        solutions.append(solution)
    
    qa = pd.DataFrame({"questions": questions, "correct answers": solutions})
    print(qa)
    qa.to_csv("mathbench_info.csv")

def load_mathbench_info():
    data = pd.read_csv("mathbench_info.csv")
    return data

def load_mathbench_questions():
    data = pd.read_csv("mathbench_info.csv")
    return data["questions"]

#MathQA Stuff
def save_mathQA(num_questions):
    data = pd.read_json('math_qa/train.json')
    numbers = np.random.choice(data.index, num_questions, replace = False)
    
    questions_list = []
    solutions_list = []
    for i in numbers:
        questions_list.append(data['Problem'][i] + " This is a multiple choice question. The options are: " + data['options'][i])
        solutions_list.append(data['correct'][i])
    qa = pd.DataFrame({"questions": questions_list, "correct answers": solutions_list})
    qa.to_csv("mathQA_info.csv")