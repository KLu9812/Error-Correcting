def summarizer_comp(model_list, train_range, fold_number, model, sampling_params, tokenizer):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from generate_data import load_aime_info
    from ItFilter import extract_selection
    import pandas as pd
    import math
    import random
    import numpy as np
    
    
    MODEL = "./aya-expanse-32b"
    MAX_TOKENS = 8192
    BATCH_SIZE = 25
    PADDING_SIDE = "left"
    ITERATIONS = 1
    CROSSES = 5
    NUM_INPUTS = 3
    #"./Llama-3.3-70B-Instruct", "4o", "./Sky-T1-32B-Preview", "./Qwen3-32B", "./QwQ-32B", "gemini15pro"
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
    num_questions = len(aime_info["questions"])
    
    proposers = []
    proposer_data = {}
    for proposer in model_list:
        for i in range(1, 6):
            proposers.append(proposer + str(i))
            proposer_data[proposer + str(i)] = pd.read_csv(proposer + "responses" + str(i) + "_filtered.csv")
    proposer_choices = []
    acc_data = {}
    for proposer in proposers:
        num_correct = 0
        for i in train_range:
            if aime_info["correct answers"][i] == proposer_data[proposer]["multiple_choice_answer"][i]:
                num_correct += 1
        acc_data[proposer] = num_correct // (num_questions - num_questions // CROSSES)
    selected_proposers = []
    output_answers = {"question_ids": [], "answers": [], "input_models": []}
    available_proposers = {}
    for model1 in model_list:
        available_proposers[model1] = [1,2,3,4,5]
    for input_num in range(NUM_INPUTS):
        random_prompts = {}
        messages = []
        for model1 in model_list:
            chosen_prompts = np.random.choice(available_proposers[model1], len(train_range))
            random_prompts[model1] = chosen_prompts
            counter = 0
            for i in train_range:
                question = multiple_choice_multi_input_prompt + "\n" + "Here is the question: "
                question += aime_info['questions'][i]
                question += "\n" + "Now, here are other solutions:\n"
                solutions = []
                for proposer in selected_proposers + [model1 + str(chosen_prompts[counter])]:
                    acc = acc_data[proposer]
                    m = proposer_data[proposer]
                    solution = str(m['concise_reasoning'][i][-1800:]) + " The final answer is (" + str(m['multiple_choice_answer'][i]) + ") with accuracy " + str(acc) + "\n"
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
                counter += 1
    
        outputs = model.generate(messages, sampling_params = sampling_params)
        answers = [output.outputs[0].text for output in outputs]
        df = {"answers": answers}
        df = pd.DataFrame(df)
        df['answers'] = df['answers'].apply(extract_selection)
        counter = 0
        run_accs = {}
        for model1 in model_list:
            for prompt_num in available_proposers[model1]:
                run_accs[model1 + str(prompt_num)] = 0
            subset_answers = []
            counter2 = 0
            subsets = []
            for i in train_range:
                if df['answers'][counter] == aime_info["correct answers"][i]:
                    run_accs[model1 + str(random_prompts[model1][counter2])] += 1
                subset_answers.append(df['answers'][counter])
                subset_name = ""
                for proposer1 in selected_proposers:
                    subset_name += proposer1 + ","
                subset_name += model1 + str(random_prompts[model1][counter2])
                subsets.append(subset_name)
                counter += 1
                counter2 += 1
            output_answers["question_ids"] += train_range
            output_answers["answers"] += subset_answers
            output_answers["input_models"] += subsets
        acc_compares = {}
        for model1 in model_list:
            acc_compares[model1] = 0
        for model1 in model_list:
            for prompt_num in available_proposers[model1]:
                acc_compares[model1] += run_accs[model1 + str(prompt_num)]
        best_model = ""
        best_acc = 0
        for model1 in model_list:
            this_acc = acc_compares[model1]
            if this_acc > best_acc:
                best_acc = this_acc
                best_model = model1
        best_prompt_num = 0
        best_prompt_acc = 0
        for i in available_proposers[best_model]:
            if run_accs[best_model + str(i)] > best_prompt_acc:
                best_prompt_acc = run_accs[best_model + str(i)]
                best_prompt_num = i
        available_proposers[best_model].remove(best_prompt_num)
        selected_proposers.append(best_model + str(best_prompt_num))

    qdf = pd.DataFrame(output_answers)
    qdf.to_csv(MODEL + "multi_responses" + str(fold_number) + str(random.randint(0, 500)) + ".csv")
    print(selected_proposers)
    return selected_proposers
