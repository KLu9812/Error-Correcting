# -*- coding: utf-8 -*-
import pandas as pd
import re
def extract_scores(text, text_num, num_agents):
    patterns = [
        r'Solution\#:',
        r'Solution',
        ]
    score_patterns = [
        r'([1-5])/5',
        r'\*+([1-5])\*+'
        ]
    sol_num = 1
    index_num = 0
    scores = []
    while sol_num <= num_agents:
        d = re.search(r'\n\s*' + str(sol_num) + '\s*:\s*([1-5])\s*\n', text[index_num:])
        if d:
            scores.append(int(d.group(1)))
            sol_num = sol_num + 1
            index_num = d.end()
            continue
        for pattern in patterns:
            m = re.search(pattern, text[index_num:])
            if m:
                index_num = m.end() + index_num
                break
        d = re.search(r':', text[index_num:])
        next_colon = 0
        if d:
            next_colon = index_num + d.start()
        else:
            print(str(text_num) + ": Broke at find colon" )
            break
        numbers_string = text[index_num:next_colon]
        numbers = re.split('[^0-9]', numbers_string)
        numbers = [x for x in numbers if x]
        num_add = 0
        if len(numbers) == 0:
            print(text_num, numbers, sol_num, index_num, next_colon)
            break
        elif int(numbers[0]) == sol_num:
            if len(numbers) == 1:
                sol_num += 1
                num_add = 1
            elif len(numbers) == 2:
                sol_num = int(numbers[1]) + 1
                num_add = int(numbers[1]) - int(numbers[0]) + 1
        else:
            print(text_num, numbers, sol_num)
            break
        d = re.search(r'(?:$|\n)', text[next_colon + 1:])
        if d:
            next_newline = next_colon + 1 + d.start()
        else:
            print(str(text_num) + ": Broke at find endline")
            break
        score = 0
        numbers = re.split('[^0-9]', text[next_colon + 1:next_newline])
        numbers = [x for x in numbers if x]
        if len(numbers) == 1:
            if int(numbers[0]) <= 5 and int(numbers[0]) >= 1:
                score = int(numbers[0])
        for pattern in score_patterns:
            d = re.search(pattern, text[next_colon + 1:next_newline])
            if d:
                score = int(d.group(1))
                break
        if score == 0:
            if re.search(r'Incomplete', text[next_colon + 1: next_newline], flags = re.IGNORECASE):
                score = 1
            else:
                print(str(text_num), sol_num, text[next_colon + 1:next_newline])
                break
        for j in range(num_add):
            scores.append(score)
        index_num = next_newline + 1
    return scores

def process_file(path, num_agents, colname='answers'):
    df = pd.read_csv(path)
    all_scores = []
    for i in range(num_agents):
        all_scores.append([])
    for i in range(len(df["answers"])):
        scores = extract_scores(df["answers"][i], i, num_agents)
        if len(scores) == num_agents:
            proposer_order = df["proposer_orders"][i][1:-1]
            proposer_order = proposer_order.split(", ")
            for j in range(num_agents):
                all_scores[int(proposer_order[j])].append(scores[j])
        else:
            for j in range(num_agents):
                all_scores[j].append(0)
    out_df = {}
    for i in range(len(all_scores)):
        out_df[i] = all_scores[i]
    out_df = pd.DataFrame(out_df)
    out_path = path.replace('.csv', '_filtered.csv')
    out_df.to_csv(out_path, index=False)
    return out_path
process_file("aya-expanse-32bcladderrateresponses.csv", 40)
