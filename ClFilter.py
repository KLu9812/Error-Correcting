# -*- coding: utf-8 -*-
import pandas as pd
import re
import sys
import numpy as np

def extract_selection(text):
    if pd.isna(text):
        return ''
    text = str(text)
    if re.search(r'yes', text[-30:], flags=re.IGNORECASE):
        return 'yes'
    if re.search(r'no', text[-30:], flags=re.IGNORECASE):
        return 'no'
    return np.random.choice(['yes', 'no'])


def extract_reasoning(text):
    if pd.isna(text):
        return ''
    text = str(text)
    # 1. Look for explicit answer lines (case-insensitive)
    patterns = [
        r'Concise Reasoning',
        r'Step[- ]by[- ]Step',
        r'Summary'
        r'Reasoning:'
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return text[m.start():]
    print("Didn't find pattern")
    return text

# Helper to process a file
def process_file(path):
    df = pd.read_csv(path)
    df['concise_reasoning'] = df['answers'].apply(extract_reasoning)
    df['answers'] = df['answers'].apply(extract_selection)
    out_path = path.replace('.csv', '_filtered.csv')
    df.to_csv(out_path, index=False)
    return out_path

#"./QwQ-32B", "./Qwen3-32B", "./Sky-T1-32B-Preview", "./Llama-3.3-70B-Instruct", "./4o", "./gemini15pro", "./AceReason-Nemotron-14B"
#models = ["./AceReason-Nemotron-14B"]
#for model in models:
#    for i in range(1, 6):
#       process_file(model + "cladderresponses" + str(i) + ".csv")

#models = ["AceReason-Nemotron-14B", "./Sky-T1-32B-Preview", "./Qwen3-32B"]
#for model in models:
#    for i in range(1, 6):
#       process_file(model + "qcladderresponses" + str(i) + ".csv")
#process_file("QwQ-32Bqcladderresponses1.csv")
#process_file("QwQ-32Bqcladderresponses2.csv")
