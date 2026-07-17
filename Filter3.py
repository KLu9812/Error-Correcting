# -*- coding: utf-8 -*-
import pandas as pd
import re

def extract_selection(text):
    if pd.isna(text):
        return ''
    text = str(text)
    # 1. Look for explicit answer lines (case-insensitive)
    patterns = [
        r'Choice:\s*\**\s*([a-eA-E])\)*',                         # Choice: d) ...
        r'Answer:?\s*\**\s*([a-eA-E])\)*',                        # Answer: c)
        r'Final Answer:?\s*\**\s*([a-eA-E])\)*',                  # Final Answer: e)
        r'The correct answer is\s*\**\s*([a-eA-E])\)*',           # The correct answer is c)
        r'Single letter choice:\s*\**\s*([a-eA-E])\**',           # Single letter choice: c
        r'^[^a-zA-Z0-9]*\**\(?([a-eA-E])\)?\**\s*(?:[).])?\s*$',  # line is just "d" or "(d)" or "**d**"
        r'\b([a-eA-E])\)\s',                                      # matches "d) " inside text
        r'\b([a-eA-E])\.',                                        # matches "d." inside text
        r'option\s*\**([a-eA-E])\**',                             # option c
        r'boxed\{([a-eA-E])\}',
        r'text\{([a-eA-E])\}',
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).lower()
    # 2. Special case for "None of these"/"None of the above"
    if re.search(r'none of (these|the above)', text, re.IGNORECASE):
        return 'e'
    # 3. As fallback, look for the last "**[a-e]**" in the response
    m = re.findall(r'\*\*([a-eA-E])\*\*', text)
    if m:
        return m[-1].lower()
    m = re.search(r'boxed\{([0-9]{1,3})\}', text)
    if m:
        return m.group(1)
    # 4. As a last resort, look for a) b) c) d) e) with no prefix
    m = re.findall(r'\b([a-eA-E])\)', text)
    if m:
        return m[-1].lower()
    return ''


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
    for subset_name in df.columns:
        df[subset_name] = df[subset_name].apply(extract_selection)
        adf = pd.read_csv('aime_info.csv')
        for index in range(len(df)):
            if df[subset_name][index].isalpha():
                continue
            full_text = adf['questions'][index]
            m = re.search(df[subset_name][index], full_text)
            if m:
                s_ind = m.start()
                letter = full_text[s_ind-3]
                if letter.isalpha():
                    df.at[index, subset_name] = letter
                    continue
            df.at[index, subset_name] = ''
            
    out_path = path.replace('.csv', '_filtered.csv')
    df.to_csv(out_path, index=False)
    return out_path

# Process all your uploaded files (use correct column names if not 'answers')


process_file('aya-expanse-32bshapley.csv')

print("Done. All filtered files are saved with _filtered.csv suffix.")
