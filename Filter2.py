import pandas as pd
import re

def extract_selection(text):
    if pd.isna(text):
        return ''
    text = str(text)[-20:]
    # 1. Look for explicit answer lines (case-insensitive)
    patterns = [
        r'Choice:\s*\**\s*([0-9])\)*',                         # Choice: d) ...
        r'Answer[^0-9]+[0-9][^0-9]',                                      # Answer: 'anything' 
        r'answer[^0-9]+[0-9][^0-9]',                                      # answer 'anything'
        r'The correct answer is\s*\**\s*([0-9])\)*',           # The correct answer is c)
        r'^[^a-zA-Z0-9]*\**\(?([0-9])\)?\**\s*(?:[).])?\s*$',  # line is just "d" or "(d)" or "**d**"
        r'\b\([0-9]\)',
        r'\b\( +[0-9] +\)',
        r'boxed\{([0-9])\}'
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return re.search(r'[0-9]', m.group()).group() 
    # 3. As fallback, look for the last "**[a-e]**" in the response
    m = re.findall(r'\*\*([0-9])\*\*', text)
    if m:
        return m[-1].lower()
    # 4. As a last resort, look for a) b) c) d) e) with no prefix
    m = re.findall(r'\b([0-9])\)', text)
    if m:
        return m[-1].lower()
    return ''

# Helper to process a file
def process_file(path, colname='answers'):
    df = pd.read_csv(path)
    df['selection_filtered'] = df[colname].apply(extract_selection)
    out_path = path.replace('.csv', '_filtered.csv')
    df.to_csv(out_path, index=False)
    return out_path

# Process all your uploaded files (use correct column names if not 'answers')
files = [
    'aya-expanse-32bresponses.csv',
    'Sky-T1-32B-Previewresponses.csv',
    'QwQ-32Bresponses.csv',
    'DeepHermes-3-Mistral-24B-Previewresponses.csv',
    'Phi-4-reasoning-plusresponses.csv'
]

for f in files:
    # Most files seem to use 'answers' as the column name
    process_file(f, 'answers')

print("Done. All filtered files are saved with _filtered.csv suffix.")
