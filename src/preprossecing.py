import pandas as pd 
from pathlib import Path


txt_path = Path('data/sarc/raw/train_text.txt')
with txt_path.open(encoding="utf-8") as f:
    lines = [line.rstrip("\n") for line in f]


labels_path = Path('data/sarc/raw/train_labels.txt')
with labels_path.open(encoding='utf-8') as f:
    labels = [label.rstrip('\n') for label in f]

sarcasm = pd.DataFrame({'id': list(range(0, len(lines))),"text": lines, 'label': labels})

# count the number of words in each claim 
sarcasm['n_words'] = [len(x.split()) for x in sarcasm['text'].tolist()]

# filter out too short and too long examples
sarcasm_pr = sarcasm[(sarcasm['n_words'] >= 6) & (sarcasm['n_words'] < 500)]

# sarcasm_pr.to_csv('data/sarc/sarcasm.csv', index=False)
