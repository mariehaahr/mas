import pandas as pd


def read_jsonl(path):
    df = pd.read_json(path, lines=True)
    parsed_cols = pd.json_normalize(df["parsed"]).add_prefix("parsed.")
    df = pd.concat([df.drop(columns=["parsed"]), parsed_cols], axis=1)



qwen = read_jsonl('results/qwen-2.5-7b.jsonl')
mistral = read_jsonl('results/mistral-0.3-7b.jsonl')



print(qwen.head())
print(mistral.head())