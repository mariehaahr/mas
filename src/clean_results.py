import pandas as pd
from pathlib import Path
import yaml


# id,model_sender,label_sender,model_receiver,label_receiver_new,confidence,valid_json,raw_text
mistral_path = Path((f"/home/rp-fril-mhpe/second-mistral-0.3-7b.csv"))
qwen_path = Path((f"/home/rp-fril-mhpe/second-qwen-2.5-7b.csv"))
lookup1_path = Path("/home/rp-fril-mhpe/first-results-sarc-ratio.csv")   # round 1 lookup

out = "llama-3.2-1b" # model we remove

# output file names
output_mistral = Path("/home/rp-fril-mhpe/clean-second-mistral-0.3-7b.csv")
output_qwen    = Path("/home/rp-fril-mhpe/clean-second-qwen-2.5-7b.csv")

# load data
df_m = pd.read_csv(mistral_path)
df_q    = pd.read_csv(qwen_path)

before_m = len(df_m)
before_q = len(df_q)

# remove where sender = llama-3.2-1b, just from the data
df_mistral = df_m[df_m["model_sender"] != out].copy()
df_qwen    = df_q[df_q["model_sender"] != out].copy()

# load lookup table
lookup1 = pd.read_csv(lookup1_path)

# keep only id and valid_json_count
lookup = lookup1[["id", "valid_json_count"]]


# merge

df_m = df_m.merge(lookup, on="id", how="left")
df_q = df_q.merge(lookup, on="id", how="left")

# remove json < 5 by using the lookup from round 1
df_m = df_m[df_m["valid_json_count"] >= 5].copy()
df_q = df_q[df_q["valid_json_count"] >= 5].copy()

# remove llama
df_m_before_sender_filter = len(df_m)
df_q_before_sender_filter = len(df_q)

df_m = df_m[df_m["model_sender"] != out].copy()
df_q = df_q[df_q["model_sender"] != out].copy()

removed_mistral = before_m - len(df_m)
removed_qwen    = before_q - len(df_q)


# remove column
df_m = df_m.drop(columns=["valid_json_count"])
df_q = df_q.drop(columns=["valid_json_count"])


# save
df_m.to_csv(output_mistral, index=False)
df_q.to_csv(output_qwen, index=False)


print(f"mistral columns: {df_m.columns.tolist()}")
print(f"qwen columns: {df_q.columns.tolist()}")

print("\nRows removed (total):")
print(f" - mistral: {removed_mistral}")
print(f" - qwen   : {removed_qwen}")

print("\nDetailed sanity check:")
print(f" - mistral before merge: {before_m}, after clean: {len(df_m)}")
print(f" - qwen before merge:    {before_q}, after clean: {len(df_q)}")

print("\nSaved to:")
print(" -", output_mistral)
print(" -", output_qwen)

