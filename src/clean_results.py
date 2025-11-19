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



# remove where sender = llama-3.2-1b, just from the data
df_mistral = df_m[df_m["model_sender"] != out].copy()
df_qwen    = df_q[df_q["model_sender"] != out].copy()




# remove json < 5 by using the lookup from round 1
    # load lookup 1 
    # mask all claims where valid_json_count < 5
    # remove mask from data (overwrite)

# load lookup table
lookup1 = pd.read_csv(lookup1_path)

# keep only id and valid_json_count
lookup = lookup1[["id", "valid_json_count"]]

# merge with data
df_mistral = df_mistral.merge(lookup, on="id", how="left")
df_qwen    = df_qwen.merge(lookup, on="id", how="left")

# remove rows where round-1 valid_json_count < 5
df_mistral = df_mistral[df_mistral["valid_json_count"] >= 5].copy()
df_qwen    = df_qwen[df_qwen["valid_json_count"] >= 5].copy()

# remove valid_json_count, since this should not be in the final data
df_mistral = df_mistral.drop(columns=["valid_json_count"])
df_qwen    = df_qwen.drop(columns=["valid_json_count"])

print(f"mistral columns: {df_mistral.columns}")
print(f"qwen columns: {df_qwen.columns}")

# save data as
mistral_new_path = "/home/rp-fril-mhpe/clean-second-mistral-0.3-7b.csv"
qwen_new_path = "/home/rp-fril-mhpe/clean-second-qwen-2.5-7b.csv"

df_mistral.to_csv(output_mistral, index=False)
df_qwen.to_csv(output_qwen, index=False)

print(f"Cleaned mistral.... before: {df_m.shape}, and after: {df_mistral.shape}")
print(f"Cleaned qwen.... before: {df_q.shape}, and after: {df_qwen.shape}")

print("Saved to:")
print(" -", output_mistral)
print(" -", output_qwen)