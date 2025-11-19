import pandas as pd
from pathlib import Path

# Paths
mistral_path = Path("/home/rp-fril-mhpe/second-mistral-0.3-7b.csv")
qwen_path    = Path("/home/rp-fril-mhpe/second-qwen-2.5-7b.csv")
lookup1_path = Path("/home/rp-fril-mhpe/first-results-sarc-ratio.csv")

output_mistral = Path("/home/rp-fril-mhpe/clean-second-mistral-0.3-7b.csv")
output_qwen    = Path("/home/rp-fril-mhpe/clean-second-qwen-2.5-7b.csv")

# Load data
df_m = pd.read_csv(mistral_path)
df_q = pd.read_csv(qwen_path)

before_m = len(df_m)
before_q = len(df_q)

# Load lookup table
lookup1 = pd.read_csv(lookup1_path)

# -----------------------------
# 1. Find IDs with valid_json < 5
# -----------------------------
bad_ids = lookup1.loc[lookup1["valid_json_count"] < 5, "id"].unique()

print(f"Found {len(bad_ids)} ids with <5 JSONs in round 1.")

# -----------------------------
# 2. Remove those IDs from the round-2 data
# -----------------------------
df_m = df_m[~df_m["id"].isin(bad_ids)].copy()
df_q = df_q[~df_q["id"].isin(bad_ids)].copy()

# -----------------------------
# 3. Remove rows where sender = llama-3.2-1b
# -----------------------------
bad_sender = "llama-3.2-1b"

df_m = df_m[df_m["model_sender"] != bad_sender].copy()
df_q = df_q[df_q["model_sender"] != bad_sender].copy()

after_m = len(df_m)
after_q = len(df_q)

# Save
df_m.to_csv(output_mistral, index=False)
df_q.to_csv(output_qwen, index=False)

# Logs
print("\n=== DONE ===")
print(f"Mistral before: {before_m}, after: {after_m}, removed: {before_m - after_m}")
print(f"Qwen    before: {before_q}, after: {after_q}, removed: {before_q - after_q}")

print("Saved to:")
print(" -", output_mistral)
print(" -", output_qwen)
