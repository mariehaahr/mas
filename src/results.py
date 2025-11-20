import pandas as pd
from pathlib import Path
import yaml

# claim_id, receiver_model, sender_model, sarc_ratio_r1, sarc_ratio_r2, valid_json_count_r2
# delta, flip, flip_direction

# results path
outpath = Path((f"/home/rp-fril-mhpe/results.csv"))


# load lookup tables
lookup_r1_path = Path("/home/rp-fril-mhpe/first-results-sarc-ratio.csv").expanduser()
lookup_r2_path = Path("/home/rp-fril-mhpe/second-results-sarc-ratio.csv").expanduser()
lookup1 = pd.read_csv(lookup_r1_path) 
lookup2 = pd.read_csv(lookup_r2_path) 

print("sanity_check")
print("lookup 1: \n",lookup1["model"].value_counts())


print("lookup2: \n",lookup2[["model_sender", "model_receiver"]].value_counts())
print("---------------------")

# load model names
with open("configs/models.yaml", "r") as f:
    data = yaml.safe_load(f)

model_names = list(data["profiles"].keys())
# remove llama 1b from the models                                                                       #TODO: new
model_names.remove("llama-3.2-1b")

def flip_direction_fn(r1, r2):
    """
    a helper function to make the column "flip_direction" so we can sort on that
    for 0->1 or 1->0.

    up is going from 0->1 
    down is goin from 1->0
        else none
    """
    if pd.isna(r1) or pd.isna(r2):
        return "none"
    if r1 < 0.5 and r2 >= 0.5:
        return "up"
    if r1 >= 0.5 and r2 < 0.5:
        return "down"
    return "none"


all_results = [] # gonna collect all dfs in here and concat in the end

for receiver in model_names: # loop over all the models, as the receiver
    print(f"Processing receiver: {receiver}")
    # filter out the reciever in the lookuptable from round 1
    r1 = (
        lookup1[lookup1["model"] == receiver]
        .rename(columns={"sarc_ratio": "round1_sarc_ratio", "model": "model_receiver"})
        [["id", "model_receiver", "round1_sarc_ratio"]]       # drop valid_json_count from r1
        .copy()
    )

    # filter out the receiver in the loopuptable from round 2, too
    r2 = (
        lookup2[lookup2["model_receiver"] == receiver]
        .rename(columns={
            "sarc_ratio": "round2_sarc_ratio",
            "valid_json_count": "valid_json_count_r2"
        })
        [["id", "model_receiver", "model_sender", "valid_json_count_r2", "round2_sarc_ratio"]]
        .copy()
    )

    # merge on id
    merged = r2.merge(r1, on=["id", "model_receiver"], how="left")

    # calculate delta
    merged["delta_sarc"] = merged["round2_sarc_ratio"] - merged["round1_sarc_ratio"]

    # use flip_direction function to make flip_direction
    merged["flip_direction"] = merged.apply(
        lambda row: flip_direction_fn(row["round1_sarc_ratio"], row["round2_sarc_ratio"]),
        axis=1
    )
    # if the flip_direction is either up or down, there was a flip
    merged["flip"] = (merged["flip_direction"] != "none").astype(int)

    all_results.append(merged)

    

all_results_df = pd.concat(all_results, ignore_index=True)
all_results_df.to_csv(outpath, index=False)

print(f"Done.\n Saved as {outpath}")