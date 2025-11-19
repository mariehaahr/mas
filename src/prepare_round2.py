# to model no need for expl
# id, claim, sender_model_name, sender_model_label, sender_model_ratio,
# receiver_model_name, receiver_model_label, receiver_model_explanation, receiver_model_ratio
import yaml
import sys
import pandas as pd
from pathlib import Path

# receiver er 1 model, som skal have alle andre senders

# argument -> receiver model
receiver = sys.argv[1]  # give the name of the receiver as argument in the run file
lookup_path = Path("/home/rp-fril-mhpe/first-results-sarc-ratio.csv").expanduser()
results_dir = Path("/home/rp-fril-mhpe").expanduser()

with open("configs/models.yaml", "r") as f:
    data = yaml.safe_load(f)


model_names = list(data["profiles"].keys())
# remove llama 1b from the models                                                                       #TODO: new
model_names.remove("llama-3.2-1b")

all_models = [f"first-{m}" for m in model_names] # TODO: 

senders = [m for m in all_models if m != receiver] # all models but the receiver



lookup = pd.read_csv(lookup_path) # lookup table frida made

# receiver filter
receiver_name = receiver.replace("first-", "")
receiver_lookup = lookup[lookup["model"] == receiver_name].copy()

# filter the "extremes"
interesting_ids = receiver_lookup.loc[
    (receiver_lookup["sarc_ratio"] <= 0.25)
    | (receiver_lookup["sarc_ratio"] >= 0.75),
    "id"
].unique()

# list of dfs to concat in the end
dfs = []

# loop for all senders: 
for sender in senders: 
    sender_name = sender.replace("first-", "")
    print(f"Processing sender: {sender_name}") # for the log

    sender_lookup = lookup[lookup["model"] == sender_name].copy()
    # mask the interesting claims
    sender_lookup = sender_lookup[sender_lookup["id"].isin(interesting_ids)]

    # merge receiver and sender lookups
    merged = receiver_lookup.merge(
        sender_lookup,
        on="id",
        suffixes=("_receiver", "_sender")
    )

    # remove rows where either receiver or sender has valid_json < 5.                                    #TODO: new
    merged = merged[(merged["valid_json_count_receiver"] >= 5) & (merged["valid_json_count_sender"] >= 5)]

    # masking again, but where they disagree
    mask = (
        ((merged["sarc_ratio_receiver"] >= 0.75) & (merged["sarc_ratio_sender"] <= 0.25))
        | ((merged["sarc_ratio_sender"] >= 0.75) & (merged["sarc_ratio_receiver"] <= 0.25))
    )
    merged_disagree = merged[mask].copy()

    # now loading the results of this sender, to save the explanation
    sender_results = pd.read_csv(results_dir / f"{sender}.csv")

    # remove not valid explanation, where valid_json = False
    sender_results = sender_results[sender_results["valid_json"] == True]

    # merge explanations onto
    df = merged_disagree.merge(
        sender_results[["id", "label", "explanation"]],
        on="id",
        how="left"
    )
    # rename columns
    df = df.rename(columns={"label":"label_sender", "explanation": "explanation_sender"})
    
    # append
    dfs.append(df)
    

# concat all df into one big df
if dfs:
    final_df = pd.concat(dfs, ignore_index=True)

    # loading the claims, so we can put these in the final df
    claims_path = Path("data/sarc/sarcasm2.csv")
    claims_df = pd.read_csv(claims_path, usecols=["id", "text"])  # only load what's needed
    claims_df = claims_df.rename(columns={"text": "claim"})

    # merge onto final_df
    final_df = final_df.merge(claims_df, on="id", how="left")
    
    # save output
    output_path = results_dir / f"input_{receiver_name}.csv"
    print(final_df.shape)
    print(final_df.columns)
    final_df.to_csv(output_path, index=False)
    print(f"Saved the final dataframe to {output_path}")
else:
    print("No disagreements found.... dfs is empty")
