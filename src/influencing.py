from collections import defaultdict # storing results
import pandas as pd # dataframe
from pathlib import Path # loading data
import yaml # loading model names from the yaml files

# results file
resultspath = Path("/home/rp-fril-mhpe/results.csv").expanduser()
# load
results = pd.read_csv(resultspath)

# to store the flip percentage, count, and total count (that made that %)

# total = {"influencer1": [70%, 70, 100], "influencer2": [30%, 300, 1000] ...}
total = defaultdict(list) 
# per_sender = {influencer1: {receiver1: [x, y, z]}, receiver2...} influencer2...}}
per_receiver = defaultdict(lambda: defaultdict(list))

up = defaultdict(lambda: defaultdict(list)) # -..-
down = defaultdict(lambda: defaultdict(list)) # -..-

# load model names from yaml
with open("configs/models.yaml", "r") as f:
    data = yaml.safe_load(f)

model_names = list(data["profiles"].keys())
# remove llama 1b from the models                                                                       #TODO: new
model_names.remove("llama-3.2-1b")


# lets sanity check the model
# print("***************************************************")
# print("sanity check")

# print("value counts: ")
# print(results.value_counts(["model_receiver", "model_sender"]))


for influencer in model_names:
    # mask results where model_sender == influencer
    mask1 = results[results["model_sender"] == influencer]
    # make sure no division by 0 mistake
    if len(mask1) == 0:
        avg_flip = 0
    else:
        avg_flip = sum(mask1["flip"]) / len(mask1)

    total[influencer].append(avg_flip) # save the flip %
    total[influencer].append(sum(mask1["flip"])) # save the # of flips
    total[influencer].append(len(mask1)) # save the number of possible flips

    # now looking into each model
    for receiver in model_names:
        if receiver == influencer: # cant influence yourself, so skip
            continue
        # doing the same as above, now just for this specific receiver
        mask2 = mask1[mask1["model_receiver"] == receiver]
        if receiver == "qwen-2.5-1.5b":
            print("printing the mask",mask2)
            print("printing the unique values:")
            print(mask1["model_receiver"].unique())
        if len(mask2) == 0:
            avg_flip = 0
        else:
            avg_flip = sum(mask2["flip"]) / len(mask2)

        per_receiver[influencer][receiver].append(avg_flip)
        per_receiver[influencer][receiver].append(sum(mask2["flip"]))
        per_receiver[influencer][receiver].append(len(mask2))
    

        # now dividing into up and down (aka 0->1 and 1->0)
        upmask = mask2[mask2["flip_direction"] == "up"]
        downmask = mask2[mask2["flip_direction"] == "down"]

        upandnone = mask2[mask2["round1_sarc_ratio"] < 0.5]
        # print(upandnone)
        # print(upandnone.shape)
        downandnone = mask2[mask2["round1_sarc_ratio"] >= 0.5]
        # print(downandnone)
        # print(downandnone.shape)

        if len(upandnone) == 0:
            up_avg_flip = 0
        else:
            up_avg_flip = sum(upmask["flip"]) / len(upandnone)
        
        up[influencer][receiver].append(up_avg_flip)
        up[influencer][receiver].append(sum(upmask["flip"]))
        up[influencer][receiver].append(len(upandnone))
        
        if len(downandnone) == 0:
            down_avg_flip = 0
        else:
            down_avg_flip = sum(downmask["flip"]) / len(downandnone)
        
        down[influencer][receiver].append(down_avg_flip)
        down[influencer][receiver].append(sum(downmask["flip"]))
        down[influencer][receiver].append(len(downandnone))

# saving the results

# turning the dicts into dataframes
df_total = (
    pd.DataFrame.from_dict(total, orient="index", 
                           columns=["fliprate", "flip_count", "total_count"])
      .reset_index()
      .rename(columns={"index": "model_sender"})
)
print(f"\n\n\ndf total:")
print(df_total)
# also the dict(dict(list))
rows = []

for sender, receivers in per_receiver.items():
    for receiver, values in receivers.items():
        rows.append([sender, receiver] + values)

df_per_receiver = pd.DataFrame(rows, columns=["model_sender", "model_receiver", "fliprate", "flip_count", "total_count"])

rows = []

print(f"\n\n\ndf per receiver:")
print(df_per_receiver)

for sender, receivers in up.items():
    for receiver, values in receivers.items():
        rows.append([sender, receiver] + values)

df_up = pd.DataFrame(rows, columns=["model_sender", "model_receiver", "fliprate", "flip_count", "total_count"])

print(f"\n\n\ndf up:")
print(df_up)
rows = []



for sender, receivers in down.items():
    for receiver, values in receivers.items():
        rows.append([sender, receiver] + values)

df_down = pd.DataFrame(rows, columns=["model_sender", "model_receiver", "fliprate", "flip_count", "total_count"])
print(f"\n\n\ndf down:")
print(df_down)

df_total.to_csv(Path((f"/home/rp-fril-mhpe/total_results.csv")))
df_per_receiver.to_csv(Path((f"/home/rp-fril-mhpe/per_receiver_results.csv")))
df_up = Path((f"/home/rp-fril-mhpe/up_results.csv"))
df_down = Path((f"/home/rp-fril-mhpe/down_results.csv"))

# printing the results
print("******************************************************\n")
print("------------------------------- RESULTS ---------------------------")



for model in model_names:
    print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print("                                OVERALL RESULTS                              ")
    print(f"---------LOOKING AT MODEL {model}")
    print(f"Flip %: {(total[model][0]) * 100}%")
    print(f"Number of flips: {total[model][1]}")
    print(f"Out of: {total[model][2]}")

    print(f"                    WHICH MODELS DID {model} IT INFLUENCE MOST?              ")
    for receiver in model_names:
        if receiver == model:
            continue
        print(f"------------------------FOR RECEIVER: {receiver}")
        print(f"Flip %: {(per_receiver[model][receiver][0])*100}%")
        print(f"Number of flips: {per_receiver[model][receiver][1]}")
        print(f"Out of {per_receiver[model][receiver][2]}")

print("-----------------------------------------------------------------------------------------")
print("------------------------- LOOKING AT 0 -> 1, THE LABEL GOING UP -------------------------")

print("from literal to sarcastic")

for model in model_names:
    print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print(f"---------LOOKING AT MODEL {model}")

    for receiver in model_names:
        if receiver == model:
            continue
        print(f"------------------------FOR RECEIVER: {receiver}")
        print(f"Flip %: {(up[model][receiver][0])*100}%")
        print(f"Number of flips: {up[model][receiver][1]}")
        print(f"Out of {up[model][receiver][2]}")


print("-----------------------------------------------------------------------------------------")
print("------------------------- LOOKING AT 1 -> 0, THE LABEL GOING DOWN -------------------------")

print("from sarcastic to literal")

for model in model_names:
    print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print(f"---------LOOKING AT MODEL {model}")

    for receiver in model_names:
        if receiver == model:
            continue
        print(f"------------------------FOR RECEIVER: {receiver}")
        print(f"Flip %: {(down[model][receiver][0])*100}%")
        print(f"Number of flips: {down[model][receiver][1]}")
        print(f"Out of {down[model][receiver][2]}")