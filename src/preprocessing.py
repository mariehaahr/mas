import pandas as pd 
from pathlib import Path
import re

txt_path = Path('data/sarc/raw/train_text.txt')
with txt_path.open(encoding="utf-8") as f:
    lines = [line.rstrip("\n") for line in f]


labels_path = Path('data/sarc/raw/train_labels.txt')
with labels_path.open(encoding='utf-8') as f:
    labels = [label.rstrip('\n') for label in f]

sarcasm = pd.DataFrame({'id': list(range(0, len(lines))),"text": lines, 'label': labels})

# count the number of words in each claim 
sarcasm['n_words'] = [len(x.split()) for x in sarcasm['text'].tolist()]

######### This is the dataset with the name 'sarcasm.csv'
# filter out too short and too long examples 
sarcasm_pr = sarcasm[(sarcasm['n_words'] >= 6) & (sarcasm['n_words'] < 500)]

# remove examples that contain mentions of Israel, israeli, muslim, hillary, racism, racists, trump, russian, russia, slavery, jesus, gay, lesbian, homosexuality, communist, communism, terrorism, palestine, palestianians, rape, conservative, democrat, republican, antifa  
BASE_TERMS = [
    r"ukrain\w*", r"russi\w*", r"putin", r"zelensky\w*",
    r"iran\w*", r"iraq\w*", r"syria\w*", r"afghan\w*",
    r"gaza", r"west\s*bank", r"hamas", r"hezbollah", r"isis", r"taliban", r"idf",
    r"biden", r"obama", r"clinton", r"bush", r"reagan", r"sanders", r"aoc", r"desantis", r"pence", r"hillary",
    r"democrat\w*", r"republican\w*", r"liberal\w*", r"progressive\w*", r"conservativ\w*",
    r"libertarian\w*", r"socialist\w*", r"marxist\w*", r"fascist\w*", r"nazi\w*", r"alt-?right", r"woke",
    r"fascism", r"communism", r"socialism", r"liberalism", r"conservatism", r"antifa",
    r"islam\w*", r"muslim\w*", r"christian\w*", r"catholic\w*", r"jewish", r"judais\w*", r"zionis\w", r"israel", r"palestine", r"palestinians",
    r"hindu\w*", r"sikh\w*", r"buddhist\w*", r"buddhis\w*", r"atheis\w*",
    r"trans(?:gender)?\w*", 
    r"non-?binary", r"lgbt\w*", r"queer", r"bisexual\w*", r"pansexual\w*", r"pronoun\w*",
    r"immigra\w*", r"migrant\w*", r"refugee\w*", r"abortion", r"pro-?life", r"pro-?choice", r"suicide"
    r"gun\w*", r"firearm\w*", r"nra", r"racist\w",
    r"covid\w*", r"coronavirus", r"pandemic", r"lockdown\w*", r"vaccine\w*", r"anti-?vax\w*",
    r"hitler", r"stalin", r"genocide\w*", r"holocaust", r"apartheid",
    r"climate\s*change", r"global\s*warming", r"feminis\w*", r"#metoo",
]

pattern = re.compile(r"\b(?:%s)\b" % "|".join(BASE_TERMS), flags=re.IGNORECASE)
mask_has_banned = sarcasm_pr["text"].str.contains(pattern, na=False)


################ this is the dataset we call sarcasm2.csv ################### 
sarcasm_clean = sarcasm_pr[~mask_has_banned].copy()
flagged = sarcasm_pr[mask_has_banned].copy()

# print('Label distribution original')
# print(sarcasm['label'].value_counts())

# print('Label distribution processed')
# print(sarcasm_clean['label'].value_counts())

# print('Size of processed dataset')
# print(sarcasm_clean.shape)

# sarcasm_clean.to_csv('data/sarc/sarcasm2.csv', index=False)
# flagged.to_csv('data/sarc/flagged2.csv', index=False)


# taking a sample, with balanced labels 
SAMPLE_SIZE = 50_000

############# This is the dataset we call sarcasm50k.csv ################################## 
balanced_df = (
    sarcasm_clean.groupby('label', group_keys=False)
      .apply(lambda x: x.sample(int(SAMPLE_SIZE/2), random_state=42))
)
# print(balanced_df.shape)
# print(balanced_df['label'].value_counts())
# print(balanced_df.head(10))
# print(balanced_df.head(-10))
# print(sarcasm_clean.head(10))

# balanced_df.to_csv('data/sarc/sarcasm50k.csv')


#################################################################################
#
#     Kode brugt til at fikse mismatch mellem model resultater og data
#                 Slettes nok senere            
#                                                                                    
#                                                                                   
#################################################################################

# llama_results = pd.read_csv('results/first-llama-3.2-1b.csv')
# sarcasm2 = pd.read_csv('data/sarc/sarcasm2.csv')

# llama_results.head()
# sarcasm2.head()

# # I sort out all the examples from sarcasm 2 that are already in llama results. 
# sarcasm2_rest = sarcasm2[~sarcasm2['id'].isin(llama_results['id'].unique())]
# print('Number of datapoints in sarcasm for llama run.')
# print(sarcasm2_rest.shape)
# sarcasm2_rest.to_csv('data/sarc/sarcasm2_llama.csv', index = False)


# # I sort out all the examples from sarcasm2 that are in sarcasm50k. 
# sarcasm50k = pd.read_csv('data/sarc/sarcasm50k.csv')
# sarcasm2_50k_rest = sarcasm2[~sarcasm2['id'].isin(sarcasm50k['id'].unique())]
# print('Number of datapoints in sarcasm for the two qween run')
# print(sarcasm2_50k_rest.shape)
# sarcasm2_50k_rest.to_csv('data/sarc/sarcasm2-minus-50k.csv', index = False)