import pandas as pd 


# Kode der bruges til at concatenate vores resultater fordi de er i to runs, og gemme dem i results, efter det skal de flyttes til shared folder 

# first = pd.read_csv('results/first-llama-3.2-1b-old.csv')
# first = pd.read_csv('results/first-qwen-2.5-7b-old.csv')
first = pd.read_csv('results/first-qwen-2.5-7b-old.csv')

# second = pd.read_csv('results_extra/first-llama-3.2-1b.csv')
# second = pd.read_csv('results_extra/first-qwen-2.5-7b.csv')
second = pd.read_csv('results_extra/first-qwen-2.5-7b.csv')


# Make sure they have the same columns 
if not first.columns.equals(second.columns):
    print("The two CSV files do not have the same columns!")
    print("Columns in first but not second:", set(first.columns) - set(second.columns))
    print("Columns in second but not first:", set(second.columns) - set(first.columns))

else:
    combined = pd.concat([first, second], ignore_index=True)

    combined = combined.sort_values(by='id').reset_index(drop=True)
    combined.to_csv('results/first-qwen-2.5-7b.csv', index=False)


print(combined.shape)
sarcasm2 = pd.read_csv('data/sarc/sarcasm2.csv')

set1 = set(combined['id'])
set2 = set(sarcasm2['id'])

difference = set1.symmetric_difference(set2)

print(f'difference: {difference}')

print(f'sarcasm shape {sarcasm2.shape}')
print(f'combines shape/10 {combined.shape[0]/10}')
