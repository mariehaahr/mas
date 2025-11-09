import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Read in results
models = [
    'qwen-2.5-1.5b'
    ,'qwen-2.5-7b'
    ,'llama-3.2-1b'
    # ,'llama-3.2-3b',
    #,'mistral-0.3-7b',
    #,'mistral-0.2-7b'
]


dfs = []
for model_name in models:
    results_path = f'results/first-{model_name}.csv'
    try:
        df = pd.read_csv(results_path)
        dfs.append(df)
    except FileNotFoundError:
        print(f'File not found: {results_path} -- skipping')
    except Exception as e:
        print(f'Error loading file: {e}')

# concat all the dfs into one master df 
master_df = pd.concat(dfs, ignore_index=True)
print('Concatenated all dfs into master df')


# Group by 'model', 'id' and create aggregated measures: count(label == 'sarcastic') call this sarcastic_count, count(valid_json == True) call this valid_count, max(confidence), min(confidence), mean(confidence)
aggregated = (
    master_df.groupby(['model', 'id'], as_index=False)
    .agg(
        sarcastic_count=('label', lambda x: (x == 'sarcastic').sum()),
        valid_count=('valid_json', lambda x: (x == True).sum()),
        max_confidence=('confidence', 'max'),
        min_confidence=('confidence', 'min'),
        mean_confidence=('confidence', 'mean'),
    )
)

aggregated['sarcasm_ratio'] = aggregated['sarcastic_count'] / aggregated['valid_count']


print(aggregated.head(10))

print('Number of rows with 0 < sarcasm_ratio < 1')
print(aggregated[(aggregated['sarcasm_ratio'] != 1.0) & (aggregated['sarcasm_ratio'] != 0)].shape)
print('out of')
print(aggregated.shape[0])

model_summary = (
    aggregated.groupby('model', as_index=False)
    .agg(
        mean_sarcasm_ratio=('sarcasm_ratio', 'mean'),
        std_sarcasm_ratio=('sarcasm_ratio', 'std'),
        median_sarcasm_ratio=('sarcasm_ratio', 'median'),
        min_sarcasm_ratio=('sarcasm_ratio', 'min'),
        max_sarcasm_ratio=('sarcasm_ratio', 'max')
    )
)
print('model summary')
print(model_summary)

sentence_comparison = (
    aggregated.groupby('id', as_index=False)
    .agg(
        sarcasm_ratio_std=('sarcasm_ratio', 'std'),
        sarcasm_ratio_mean=('sarcasm_ratio', 'mean'),
        sarcasm_ratio_min=('sarcasm_ratio', 'min'),
        sarcasm_ratio_max=('sarcasm_ratio', 'max')
    )
)
print('Sentence comparison')
print(sentence_comparison)

pivot = aggregated.pivot(index='id', columns='model', values='sarcasm_ratio')
corr = pivot.corr()
print('corr - do models tend to agree on sarcasm ratio')
print(corr)

sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation of sarcasm ratios between models")
plt.tight_layout()
plt.savefig('plots/corr_models_sarcasmratio.png', dpi=300)

# Plot 
# Set Seaborn style
# sns.set(style='whitegrid', context='talk')

# # Create the scatter plot
# plt.figure(figsize=(12, 6))
# sns.scatterplot(
#     data=aggregated,
#     x='id',
#     y='sarcasm_ratio',
#     hue='model',
#     alpha=0.8,
#     s=60,
#     edgecolor=None
# )

# # Labels and title
# plt.xlabel('ID')
# plt.ylabel('Sarcastic Count / Valid Count')
# plt.title('Sarcasm Ratio per ID for Each Model')
# plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()

# # Save and show
# plt.savefig('results/sarcasm_ratio_scatter.png', dpi=300)
# plt.show()