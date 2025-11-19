import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pathlib, yaml
import matplotlib as mlp

# For displaying results in output file 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Set theme across all plots 
sns.set_theme(
    style="whitegrid",
    palette="Paired",     
    font_scale=1.2       
)

# matplotlib params 
mpl.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 10,
})


def plot_sarc_distribution(df, kde = True):
    '''
    Plotting the sarcasm ratio distribution of results in round 1. 
    '''

    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(16,12), sharey=True)

    models = df['model'].unique()
    for model_name, ax in zip(models, axs.ravel()):
        model_res = df[df['model'] == model_name].copy()

        if kde:
            sns.kdeplot(data=model_res, ax=ax, x = 'sarc_ratio')
        else:
            sns.histplot(data=model_res, ax=ax, x = 'sarc_ratio')

        ax.set_title(f'{model_name}')

    plt.tight_layout()
    sns.despine()
    plt.savefig('results/plots/label-dist-all.png', dpi = 300, bbox_inches='tight')


def plot_label_distribution(df, perc=True):
    df = df.copy()

    df['label'] = df['sarc_ratio'].apply(
        lambda x: 'sarcastic' if x >= 0.5 else 'literal'
    )
    df['label'] = df['label'].astype(
        pd.CategoricalDtype(categories=['sarcastic', 'literal'])
    )

    counts = df.groupby(['model', 'label'], observed=False).size().reset_index(name='count')

    if perc:
        counts['value'] = (counts['count'] / counts.groupby('model')['count'].transform('sum'))

    else:
        counts['value'] = counts['count']

    # sort according to value
    sorted_models = counts.sort_values(by = 'value', ascending=False).model.to_list()
    print(f'Label counts for each model: {counts}')

    plt.figure(figsize=(12, 6))

    ax = sns.barplot(
        data=counts,
        x='model',
        y='value',
        hue='label',
        hue_order=['sarcastic', 'literal'],
        order=sorted_models)

    
    sns.despine()

    ax.ylabel('Percentage' if perc else 'Count')
    ax.xlabel('Model')
    ax.title('Prediction Label Distribution by Model')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig('results/plots/label-distr-r1.png', dpi = 300, bbox_inches='tight')

def plot_valid_json_distribution(df, perc=True):
    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(16,12), sharey=True)
    
    models = df['model'].unique()
    
    for model_name, ax in zip(models, axs.ravel()):
        model_res = df[df['model'] == model_name].copy()

        if perc: 
            counts = model_res['valid_json_count'].value_counts(normalize=True).sort_index()
            sns.barplot(x=counts.index, y=counts.values, ax=ax)
            ax.set_ylabel('Percentage')

        else: 
            sns.countplot(data=model_res, ax=ax, x = 'valid_json_count')
            ax.set_ylabel('Count')

        ax.set_title(f'{model_name}')
        ax.set_xlabel('Valid output count')

    sns.despine()
    plt.tight_layout()
    plt.savefig('results/plots/valid-json-dist-r1.png', dpi = 300, bbox_inches='tight')

    

def check_input_r2(df):
    pass 

def main():
    sarc_ratio_df = pd.read_csv('/home/rp-fril-mhpe/sarcasm-ratio.csv')

    plot_sarc_distribution(sarc_ratio_df)
    plot_label_distribution(sarc_ratio_df)
    plot_valid_json_distribution(sarc_ratio_df)


if __name__ == '__main__':
    main()
