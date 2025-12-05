import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pathlib, yaml
import matplotlib as mpl
from itertools import combinations
import math

# For displaying results in output file 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Set theme across all plots 
sns.set_theme(
    style="ticks",
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


def prepare_heatmap_df():
    profiles_root = yaml.safe_load(pathlib.Path('configs/models.yaml').read_text())
    profiles = profiles_root.get('profiles', {})
    model_names = list(profiles.keys())
    model_names.remove('llama-3.2-1b')

    dfs = [] 
    for model_n in model_names:
        p = f'/home/rp-fril-mhpe/input_{model_n}.csv'
        try:
            df = pd.read_csv(p, low_memory=False)
            dfs.append(df)
        except FileNotFoundError:
            print(f'File not found: {p}')
        except Exception as e:
            print(f'Error loading file: {p}, {e}')

    combined = pd.concat(dfs, ignore_index=True)


    combined['label_sender_agg'] = df['sarc_ratio_receiver'].apply(
            lambda x: 'sarcastic' if x >= 0.5 else 'literal'
             )
    heatmap_df = (
    combined
    .groupby(['model_sender', 'model_receiver', 'label_receiver_agg', 'id'])
    .size()
    .reset_index(name='count')
    )
    heatmap_df.to_csv('/home/fril/mas/results/input-r2-claim-count.csv', index= False)
    print(f'Saved df with {heatmap_df.shape[0]} rows.')

def plot_sarc_distribution(df, kde = True):
    '''
    Plotting the sarcasm ratio distribution of results in round 1. 
    '''

    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(16,12))

    models = df['model'].unique()
    for model_name, ax in zip(models, axs.ravel()):
        model_res = df[df['model'] == model_name].copy()

        if kde:
            sns.kdeplot(data=model_res, ax=ax, x = 'sarc_ratio', fill=True)
        else:
            sns.histplot(data=model_res, ax=ax, x = 'sarc_ratio')

        ax.set_title(f'{model_name}')
        ax.set_xlabel('Sarcasm Ratio')

    plt.tight_layout()
    sns.despine()
    plt.savefig('plots/label-dist-all.png', dpi = 300, bbox_inches='tight')


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
        counts['value'] = (counts['count'] / counts.groupby('model')['count'].transform('sum')) * 100

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

    ax.set_ylabel('Percentage' if perc else 'Count')
    ax.set_xlabel('Model')
    plt.legend(title= "Label", loc= "upper right")
    ax.set_title('Prediction Label Distribution by Model')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig('plots/label-distr-r1.png', dpi = 300, bbox_inches='tight')

def plot_valid_json_distribution(df, perc=True):
    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(16,12), sharey=True)
    
    models = df['model'].unique()
    
    for model_name, ax in zip(models, axs.ravel()):
        model_res = df[df['model'] == model_name].copy()

        if perc: 
            counts = model_res['valid_json_count'].value_counts(normalize=True).sort_index()
            counts_perc = counts * 100
            sns.barplot(x=counts.index, y=counts_perc, ax=ax)
            ax.set_ylabel('Percentage')

        else: 
            sns.countplot(data=model_res, ax=ax, x = 'valid_json_count')
            ax.set_ylabel('Count')

        ax.set_title(f'{model_name}')
        ax.set_xlabel('Valid output count')

    sns.despine()
    plt.tight_layout()
    plt.savefig('plots/valid-json-dist-r1.png', dpi = 300, bbox_inches='tight')
    print('Valid output counts: \n')
    print(counts)
    

def plot_input_r2():
    profiles_root = yaml.safe_load(pathlib.Path('configs/models.yaml').read_text())
    profiles = profiles_root.get('profiles', {})
    model_names = list(profiles.keys())
    model_names.remove('llama-3.2-1b')

    dfs = [] 
    for model_n in model_names:
        p = f'/home/rp-fril-mhpe/input_{model_n}.csv'
        try:
            df = pd.read_csv(p, low_memory=False)
            dfs.append(df)


        except FileNotFoundError:
            print(f'File not found: {p}')
        except Exception as e:
            print(f'Error loading file: {p}, {e}')

    combined = pd.concat(dfs, ignore_index=True)

    heatmap_df = pd.crosstab(
        combined['model_sender'],
        combined['model_receiver']
    )

    print('\nSender x Receiver count matrix:')
    print(heatmap_df)


    plt.figure()

    ax = sns.heatmap(
        heatmap_df,
        cmap='Blues',
        annot=True,
        annot_kws={"fontsize":6},
        fmt=".0f",
        cbar_kws={'shrink': 0.8}
    )


    # ax.xaxis.set_label_position('top')
    # ax.xaxis.tick_top()

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha='right',
        rotation_mode='anchor'
    )

    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)

    
    plt.title('Count of (sender, receiver) pairs', pad=30)
    plt.xlabel('Receiver')
    plt.ylabel('Sender')

    plt.tight_layout()
    plt.savefig('plots/heatmap-input-second.png', dpi = 300, bbox_inches="tight")



def plot_heatmaps_sarc_ratio(df):
    df = df.copy()
    df.dropna(inplace=True)
    df['sarc_ratio_round'] = (df['sarc_ratio'] * 10).astype(int) / 10

    pivot = df.pivot_table(
    index='id',
    columns='model',
    values='sarc_ratio_round'
    )

    bins = [i/10 for i in range(11)]

    heatmaps = {}
    models = pivot.columns

    for m1, m2 in combinations(models, 2):
        
        sub = pivot[[m1, m2]].dropna()

        # get counts 
        H = pd.crosstab(sub[m1], sub[m2])
        
        # make sure all ranges are present even though no occurences 
        H = H.reindex(index=bins, columns=bins, fill_value=0)

        total = H.to_numpy().sum()
        if total > 0:
            H_pct = H / total
        else:
            H_pct = H.astype(float) #all values zero

        heatmaps[(m1, m2)] = H_pct


    n = len(heatmaps)
    n_cols = 3
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 6 * n_rows),
                             squeeze=False)
    
    
    for ax, ((m1, m2), H_pct) in zip(axes.flat, heatmaps.items()):
        
        annot_matrix = H_pct.copy().astype(object)
        annot_matrix = annot_matrix.map(
            lambda v: "" if round(v, 5) == 0 else f"{v:.0e}"
            )

        hm = sns.heatmap(
            H_pct,
            ax=ax,
            annot=annot_matrix,
            fmt='',
            cmap='Blues',
            vmin=0,
            vmax=1,
            cbar = True,
            annot_kws={"fontsize":6},
        )

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        
        ax.set_xlabel(m2)
        ax.set_ylabel(m1)
        
    
    for ax in axes.flat[len(heatmaps):]:
        ax.set_visible(False)


    fig.tight_layout()
    plt.savefig('plots/heatmaps-first-results.png', dpi=300, bbox_inches="tight")




def main():
    sarc_ratio_df = pd.read_csv('/home/rp-fril-mhpe/first-results-sarc-ratio.csv')

    # plot_sarc_distribution(sarc_ratio_df)
    # plot_label_distribution(sarc_ratio_df)
    # plot_valid_json_distribution(sarc_ratio_df)
    # plot_heatmaps_sarc_ratio(sarc_ratio_df)
    # plot_input_r2()
    
    prepare_heatmap_df()

if __name__ == '__main__':
    main()

