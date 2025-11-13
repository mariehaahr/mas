import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pathlib, yaml

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def label_distribution(sarcasm_ratio_df):
    sarcasm_ratio_df["label"] = np.where(sarcasm_ratio_df["sarc_ratio"] >= 0.5, "sarcastic", "literal")

    
    pass




def main():
    profiles_root = yaml.safe_load(pathlib.Path('configs/models.yaml').read_text())
    profiles = profiles_root.get('profiles', {})
    model_names = profiles.keys()


    sarc_ratio_df = pd.read_csv('/home/rp-fril-mhpe/sarcasm-ratio.csv')


    

    dfs = []
    # read in all results
    for m_name in model_names:
        p = f'/home/rp-fril-mhpe/first-{m_name}.csv'

        try:
            df = pd.read_csv(p)
            dfs.append(df)
        except FileNotFoundError:
            print(f'File not found: {p}')
        except Exception as e:
            print(f'Error loading file: {p}, {e}')

    combined = pd.concat(dfs, ignore_index=True)



    # group by model, id
    aggregated = (
    combined.groupby(['model', 'id'], as_index=False)
    .agg(
        sarc_count=('label', lambda x: (x == 'sarcastic').sum()),
        valid_json_count=('valid_json', lambda x: (x == True).sum()),
        )
    )
    aggregated['sarc_ratio'] = aggregated['sarc_count'] / aggregated['valid_json_count']
    aggregated.drop('sarc_count', inplace=True, axis = 1)


if __name__ == '__main__':
    main()