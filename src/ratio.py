import pathlib, yaml
import pandas as pd 
import argparse

def make_ratio(round_num: int):

    if round_num not in (1, 2):
        raise ValueError(f'round_num must be either 1 or 2, got {round_num}')
    

    profiles_root = yaml.safe_load(pathlib.Path('configs/models.yaml').read_text())
    profiles = profiles_root.get('profiles', {})
    model_names = profiles.keys()

    dfs = []
    # read in all results
    for m_name in model_names:
        if round == 1: 
            p = f'/home/rp-fril-mhpe/first-{m_name}.csv'
        else:
            p = f'/home/rp-fril-mhpe/second-{m_name}.csv'

        try:
            df = pd.read_csv(p)
            dfs.append(df)
        except FileNotFoundError:
            print(f'File not found: {p}')
        except Exception as e:
            print(f'Error loading file: {p}, {e}')

    combined = pd.concat(dfs, ignore_index=True)

    if round_num == 1:
        group_cols = ['model', 'id']
        label_col = 'label'
        save_as = 'first-results-sarc-ratio.csv'

    else:
        group_cols = ['model_sender', 'id', 'model_receiver']
        label_col = 'label_receiver_new'
        save_as = 'second-results-sarc-ratio.csv'

    # group by cols
    aggregated = (
    combined.groupby(group_cols, as_index=False)
    .agg(
        sarc_count=(label_col, lambda x: (x == 'sarcastic').sum()),
        valid_json_count=('valid_json', lambda x: (x == True).sum()),
        )
    )
    
    aggregated['sarc_ratio'] = aggregated['sarc_count'] / aggregated['valid_json_count'].replace({0: pd.NA})

    aggregated.drop('sarc_count', inplace=True, axis = 1)

    out_path = f'/home/rp-fril-mhpe/{save_as}'
    # save 
    aggregated.to_csv(out_path, index=False)
    print(f'Saved ratios to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute sarcastic ratios for round 1 and round 2')

    parser.add_argument(
        '--round',
        type=int,
        required=True,
        choices=[1,2],
        help='Which round to compute results for (1 or 2).',
    )

    args = parser.parse_args()
    
    make_ratio(args.round)
