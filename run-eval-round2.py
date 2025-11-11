from dotenv import load_dotenv
import argparse, pathlib
from utils.data import load_claims_text, load_claims_batches
from utils.prompts import build_conversations_round2, SYSTEM_JSON_GUIDED_R2, USER_R2
from utils.models import load_yaml, init_llm, init_sampling_params, ensure_local_model
from utils.runner import run_inference
from utils.io import write_csv, write_jsonl
from utils.prompts import OutputSarcRound2
import time

home_env = pathlib.Path.home() / ".env"
if home_env.exists():
    load_dotenv(home_env, override=False)

def main(args):

    # load model configs
    default_cfg = load_yaml('configs/default-model.yaml')
    profiles_root = load_yaml('configs/models.yaml')
    profiles = profiles_root.get('profiles', {})

    model_name = args.model_name
    if model_name not in profiles:
        raise SystemExit(f"Unknown --model_name '{model_name}'. Available: {', '.join(profiles.keys())}")
    
    model_cfg = {**default_cfg, **profiles[model_name]}

    # making sure to run local model, and download if not there 
    repo_id = model_cfg['model']
    print(repo_id)
    local_path = ensure_local_model(repo_id=repo_id)
    print(local_path)
    model_cfg['model'] = str(local_path)


    # init model and sampling 
    llm = init_llm(model_cfg=model_cfg)                      
    
    # get sampling params
    decoding_cfg = load_yaml(args.decoding_cfg)

    if model_cfg['has_default_sampling_params']:        
        # if sampling params specified in huggingface repo
        sampling = init_sampling_params(decoding_cfg, default = llm.get_default_sampling_params())
    else:
        # params we have specified
        decoding_cfg = {**decoding_cfg, **model_cfg['sampling']}
        sampling = init_sampling_params(decoding_cfg, default = None) # var uden default før
    
    # print params to output
    print('###### SAMPLING PARAMS ######')
    print(sampling)


    # write results 
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents = True, exist_ok = True)
    
    jsonl_path = outdir / f'{args.model_name}_round2.jsonl'
    csv_path = outdir / f'{args.model_name}_round2.csv'

    no_rows = 0

    # first load in the sender-reciever data for model 

    # adjust load claims batches to only take the claims that are present in sender-receiver df
    for batch in load_claims_batches(path = args.dataset_path, start = args.idx_start, batch_size = args.batch_size, limit=args.limit):

        # build the prompts

        print(batch)
        break 
    #     # TODO: Change build conversations to take label from sender
    #     conversations = build_conversations_round2(
    #         examples=batch, 
    #         system_prompt=args.system, 
    #         user_template=args.user)
        
    #     for i in range(args.repetition):
    #         # run inference 
    #         texts, parsed = run_inference(llm, conversations=conversations, sampling=sampling, json_format=OutputSarcRound2)
    #         valid_json = []

    #         rows = []
    #         for ex, t, p in zip(batch, texts, parsed):
    #             valid_json = False
    #             if p is not None:
    #                 valid_json = True
    #                 rows.append({'id': ex['id'], 'from_model': ex['model'], 'to_model': model_name, 'repetition': i, 'label': p['label'], 'confidence': p['confidence'], 'valid_json': valid_json, 'raw_text': t})
    #             else:
    #                 rrows.append({'id': ex['id'], 'from_model': ex['model'], 'to_model': model_name, 'repetition': i, 'label': None, 'confidence': None, 'valid_json': valid_json, 'raw_text': t})

    #         no_rows += len(rows)
    #         write_csv(rows, csv_path, ['id', 'from_model', 'to_model', 'repetition', 'label', 'confidence', 'valid_json', 'raw_text'])

    # print(f'Wrote {no_rows} rows to {outdir}')

if __name__ == '__main__':
    t0 = time.perf_counter()
    try:
        ap = argparse.ArgumentParser(description='Run offline inference on dataset (one example per line)')
        ap.add_argument('--model_name',
                        help = 'Short name of model from configs/models.yaml')
        ap.add_argument('--dataset_path', 
                        help='Path to dataset', 
                        default='results_small/first-llama-3.2-3b.csv')
        ap.add_argument('--repetition',
                        help='Number of times a model is presented a specific claim.',
                        type=int,
                        default=1)
        ap.add_argument('--decoding_cfg', 
                        help='Path to YAML file with sampling params and guided decoding toggle',
                        default='configs/decoding.yaml')
        ap.add_argument('--outdir',
                        help='Directory to write results',
                        default='/results/'),
        ap.add_argument('--system', 
                        help = 'System prompt string',
                        default=SYSTEM_JSON_GUIDED_R2)
        ap.add_argument('--user', 
                        help= 'User prompt string',
                        default=USER_R2)
        ap.add_argument('--batch_size',
                        help='Batch size to process dataset in',
                        type = int,
                        default=256)
        ap.add_argument('-limit', 
                        help='Limit number of examples for inference',
                        type=int)
        ap.add_argument('-idx_start',
                        help='Idx of row to start from in dataset',
                        type=int,
                        default=0)

        args = ap.parse_args()

        main(args)
    finally:
        dt = time.perf_counter() - t0
        print(f'[timing] total runtime: {dt:.3f}s')

