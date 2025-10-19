from dotenv import load_dotenv
import argparse, pathlib
from utils.data import load_claims_text
from utils.prompts import build_conversations, SYSTEM_JSON_GUIDED_R0, USER_R0
from utils.models import load_yaml, init_llm, init_sampling_params, ensure_local_model
from utils.runner import run_inference
from utils.io import write_csv, write_jsonl

home_env = "~/.env"
if home_env.exists():
    load_dotenv(home_env, override=False)

def main():
    ap = argparse.ArgumentParser(description='Run online inference over .txt dataset (one example per line)')
    ap.add_argument('--model_name',
                    help = 'Short name of model from configs/models.yaml')
    ap.add_argument('--dataset_path', 
                    help='Path to dataset', 
                    default='data/sarc/train_text.txt')
    ap.add_argument('--decoding_cfg', 
                    help='Path to YAML file with sampling params and guided decoding toggle',
                    default='configs/decoding.yaml')
    ap.add_argument('outdir',
                    help='Directory to write results',
                    default='results/'),
    ap.add_argument('--system', 
                    help = 'System prompt string',
                    default=SYSTEM_JSON_GUIDED_R0)
    ap.add_argument('--user', 
                    help= 'User prompt string',
                    default=USER_R0)
    ap.add_argument('--limit', 
                    help='Limit number of examples for inference',
                    type=int, 
                    default=0)

    args = ap.parse_args()

    # load data 
    examples = load_claims_text(args.dataset_path)
    # option to choose a subset 
    if args.limit > 0:
        examples = examples[:args.limit]
    # build the prompts
    conversations = build_conversations(
        examples=examples, 
        system_prompt=args.system, 
        user_template=args.user)
    
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
    local_path = ensure_local_model(repo_id=repo_id)
    model_cfg['model'] = str(local_path)
    
    # init model and sampling 
    llm = init_llm(model_cfg=model_cfg)
    #TODO: make it easier to switch to other task, SARCASM scheme hardcoded in init_sampling_params
    sampling = init_sampling_params(load_yaml(args.decoding_cfg))

    # run inference 
    texts, parsed, per_item = run_inference(llm, conversations=conversations, sampling=sampling)

    # write results 
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents = True, exists_ok = True)
    rows = [
        {'id': ex['id'], 'input_text': ex['text'], 'output_text': t, 'valid_json': p is not None, 'parsed': p} for ex, t, p in zip(examples, texts, parsed)
    ]

    model_full_name = model_cfg['model'].strip().replace('/','_').replace(' ', '_')
    

    write_jsonl(rows, outdir / f'{model_full_name}.jsonl')
    write_csv(rows, outdir / f'{model_full_name}.csv', ['id', 'input_text', 'output_text', 'valid_json'])

    print(f'Wrote {len(rows)} rows to {outdir}')
    print(f'Avg latency/item approx. {per_item:3f}s')

    #TODO: Adjust so it is the downloaded model that is getting used.


if __name__ == '__main__':
    main()