import argparse, pathlib
from utils.data import load_claims_text
from utils.prompts import build_conversations, SYSTEM_JSON_GUIDED_R0, USER_R0
from utils.models import load_yaml, init_llm, init_sampling_params
from utils.runner import run_inference
from utils.io import write_csv, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--claims')
    ap.add_argument('--model_cfg', default='configs/model.yaml')
    ap.add_argument('--decoding_cfg', default='configs/decoding.yaml')
    ap.add_argument('--outdir', default='artifacts/results/')
    ap.add_argument('--system', default=SYSTEM_JSON_GUIDED_R0)
    ap.add_argument('--user', default=USER_R0)
    ap.add_argument('--limit', type=int, default=0)

    args = ap.parse_args()

    examples = load_claims_text(args.claims)

    if args.limit > 0:
        examples = examples[:args.limit]

    conversations = build_conversations(examples=examples, system_prompt=args.system, user_template=args.user)

    llm = init_llm(load_yaml(args.model_cfg))
    sampling = init_sampling_params(load_yaml(args.decoding_cfg))

    texts, parsed, per_item = run_inference(llm, conversations=conversations, sampling=sampling)

    rows = [
        {'id': ex['id'], 'input_text': ex['text'], 'output_text': t, 'valid_json': p is not None, 'parsed': p} for ex, t, p in zip(examples, texts, parsed)
    ]

    outdir = pathlib.Path(args.outdir)
    write_jsonl(rows, outdir / 'results.jsonl')
    write_csv(rows, outdir / 'results.csv', ['id', 'input_text', 'output_text', 'valid_json'])


    print(f'Wrote {len(rows)} rows to {outdir}')
    print(f'Avg latency/item approx. {per_item:3f}s')


if __name__ == '__main__':
    main()