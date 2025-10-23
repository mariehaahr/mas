import json, csv, pathlib
from typing import List, Dict, Any


def write_jsonl(records: List[Dict[str, Any]], path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True) 

    with path.open('w', encoding='utf-8') as file: 
        for r in records:
            file.write(json.dumps(r, ensure_ascii=False) + '\n')


def write_csv(records: List[Dict[str, Any]], path: pathlib.Path, fields):
    path.parent.mkdir(parents=True, exist_ok=True) 
    with path.open('w', newline='', encoding='utf-8') as file:
        w = csv.DictWriter(file, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in fields})
            