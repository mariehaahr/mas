import json, csv, pathlib
from typing import List, Dict, Any


def write_jsonl(records: List[Dict[str, Any]], path: pathlib.Path):
    with path.open('a', encoding='utf-8') as file: 
        for r in records:
            file.write(json.dumps(r, ensure_ascii=False) + '\n')


def write_csv(records: List[Dict[str, Any]], path: pathlib.Path, fields):
    # checking if file needs header 
    needs_header = (not pathlib.Path.exists()) or (path.stat().st_size == 0)
    with path.open('a', newline='', encoding='utf-8') as file:
        w = csv.DictWriter(file, fieldnames=fields)
        if needs_header:
            w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in fields})
            