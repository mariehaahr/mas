#helper functions for reading in data 
from typing import List, Dict


def load_claims_text(path: str) -> List[Dict[str, str]]:
    items = []
    with open(path, 'r', encoding='utf-8') as file: 
        for i, line in enumerate(file):
            claim = line.strip()
            if not claim:
                continue
            
            items.append({'id': i, 'text': claim.strip('"').strip("'")})
    return items 



