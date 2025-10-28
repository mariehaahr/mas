#helper functions for reading in data 
from typing import List, Dict, Optional, Iterator
from itertools import islice

def load_claims_text(path: str) -> List[Dict[str, str]]:
    items = []
    with open(path, 'r', encoding='utf-8') as file: 
        for i, line in enumerate(file):
            claim = line.strip()
            if not claim:
                continue
            
            items.append({'id': i, 'text': claim.strip('"').strip("'")})
    return items 



def load_claims_batches(path: str, 
                start: int=0,
                batch_size: int,
                limit: Optional[int] = None ) -> Iterator[List[Dict[str, str]]]:
    emitted = 0
    buf = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if limit is not None and emitted >= limit:
                break
            
            claim = line.strip()
            if not claim:
                continue
            buf.append({'id': i, 'text': claim.strip('"').strip("'")})
            emitted += 1
            
            if len(buf) == batch_size:
                yield buf
                buf = []
    if buf:
        yield buf