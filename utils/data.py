#helper functions for reading in data 
from typing import List, Dict, Optional, Iterator
from itertools import islice
import pandas as pd

def load_claims_text(path: str) -> List[Dict[str, str]]:
    items = []
    with open(path, 'r', encoding='utf-8') as file: 
        for i, line in enumerate(file):
            claim = line.strip()
            if not claim:
                continue
            
            items.append({'id': i, 'text': claim.strip('"').strip("'")})
    return items 



def load_claims_batches(
                path: str, 
                start: int=0,
                batch_size: int=256,
                limit: Optional[int] = None ) -> Iterator[List[Dict[str, str]]]:
    
    data = pd.read_csv(path)

    if start >= len(data):
        raise Valuerror('Start is larger than size of data.')
    
    end = len(data) if limit is None else min(len(data), start+limit)
    
    if end <= start:
        return #nothing to yield 

    window = data.iloc[start:end]

    for i in range(start, len(window), batch_size):
        chunk = window.iloc[i : i+batch_size]
        buf = chunk.to_dict()
        
        buf = [r.to_dict() for _, r in chunk.iterrows()]

        yield buf



# def load_claims_batches_old(
#                 path: str, 
#                 start: int=0,
#                 batch_size: int,
#                 limit: Optional[int] = None ) -> Iterator[List[Dict[int, str]]]:
    
#     data = pd.read_csv(path)

#     if start >= len(data):
#         raise Valuerror('Start is larger than size of data.')
    
#     end = len(data) if limit is None else min(len(df), start+limit)
    
#     if end <= start:
#         return #nothing to yield 

#     window = df.iloc[start:end]

#     for i in range(start, len(window), batch_size):
#         chunk = window.iloc[i : i+batch_size]
#         buf = chunk.to_dict()
        
#         buf = [r.to_dict() for _, r in chunk.iterrows()]

#         yield buf