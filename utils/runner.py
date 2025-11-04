import time, json 
from typing import List, Dict, Any, Tuple
from pydantic import ValidationError
from vllm import LLM 
from vllm.sampling_params import SamplingParams 

# check, works for both round 1 and round 2
def run_inference(
    llm: LLM, 
    conversations: List[List[Dict[str, str]]],
    sampling: SamplingParams,
    json_format) -> Tuple[List[str], List[Dict[str, Any] | None], float]:
    

    outs = llm.chat(messages=conversations, sampling_params=sampling)

    texts = [o.outputs[0].text if o.outputs else "" for o in outs]
    parsed = []

    for txt in texts:
        try:
            obj = json.loads(txt)
            parsed.append(json_format(**obj).model_dump())
        except(json.JSONDecodeError, ValidationError, KeyError, TypeError):
            parsed.append(None)

    return texts, parsed