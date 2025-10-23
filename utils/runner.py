import time, json 
from typing import List, Dict, Any, Tuple
from pydantic import ValidationError
from vllm import LLM 
from vllm.sampling_params import SamplingParams 
from .prompts import OutputSarc


def run_inference(
    llm: LLM, 
    conversations: List[List[Dict[str, str]]],
    sampling: SamplingParams) -> Tuple[List[str], List[Dict[str, Any] | None], float]:
    
    t0 = time.time()
    outs = llm.chat(messages=conversations, sampling_params=sampling)
    latency = time.time() - t0 

    texts = [o.outputs[0].text if o.outputs else "" for o in outs]
    parsed = []

    for txt in texts:
        try:
            obj = json.loads(txt)
            parsed.append(OutputSarc(**obj).model_dump())
        except(json.JSONDecodeError, ValidationError, KeyError, TypeError):
            parsed.append(None)


    per_item = latency / max(1, len(conversations))
    return texts, parsed, per_item