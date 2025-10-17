#init llm + samplingparams + guidedDecodingparams 
import pathlib, yaml, json 
from vllm import LLM
from vllm.sampling_params import SamplingParams, GuidedDecodingParams
from .prompts import SARCASTIC_SCHEMA 


def load_yaml(path: str) -> dict:
    return yaml.safe_load(pathlib.Path(path).read_text())



def init_llm(model_cfg: dict) -> LLM:
    return LLM(
        model_cfg=model_cfg['model'],
        quantization=model_cfg['quantization'],
        seed=model_cfg.get('seed', 0),
        dtype=model_cfg.get('dtype', None)
    )


def init_sampling_params(decoding_cfg: dict) -> SamplingParams:
    guided = GuidedDecodingParams(json = SARCASTIC_SCHEMA) if decoding_cfg['use_guided_json'] else None
    
    return SamplingParams(
        max_tokens=decoding_cfg.get('max_tokens', 100),
        temperature = decoding_cfg.get('temperature', 1.0),
        top_p = decoding_cfg.get('top_p', 1.0),
        guided_decoding = guided
        )
