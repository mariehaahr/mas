#init llm + samplingparams + guidedDecodingparams 
import pathlib, yaml, json 
from vllm import LLM
from vllm.sampling_params import SamplingParams, GuidedDecodingParams
from .prompts import SARCASTIC_SCHEMA 
import os, subprocess

def load_yaml(path: str) -> dict:
    return yaml.safe_load(pathlib.Path(path).read_text())

def _sanitize_repo_id(repo_id: str) -> str:
    # HF hub’s cache uses "models--org--name" style
    return "models--" + repo_id.replace("/", "--")


def ensure_local_model(
    repo_id: str
) -> pathlib.Path:
    """
    Ensure a model repo is present locally and return its path.
    - repo_id: e.g. "meta-llama/Meta-Llama-3-8B-Instruct"
    """

    local_dir = f'~/.cache/huggingface/hub/{_sanitize_repo_id(repo_id)}'

    # If the directory already exists and is non-empty, assume it’s usable
    if local_dir.exists() and any(local_dir.iterdir()):
        return local_dir
    


    cmd = [
        "huggingface-cli", "download", repo_id,
        "--local-dir", str(local_dir),
    ]

    subprocess.run(cmd, check=True)

    return local_dir


def init_llm(model_cfg: dict) -> LLM:
    return LLM(
        model_cfg=model_cfg['model'],
        quantization=model_cfg['quantization'],
        seed=model_cfg['seed'],
        dtype=model_cfg.get('dtype', None),
    )


def init_sampling_params(decoding_cfg: dict) -> SamplingParams:
    guided = GuidedDecodingParams(json = SARCASTIC_SCHEMA) if decoding_cfg['use_guided_json'] else None
    
    return SamplingParams(
        max_tokens=decoding_cfg.get('max_tokens', 100),
        temperature = decoding_cfg.get('temperature', 1.0),
        top_p = decoding_cfg.get('top_p', 1.0),
        guided_decoding = guided
        )


