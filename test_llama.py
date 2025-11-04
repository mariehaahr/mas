from vllm import LLM, SamplingParams
import torch, vllm
import time 
from dotenv import load_dotenv
from utils.models import load_yaml
# access token from hf
load_dotenv("/home/mhpe/.env")

def main():
    print("torch", torch.__version__, "cuda?", torch.cuda.is_available(), "vllm", vllm.__version__)
    start = time.time()
    # meta-llama/Llama-3.3-70B-Instruct
    # unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
    # LLM(model="nvidia/Llama-3.3-70B-Instruct-FP8", quantization="fp8")
    llm = LLM(model="unsloth/mistral-7b-instruct-v0.3-bnb-4bit", quantization="bitsandbytes")
    
    profiles_root = load_yaml('configs/models.yaml')
    profiles = profiles_root.get('profiles', {})

    print(profiles_root['profiles']['mistral-0.3-7b'])
    # Returns the effective defaults (merged from the model’s generation_config.json)
    sp = llm.get_default_sampling_params()

    print("Temperature:", sp.temperature)
    print("Top-p:", sp.top_p)
    print("Max tokens:", sp.max_tokens)

    try:
        print(sp.model_dump())   # pydantic v2 style
    except Exception:
        print(vars(sp))


    # default parameters from the documentation 
    params = SamplingParams(max_tokens=40) 


    prompt = """
    Task: Is this claim below sarcastic or literal?
    Claim: "wow so funny"
    
    Your output format should be:
        1) A label: "Sarcastic" or "Literal"
        2) An explanation why you chose that label: max 30 words
        3) A confidence score: [1,100]
    """

    # Query
    outputs = llm.generate(prompt, params) # true label is sarcastic for this example

    for output in outputs:
        print('This is output:')
        print(output.outputs[0].text)
    
    print('ENDED, IT TOOK:')
    print((time.time() - start))
    
    

if __name__ == "__main__":
    main()