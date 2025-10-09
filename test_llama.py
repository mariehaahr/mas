from vllm import LLM, SamplingParams
import torch, vllm
import time 
from dotenv import load_dotenv

# access token from hf
load_dotenv("/home/mhpe/.env")


def main():
    print("torch", torch.__version__, "cuda?", torch.cuda.is_available(), "vllm", vllm.__version__)
    start = time.time()
    # meta-llama/Llama-3.3-70B-Instruct
    # unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
    # LLM(model="nvidia/Llama-3.3-70B-Instruct-FP8", quantization="fp8")
    llm = LLM(model="nvidia/Llama-3.3-70B-Instruct-FP8", quantization="fp8")

    # default parameters from the documentation
    params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=40) 


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