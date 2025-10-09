from vllm import LLM, SamplingParams
import torch

def main():
    print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
    llm = LLM(model="facebook/opt-125m")

    params = SamplingParams(temperature=0.7, max_tokens=128)

    # Query
    outputs = llm.generate(["Hello, how are you today?"], params)

    for output in outputs:
        print(output.outputs[0].text)

if __name__ == "__main__":
    main()
