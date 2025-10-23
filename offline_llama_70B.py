from vllm import LLM, SamplingParams
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

# model_path = "/home/mhpe/.cache/huggingface/hub/models--unsloth--Llama-3.3-70B-Instruct-bnb-4bit/snapshots/75779cdfa1240a4d048c6ecf65e97ff31b707214"

model_path = "/home/mhpe/.cache/huggingface/hub/models--unsloth--Llama-3.3-70B-Instruct-bnb-4bit"
# llm = LLM(model=model_path)
llm = LLM(model=model_path)


# llm = LLM(model=model_path)
params = SamplingParams(temperature=0.7, max_tokens=128)

outputs = llm.generate(["Hello world!"], params)
print(outputs[0].outputs[0].text)
