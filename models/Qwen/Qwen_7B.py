###############################################################################################

                                    # 8B parameters
                                    # 4 bit quantisation

###############################################################################################

from vllm import LLM, SamplingParams
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

model_path = "/home/mhpe/.cache/huggingface/hub/models--osllmai-community--Qwen2.5-7B-Instruct-bnb-4bit"
llm = LLM(model=model_path)


params = SamplingParams(temperature=0.7, max_tokens=128)

outputs = llm.generate(["Hello world!"], params)
print(outputs[0].outputs[0].text)
