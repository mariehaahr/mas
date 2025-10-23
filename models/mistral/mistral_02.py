###############################################################################################

                                    # 7B parameters
                                    # version 0.2
                                    # 4 bit quantisation

###############################################################################################

from vllm import LLM, SamplingParams
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

model_path = "/home/mhpe/.cache/huggingface/hub/models--unsloth--Mistral-7B-Instruct-v0.2-bnb-4bit"
llm = LLM(model=model_path)


params = SamplingParams(temperature=0.7, max_tokens=128)

outputs = llm.generate(["Hello world!"], params)
print(outputs[0].outputs[0].text)
