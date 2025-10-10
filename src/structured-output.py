from pydantic import BaseModel
from enum import Enum
from vllm import LLM
import json
from vllm.sampling_params import GuidedDecodingParams

class LabelSarc(str, Enum):
    sarcastic = 'sarcastic'
    literal = 'literal'

class OutputSarc(BaseModel):
    label: LabelSarc
    explanation: str

def main():
    json_schema = OutputSarc.model_json_schema()

    user_msg = """Task: Is this claim sarcastic or literal?
                Claim: "wow so funny"
                Output format:
                Label: sarcastic/literal
                Explanation: max 30 words"""


    system_msg =    ("Return ONLY a minified JSON object that conforms to this schema:\n" 
                f"{json.dumps(json_schema, ensure_ascii=False)}\n\n"
                "Rules:\n"
                "- Keys: label, explanation.\n"
                "- label MUST be one of: 'sarcastic', 'literal'.\n"
                "- explanation MUST be 30 words or fewer.\n"
                "- Output JSON only. No prose, no Markdown, no extra text.")
    

    messages = [{'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': user_msg}]

    llm = LLM(model="unsloth/Llama-3.2-3B-Instruct-bnb-4bit", seed = 0, quantization = 'bitsandbytes')

    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    sampling_params = SamplingParams(guided_decoding=guided_decoding_params)

    outputs = llm.chat(messages=messages, sampling_params=sampling_params)

    for output in outputs:
        print(output.outputs[0].text)

if __name__ == '__main__':
    main()