from .schemas import OutputSarc
import json 
from typing import List, Dict


SARCASTIC_SCHEMA = OutputSarc.model_json_schema()

SYSTEM_JSON_GUIDED_R0 = (
    "Return ONLY a minified JSON object that conforms to this schema:\n"
    f"{json.dumps(SARCASTIC_SCHEMA, ensure_ascii=False)}\n\n"
    "Rules:\n"
    "- Keys: label, explanation.\n"
    "- label MUST be one of: 'sarcastic', 'literal'.\n"
    "- explanation MUST be 30 words or fewer.\n"
    "- Output JSON only. No prose, no Markdown, no extra text."
)

USER_R0 = ( "Task: Is this claim sarcastic or literal?\n"
            'Claim: "{claim}"\n'
            "Output format:\n"
            "Label: sarcastic/literal\n"
            "Explanation: Describe your reasoning."
    )

def build_conversations(
    examples: List[Dict[str, str]],
    system_prompt: str = SYSTEM_JSON_GUIDED_R0,
    user_template: str = USER_R0) -> List[List[Dict[str, str]]]:

    '''
    Several conversations will be a list of lists containing a dict for each user. 
    '''
    convs: List[List[Dict[str, str]]] = []

    for ex in examples:
       convs.append([
          {'role': 'system', 'content': system_prompt},
          {'role': 'user', 'content': user_template.format(claim=ex['text'])},
       ])

    return convs 
