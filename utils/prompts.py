from .schemas import OutputSarc, OutputSarcRound2
import json 
from typing import List, Dict


SARCASTIC_SCHEMA = OutputSarc.model_json_schema()
SARCASTIC_SCHEMA_ROUND2 = OutputSarcRound2.model_json_schema()


SYSTEM_JSON_GUIDED_R1 = (
    "Return ONLY a minified JSON object that conforms to this schema:\n"
    f"{json.dumps(SARCASTIC_SCHEMA, ensure_ascii=False)}\n\n"
    "Rules:\n"
    "- Keys: label, explanation, confidence.\n"
    "- label MUST be one of: 'sarcastic', 'literal'.\n"
    "- explanation MUST be 30 words or fewer.\n"
    "- confidence MUST be an integer between 0 and 100 (inclusive), no percent sign, no text.\n"
    "- Output JSON only. No prose, no Markdown, no extra text."
)

SYSTEM_JSON_GUIDED_R2 = (
    "Return ONLY a minified JSON object that conforms to this schema:\n"
    f"{json.dumps(SARCASTIC_SCHEMA_ROUND2, ensure_ascii=False)}\n\n"
    "Rules:\n"
    "- Keys: label, confidence.\n"
    "- label MUST be one of: 'sarcastic', 'literal'.\n"
    "- confidence MUST be an integer between 0 and 100 (inclusive), no percent sign, no text.\n"
    "- Output JSON only. No prose, no Markdown, no extra text."
)

USER_R1 = ( "Task: Is this claim sarcastic or literal?\n"
            'Claim: "{claim}"\n'
            "Output format:\n"
            "Label: sarcastic/literal\n"
            "Explanation: Describe your reasoning.\n"
            "Confidence (0-100): How confident are you about your answer?"
    )

USER_R2 = ( "Task: Is this claim sarcastic or literal?\n"
            'Claim: "{claim}"\n'
            "A peer of yours think the claim is {other_label}, with the following explanation: '{other_explanation}' \n"
            "Output format:\n"
            "Label: sarcastic/literal\n"
            "Confidence (0-100): How confident are you about your answer?"
    )

def build_conversations(
    examples: List[Dict[str, str]],
    system_prompt: str = SYSTEM_JSON_GUIDED_R1,             # this can be changed in run eval to R2
    user_template: str = USER_R1) -> List[List[Dict[int, str]]]:

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

def build_conversations_round2(
    examples: List[Dict[int, str]],
    system_prompt: str = SYSTEM_JSON_GUIDED_R2,             # this can be changed in run eval to R2
    user_template: str = USER_R2) -> List[List[Dict[str, str]]]:

    '''
    Several conversations will be a list of lists containing a dict for each user.
    This is for round 2, so we append label and explanation from the previous round. 
    '''
    convs: List[List[Dict[str, str]]] = []

    for ex in examples:
        
        convs.append([
          {'role': 'system', 'content': system_prompt},
          {'role': 'user', 'content': user_template.format(claim=ex['claim'], other_label=ex['label'], other_explanation=ex['explanation'])},
          ])

    return convs 
