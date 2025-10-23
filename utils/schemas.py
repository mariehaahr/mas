import json
from enum import Enum
from pydantic import BaseModel

class LabelSarc(str, Enum):
    sarcastic = 'sarcastic'
    literal = 'literal'

class OutputSarc(BaseModel):
    label: LabelSarc
    explanation: str