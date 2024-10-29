from pydantic import BaseModel
from typing import List


class Query(BaseModel):
    text: str
    
class NltkData(BaseModel):
    freq: int
    word: str
    info: str
    
class NltkResponse(BaseModel):
    nltk: List[NltkData]
    count: int
    
class SyntacticTreeRequest(BaseModel):
    order: str
    
class TranslationResponse(BaseModel):
    text: str
    information: NltkResponse