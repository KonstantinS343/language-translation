from nltk import sent_tokenize
import json
import re
from nltk_module.core import NltkProcess
from models.model import TranslationResponse

class Translation:
    
    @classmethod
    async def translate(cls, text: str) -> str:
        text = """
                Anne Frank war ein jüdisches Mädchen, das während des Zweiten Weltkriegs in Amsterdam lebte. Als die Nazis die Niederlande besetzten, versteckte sich Anne zusammen mit ihrer Familie, damit die Nazis sie nicht finden. Während ihres Verstecks schrieb Anne ihr Tagebuch, in dem sie ihre Gedanken und Gefühle aufschrieb. Ihr Tagebuch wurde später unter dem Titel 'Das Tagebuch der Anne Frank' veröffentlicht und ist zu einem der bekanntesten Bücher über den Zweiten Weltkrieg geworden.
               """
        text=re.sub(r'[\n\r\t]', ' ', text).strip()
        return await cls.post_process(text)

               
    @classmethod
    async def post_process(cls, text: str) -> TranslationResponse:
        
        translation = dict()
        
        for i, j in enumerate(sent_tokenize(text, 'german')):
            translation[i + 1] = j.strip()
        
        with open('translation.json', 'w') as file:
            json.dump(translation, file, indent=4)
            
        output = await NltkProcess.process(text, 'german')
        
        
        output.nltk = sorted(output.nltk, key=lambda x: x.freq, reverse=True)
        
        with open('output.json', 'w') as file:
            json.dump([item.dict() for item in output.nltk], file, indent=4, ensure_ascii=False)
            
        
        return TranslationResponse(information=output, text=text)
        
        