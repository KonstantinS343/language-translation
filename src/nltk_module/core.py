import string
import io
import nltk
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget
from PIL import Image

from models.model import NltkResponse, NltkData

class NltkProcess:
    
    metadata = {
        "CC": "координирующая конъюнкция",
        "CD": "цифра",
        "DT": "артикль",
        "TO": "предлог to",
        "MD": "модальный глагол",
        "IN": "предлог/подчинительный союз",
        "JJ": "прилагательное",
        "JJR": "прилагательное сравнительное",
        "JJS": "прилагательное в превосходной ",
        "NN": "существительное в единственном числе",
        "NNS": "существительное во множественном числе",
        "NNP": "имя собственное в единственном числе",
        "NNPS": "имя собственное, множественное число",
        "PRP": "местоимение",
        "POS": "притяжательное окончание",
        "PRP$": "местоимение",
        "RB": "наречие",
        "RBR": "наречие в сравнительной степени",
        "RBS": "наречие в превосходной степени",
        "RP": "частица",
        "UH": "междометие",
        "VB": "глагол, базовая форма",
        "VBD": "глагол, прошедшее время",
        "VBG": "глагол, герундий/причастие настоящего времени",
        "VBN": "глагол, причастие прошедшего времени",
        "VBP": "глагол, петь. настоящее, не трехмерное действие",
        "VBZ": "глагол, поющий в 3-м лице. настоящее принимает",
        "WDT": "вопрос, который",
        "WP": "вопрос кто, что",
        "WRB": "вопрос, где, когда",
    }
    
    grammer = nltk.RegexpParser(
        """
        NP: {<DT>?<JJ>*<NN.*>}
        P: {<IN>}
        V: {<V.*>}
        PP: {<P> <NP>}
        VP: {<V> <NP|PP>*}
        S:  {<NP> <VP>}
        """
    )
    
    @classmethod
    async def process(cls, text: str):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        
        filtered_sentence = []
        
        for w in word_tokens:
            if w.lower() not in stop_words and w.lower() not in (string.punctuation + '“”'):
                filtered_sentence.append(
                    NltkData(
                        freq=word_tokens.count(w),
                        word=w,
                        info=cls.metadata[nltk.pos_tag([w])[0][1]]
                    )
                )
        
        with open('input.json', 'w') as f:
            json.dump([item.dict() for item in filtered_sentence], f, indent=4, ensure_ascii=False)
        
        return NltkResponse(nltk=filtered_sentence, count=len(filtered_sentence))
    
    @classmethod
    async def syntactic_tree(cls, text: str):
        tagged = nltk.pos_tag(
            [word for word in word_tokenize(text) if word not in string.punctuation]
        )
        window = CanvasFrame(width=1500, height=500)
        output = cls.grammer.parse(tagged)
        tree = TreeWidget(window.canvas(), output)
        window.add_widget(tree, 10, 10)
        window.print_to_file('tree.ps')
        window.destroy()
        
        psimage=Image.open('tree.ps')
        psimage.load(scale=2) 
        img_byte_arr = io.BytesIO()
        psimage.save(img_byte_arr, format='PNG', dpi=(300, 300))
        img_byte_arr = img_byte_arr.getvalue()
        
        return img_byte_arr

                
    