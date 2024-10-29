import string
import io
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget
from PIL import Image

from models.model import NltkResponse, NltkData

class NltkProcess:
    
    nlp = spacy.load('de_core_news_sm')
    
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
        "ADJA": "прилагательное, атрибутивное",
        "ADJD": "прилагательное, наречное или предикативное",
        "APPO": "постпозиция",
        "APPR": "предлог; циркумпозиция слева",
        "APPRART": "предлог с артиклем",
        "APZR": "циркумпозиция справа",
        "ART": "определённый или неопределённый артикль",
        "CARD": "количественное числительное",
        "FM": "материал на иностранном языке",
        "ITJ": "междометие",
        "KOKOM": "сравнительный союз",
        "KON": "координирующий союз",
        "KOUI": 'подчинительный союз с "zu" и инфинитивом',
        "KOUS": "подчинительный союз с предложением",
        "NE": "имя собственное",
        "NNE": "имя собственное",
        "PAV": "местоимённое наречие",
        "PROAV": "местоимённое наречие",
        "PDAT": "атрибутивное указательное местоимение",
        "PDS": "замещающее указательное местоимение",
        "PIAT": "атрибутивное неопределённое местоимение без определителя",
        "PIDAT": "атрибутивное неопределённое местоимение с определителем",
        "PIS": "замещающее неопределённое местоимение",
        "PPER": "не-рефлексивное личное местоимение",
        "PPOSAT": "атрибутивное притяжательное местоимение",
        "PPOSS": "замещающее притяжательное местоимение",
        "PRELAT": "атрибутивное относительное местоимение",
        "PRELS": "замещающее относительное местоимение",
        "PRF": "рефлексивное личное местоимение",
        "PTKA": "частица с прилагательным или наречием",
        "PTKANT": "частица для ответа",
        "PTKNEG": "отрицательная частица",
        "PTKVZ": "отделяемая глагольная частица",
        "PTKZU": '"zu" перед инфинитивом',
        "PWAT": "атрибутивное вопросительное местоимение",
        "PWAV": "наречное вопросительное или относительное местоимение",
        "PWS": "замещающее вопросительное местоимение",
        "TRUNC": "остаток слова",
        "VAFIN": "финитный глагол, вспомогательный",
        "VAIMP": "повелительное наклонение, вспомогательное",
        "VAINF": "инфинитив, вспомогательное",
        "VAPP": "причастие прошедшего времени, вспомогательное",
        "VMFIN": "финитный глагол, модальный",
        "VMINF": "инфинитив, модальный",
        "VMPP": "причастие прошедшего времени, модальный",
        "VVFIN": "финитный глагол, полный",
        "VVIMP": "повелительное наклонение, полный",
        "VVINF": "инфинитив, полный",
        "VVIZU": 'инфинитив с "zu", полный',
        "VVPP": "причастие прошедшего времени, полный",
        "XY": "не-слово, содержащее не-букву"
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
    async def process(cls, text: str, language: str = 'english'):
        stop_words = set(stopwords.words(language))
        word_tokens = word_tokenize(text)
        
        filtered_sentence = []
        
        for w in set(word_tokens):
            if w.lower() not in stop_words and w.lower() not in (string.punctuation + '“”'):
                info = await cls.pos_tag(w, language)
                if info:
                    filtered_sentence.append(
                        NltkData(
                            freq=word_tokens.count(w),
                            word=w,
                            info=info
                        )
                    )
        
        return NltkResponse(nltk=filtered_sentence, count=len(filtered_sentence))
    
    @classmethod
    async def pos_tag(cls, word: str, language: str):
        if language == 'english':
            return cls.metadata[nltk.pos_tag([word])[0][1]]
        else:
            token = cls.nlp(word)[0]
            return cls.metadata.get(token.tag_)
    
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

                
    