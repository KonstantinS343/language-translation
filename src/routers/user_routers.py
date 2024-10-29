from models.model import Query, NltkResponse, SyntacticTreeRequest, TranslationResponse
from fastapi import APIRouter, Response, Request
from fastapi.responses import FileResponse, Response
import json

from nltk_module.core import NltkProcess
from report.core import Report
from translation.translation import Translation

nltk_router = APIRouter(prefix='/nltk', tags=["Nltk"])
user_router = APIRouter(prefix='/query', tags=["Query"])


@nltk_router.post('/')
async def post_nltk(message: Query) -> NltkResponse:
    text = await NltkProcess.process(message.text)
    
    with open('input.json', 'w') as f:
        json.dump([item.dict() for item in text.nltk], f, indent=4, ensure_ascii=False)
    return text

@nltk_router.post('/tree/')
async def post_tree(message: SyntacticTreeRequest) -> Response:
    with open('translation.json', 'r') as file:
        try:
            sentence = (json.load(file))[message.order]
        except Exception as e:
            return Response(content=str(e.__class__.__name__),  status_code=400)
    
    image = await NltkProcess.syntactic_tree(sentence)
    return Response(content=image, media_type="image/png")

@user_router.get('/report/')
async def get_repost(request: Request) -> Response:
    report = await Report.create_report(request)
    return FileResponse(report, media_type="application/pdf", filename="report.pdf") 

@user_router.post('/')
async def translate(query: Query) -> TranslationResponse:
    report = await Translation.translate(query.text)
    return report