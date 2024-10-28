from models.model import Query, NltkResponse, SyntacticTreeRequest
from fastapi import APIRouter, Response, Request
from fastapi.responses import FileResponse

from nltk_module.core import NltkProcess
from report.core import Report

nltk_router = APIRouter(prefix='/nltk', tags=["Nltk"])
user_router = APIRouter(prefix='/query', tags=["Query"])


@nltk_router.post('/')
async def post_nltk(message: Query) -> NltkResponse:
    text = await NltkProcess.process(message.text)
    return text

@nltk_router.post('/tree/')
async def post_tree(message: SyntacticTreeRequest) -> Response:
    image = await NltkProcess.syntactic_tree('The provided Python code demonstrates stopword removal using the Natural Language Toolkit (NLTK) library.')
    return Response(content=image, media_type="image/png")

@user_router.get('/report/')
async def get_repost(request: Request) -> Response:
    report = await Report.create_report(request)
    return FileResponse(report, media_type="application/pdf", filename="report.pdf") 



