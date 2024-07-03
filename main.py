# from fastapi import FastAPI, Form, Request, status
# from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# import uvicorn


# app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     print('Request for index page received')
#     return templates.TemplateResponse('index.html', {"request": request})

# @app.get('/favicon.ico')
# async def favicon():
#     file_name = 'favicon.ico'
#     file_path = './static/' + file_name
#     return FileResponse(path=file_path, headers={'mimetype': 'image/vnd.microsoft.icon'})

# @app.post('/hello', response_class=HTMLResponse)
# async def hello(request: Request, name: str = Form(...)):
#     if name:
#         print('Request for hello page received with name=%s' % name)
#         return templates.TemplateResponse('hello.html', {"request": request, 'name':name})
#     else:
#         print('Request for hello page received with no name or blank name -- redirecting')
#         return RedirectResponse(request.url_for("index"), status_code=status.HTTP_302_FOUND)

# if __name__ == '__main__':
#     uvicorn.run('main:app', host='0.0.0.0', port=8000)

# FastAPI Azure QA Application

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

app = FastAPI()

# OpenAI APIキーを環境変数から取得
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

# ベクトルストアの初期化
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_store = FAISS.from_texts(["今日の天気は曇りのち雨", "Sample text 2", "Sample text 3"], embeddings)

# RAGチェーンの作成
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    try:
        response = qa_chain.run(query.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)