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
import logging
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import uvicorn
from langchain.vectorstores import InMemoryDocstore
from langchain.docstore import InMemoryDocstore as InMemoryDocstoreBase


# ロギングの設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# 静的ファイルの設定
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# OpenAI APIキーを環境変数から取得
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY is not set in the environment variables")
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

try:
    # シンプルな辞書ベースのベクトルストア
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    texts = ["今日の天気は曇りのち雨", "Sample text 2", "Sample text 3"]
    vectors = embeddings.embed_documents(texts)
    text_to_vector = dict(zip(texts, vectors))
    docstore = InMemoryDocstoreBase(text_to_vector)
    vector_store = InMemoryDocstore(docstore, embeddings)

    # RAGチェーンの作成
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    logger.info("RAG chain initialized successfully")
except Exception as e:
    logger.error(f"Error initializing RAG chain: {str(e)}")
    logger.error(traceback.format_exc())
    qa_chain = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    logger.debug("Accessing root route")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    logger.debug(f"Received question: {question}")
    try:
        answer = qa_chain.run(question)
        logger.info(f"Generated answer for question: {question}")
        return templates.TemplateResponse("index.html", {"request": request, "answer": answer, "question": question})
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return templates.TemplateResponse("index.html", {"request": request, "answer": f"Error: {str(e)}", "question": question})

# if __name__ == "__main__":
#     # import uvicorn
#     logger.info("Starting the application")
#     uvicorn.run(app, host="0.0.0.0", port=8000)
if __name__ == '__main__':
      uvicorn.run('main:app', host='0.0.0.0', port=8000)