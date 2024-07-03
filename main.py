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

import traceback
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

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

# カスタムベクトルストアクラス
class CustomVectorStore:
    def __init__(self, texts, embeddings):
        self.texts = texts
        self.embeddings = embeddings
        self.vectors = None

    def embed(self):
        self.vectors = self.embeddings.embed_documents(self.texts)

    def similarity_search(self, query, k=1):
        query_vector = self.embeddings.embed_query(query)
        similarities = cosine_similarity([query_vector], self.vectors)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [self.texts[i] for i in top_k_indices]

try:
    # embeddings と LLM の初期化
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

    # カスタムベクトルストアの初期化
    texts = ["今日の天気は曇りのち雨", "東京の人口は約1400万人です", "富士山の高さは3776メートルです"]
    vector_store = CustomVectorStore(texts, embeddings)
    vector_store.embed()

    # プロンプトテンプレートの作成
    prompt_template = """以下のコンテキストを使用して、質問に答えてください。

    コンテキスト:
    {context}

    質問: {question}

    回答:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # LLMチェーンの作成
    qa_chain = LLMChain(llm=llm, prompt=prompt)

    logger.info("RAG chain initialized successfully")
except Exception as e:
    logger.error(f"Error initializing RAG chain: {str(e)}")
    logger.error(traceback.format_exc())
    qa_chain = None
    vector_store = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    logger.debug("Accessing root route")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    logger.debug(f"Received question: {question}")
    if qa_chain is None or vector_store is None:
        logger.error("RAG chain or vector store is not initialized")
        raise HTTPException(status_code=500, detail="System is not properly initialized. Check server logs for more details.")
    try:
        # 類似度検索を実行
        relevant_texts = vector_store.similarity_search(question, k=1)
        context = "\n".join(relevant_texts)

        # 質問に回答
        answer = qa_chain.run(context=context, question=question)

        logger.info(f"Generated answer for question: {question}")
        return templates.TemplateResponse("index.html", {"request": request, "answer": answer, "question": question})
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# if __name__ == "__main__":
#     # import uvicorn
#     logger.info("Starting the application")
#     uvicorn.run(app, host="0.0.0.0", port=8000)
if __name__ == '__main__':
      uvicorn.run('main:app', host='0.0.0.0', port=8000)