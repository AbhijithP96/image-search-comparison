import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from search.baseline import BaselineSearcher
from contextlib import asynccontextmanager
import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.searcher = BaselineSearcher()
    yield
    app.state.searcher.close()


app = FastAPI(
    title="Image Retriever API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "Welcome to Image Search"}


@app.post("/search")
async def search_db(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    ids = request.app.state.searcher.search(image_bytes=image_bytes)

    return {"ids": ids}


if __name__ == "__main__":
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
