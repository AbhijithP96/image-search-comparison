import uvicorn
from fastapi import FastAPI, UploadFile, File
from search import hybrid

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Welcome to Image Search"}


@app.post("/search")
async def search_db(file: UploadFile = File(...)):
    image_bytes = await file.read()
    ids = hybrid.get_top_5(image_bytes)

    return {"ids": ids}


if __name__ == "__main__":
    uvicorn.run(app)
