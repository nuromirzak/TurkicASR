from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from audio_service import AudioService

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

audio_service = AudioService()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, query: str = None):
    return {"item_id": item_id, "query": query}


@app.post("/items/")
def create_item(item: dict):
    return {"item_name": item["name"], "item_value": item["value"]}


@app.post("/recognize-audio/")
async def recognize_audio(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, 'wb+') as f:
        f.write(await file.read())
    wav_file = audio_service.convert_audio(temp_file_path)
    recognized_text = audio_service.recognize(wav_file)
    os.remove(temp_file_path)  # Clean up the temporary file
    os.remove(wav_file)  # Clean up the converted file
    return JSONResponse(content={"recognized_text": recognized_text})
