from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:5500",  # if using VSCode Live Server
    "*",  # optional: allows all origins (use cautiously in production)
]



app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "saved_models", "1.keras")

MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["asl_alphabet_test", "asl_alphabet_train"]


@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    image= np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image= read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)

    prediction = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence =np.max(prediction[0])



    return {
            "class": predicted_class,
            "confidence": float(confidence)}

if __name__=="__main__":
    uvicorn.run(app, host='localhost',port=8000)