import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  

from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
from keras.models import load_model
from keras.applications.efficientnet import preprocess_input

app = FastAPI()

MODEL_PATH = "/best_model_selected_9.keras"
CLASSES = ['бабочка', 'паук', 'слон']

try:
    model = load_model(MODEL_PATH)
    print("Модель загружена успешно")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")

        img = cv2.resize(img, (224, 224))
        img = img.astype('float32')
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)[0]
        predicted_class_index = np.argmax(preds)
        predicted_class_name = CLASSES[predicted_class_index]

        probabilities = {cls: float(round(prob, 4)) for cls, prob in zip(CLASSES, preds)}

        return {
            "prediction": predicted_class_name,
            "probabilities": probabilities
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
