from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from measurement.measurement import get_measurements

app = FastAPI()

@app.post("/measurements")
async def measure(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    result = get_measurements(image)
    return JSONResponse(content=result)
