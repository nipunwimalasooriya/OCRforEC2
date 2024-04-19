from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import pytesseract
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Adjust allowed methods as needed
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
def home():
    return {"message": "Server started successfully!"}


async def read_image(img_path, lang='sin'):
    try:
        # Read image using OpenCV
        img = cv2.imread(img_path)

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to the image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Apply OCR using Pytesseract to detect Sinhala Unicode language characters
        return pytesseract.image_to_string(opening, lang=lang)
    except Exception as e:
        return f"[ERROR] Unable to process file: {img_path}. Error: {str(e)}"

@app.post("/api/v1/extract_text")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open("temp.jpg", "wb") as f:
            f.write(contents)
        text = await read_image("temp.jpg")
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
