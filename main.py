import requests
import pytesseract
from PIL import Image
import io
from bs4 import BeautifulSoup
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from engine.verifier import Verifier

app = FastAPI(title="VeritasAI API")

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = Verifier()

class ClaimRequest(BaseModel):
    text: str

@app.post("/api/verify")
def verify_text(request: ClaimRequest):
    try:
        input_text = request.text.strip()
        extracted_message = None

        # Handle pasted URLs directly
        if input_text.startswith("http"):
            headers = {"User-Agent": "Mozilla/5.0"}
            res = requests.get(input_text, headers=headers, timeout=5)
            soup = BeautifulSoup(res.text, "html.parser")
            paragraphs = soup.find_all("p")
            input_text = " ".join([p.get_text() for p in paragraphs])[:2000]
            extracted_message = f"🔗 Scraped from URL: {request.text}"
            
        result = engine.verify_claim(input_text)
        if extracted_message:
            result["extracted_text"] = extracted_message
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/verify-image")
def verify_image(file: UploadFile = File(...)):
    try:
        # Read image directly from memory, no saving to disk!
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        extracted_text = pytesseract.image_to_string(image).strip()
        if len(extracted_text) < 5:
            raise HTTPException(status_code=400, detail="Could not read text from image.")
            
        result = engine.verify_claim(extracted_text)
        result["extracted_text"] = extracted_text
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))