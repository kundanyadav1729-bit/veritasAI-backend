import io
import requests
import pytesseract
from PIL import Image, UnidentifiedImageError
from bs4 import BeautifulSoup
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from engine.verifier import Verifier   # requires engine/ package; see README

app = FastAPI(title="VeritasAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = Verifier()

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}


class ClaimRequest(BaseModel):
    text: str


@app.post("/api/verify")
def verify_text(request: ClaimRequest):
    try:
        input_text = request.text.strip()
        extracted_message = None

        if input_text.startswith("http"):
            headers = {"User-Agent": "Mozilla/5.0"}
            try:
                res = requests.get(input_text, headers=headers, timeout=8)
                res.raise_for_status()
            except requests.exceptions.RequestException as e:
                raise HTTPException(status_code=400, detail=f"Could not fetch URL: {e}")

            soup = BeautifulSoup(res.text, "html.parser")
            paragraphs = soup.find_all("p")
            scraped = " ".join(p.get_text(strip=True) for p in paragraphs)[:2000].strip()

            if not scraped:
                raise HTTPException(
                    status_code=400,
                    detail="No readable text found at that URL. Try pasting the article text directly."
                )

            input_text = scraped
            extracted_message = f"🔗 Scraped from URL: {request.text}"

        result = engine.verify_claim(input_text)
        if extracted_message:
            result["extracted_text"] = extracted_message

        return result

    except HTTPException:
        raise   # re-raise 4xx errors as-is — don't wrap in 500
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/verify-image")
def verify_image(file: UploadFile = File(...)):
    # ── 1. Validate MIME type ────────────────────────────────────────────────
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Upload a JPEG, PNG, WEBP, GIF, or BMP image."
        )

    try:
        image_bytes = file.file.read()

        # ── 2. Open image — guard against corrupt / non-image bytes ─────────
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()                    # checks integrity without decoding fully
            image = Image.open(io.BytesIO(image_bytes))   # re-open after verify()
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="File could not be read as an image.")

        # ── 3. OCR ────────────────────────────────────────────────────────────
        extracted_text = pytesseract.image_to_string(image).strip()
        if len(extracted_text) < 5:
            raise HTTPException(
                status_code=400,
                detail="Could not extract readable text from the image. Try a clearer screenshot."
            )

        result = engine.verify_claim(extracted_text)
        result["extracted_text"] = extracted_text
        return result

    except HTTPException:
        raise   # re-raise 4xx errors — don't wrap in 500
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))