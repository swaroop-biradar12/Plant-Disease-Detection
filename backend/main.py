from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import io
import os
import time
from google import genai
from google.genai import types
from gtts import gTTS
from dotenv import load_dotenv

from onnx_infer import ONNXDetector

load_dotenv()  # ← reads .env file and loads values into os.environ

# ─── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Plant Disease Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load ONNX model once at startup (no torch, no ultralytics) ─────────────────
MODEL_PATH = "best.onnx"
detector = None

@app.on_event("startup")
def load_model():
    global detector
    if os.path.exists(MODEL_PATH):
        detector = ONNXDetector(MODEL_PATH)
        print("✅ ONNX model loaded successfully (lightweight, no torch)")
    else:
        print("⚠️ best.onnx not found! Place it in the backend folder.")


# ─── Health check endpoint ──────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Plant Disease Detection API is live",
        "model_loaded": detector is not None
    }


# ─── Main prediction endpoint ───────────────────────────────────────────────────
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = Form(0.5),
    gemini_key: str = Form(default=""),
    language: str = Form("English")
):
    if detector is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded"})

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ── Run lightweight ONNX detection ───────────────────────────────────────────
    detections = detector.predict(image, conf_threshold=confidence)
    detected_diseases = list({d["disease"] for d in detections})

    response_data = {
        "detections": detections,
        "total_detections": len(detections),
        "healthy": len(detections) == 0
    }

    # ── Gemini VLM advice (always runs — uses your key or server fallback) ──────
    key_to_use = gemini_key.strip() if gemini_key else ""
    if not key_to_use:
        key_to_use = os.getenv("GEMINI_API_KEY", "")

    if key_to_use:
        advice = get_gemini_advice(image, detected_diseases, language, key_to_use)
        response_data["ai_advice"] = advice
    else:
        response_data["ai_advice"] = "Gemini error: No API key available (server key not configured)."

    return response_data


# ─── Gemini helper function ─────────────────────────────────────────────────────
def get_gemini_advice(image, detected_diseases, language, api_key):
    import time as time_module

    try:
        client = genai.Client(api_key=api_key)

        disease_list = ", ".join(detected_diseases) if detected_diseases else "No disease detected (note: this could mean a healthy leaf, or the image may not contain a plant leaf at all — assess the image content before concluding it's a healthy plant)"

        lang_instruction = ""
        if language == "Hindi":
            lang_instruction = "Respond entirely in Hindi."
        elif language == "Marathi":
            lang_instruction = "Respond entirely in Marathi."

        prompt = f"""You are an expert agricultural plant pathologist.

A plant disease detection AI has analyzed this leaf image and detected: {disease_list}

IMPORTANT: Before writing your diagnosis, first check whether the image actually shows a plant leaf at all.
If the image does NOT show a plant leaf (e.g. it shows a person, document, object, animal, or anything unrelated to plants),
you MUST respond with exactly this format instead:

OVERVIEW: This image does not appear to show a plant leaf, so a disease diagnosis cannot be provided.
SEVERITY: N/A
CAUSE: N/A
SYMPTOMS: No leaf detected in the image.
TREATMENT: Please upload a clear photo of a plant leaf | Ensure the leaf fills most of the frame | Use good lighting for best results
PREVENTION: N/A

If the image DOES show a plant leaf, proceed with a normal diagnosis using this exact compact format, no extra text before or after:

OVERVIEW: <one sentence on what the disease is and its cause, or that the leaf looks healthy>
SEVERITY: <one or two words only, e.g. "Moderate to severe", or "None" if healthy>
CAUSE: <2-4 words, e.g. "Fungal infection", or "N/A" if healthy>
SYMPTOMS: <one sentence describing key visible signs, or lack thereof>
TREATMENT: <bullet 1> | <bullet 2> | <bullet 3> | <bullet 4>
PREVENTION: <bullet 1> | <bullet 2> | <bullet 3>

{lang_instruction}

Each bullet must be a short actionable phrase, max 12 words. Do not add headings, markdown, or any text outside this format."""

        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        types.Part.from_text(text=prompt)
                    ]
                )
                return response.text
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "503" in error_str or "UNAVAILABLE" in error_str or "overloaded" in error_str.lower():
                    if attempt < max_retries - 1:
                        time_module.sleep(2 * (attempt + 1))
                        continue
                raise last_error

        raise last_error

    except Exception as e:
        return f"Gemini error: {str(e)}"


# ─── Text-to-speech endpoint ─────────────────────────────────────────────────────
@app.post("/speak")
async def speak(
    text: str = Form(...),
    language: str = Form("English")
):
    try:
        lang_codes = {"English": "en", "Hindi": "hi", "Marathi": "mr"}
        lang_code = lang_codes.get(language, "en")

        tts = gTTS(text=text, lang=lang_code, slow=False)

        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return StreamingResponse(audio_buffer, media_type="audio/mpeg")

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": f"{type(e).__name__}: {str(e)}"})