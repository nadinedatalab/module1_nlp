import re
import os
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Model 1 - NLP Skin Disease Classification")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("model_nlp.pkl", "rb") as f:
    model = pickle.load(f)

# ============================================================
# DATABASE
# ============================================================

DATABASE_URL = (
    f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
    f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DATABASE')}"
    f"?ssl_verify_cert=false"
)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def get_description(disease_name: str) -> str:
    db = SessionLocal()
    try:
        result = db.execute(
            text("SELECT description FROM diseases WHERE disease_name = :name"),
            {"name": disease_name}
        ).fetchone()
        return result[0] if result else "Không có mô tả"
    except Exception:
        return "Không có mô tả"
    finally:
        db.close()

# ============================================================
# PREPROCESSING
# ============================================================

ABBREVIATIONS = {
    "ko": "không", "k": "không", "kg": "không", "khong": "không",
    "bj": "bị", "dc": "được", "đc": "được",
    "mun": "mụn", "ngua": "ngứa", "nam": "nấm",
    "dau": "đau", "sung": "sưng", "do": "đỏ",
    "san": "sần", "sui": "sùi", "day": "dày",
}

NEGATION_WORDS_EN = ["no", "not", "never", "without", "don't", "doesn't",
                     "isn't", "aren't", "none", "neither", "nor"]

NEGATION_WORDS_VI = ["không", "chưa", "chẳng", "chả"]

SKIP_WORDS = {"tôi", "mình", "tớ", "bạn", "em", "anh", "chị"}

def normalize_vi(text: str) -> str:
    text = text.lower().strip()
    words = text.split()
    words = [ABBREVIATIONS.get(w, w) for w in words]
    return " ".join(words)

def handle_negation(text: str) -> str:
    words = text.lower().split()
    result = []
    negate_count = 0
    for word in words:
        clean = re.sub(r"[^\w]", "", word)
        if clean in NEGATION_WORDS_EN or clean in NEGATION_WORDS_VI:
            negate_count = 3
            result.append(clean)
            continue
        if clean in SKIP_WORDS:
            result.append(clean)
            continue
        if negate_count > 0:
            result.append(f"NOT_{clean}")
            negate_count -= 1
        else:
            result.append(clean)
    return " ".join(result)

def preprocess(text: str) -> str:
    text = str(text).lower().strip()
    text = normalize_vi(text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = handle_negation(text)
    return text

# ============================================================
# API
# ============================================================

class TextRequest(BaseModel):
    text: str

@app.post("/predict-text")
def predict_text(request: TextRequest):
    raw = request.text.strip()

    if not raw:
        return {
            "prediction": "Unknown",
            "confidence": 0.0,
            "probabilities": {},
            "description": get_description("Unknown")
        }

    text = normalize_vi(raw)
    text = preprocess(text)

    prediction = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    confidence = float(max(proba))

    classes = model.classes_

    probabilities = {
        cls: round(float(prob), 5)
        for cls, prob in zip(classes, proba)
    }

    if confidence < 0.20:
        prediction = "Unknown"

    description = get_description(prediction)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "description": description
    }

@app.get("/health")
def health():
    return {"status": "ok"}