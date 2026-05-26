import re
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Model 1 - NLP Skin Disease Classification")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("model_nlp.pkl", "rb") as f:
    model = pickle.load(f)

DISEASE_INFO = {
    "Acne": {
        "description": "Mụn trứng cá là tình trạng viêm da mãn tính xảy ra khi lỗ chân lông bị tắc nghẽn bởi dầu nhờn và tế bào da chết. Vi khuẩn Cutibacterium acnes phát triển gây viêm nhiễm. Bệnh thường gặp ở tuổi dậy thì nhưng có thể xảy ra ở mọi lứa tuổi. Biểu hiện gồm mụn đầu trắng, mụn đầu đen, mụn viêm đỏ và mụn bọc.",
    },
    "Tinea": {
        "description": "Tinea là bệnh nhiễm nấm ngoài da do các loài nấm dermatophyte gây ra. Biểu hiện điển hình là vùng da ngứa, có hình tròn hoặc bầu dục với bờ rõ ràng, bong vảy. Bệnh lây qua tiếp xúc trực tiếp hoặc dùng chung đồ dùng cá nhân. Các thể phổ biến: hắc lào, nấm chân, nấm bẹn.",
    },
    "Keratosis": {
        "description": "Keratosis là tình trạng da dày lên bất thường do tích tụ keratin. Seborrheic keratosis là dạng lành tính phổ biến, xuất hiện dưới dạng mảng nâu, vàng hoặc đen, bề mặt sần như sáp. Actinic keratosis do tia UV gây ra, biểu hiện là mảng da thô ráp, có vảy.",
    },
    "Nevus": {
        "description": "Nevus là tình trạng tăng sinh sắc tố melanin tạo thành đốm hoặc nốt trên da. Thường lành tính, màu nâu đến đen, kích thước vài mm đến vài cm. Cần theo dõi theo nguyên tắc ABCDE: Asymmetry, Border, Color, Diameter, Evolution để phát hiện sớm nguy cơ.",
    },
    "Unknown": {
        "description": "Hệ thống không đủ thông tin để xác định bệnh da liễu cụ thể. Vui lòng mô tả chi tiết hơn về triệu chứng hoặc tải ảnh lên để được phân tích chính xác hơn.",
    },

}

# ============================================================
# PREPROCESSING — giống hệt train
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
            "description": DISEASE_INFO["Unknown"]["description"],
            "probabilities": {}
        }

    text = normalize_vi(raw)   
    text = preprocess(text)    

    prediction = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    confidence = float(max(proba))

    classes = model.classes_

    # Convert sang dict
    probabilities = {
        cls: round(float(prob), 5)
        for cls, prob in zip(classes, proba)
    }
    if confidence < 0.20:
        prediction = "Unknown"

    info = DISEASE_INFO[prediction]

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "description": info["description"],
        "probabilities": probabilities
    }

@app.get("/health")
def health():
    return {"status": "ok"}