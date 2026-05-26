import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

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
    """Xử lý phủ định cả tiếng Anh và tiếng Việt."""
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
# LOAD DATA — NLP_bilingual.xlsx (sau khi chạy translate_data.py)
# ============================================================

df = pd.read_excel("NLP_bilingual.xlsx")
df.columns = ["Disease_Class", "Disease_Definition", "Patient_Statement", "Patient_Statement_VI"]

CLASS_MAP = {
    "Moles": "Nevus", "Seborrh_Keratoses": "Keratosis",
    "Tinea": "Tinea", "Acne": "Acne", "Unknown": "Unknown",
}
df["label"] = df["Disease_Class"].map(CLASS_MAP)
df = df.dropna(subset=["label"])

# Tạo 2 bộ data: tiếng Anh và tiếng Việt đã dịch
df_en = df[["Patient_Statement", "label"]].copy()
df_en.columns = ["text", "label"]

df_vi = df[["Patient_Statement_VI", "label"]].copy()
df_vi.columns = ["text", "label"]

# Ghép lại — gấp đôi data, model học cả 2 ngôn ngữ
df_all = pd.concat([df_en, df_vi], ignore_index=True)
df_all = df_all.dropna(subset=["text"])
df_all["processed"] = df_all["text"].apply(preprocess)

print("Phân bố class:")
print(df_all["label"].value_counts())
print(f"Tổng mẫu: {len(df_all)}")

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X = df_all["processed"]
y = df_all["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ============================================================
# PIPELINE TF-IDF + SVM
# ============================================================

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        min_df=2,
        max_df=0.90,
        analyzer="word",
    )),
    ("svm", SVC(
        kernel="linear",
        C=0.2,
        probability=True,
        class_weight="balanced",
    )),
])

pipeline.fit(X_train, y_train)

# ============================================================
# ĐÁNH GIÁ
# ============================================================

y_pred = pipeline.predict(X_test)
print("\n=== KẾT QUẢ ĐÁNH GIÁ ===")
print(classification_report(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy test set: {acc*100:.1f}%")

cv = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
print(f"CV Accuracy (5-fold): {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%")
print(f"Từng fold: {[round(x*100,1) for x in cv]}")
print(f"\n=> Độ chính xác thực tế: ~{cv.mean()*100:.0f}%")

cv_f1 = cross_val_score(pipeline, X, y, cv=5, scoring="f1_macro")
print(f"CV F1 (5-fold): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

# ============================================================
# LƯU MODEL
# ============================================================

with open("model_nlp.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\nModel đã lưu vào model_nlp.pkl")