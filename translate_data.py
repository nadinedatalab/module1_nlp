import pandas as pd
import time
from deep_translator import GoogleTranslator

df = pd.read_excel("NLP.xlsx")
df.columns = ["Disease_Class", "Disease_Definition", "Patient_Statement"]

def translate_text(text: str) -> str:
    """Dịch 1 đoạn text, retry nếu lỗi."""
    if not text or pd.isna(text):
        return ""
    try:
        result = GoogleTranslator(source="en", target="vi").translate(str(text))
        return result
    except Exception as e:
        print(f"[ERROR] {e}")
        time.sleep(2)
        try:
            return GoogleTranslator(source="en", target="vi").translate(str(text))
        except:
            return str(text)  # Fallback giữ nguyên

# Dịch cột Patient_Statement
print(f"Đang dịch {len(df)} dòng Patient_Statement...")
vi_statements = []
for i, text in enumerate(df["Patient_Statement"]):
    translated = translate_text(text)
    vi_statements.append(translated)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(df)} xong")
    time.sleep(0.3)  # Tránh rate limit

df["Patient_Statement_VI"] = vi_statements

# Lưu ra file mới
df.to_excel("NLP_bilingual.xlsx", index=False)
print("Đã lưu NLP_bilingual.xlsx")