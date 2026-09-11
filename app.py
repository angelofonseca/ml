import io
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

artifact = joblib.load(BASE_DIR / "knn_model.joblib")
scaler = artifact["scaler"]
model = artifact["model"]
FEATURES = list(scaler.feature_names_in_)

app = FastAPI(title="Classificador de exame")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(400, "O arquivo enviado está vazio.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(
            400, "Não foi possível ler o arquivo. Envie um CSV com cabeçalho.")

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Faltam as colunas: {', '.join(missing)}")

    if len(df) == 0:
        raise HTTPException(400, "O arquivo não tem nenhuma linha de exame.")
    if len(df) > 1:
        raise HTTPException(
            400, f"O arquivo deve ter exatamente uma linha de exame (encontrei {len(df)}).")

    exam = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    invalid = [c for c in FEATURES if exam[c].isna().any()]
    if invalid:
        raise HTTPException(
            400, f"Valores ausentes ou não numéricos em: {', '.join(invalid)}")

    proba = model.predict_proba(scaler.transform(exam))[0]
    malignant_index = list(model.classes_).index(1)
    prob_malignant = float(proba[malignant_index])

    return {
        "diagnosis": "Maligno" if prob_malignant >= 0.5 else "Benigno",
        "prob_malignant": prob_malignant,
        "prob_benign": 1 - prob_malignant,
        "neighbors": model.n_neighbors,
    }
