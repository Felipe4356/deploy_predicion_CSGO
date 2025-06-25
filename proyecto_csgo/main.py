from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Inicializar FastAPI
app = FastAPI(title="Simulador CSGO")

# Cargar modelos entrenados
modelo_regresion = joblib.load("models/modelo_regresion.pkl")
modelo_clasificacion = joblib.load("models/modelo_clasificacion.pkl")
columnas_clasificacion = joblib.load("models/columnas_clasificacion.pkl")

# Configurar carpeta de plantillas HTML
templates = Jinja2Templates(directory="templates")

# Ruta principal
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -----------------------------
# Esquemas de entrada
# -----------------------------
class InputRegresion(BaseModel):
    TeamStartingEquipmentValue: float

class InputClasificacion(BaseModel):
    TeamStartingEquipmentValue: float
    MatchKills: int
    MatchAssists: int
    Map: int

# -----------------------------
# Endpoints
# -----------------------------
@app.post("/predict/regresion")
def predict_regresion(data: InputRegresion):
    try:
        entrada = pd.DataFrame(
            [[data.TeamStartingEquipmentValue]],
            columns=["TeamStartingEquipmentValue"]
        )
        prediccion = modelo_regresion.predict(entrada)
        return {"RoundStartingEquipmentValue_predicho": float(prediccion[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción de regresión: {str(e)}")

@app.post("/predict/clasificacion")
def predict_clasificacion(data: InputClasificacion):
    try:
        entrada = pd.DataFrame(
            [[
                data.TeamStartingEquipmentValue,
                data.MatchKills,
                data.MatchAssists,
                data.Map
            ]],
            columns=columnas_clasificacion
        )
        print("Entrada al modelo de clasificación:", entrada)
        prediccion = modelo_clasificacion.predict(entrada)
        return {"MatchWinner_predicho": int(prediccion[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción de clasificación: {str(e)}")
