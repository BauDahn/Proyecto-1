from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.app.main import predict
from pydantic import BaseModel
from typing import Literal

app = FastAPI()

origins = [
    "https://dkg5bcnx-5173.uks1.devtunnels.ms", 
    "http://localhost:5173", # Para trabajar de forma local
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Muestra(BaseModel):
    edad : int
    sexo : bool
    fumador: Literal['No fuma', 'Fumador activo', 'Ex-fumador']
    dislipemia: bool
    diabetes : bool
    hta:  bool
    iam: bool
    erc: bool
    epoc: bool
    acv: bool
    fa: bool
    cancer: bool
    icc: bool
    eap: bool

# Edad, Sexo, Fumador, Dislipemia, Diabetes, HTA, IAM, ERC, EPOC, ACV, FA, Cancer, ICC, EAP
@app.post("/predictor")
async def predictor(datos: Muestra):
    return predict(datos)

