from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.src.app.main import predict
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Muestra(BaseModel):
    edad : int
    sexo : bool
    fumador: int
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

