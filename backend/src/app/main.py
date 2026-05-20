import json
import math

with open("src/models/coeficientes.json", "r") as f:
    MODELO = json.load(f)

def predict(datos):
    suma_comorbilidades = (
        datos.dislipemia + datos.hta + datos.iam + datos.erc +
        datos.epoc + datos.acv + datos.fa + datos.cancer + datos.icc + datos.eap
    )

    if 3 <= suma_comorbilidades <= 5:
        x_comorb_medio = 1
        x_comorb_alto = 0
    elif suma_comorbilidades > 5:
        x_comorb_medio = 0
        x_comorb_alto = 1
    else:
        x_comorb_medio = 0
        x_comorb_alto = 0

    x_edad = datos.edad
    x_sexo = 1 if datos.sexo else 0
    x_fumador1 = 1 if datos.fumador == "Fumador activo" else 0
    x_fumador2 = 1 if datos.fumador == "Ex-fumador" else 0

    z = MODELO["intercepto"]
    z += MODELO["betas"]["Edad"] * x_edad
    z += MODELO["betas"]["SexoTRUE"] * x_sexo
    z += MODELO["betas"]["Fumador1"] * x_fumador1
    z += MODELO["betas"]["Fumador2"] * x_fumador2
    z += MODELO["betas"]["comorb_gruposMedio"] * x_comorb_medio
    z += MODELO["betas"]["comorb_gruposAlto"] * x_comorb_alto

    probabilidad = 1 / (1 + math.exp(-z))

    if probabilidad <= 0.3:
        riesgo = 'Bajo'
    elif probabilidad <= 0.6:
        riesgo = 'Medio'
    else:
        riesgo = 'Alto'

    return {
        "riesgo": riesgo,
        "probabilidad": probabilidad,
        "nota": "Conexión exitosa con el modelo"
    }