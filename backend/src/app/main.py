def predict(datos):
    print(f"Recibidos datos de paciente de {datos.edad} años")
        
    return {
        "riesgo": "Bajo",
        "probabilidad": 0.15,
        "nota": "Conexión exitosa con el modelo"
    }