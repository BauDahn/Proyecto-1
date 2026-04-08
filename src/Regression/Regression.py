import numpy as np
import pandas as pd

# Función para inicializar el modelo
def inicializar_parametros(num_variables):
    # En nuestro caso, 14 variables
    # 'W' (los pesos) es un vector simple (una lista de 14 ceros)
    # Queremos predecir una sola cosa, si es EVAR o no
    W = np.zeros(num_variables)

    # El intercepto (término independiente) es un solo número escalar.
    b = 0.0

    return W, b


# Función sigmoid
def sigmoide(z):
    # Convertir cualquier número 'z' para que esté exactamente entre 0 y 1
    return 1 / (1 + np.exp(-z))


# 3. Función para predecir (Forward Pass)
def predecir_probabilidades(X, W, b):
    # X es nuestra matriz de pacientes
    # Calculamos el "score clínico" (z) multiplicando cada variable por su peso
    # Matemáticamente es el producto punto (dot product)
    z = np.dot(X, W) + b

    # Pasamos los scores por la sigmoide para obtener la probabilidad de EVAR (clase 1)
    probabilidades = sigmoide(z)

    return probabilidades

def calcular_coste(y_real, y_predicha):
    # m es el número total de pacientes (filas)
    m = y_real.shape[0]

    # Añadimos un valor pequeño (epsilon) a las predicciones para evitar
    # calcular el logaritmo de 0, lo cual nos daría error.
    epsilon = 1e-15
    #Utilizamos la función np.clip para evitar el calculo de ln(0)
    y_predicha = np.clip(y_predicha, epsilon, 1 - epsilon)

    # Calculamos el coste aplicando la fórmula exacta
    coste = - (1 / m) * np.sum(y_real * np.log(y_predicha) + (1 - y_real) * np.log(1 - y_predicha))

    return coste


def calcular_gradientes(X, y_real, y_predicha):
    m = X.shape[0]

    # Calculamos el error de nuestra predicción
    error = y_predicha - y_real

    # Calculamos el gradiente de los pesos (dW)
    # np.dot(X.T, error) multiplica cada variable clínica por el error que ha causado
    dW = (1 / m) * np.dot(X.T, error)

    # 3. Calculamos el gradiente del sesgo (db)
    db = (1 / m) * np.sum(error)

    return dW, db


def entrenar_regresion_logistica(X, y, learning_rate=0.01, num_iteraciones=1000):
    # Inicializamos W y b con ceros
    num_variables = X.shape[1]
    W = np.zeros(num_variables)
    b = 0.0

    # Lista para ir guardando el error y ver si el modelo realmente aprende
    historial_coste = []

    # Bucle de entrenamiento (Descenso de Gradiente)
    for i in range(num_iteraciones):

        # PASO A: El modelo intenta predecir con sus pesos actuales
        z = np.dot(X, W) + b
        y_predicha = 1 / (1 + np.exp(-z))  # Función Sigmoide

        # PASO B: Calculamos cuánto se ha equivocado (Coste)
        coste = calcular_coste(y, y_predicha)
        historial_coste.append(coste)

        # PASO C: Calculamos los gradientes (la dirección para corregir)
        dW, db = calcular_gradientes(X, y, y_predicha)

        # PASO D: actualizamos los pesos
        # Restamos el gradiente multiplicado por el learning_rate
        W = W - learning_rate * dW
        b = b - learning_rate * db

        # Imprimimos el coste cada 100 iteraciones para monitorizar
        if i % 400 == 0:
            print(f"Iteración {i}: Coste = {coste:.4f}")

    return W, b, historial_coste

if __name__ == '__main__':
    X = np.array([
        [-1.09388026, 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0.],
        [0.47743945, 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 1.],
        [1.56527617, 1., 1., 0., 0., 1., 1., 1., 1., 1., 0., 1., 1., 0.],
        [-0.12691428, 1., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 0., 0.],
        [-0.61039727, 1., 1., 0., 0., 1., 0., 1., 1., 0., 1., 1., 0., 0.],
        [-0.97300951, 1., 1., 1., 0., 1., 1., 0., 0., 0., 0., 1., 1., 1.],
        [1.56527617, 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 0., 1.],
        [0.5983102, 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 1.],
        [-1.09388026, 1., 1., 1., 1., 1., 1., 0., 0., 0., 1., 0., 0., 0.],
        [1.20266393, 1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 0.],
        [0.3565687, 0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 1., 1., 1.],
        [0.84005169, 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1.],
        [-0.61039727, 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1.],
        [-0.61039727, 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0., 1.],
        [0.96092244, 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
        [0.5983102, 0., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 0., 1.],
        [-1.4564925, 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 0., 0.],
        [-0.97300951, 0., 0., 1., 1., 1., 0., 1., 0., 1., 0., 1., 1., 1.],
        [0.96092244, 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0.],
        [-1.57736324, 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1.]])

    y = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1])

    W, b, historial_coste = entrenar_regresion_logistica(X, y, learning_rate=0.01, num_iteraciones=1000)

    # 1. HACER PREDICCIONES CON LOS PESOS APRENDIDOS
    # (Asumiendo que tus variables del modelo se llaman W y b, y tus datos X e y)
    z_final = np.dot(X, W) + b
    probabilidades_finales = 1 / (1 + np.exp(-z_final)) # Sigmoide

    # Si la probabilidad es mayor a 0.5, decimos que es EVAR (1), si no, Cirugía (0)
    predicciones_firmes = (probabilidades_finales >= 0.5).astype(int)

    # 2. CALCULAR EL PORCENTAJE DE ACIERTO (ACCURACY)
    aciertos = np.sum(predicciones_firmes == y)
    total_pacientes = len(y)
    precision_global = (aciertos / total_pacientes) * 100

    print(f"\n--- EVALUACIÓN CLÍNICA ---")
    print(f"El modelo ha acertado el diagnóstico de {aciertos} de {total_pacientes} pacientes.")
    print(f"Precisión Global (Accuracy): {precision_global:.2f}%")

    # 3. EXTRAER LOS ODDS RATIOS (INTERPRETABILIDAD CLÍNICA)
    # Convertimos los pesos (Log-Odds) a Odds Ratios con la exponencial
    odds_ratios = np.exp(W)

    # Nombres de tus columnas originales (excepto TTO y FORMA INTERV)
    nombres_variables = ['Edad_Estandarizada', 'Sexo', 'Fumador', 'Dislipemia', 'Diabetes',
                         'HTA', 'IAM', 'ERC', 'EPOC', 'ACV', 'FA', 'CANCER', 'ICC', 'EAP']

    print("\n--- PESOS DE LAS VARIABLES (ODDS RATIO PARA EVAR) ---")
    for nombre, or_val in zip(nombres_variables, odds_ratios):
        print(f"{nombre}: {or_val:.2f}")