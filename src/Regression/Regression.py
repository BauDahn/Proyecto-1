import numpy as np
import pandas as pd

def estandarizar_datos(X, medias=None, desviaciones=None):
    # Hacemos una copia para no sobreescribir los datos originales
    X_escalado = np.copy(X).astype(float)

    # 1. Detectar qué columnas son continuas (más de 2 valores diferentes)
    columnas_continuas = [i for i in range(X.shape[1]) if len(np.unique(X[:, i])) > 2]

    # Si todo es 0 o 1 no hacemos nada
    if len(columnas_continuas) == 0:
        return X_escalado, None, None

        # 2. Si estamos Entrenando (no nos han pasado medias previas)
    if medias is None or desviaciones is None:
        medias = np.mean(X_escalado[:, columnas_continuas], axis=0)
        desviaciones = np.std(X_escalado[:, columnas_continuas], axis=0)
        desviaciones[desviaciones == 0] = 1e-8  # Protección para evitar dividir por cero

    # 3. Aplicamos la fórmula matemática Z-Score a esas columnas concretas
    X_escalado[:, columnas_continuas] = (X_escalado[:, columnas_continuas] - medias) / desviaciones

    return X_escalado, medias, desviaciones

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
def predecir_probabilidades(X, W, b, medias=None, desviaciones=None):
    X = np.asarray(X, dtype=float)
    # Antes de predecir, estandarizamos al nuevo paciente con las reglas del entrenamiento
    if medias is not None and desviaciones is not None:
        X_escalado, _, _ = estandarizar_datos(X, medias, desviaciones)
    else:
        X_escalado = np.copy(X).astype(float)

    z = np.dot(X_escalado, W) + b
    return sigmoide(z)

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
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # El modelo detecta columnas raras y las estandariza él solo.
    X_estandarizado, medias, desviaciones = estandarizar_datos(X)

    num_variables = X_estandarizado.shape[1]
    W = np.zeros(num_variables)
    b = 0.0
    historial_coste = []

    for i in range(num_iteraciones):
        # OJO: Usamos X_estandarizado para todo el entrenamiento
        z = np.dot(X_estandarizado, W) + b
        y_predicha = sigmoide(z)

        coste = calcular_coste(y, y_predicha)
        historial_coste.append(coste)

        dW, db = calcular_gradientes(X_estandarizado, y, y_predicha)

        W = W - learning_rate * dW
        b = b - learning_rate * db

        if i % 400 == 0:
            print(f"Iteración {i}: Coste = {coste:.4f}")

    # Ahora devolvemos también las medias y desviaciones aprendidas
    return W, b, historial_coste, medias, desviaciones

if __name__ == '__main__':
    X = np.array([
        [55., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0.],
        [72., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 1.],
        [81., 1., 1., 0., 0., 1., 1., 1., 1., 1., 0., 1., 1., 0.],
        [64., 1., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 0., 0.],
        [59., 1., 1., 0., 0., 1., 0., 1., 1., 0., 1., 1., 0., 0.],
        [52., 1., 1., 1., 0., 1., 1., 0., 0., 0., 0., 1., 1., 1.],
        [83., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 0., 1.],
        [74., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 1.],
        [51., 1., 1., 1., 1., 1., 1., 0., 0., 0., 1., 0., 0., 0.],
        [78., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 0.],
        [70., 0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 1., 1., 1.],
        [76., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1.],
        [60., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1.],
        [61., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0., 1.],
        [77., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
        [73., 0., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 0., 1.],
        [48., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 0., 0.],
        [54., 0., 0., 1., 1., 1., 0., 1., 0., 1., 0., 1., 1., 1.],
        [75., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0.],
        [45., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1.]])

    y = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1])

    W, b, historial_coste, medias, std = entrenar_regresion_logistica(X, y, learning_rate=0.01, num_iteraciones=1000)

    # Recogemos las medias y desviaciones también
    W, b, historial_coste, medias, desviaciones = entrenar_regresion_logistica(X, y, learning_rate=0.01,
                                                                               num_iteraciones=1000)

    # Pasamos las medias y desviaciones para que prediga correctamente
    probabilidades_finales = predecir_probabilidades(X, W, b, medias, desviaciones)

    predicciones_firmes = (probabilidades_finales >= 0.5).astype(int)

    aciertos = np.sum(predicciones_firmes == y)
    total_pacientes = len(y)
    precision_global = (aciertos / total_pacientes) * 100

    print(f"\n--- EVALUACIÓN CLÍNICA ---")
    print(f"El modelo ha acertado el diagnóstico de {aciertos} de {total_pacientes} pacientes.")
    print(f"Precisión Global (Accuracy): {precision_global:.2f}%")

    odds_ratios = np.exp(W)
    nombres_variables = ['Edad_Auto_Estandarizada', 'Sexo', 'Fumador', 'Dislipemia', 'Diabetes',
                         'HTA', 'IAM', 'ERC', 'EPOC', 'ACV', 'FA', 'CANCER', 'ICC', 'EAP']

    print("\n--- PESOS DE LAS VARIABLES (ODDS RATIO PARA EVAR) ---")
    for nombre, or_val in zip(nombres_variables, odds_ratios):
        print(f"{nombre}: {or_val:.2f}")
