"""
Regresión Logística — módulo de entrenamiento y persistencia
=============================================================
Entrena un modelo de clasificación binaria para predecir el tipo de
tratamiento de un aneurisma aórtico: Cirugía Abierta (0) vs EVAR (1).

El modelo se guarda en disco como archivo .pkl y puede ser cargado
directamente por validation.py, que espera un objeto con .predict().

Arquitectura elegida: Pipeline de sklearn
  StandardScaler → LogisticRegression
El escalado va dentro del pipeline para que cualquier llamada a
.predict() o .predict_proba() funcione con datos en bruto, sin
necesidad de escalar manualmente antes.
"""

import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.exceptions import ConvergenceWarning


# ==============================================================
# RUTAS DEL PROYECTO
# ==============================================================
# Calculadas a partir de la ubicación de este archivo para que
# funcionen independientemente del directorio de trabajo.

_AQUI   = os.path.dirname(os.path.abspath(__file__))   # src/Regression/
_SRC    = os.path.dirname(_AQUI)                        # src/
_RAIZ   = os.path.dirname(_SRC)                         # raíz del proyecto

RUTA_DATOS  = os.path.join(_RAIZ, 'data', 'processed', 'dataset_clean.csv')
RUTA_MODELO = os.path.join(_SRC, 'modelo_regresion.pkl')


# ==============================================================
# FUNCIÓN 1: ENTRENAR EL MODELO
# ==============================================================

def entrenar_modelo(X, y, num_iteraciones=1000):
    """
    Crea y entrena un Pipeline con estandarización y regresión logística.

    El Pipeline encadena dos pasos:
      1. StandardScaler: lleva todas las variables a la misma escala
         (media 0, desviación 1) para que el optimizador converja bien.
      2. LogisticRegression: aprende qué combinación de variables
         separa mejor Cirugía Abierta de EVAR.

    Al estar todo en un Pipeline, cualquier llamada posterior a
    .predict() o .predict_proba() escala los datos automáticamente,
    lo que hace al modelo compatible con validate_model() de validation.py.

    Parámetros:
        X               → Matriz de pacientes (una fila por paciente)
        y               → Etiquetas: 0 = Cirugía Abierta, 1 = EVAR
        num_iteraciones → Límite de iteraciones del optimizador

    Devuelve:
        Pipeline entrenado con .predict() y .predict_proba()
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    pipeline = Pipeline([
        ('escalado',     StandardScaler()),
        ('clasificador', LogisticRegression(
            C=1.0,
            max_iter=num_iteraciones,
            solver='lbfgs',     # Eficiente para datasets pequeños como el nuestro
            random_state=42     # Resultados reproducibles
        ))
    ])

    # Capturamos el aviso de convergencia para mostrarlo en español
    # en vez del mensaje técnico que genera sklearn por defecto.
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(X, y)

    if any(issubclass(a.category, ConvergenceWarning) for a in capturados):
        print(f"  Aviso: el modelo no convergió en {num_iteraciones} iteraciones. "
              f"Considera aumentar 'num_iteraciones'.")

    return pipeline


# ==============================================================
# FUNCIÓN 2: GUARDAR EL MODELO EN DISCO
# ==============================================================

def guardar_modelo(modelo, ruta=RUTA_MODELO):
    """
    Serializa el modelo entrenado en un archivo .pkl.

    El archivo resultante puede cargarse con joblib.load() o con
    cargar_modelo() de este mismo módulo, y es compatible con
    validate_model() de validation.py.

    Parámetros:
        modelo → Pipeline entrenado que queremos guardar
        ruta   → Ruta completa del archivo de salida
    """
    joblib.dump(modelo, ruta)
    print(f"Modelo guardado en: {ruta}")


# ==============================================================
# FUNCIÓN 3: CARGAR UN MODELO DESDE DISCO
# ==============================================================

def cargar_modelo(ruta=RUTA_MODELO):
    """
    Carga un modelo previamente guardado con guardar_modelo().

    Devuelve el Pipeline completo con el escalador y el clasificador
    ya entrenados, listo para hacer predicciones.

    Parámetros:
        ruta → Ruta del archivo .pkl a cargar
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el modelo en: {ruta}\n"
            f"Ejecuta primero este módulo para generar el archivo."
        )
    return joblib.load(ruta)


# ==============================================================
# ENTRENAMIENTO Y COMPROBACIÓN
# ==============================================================
# Este bloque entrena el modelo, lo guarda como .pkl y llama a
# validate_model() de validation.py para mostrar las métricas,
# demostrando la integración completa entre los dos módulos.

if __name__ == '__main__':

    # Añadimos src/ al path para poder importar validation.py
    sys.path.insert(0, _SRC)
    from validation import validate_model

    print("=" * 62)
    print("   ENTRENAMIENTO — REGRESIÓN LOGÍSTICA")
    print("=" * 62)

    # ----------------------------------------------------------
    # 1. Cargar datos
    # ----------------------------------------------------------
    print("\n[1/4] Cargando datos clínicos...")
    try:
        df = pd.read_csv(RUTA_DATOS, sep=';')
    except FileNotFoundError:
        print(f"\nError: no se encontró el dataset en:\n  {RUTA_DATOS}")
        sys.exit(1)

    X_df = df.drop(columns=['TTO'])
    y    = df['TTO'].to_numpy().astype(int)
    X    = X_df.to_numpy(dtype=float)

    print(f"   {X.shape[0]} pacientes | {X.shape[1]} variables")
    print(f"   Cirugía Abierta: {int(np.sum(y == 0))} | EVAR: {int(np.sum(y == 1))}")

    # ----------------------------------------------------------
    # 2. Dividir en entrenamiento y test
    # ----------------------------------------------------------
    print("\n[2/4] Dividiendo en entrenamiento (80%) y test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Entrenamiento: {len(X_train)} | Test: {len(X_test)}")

    # ----------------------------------------------------------
    # 3. Entrenar y guardar
    # ----------------------------------------------------------
    print("\n[3/4] Entrenando el modelo...")
    modelo = entrenar_modelo(X_train, y_train)
    guardar_modelo(modelo)

    # ----------------------------------------------------------
    # 4. Validar con validate_model() de validation.py
    # ----------------------------------------------------------
    print("\n[4/4] Evaluando con validation.py...\n")
    exactitud = validate_model(RUTA_MODELO, X_test, y_test)
    print(f"\n   Exactitud final: {exactitud * 100:.2f}%")

    # ----------------------------------------------------------
    # Extra: Validación cruzada sobre todo el dataset
    # ----------------------------------------------------------
    # La validación cruzada evalúa el modelo en 5 particiones distintas
    # y promedia el resultado, lo que es más fiable que un único split.
    # Usamos el pipeline completo para que el escalado no "contamine"
    # los datos de test dentro de cada fold.
    print("\n" + "=" * 62)
    print("   VALIDACIÓN CRUZADA (5-fold)")
    print("=" * 62)

    pipeline_cv = Pipeline([
        ('escalado',     StandardScaler()),
        ('clasificador', LogisticRegression(C=1.0, max_iter=1000,
                                            solver='lbfgs', random_state=42))
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        puntuaciones = cross_val_score(pipeline_cv, X, y, cv=5, scoring='accuracy')

    print(f"\n   Por fold: {' | '.join(f'{p*100:.1f}%' for p in puntuaciones)}")
    print(f"   Media:    {puntuaciones.mean()*100:.2f}%  ±{puntuaciones.std()*100:.2f}%")

    print("\n" + "=" * 62)
    print("   Proceso completado.")
    print("=" * 62)
