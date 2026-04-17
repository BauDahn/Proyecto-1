import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Regression.Regression import entrenar_regresion_logistica, predecir_probabilidades
from validation import evaluar_con_bootstrap

def fine_tuning_logistica():
    # 2. Carga de datos
    path_data = "../../data/processed/dataset_clean.csv"
    if not os.path.exists(path_data):
        print(f"Error: No se encuentra el archivo en {path_data}")
        return

    df = pd.read_csv(path_data, sep=";")
    
    # Separación de variables
    X = df.drop("TTO", axis=1).values
    y = df["TTO"].values

    learning_rates = [0.1, 0.01] # Reducimos la rejilla para que no tarde horas
    iterations_list = [1000, 2000]
    umbrales = [0.45, 0.5, 0.55] # Probamos solo los más lógicos

    mejor_precision_media = 0
    mejores_params = {}

    # Reducimos iteraciones de bootstrap para el tuning (ej. 20 o 50)
    # 200 iteraciones por cada combinación sería demasiado lento para probar
    ITER_BOOTSTRAP = 30 

    for lr in learning_rates:
        for iters in iterations_list:
            for umbral in umbrales:
                print(f"Evaluando con Bootstrap: LR={lr}, Iters={iters}, Umbral={umbral}...")
                
                # Llamamos a tu función de validación
                media, desviacion = evaluar_con_bootstrap(
                    X, y, tipo_modelo="logistic", 
                    iteraciones=ITER_BOOTSTRAP, 
                    lr=lr, iters=iters, umbral=umbral
                ) #

                if media > mejor_precision_media:
                    mejor_precision_media = media
                    mejores_params = {
                        'LR': lr, 
                        'Iters': iters, 
                        'Umbral': umbral,
                        'Std': desviacion
                    }

    print("-" * 50)
    print(f"Mejores Parámetros encontrados: {mejores_params}")
    print(f"Precisión media esperada: {mejor_precision_media*100:.2f}%")

if __name__ == "__main__":
    fine_tuning_logistica()
