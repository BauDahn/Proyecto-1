import numpy as np
import pandas as pd
#from Tree.tree import DecisionTree
from Tree.explainability import tree_visualizer
from Regression import Regression
import random

def generar_muestra_bootstrap(X, y):
    '''
    X es el dataset de pacientes sin el tipo de tratamiento que deben recibir
    y es el dataset de 1 columna por len(X) filas donde está el tipo de tratamiento que recibió esa persona
    '''
    
    tamaño_muestral = len(X)
    
    '''
    X_train = Muestra bootstrap sin tipo de tratamiento recibido
    y_train = Tipo de tratamiento que recibió esa persona

    X_test = Matriz con datos de los pacientes que no están en la muestra bootstrap
    y_test = Tipo de tratamiento que recibió cada paciente que no está en la muestra bootstrap
    '''

    X_train, y_train = [], []
    X_test, y_test = [], []
    
    # Lista para llevar el registro de qué índices hemos seleccionado
    indices_seleccionados = set()
    
    # Crear el set de entrenamiento
    for i in range(tamaño_muestral):
        idx_aleatorio = random.randint(0, tamaño_muestral - 1)
        
        X_train.append(X[idx_aleatorio]) # Metemos en la muestra bootstrap al paciente
        y_train.append(y[idx_aleatorio]) # Metemos el tipo de tratamiento de ese paciente
        
        indices_seleccionados.add(idx_aleatorio) # Guardamos el índice que se seleccionó
        
    # Crear el set de pacientes que no fueron recogidos en la muestra bootsrap
    for i in range(tamaño_muestral):
        if i not in indices_seleccionados: # Si el índice no se seleccionó
            X_test.append(X[i]) # Metemos al paciente en la matriz de pacientes que no salieron
            y_test.append(y[i]) # Metemos el tipo de tratamiento que le correspondió a ese paciente
            
    return X_train, y_train, X_test, y_test


def evaluar_con_bootstrap(X, y, tipo_modelo="tree", iteraciones=200, max_depth=10, min_samples=5, cantidad_minima_en_nodo=1):
    metricas = []
    
    for i in range(iteraciones):
        # Generamos los datos de esta simulación
        X_tr, y_tr, X_te, y_te = generar_muestra_bootstrap(X, y)

        X_tr, y_tr = np.array(X_tr), np.array(y_tr)
        X_te, y_te = np.array(X_te), np.array(y_te)

        if tipo_modelo == "tree":
            # Lógica para el modelo de árbol
            modelo = DecisionTree(min_samples=10, max_depth=15, cantidad_minima_en_nodo=cantidad_minima_en_nodo)
            modelo.fit(X_tr, y_tr)
            predicciones = modelo.predict(X_te)
            exactitud = np.mean(predicciones == y_te)

        elif tipo_modelo == "logistic":
            # Recogemos W, b, ignoramos coste (_), y recogemos medias y stds
            pesos, sesgos, _, medias, stds = Regression.entrenar_regresion_logistica(X_tr, y_tr, learning_rate=0.01, num_iteraciones=1000)

            # Le pasamos las medias y stds del entrenamiento a los pacientes de test
            probabilidades = Regression.predecir_probabilidades(X_te, pesos, sesgos, medias, stds)

            # Las convertimos a decisiones firmes (0 o 1) con el umbral de 0.5
            predicciones_firmes = (probabilidades >= 0.5).astype(int)

            # Calculamos la exactitud (Accuracy) comparando con la realidad (y_te)
            exactitud = np.mean(predicciones_firmes == y_te)
        
        metricas.append(exactitud)
        

        if (i + 1) % 10 == 0:
            print(f'\nIteracion {i} completada...\n')
            print("----------\n")

    # Ahora hacemos la media de las métricas como el optimismo que tenemos que restarle al RMSE de la muestra ajustada

    return np.mean(metricas), np.std(metricas)


if __name__ == '__main__':
    df = pd.read_csv('../data/processed/dataset_clean.csv', sep=';')

    X_df = df.drop(columns=["TTO"])
    y_series = df["TTO"]


    X = X_df.to_numpy(dtype=float)
    y = y_series.to_numpy().astype(int)

    '''
    # Testeo del árbol
    print("Evaluación del primer árbol")
    media_tree, std_tree = evaluar_con_bootstrap(X, y, tipo_modelo="tree", max_depth=5, min_samples=10, cantidad_minima_en_nodo=4)
    print("Primer árbol entrenado...")
    print("-----------------------")
    print("-----------------------")
    print("\nResultados:\n")
    print(f"Modelo: Árbol con todos los pacientes.\nPrecisión: {media_tree*100:.2f}% (+/- {std_tree*100:.2f}%)")
    '''
    '''
    print("\nVisualización del árbol generado...")

    # Entrenamos un árbol final para visualizarlo
    arbol_final = DecisionTree(max_depth=7, min_samples=8, cantidad_minima_en_nodo=5)
    arbol_final.fit(X, y)

    # Primero agarramos los nombres de las columnas del dataset
    nombres_columnas = X_df.columns.tolist()

    # Ahora generamos el código dot para visualizar
    dot_code = tree_visualizer(arbol_final, feature_names=nombres_columnas)

    # Lo guardamos en un archivo para poder verlo después
    with open("resultado_arbol.dot", "w") as f:
        f.write(dot_code)

    print("\nVisualización generada en 'resultado_arbol.dot'.")
    '''

    # ==========================================


    # Testeo de la Regresión Logística
    print("\nEvaluación de la primera Regresión Logística")
    media_log, std_log = evaluar_con_bootstrap(X, y, tipo_modelo="logistic")
    print("Primera Regresión Logística evaluada...")
    print("-----------------------")

    print("\nResultados Regresión Logística:\n")
    print(
        f"Modelo: Regresión Logística con todos los pacientes.\nPrecisión: {media_log * 100:.2f}% (+/- {std_log * 100:.2f}%)")
    print("\n----------------\n")