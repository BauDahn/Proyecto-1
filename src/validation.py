import numpy as np
import pandas as pd
from tree import DecisionTree
import train_model
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


def evaluar_con_bootstrap(X, y, tipo_modelo="tree", iteraciones=200, max_depth=10, min_samples=5):
    metricas = []
    
    for i in range(iteraciones):
        # Generamos los datos de esta simulación
        X_tr, y_tr, X_te, y_te = generar_muestra_bootstrap(X, y)

        X_tr, y_tr = np.array(X_tr), np.array(y_tr)
        X_te, y_te = np.array(X_te), np.array(y_te)

        if tipo_modelo == "tree":
            # Lógica para el modelo de árbol
            modelo = DecisionTree(min_samples=10, max_depth=15)
            modelo.fit(X_tr, y_tr)
            predicciones = modelo.predict(X_te)
            exactitud = np.mean(predicciones == y_te)
        
        elif tipo_modelo == "logistic":
            # Lógica para la regresión logística.
            pesos, sesgos = train_model.entrenar_regresion_logistica(X_tr, y_tr)
        
            # Evaluamos con el set de pacientes que no fue recogido en la muestra bootstrap
            exactitud = train_model.calcular_exactitud(X_te, y_te, pesos, sesgos)
        
        metricas.append(exactitud)
        

        if (i + 1) % 10 == 0:
            print(f'Iteracion {i} completada...')

    # Ahora hacemos la media de las métricas como el optimismo que tenemos que restarle al RMSE de la muestra ajustada

    return np.mean(metricas), np.std(metricas)


if __name__ == '__main__':
    df = pd.read_csv('../data/processed/processed_data.csv')
    df_alt = pd.read_csv('../data/processed/alternative_data.csv')

    X_df = df.drop(columns=["TTO"])
    y_series = df["TTO"]

    X_alt_df = df_alt.drop(columns=["TTO"])
    y_alt_series = df_alt["TTO"]


    X = X_df.to_numpy()
    y = y_series.to_numpy().astype(int)

    X_alt = X_alt_df.to_numpy()
    y_alt = y_alt_series.to_numpy().astype(int)
    

    # Testeo del árbol
    print("Evaluación del primer árbol")
    media_tree, std_tree = evaluar_con_bootstrap(X, y, tipo_modelo="tree", max_depth=5, min_samples=10)
    print("Primer árbol entrenado...")
    print("-----------------------")
    print("Evaluación del segundo árbol")
    media_alt_tree, std_alt_tree = evaluar_con_bootstrap(X_alt, y_alt, tipo_modelo="tree", max_depth=3, min_samples=10)
    print("Segundo árbol entrenado")
    print("-----------------------")
    print("\nResultados:\n")
    print(f"Modelo: Árbol con todos los pacientes.\nPrecisión: {media_tree*100:.2f}% (+/- {std_tree*100:.2f}%)")
    print("\n----------------\n")
    print(f"Modelo: Árbol solo con pacientes con diámetro.\nPrecisión: {media_alt_tree*100:.2f}% (+/- {std_alt_tree*100:.2f}%)")

