import random
import train_model.py

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

# Integración a la validación
def evaluar_con_bootstrap(X, y, iteraciones=100):
    metricas = []
    
    for i in range(iteraciones):
        # Generamos los datos de esta simulación
        X_tr, y_tr, X_te, y_te = generar_muestra_bootstrap(X, y)
        
        pesos, sesgos = entrenar_regresion_logistica(X_tr, y_tr)
        
        # Evaluamos con el set de pacientes que no fue recogido en la muestra bootstrap
        exactitud = calcular_exactitud(X_te, y_te, pesos, sesgos)
        metricas.append(exactitud)

    # Ahora hacemos la media de las métricas como el optimismo que tenemos que restarle al RMSE de la muestra ajustada
    media_diferencias = sum(metricas) / len(metricas) 

    return media_diferencias