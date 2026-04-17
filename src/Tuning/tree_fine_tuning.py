import numpy as np
import pandas as pd
from Tree.tree import DecisionTree
from validation import evaluar_con_bootstrap

# 1. Definimos la "Rejilla" (Grid) de parámetros a probar
param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'min_samples': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'cantidad_minima_en_nodo': [2, 3, 4, 5, 6]
}

def hyperparameter_tuning(X, y):
    mejor_score = 0
    mejores_params = {}
    
    # Recorremos todas las combinaciones posibles
    for depth in param_grid['max_depth']:
        for min_s in param_grid['min_samples']:
            for min_leaf in param_grid['cantidad_minima_en_nodo']:
                
                print(f"Probando: Depth={depth}, MinSamples={min_s}, MinLeaf={min_leaf}")
                
                # Usamos el Bootstrap para una evaluación robusta
                # Reducimos las iteraciones a 50 para que el tuning sea más rápido
                media, std = evaluar_con_bootstrap(
                    X, y, 
                    tipo_modelo="tree", 
                    iteraciones=50, 
                    max_depth=depth, 
                    min_samples=min_s, 
                    cantidad_minima_en_nodo=min_leaf
                )
                
                # Si encontramos un resultado mejor, lo guardamos
                if media > mejor_score:
                    mejor_score = media
                    mejores_params = {
                        'max_depth': depth,
                        'min_samples': min_s,
                        'cantidad_minima_en_nodo': min_leaf
                    }
    
    return mejores_params, mejor_score

if __name__ == '__main__':
    df = pd.read_csv('../data/processed/dataset_clean.csv', sep=';')
    X = df.drop(columns=["TTO"]).to_numpy(dtype=float)
    y = df["TTO"].to_numpy().astype(int)
    
    params, score = hyperparameter_tuning(X, y)
    print("\n" + "="*30)
    print(f"MEJORES PARÁMETROS ENCONTRADOS:")
    print(f"{params}")
    print(f"Precisión esperada (OOB): {score*100:.2f}%")