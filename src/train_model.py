import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib # Guarda el modelo

def train():
    # 1. Cargar y limpiar datos usando tu lógica existente
    df = pd.read_csv('../data/processed/dataset_clean.csv', sep=';')
    df_final = pd.get_dummies(df, columns=['comorbilidades'], drop_first=True)
    y = df_final["TTO"]
    X = df_final.loc[:, df_final.columns != 'TTO']

    # 2. Implementar Scikit-Learn
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X, y)
    
    # 3. Guardar el modelo entrenado
    joblib.dump(model, 'modelo_arbol.pkl')
    print("Modelo entrenado y guardado.")

if __name__ == "__main__":
    train()