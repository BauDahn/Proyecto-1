import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def validate_model(model_path, X_test, y_test):
    # 1. Cargar el modelo .pkl
    model = joblib.load(model_path)
    
    # 2. Realizar predicciones
    y_pred = model.predict(X_test)
    
    # 3. Métricas principales
    print("--- Reporte de Clasificación ---")
    print(classification_report(y_test, y_pred))
    
    # 4. Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Matriz de Confusión")
    plt.show()

    return accuracy_score(y_test, y_pred)

if __name__ == '__main__':
    df = pd.read_csv('../data/processed/dataset_clean.csv', sep=';')
    y = df["TTO"]
    X = df.loc[:, df.columns != 'TTO']

    
    validate_model('src/modelo_arbol.pkl', )