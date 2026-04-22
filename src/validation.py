import os
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt

_SRC  = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_SRC)

RUTA_DATOS     = os.path.join(_RAIZ, 'data', 'processed', 'dataset_clean.csv')
RUTA_ARBOL     = os.path.join(_SRC, 'modelo_arbol.pkl')
RUTA_REGRESION = os.path.join(_SRC, 'modelo_regresion.pkl')

CV_FOLDS = 5
SCORING  = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

def validate_model(model_path, X, y, titulo="Modelo"):
    model     = joblib.load(model_path)
    estimator = clone(model)  # copia sin entrenar, mismos hiperparámetros

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        results    = cross_validate(estimator, X, y, cv=cv, scoring=SCORING)
        y_pred_cv  = cross_val_predict(clone(model), X, y, cv=cv)

    print(f"\n{'=' * 58}")
    print(f"  {titulo}  —  Validación Cruzada Estratificada ({CV_FOLDS} folds)")
    print(f"{'=' * 58}")

    etiquetas = [
        ('Accuracy',  'test_accuracy'),
        ('Precision', 'test_precision_weighted'),
        ('Recall',    'test_recall_weighted'),
        ('F1',        'test_f1_weighted'),
    ]
    for nombre, clave in etiquetas:
        s = results[clave]
        print(f"  {nombre:<12} media={s.mean()*100:.2f}%  "
              f"±{s.std()*100:.2f}%  "
              f"[{s.min()*100:.2f}% – {s.max()*100:.2f}%]")
        fold_str = "  ".join(f"{v*100:.1f}%" for v in s)
        print(f"  {'':12} folds: {fold_str}")

    print("\n--- Reporte de Clasificación (predicciones CV) ---")
    print(classification_report(y, y_pred_cv,
                                target_names=["Cirugía Abierta", "EVAR"]))

    cm   = confusion_matrix(y, y_pred_cv)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Cirugía Abierta", "EVAR"])
    disp.plot()
    plt.title(f"Matriz de Confusión — {titulo} ({CV_FOLDS}-fold CV)")
    plt.tight_layout()
    plt.show()

    return results['test_accuracy'].mean()


if __name__ == '__main__':
    df = pd.read_csv(RUTA_DATOS, sep=';')
    y  = df["TTO"].to_numpy().astype(int)
    X  = df.loc[:, df.columns != 'TTO'].to_numpy(dtype=float)

    acc_arbol = validate_model(RUTA_ARBOL, X, y, titulo="Árbol de Decisión")
    acc_reg   = validate_model(RUTA_REGRESION, X, y, titulo="Regresión Logística")

    print(f"\n{'=' * 58}")
    print(f"  Exactitud CV media — Árbol:     {acc_arbol * 100:.2f}%")
    print(f"  Exactitud CV media — Regresión: {acc_reg   * 100:.2f}%")
    print(f"{'=' * 58}")
