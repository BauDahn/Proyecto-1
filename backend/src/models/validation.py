import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, StratifiedKFold
import matplotlib.pyplot as plt

def validate_model(model_path, X, y):
    model = joblib.load(model_path)

    # =====================================================================
    # CROSS-VALIDATION (5 folds estratificados)
    # =====================================================================
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(
        model, X, y, cv=cv,
        scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'],
        return_train_score=True
    )

    print("--- Resultados Cross-Validation (5 folds) ---")
    print(f"Accuracy   : {scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}")
    print(f"F1 weighted: {scores['test_f1_weighted'].mean():.3f} ± {scores['test_f1_weighted'].std():.3f}")
    print(f"Precision  : {scores['test_precision_weighted'].mean():.3f} ± {scores['test_precision_weighted'].std():.3f}")
    print(f"Recall     : {scores['test_recall_weighted'].mean():.3f} ± {scores['test_recall_weighted'].std():.3f}")

    # =====================================================================
    # REPORTE DETALLADO EN EL FOLD COMPLETO (para ver True/False por separado)
    # =====================================================================
    from sklearn.model_selection import cross_val_predict
    y_pred = cross_val_predict(model, X, y, cv=cv)

    print("\n--- Reporte de Clasificación (agregado de todos los folds) ---")
    print(classification_report(y, y_pred))

    # Matriz de confusión
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Matriz de Confusión (Cross-Validation)")
    plt.show()

    return scores['test_accuracy'].mean()


if __name__ == '__main__':
    df = pd.read_csv('EDA/data/processed/dataset_clean.csv', sep=';')
    df['comorb_grupos'] = df['comorb_grupos'].map({"Bajo": 0, "Medio": 1, "Alto": 2})
    y = df["TTO"]
    X = df.loc[:, df.columns != 'TTO']

    validate_model('src/models/decision_tree_model.pkl', X, y)