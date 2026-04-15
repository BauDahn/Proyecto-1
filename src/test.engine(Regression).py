import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from Regression.Regression import entrenar_regresion_logistica
from Regression.Regression import predecir_probabilidades


def comparar():
    # 1. CARGAR LOS DATOS REALES
    print("Cargando base de datos...")
    try:
        # Usamos sep=None para que Pandas autodetecte si el archivo usa comas (,) o puntos y comas (;)
        df = pd.read_csv('../data/processed/dataset_clean.csv', sep=None, engine='python')

        # Limpiamos cualquier espacio en blanco invisible que pueda haber en los nombres de las columnas
        df.columns = df.columns.str.strip()

        # Comprobación de seguridad para ver qué ha leído realmente
        print(f"Columnas detectadas: {df.columns.tolist()}")
    except FileNotFoundError:
        print("Error: No se encuentra el archivo procesado. Asegúrate de ejecutar esto desde la carpeta 'src/'.")
        return

    # Separar variables (X) de la respuesta (y)
    X = df.drop(columns=["TTO"])
    y = df["TTO"]

    # 2. SEPARAR EN ENTRENAMIENTO (80%) Y TEST (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Pacientes de Entrenamiento: {len(X_train)} | Pacientes de Test (Examen): {len(X_test)}\n")

    # ==========================================
    # 3. EVALUAR TU MODELO (Regression.py)
    # ==========================================
    print("--- ENTRENANDO TU MODELO ARTESANAL ---")
    # Entrenamos con el 80% de los datos
    W, b, _, medias, stds = entrenar_regresion_logistica(X_train, y_train, learning_rate=0.01,
                                                                    num_iteraciones=1000)

    # Hacemos el examen con el 20% restante
    prob_tu_modelo = predecir_probabilidades(X_test, W, b, medias, stds)
    pred_tu_modelo = (prob_tu_modelo >= 0.5).astype(int)

    # Calculamos aciertos
    exactitud_tu_modelo = np.mean(pred_tu_modelo == y_test)
    print(f"✅ Precisión de TU modelo: {exactitud_tu_modelo * 100:.2f}%\n")

    # ==========================================
    # 4. EVALUAR EL MODELO DE SCIKIT-LEARN
    # ==========================================
    print("--- ENTRENANDO EL MODELO DE SCIKIT-LEARN ---")
    # Nota: Scikit-learn NO estandariza automáticamente, tenemos que usar su herramienta
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Usamos las reglas del entrenamiento para el test

    # Inicializar y entrenar el modelo oficial
    modelo_sklearn = LogisticRegression(max_iter=1000)
    modelo_sklearn.fit(X_train_scaled, y_train)

    # Evaluar con el examen (Scikit-Learn tiene una función .score que hace esto directo)
    exactitud_sklearn = modelo_sklearn.score(X_test_scaled, y_test)
    print(f"🤖 Precisión de SCIKIT-LEARN: {exactitud_sklearn * 100:.2f}%\n")

    # 5. VEREDICTO FINAL
    print("==========================================")
    diferencia = abs(exactitud_tu_modelo - exactitud_sklearn) * 100
    if exactitud_tu_modelo > exactitud_sklearn:
        print(f"🏆 ¡TU MODELO GANA por un {diferencia:.2f}% de margen!")
    elif exactitud_sklearn > exactitud_tu_modelo:
        print(f"🥇 Scikit-Learn gana por un {diferencia:.2f}% de margen.")
    else:
        print("🤝 ¡EMPATE TÉCNICO! Tu modelo es igual de bueno que el de la industria.")
    print("==========================================")


if __name__ == '__main__':
    comparar()