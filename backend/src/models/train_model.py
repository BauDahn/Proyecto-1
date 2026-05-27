import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
# 1. AGREGAMOS ESTAS IMPORTACIONES PARA PODER DIBUJAR
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os

# Carga de datos procesados
data_path = "backend/EDA/data/processed/dataset_clean.csv"
df = pd.read_csv(data_path, delimiter=";")

df['comorb_grupos'] = df['comorb_grupos'].map({"Bajo": 0, "Medio": 1, "Alto": 2})

# Separación de características y variable objetivo
X = df.drop(columns=["TTO"])  # Asegúrate de usar el nombre correcto de tu columna objetivo
y = df["TTO"]

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicialización y entrenamiento del modelo
# (Le ponemos max_depth=4 para que el gráfico no sea gigante y sea legible en la web)
decision_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
decision_tree.fit(X_train, y_train)

# Guardar el modelo entrenado
os.makedirs("backend/src/models", exist_ok=True)
joblib.dump(decision_tree, "backend/src/models/decision_tree_model.pkl")
print("Modelo de árbol de decisión guardado exitosamente.")

# =====================================================================
# 2. AQUÍ CONECTAMOS LA GRÁFICA DIRECTAMENTE CON EL FRONTEND
# =====================================================================
print("Generando diagrama estructural del árbol para la interfaz web...")

# Definimos la ruta de la carpeta pública de tu frontend
frontend_plots_dir = "frontend/public/plots"
os.makedirs(frontend_plots_dir, exist_ok=True)

# Configuramos el lienzo de matplotlib con alta definición (300 DPI)
plt.figure(figsize=(22, 10), dpi=300)

# Dibujamos el mapa del árbol de Scikit-Learn
plot_tree(
    decision_tree,
    feature_names=X_train.columns.tolist(),
    class_names=["Sin intervención", "Intervención"], # Tus categorías de diagnóstico
    filled=True,                              # Da color a los nodos según su pureza
    rounded=True,                             # Bordes redondeados más estéticos
    proportion=True                           # Muestra porcentajes en lugar de valores absolutos
)

# Guardamos el archivo .png directamente donde React lo puede leer
ruta_final_grafico = os.path.join(frontend_plots_dir, "diagrama_arbol.png")
plt.savefig(ruta_final_grafico, bbox_inches='tight')
plt.close()

print(f"¡Gráfica exportada con éxito! Ya puedes verla en el frontend: {ruta_final_grafico}")
# =====================================================================