# Modelado Analítico y Machine Learning (R & Scikit-Learn)

Para la inveestigación de datos decidimos usar dos softwares distintos. El árbol fue hecho usando SciKit-Learn, mientras que el modelo de regresión logística fue hecho en R, dado que en la asignatura proyectos de regresión hemos aprendido a hacerlo en este software. La idea era tener dos modelos para comparar y ver cuál era mejor.

## 1. Regresión Logística (R a Python)
Para garantizar la solidez estadística y la interpretabilidad, se entrenaron modelos de regresión utilizando **R**.
* **Estudio de Datos**: Todo el estudio de dats necesario para hacer la regresión logística en R se hizo en (`EDA/notebooks`), donde comparamos varios modelos, vimos relaciones entre variables o interacciones. Luego nos quedamos con el mejor modelo.
* **Exportación de Pesos**: Los coeficientes obtenidos en R (Betas, Intercepto) se han exportado en un json (`coeficientes.json`).
* **Inferencia**: El backend en Python (`app/main.py`) carga este JSON y calcula la probabilidad de riesgo mediante la función sigmoide estandarizada matemática: `1 / (1 + math.exp(-z))`. 

## 2. Árboles de Decisión (Python / Scikit-Learn)
Para obtener predicciones no lineales y altamente explicables visualmente para los médicos, se utiliza la librería Scikit-Learn.
* **Entrenamiento**: El script `train_model.py` entrena un `DecisionTreeClassifier` utilizando los datos limpios (`dataset_clean.csv`).
* **Explicabilidad Visual Restringida**: Se ha configurado el árbol siguiendo un tuning hecho con SciKit-Learn usando la función hecha en (`tuning.py`).
* **Exportación Directa a UI**: Utilizando la librería `matplotlib`, el modelo guarda una visualización del mapa de decisión (`diagrama_arbol.png`) en alta definición (300 DPI) y lo inyecta automáticamente en la carpeta `/public` del frontend, enlazando directamente el ciclo de entrenamiento con la UI.
* **Persistencia**: El modelo se guarda serializado usando `joblib` (`decision_tree_model.pkl`) para poder predecir sobre nuevos pacientes sin tener que reentrenar.