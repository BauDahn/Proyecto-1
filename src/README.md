# Pipeline de Datos y Validación

Este conjunto de scripts constituye el entorno de experimentación principal. Coordina la limpieza de datos, el entrenamiento de los modelos y la evaluación estadística mediante técnicas de remuestreo.

## 📂 Archivos Principales

* **`data_preparation.py`**: Script fundacional de *Data Engineering*. Carga los datos crudos, estandariza los nombres de las columnas y mapea las variables clínicas de interés. Aplica reglas de descarte clínico y limpieza de valores nulos para garantizar que los modelos reciban matrices densas y limpias.
* **`test.engine.py`**: Entorno de "Sanity Check". Compara directamente nuestra implementación nativa (`Tree.tree.DecisionTree`) contra el estándar de la industria (`sklearn.tree.DecisionTreeClassifier`) utilizando el dataset *Breast Cancer*. Sirve para validar la integridad matemática del motor de C++ aislando el problema de los datos.
* **`validation.py`**: Núcleo de evaluación comparativa. Instancia los distintos modelos (Árboles de Decisión y Regresión Logística) y evalúa su capacidad predictiva utilizando un sistema de *Bootstrap*.

## 🧪 Metodología de Validación (`validation.py`)

Debido a la naturaleza de los datasets médicos, una partición simple (`train_test_split`) posee demasiada varianza. Por ello, empleamos un remuestreo aleatorio con reemplazo:

1.  **Generación Bootstrap**: Se crean submuestras del mismo tamaño que el dataset original ($N$), extrayendo pacientes con reemplazo.
2.  **Out-of-Bag (OOB)**: Los pacientes que por probabilidad matemática ($\approx 36.8\%$) no entraron en la muestra de entrenamiento, se utilizan como conjunto de prueba estricto.
3.  **Métricas**: El proceso se itera $X$ veces para calcular un vector de exactitudes (*accuracies*). Al final, se reporta la media estadística de las predicciones, reduciendo el ruido inherente de la matriz de datos y mostrando el rendimiento real esperado del modelo.