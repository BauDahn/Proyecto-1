# Pipeline de Validación (`validation.py`)

Este script es el entorno de simulación y testeo para evaluar la robustez de los modelos predictivos frente a datos clínicos de aneurismas.

## 1. Metodología de Evaluación: Bootstrap

Dado que en los datasets médicos la cantidad de pacientes suele ser limitada, utilizamos una técnica de remuestreo llamada **Bootstrap**:
* El script genera **200 muestras** aleatorias con reemplazo a partir del dataset original.
* Para cada iteración, entrena el modelo con la muestra Bootstrap y lo evalúa contra los pacientes que **no** fueron seleccionados en ese sorteo.
* Al finalizar, calcula la **precisión media** y la **desviación estándar**, proporcionando un intervalo de confianza mucho más realista que un simple `train_test_split`.

## 2. Datasets de Evaluación

El sistema prueba los modelos contra dos realidades de datos distintas para analizar el impacto de las variables faltantes:

* **Dataset 1 (Matriz Completa - `X`):** Utiliza a todos los pacientes del estudio. Dependiendo del preprocesamiento, incluye datos imputados para rellenar vacíos. Evalúa la capacidad general del modelo.
* **Dataset 2 (Subconjunto de Diámetro - `X_alt`):** Un dataset estricto (`df.dropna`) que solo contiene a los pacientes que tienen la variable `Diametro` registrada. Sirve para comparar si la certeza de esta variable clínica supera a la pérdida de volumen de datos.

## 3. Explicabilidad Integrada

Al finalizar las métricas de Bootstrap, el pipeline entrena un **Árbol Final** utilizando todos los datos disponibles y extrae los nombres reales de las columnas mediante Pandas. 
Luego, invoca al módulo de explicabilidad (`explainability.py`) para exportar un archivo `.dot`. Este archivo representa visualmente las decisiones clínicas tomadas por la IA, permitiendo una auditoría médica transparente de los umbrales.