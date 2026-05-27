# Backend: API, Datos y Modelos

Este directorio engloba toda la lógica del servidor, el procesamiento de la información, el análisis de datos (EDA) y la API que nutre al frontend.

## Estructura del Backend

* `/EDA`: Contiene los notebooks de Jupyter y scripts (tanto en Python como en R) utilizados para la limpieza, análisis exploratorio y preparación de los conjuntos de datos originales.
* `/src/models`: Es el lugar donde se entrenó el árbol con SciKit. Aloja los scripts de entrenamiento en (`train_model.py`), extracción de métricas, y los modelos guardados o sus hiperparámetros (ej. `coeficientes.json`, `decision_tree_model.pkl`). Además cuenta con el archivo de validación (`validation.py`) dónde se obtienen métricas del funcionamiento del árbol.
* `/src/app`: Carpeta principal de la . Aquí reside la lógica en tiempo real (`main.py`) que recibe las peticiones del frontend, procesa la entrada del paciente y emite una predicción calculada con los modelos.

## Lógica de Inferencia de la API
El script principal (`main.py`) agrupa las comorbilidades del paciente en categorías (Bajo, Medio, Alto) y aplica fórmulas de probabilidad (regresión logística) utilizando coeficientes pre-calculados que fueron exportados desde R al archivo "coeficientes.json". La API se apoya en la función de (`app/main.py`), que lo que hace es combinar los coeficientes de la regresión logística con los datos provistos por el paciente desde la página.