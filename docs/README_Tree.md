# Modelo: Árbol de Decisión Optimizado (Tree)

Dentro de src está tree.py y tree_core.cpp. Estos módulos implementan un Árbol de Decisión personalizado para predecir el tratamiento de aneurismas. Está diseñado con una arquitectura híbrida para maximizar la velocidad sin perder la flexibilidad de Python.

## 1. Estructura del Modelo

El modelo está dividido en dos capas principales:

* **Capa de Interfaz (`src/tree.py`):** Escrita en Python usando Programación Orientada a Objetos. Contiene las clases `Node` (que almacena la estructura gráfica, umbrales y Gini) y `DecisionTree` (que gestiona el ciclo de vida del entrenamiento y las predicciones).
* **Capa de Computación (`src/tree_core.cpp`):** Escrita en C++17. Se encarga exclusivamente de la matemática pesada (calcular la impureza de Gini y encontrar los mejores umbrales de corte). Se comunica con Python mediante la librería `pybind11`.

## 2. Parámetros Modificables

Al instanciar el modelo (`arbol = DecisionTree(...)`), el equipo de datos puede ajustar los siguientes hiperparámetros para evitar el sobreajuste (overfitting) o el subajuste (underfitting):

* **`max_depth` (int):** La profundidad máxima del árbol. Un valor muy alto permite aprender patrones complejos pero puede memorizar el ruido. Un valor muy bajo hará que el modelo sea demasiado simple.
* **`min_samples` (int):** El número mínimo de pacientes que debe haber en un nodo para intentar dividirlo. Si un nodo tiene menos pacientes que este valor, se convierte en hoja automáticamente.
* **`cantidad_minima_en_nodo` (int):** El número mínimo de pacientes que **debe quedar** en una hoja tras un corte. Es un freno matemático: el algoritmo ignorará cualquier regla clínica que aísle a muy pocos pacientes, forzando al modelo a buscar reglas más generales.

## 3. Optimización en C++

El cálculo del mejor punto de corte (split) no se hace por fuerza bruta. El motor de C++ realiza un **escaneo lineal ordenado**:
1.  Recibe la matriz de datos directamente en memoria sin hacer copias.
2.  Ordena los pacientes según la variable clínica a evaluar en tiempo $O(N \log N)$.
3.  Utiliza una "ventana deslizante" para actualizar la impureza de Gini en tiempo real mientras mueve pacientes de la rama derecha a la izquierda.
4.  Devuelve a Python el umbral exacto que maximiza la Ganancia de Información.