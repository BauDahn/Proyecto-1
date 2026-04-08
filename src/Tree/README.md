# Módulo: Árbol de Decisión Híbrido (C++ / Python)

Este módulo contiene la implementación nativa de un Árbol de Decisión optimizado para problemas de clasificación médica. Utiliza una arquitectura en dos capas para combinar la facilidad de uso de Python con el rendimiento de bajo nivel de C++.

## 📂 Estructura del Módulo

* **`tree.py`**: Es la API orientada a objetos. Contiene las clases `Node` y `DecisionTree`. Gestiona el crecimiento recursivo del árbol y la lógica de parada (`max_depth`, `min_samples`, `cantidad_minima_en_nodo`), preparándolos para interactuar con la capa computacional.
* **`tree_core.cpp`**: Es el motor matemático escrito en C++17. Utiliza la librería `pybind11` para integrarse como un módulo nativo en Python. En lugar de iterar por fuerza bruta, ordena los datos ($O(N \log N)$) y calcula las impurezas de Gini mediante una ventana deslizante. Esto resuelve la sobrecarga computacional de la búsqueda del mejor punto de corte.
* **`explainability.py`**: Proporciona herramientas de IA Explicable (XAI). Su función principal, `tree_visualizer`, traduce la estructura recursiva de los nodos entrenados en un archivo `.dot` compatible con Graphviz. Esto permite auditar visualmente las reglas clínicas que el algoritmo ha descubierto.

## 🛠️ Parámetros de Ajuste (`DecisionTree`)

El equipo de modelado puede modificar los siguientes hiperparámetros durante el instanciamiento para controlar la regularización del modelo:

* **`max_depth`**: Límite de profundidad de la búsqueda DFS. Útil para mitigar el sobreajuste.
* **`min_samples`**: Cantidad mínima de pacientes que deben llegar a un nodo para que el algoritmo intente particionarlo.
* **`cantidad_minima_en_nodo`**: Límite matemático impuesto desde C++. Descarta cualquier corte (split) que deje una rama con menos pacientes que este valor, forzando la generalización geométrica de las clases.

## 🚀 Compilación del Motor

Para compilar las actualizaciones en el motor C++, ejecuta el siguiente comando desde el directorio padre:
```bash
g++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) Tree/tree_core.cpp -o Tree/tree_core$(python3-config --extension-suffix)