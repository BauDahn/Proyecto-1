# Documentación del Módulo de Explicabilidad (XAI)

Este módulo, ubicado en `src/explainability.py`, proporciona las herramientas necesarias para visualizar y entender las decisiones tomadas por el modelo de árbol de decisión entrenado para el estudio de aneurismas.

## Objetivo
Transformar la estructura jerárquica de datos del árbol en una representación visual (formato DOT) que sea fácil de interpretar y entender por los médicos usuarios.

## Componentes Técnicos

### 1. Función `visualize_tree(model, feature_names=None)`
Es la función principal que actúa como interfaz para el usuario.
* **Entrada**: Recibe una instancia de `DecisionTree` y, opcionalmente, los nombres de las variables clínicas.
* **Proceso**: Inicializa la cadena de texto en formato DOT y orquesta el recorrido del árbol.
* **Salida**: Un string formateado para ser interpretado por herramientas de Graphviz.

### 2. Función interna `traverse(node, node_id)`
Es una función recursiva anidada que realiza un recorrido en profundidad (DFS) por los nodos del árbol.
* **Lógica de Hoja**: Si el nodo actual es una hoja (`is_leaf_node()`), genera una caja con la predicción final.
* **Lógica de Rama**: Si es un nodo de decisión, extrae el índice de la característica (`feature`) y el umbral (`threshold`) para mostrar el criterio de división.

### 3. Uso de `nonlocal dot`
Se utiliza la palabra clave `nonlocal` para permitir que la función interna `traverse` modifique la variable `dot` definida en el ámbito de `visualize_tree`. Esto evita la creación de múltiples copias del string y permite una acumulación eficiente de la descripción del gráfico a medida que la recursión avanza por las ramas.

## Atributos Visualizados
Para que la explicación sea completa, cada nodo muestra:
* **Gini**: El índice de impureza en ese punto específico del entrenamiento.
* **Samples**: La cantidad de pacientes (muestras) que alcanzan ese nodo.
* **Threshold**: El valor exacto por el cual se divide la población (ej. Diámetro <= 55mm).

## Cómo generar la imagen final
Una vez obtenido el texto DOT, se puede convertir en una imagen PNG usando visualizadores como WebGraphviz o también usando la terminal:
```bash
dot -Tpng tree_viz.dot -o tree_viz.png