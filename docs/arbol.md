# Árbol de Decisión — Explicación completa

## Qué problema resuelve

Al igual que la Regresión Logística, el árbol de decisión resuelve el mismo problema de **clasificación binaria**: dado el perfil clínico de un paciente con aneurisma aórtico, predecir si el tratamiento será **Cirugía Abierta (clase 0)** o **EVAR (clase 1)**.

Se elige un árbol de decisión como segundo modelo porque tiene una propiedad que la Regresión Logística no tiene: **captura relaciones no lineales y jerarquías entre variables de forma natural**. La Regresión Logística busca una frontera de decisión que sea una combinación lineal de las variables. Un árbol puede crear fronteras mucho más complejas y puede descubrir, por ejemplo, que la edad solo importa en combinación con ciertas comorbilidades.

Además, los árboles son extremadamente interpretables: el modelo entrenado se puede visualizar como un diagrama de flujo que cualquier médico puede seguir.

---

## Cómo funciona un Árbol de Decisión

Un árbol de decisión aprende una secuencia de preguntas de tipo "¿esta variable supera este umbral?" que llevan hasta una predicción final. Cada pregunta es un **nodo**, cada posible respuesta es una **rama**, y los nodos finales que dan la predicción son las **hojas**.

Por ejemplo, el árbol podría aprender algo como:

```
¿Edad > 75?
  SÍ → ¿FA = Verdadero?
         SÍ → EVAR
         NO → ¿ICC = Verdadero?
                SÍ → EVAR
                NO → Cirugía Abierta
  NO → ¿Fumador = 2?
         SÍ → Cirugía Abierta
         NO → EVAR
```

El entrenamiento consiste en encontrar, en cada nodo, **qué variable** y **qué umbral** hace la división más pura (que separa mejor las dos clases). La métrica de pureza más común es el **índice Gini**, que mide qué tan mezcladas están las clases en cada subgrupo resultante. Una división perfecta (todo un lado clase 0, todo el otro lado clase 1) tiene Gini = 0. Una división completamente mezclada tiene Gini = 0.5.

El algoritmo es **codicioso** (greedy): en cada nodo elige la mejor división posible para ese nodo sin mirar hacia adelante. No garantiza el árbol globalmente óptimo, pero es computacionalmente viable y funciona bien en la práctica.

---

## El dataset de entrada

El árbol trabaja con los mismos **246 pacientes** y **8 variables** que la Regresión Logística. Sin embargo, el árbol **no necesita StandardScaler** porque no usa gradientes ni distancias — simplemente busca el mejor punto de corte en cada variable, y ese proceso es independiente de si `Edad` va de 39 a 120 o de -1.5 a +1.5. La escala no le afecta.

---

## Los hiperparámetros explicados

```python
DecisionTreeClassifier(max_depth=5, random_state=42)
```

### `max_depth=5` — El parámetro más importante

`max_depth` es el número máximo de niveles que puede tener el árbol. Sin este límite, el árbol crecería hasta memorizar perfectamente el dataset de entrenamiento: en el extremo, haría una hoja por paciente y tendría 100% de accuracy en entrenamiento pero sería completamente inútil con datos nuevos. Eso se llama **sobreajuste u overfitting**.

Con `max_depth=5`, el árbol puede hacer como máximo 5 divisiones en serie desde la raíz hasta una hoja. Esto limita la complejidad del modelo y le obliga a encontrar patrones que generalicen.

¿Por qué 5 y no 3 o 10? Es una elección de compromiso:
- Demasiado poco (`max_depth=2`) → el modelo no captura suficiente complejidad, **underfitting**
- Demasiado grande (`max_depth=20`) → el modelo memoriza el dataset, **overfitting**
- `max_depth=5` es un valor habitual como punto de partida razonable para datasets pequeños

Para elegirlo formalmente se usaría validación cruzada con varias opciones de `max_depth` y se escogería el que mejor generaliza.

### `random_state=42` — Reproducibilidad

Cuando hay empates (dos divisiones igualmente buenas), el árbol desempata de forma aleatoria. Fijar `random_state=42` garantiza que siempre se resuelvan igual y los resultados sean reproducibles. No tiene ningún efecto sobre la lógica del algoritmo, solo sobre los empates.

---

## Cómo se entrena

```python
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X, y)
```

El entrenamiento en sklearn es una sola llamada a `.fit()`. Internamente, el algoritmo:

1. Empieza en la raíz con todos los 246 pacientes
2. Prueba todas las variables y todos los umbrales posibles, calcula el Gini resultante de cada división
3. Elige la división con menor Gini (mayor pureza)
4. Divide los datos en dos grupos y repite el proceso recursivamente en cada subgrupo
5. Para cuando alcanza `max_depth=5` o cuando un nodo ya es puro (todos de la misma clase)

Una diferencia importante con la Regresión Logística: **el árbol entrena con todo el dataset sin separar en train/test**. Esto significa que el modelo guardado ha visto todos los datos. Esto es válido para el modelo final (que se quiere entrenar con todos los datos disponibles), pero implica que la evaluación debe hacerse con cross-validation (como hace `validation.py`) y no comparando las predicciones del modelo guardado contra esos mismos datos.

---

## Por qué el árbol funciona peor que la Regresión Logística aquí

Los resultados de validación cruzada muestran que el árbol (64.22%) es significativamente peor que la Regresión Logística (71.13%). Esto puede parecer contraintuitivo porque los árboles son modelos más flexibles, pero tiene varias explicaciones:

**1. El dataset es pequeño.** Con 246 pacientes y `max_depth=5`, el árbol puede crear hasta 2⁵ = 32 hojas. Con ~246/32 ≈ 7-8 pacientes por hoja en promedio, las decisiones en hojas profundas están basadas en muy pocos ejemplos y son frágiles.

**2. La relación entre variables y target parece aproximadamente lineal.** Las correlaciones con `TTO` son moderadas y positivas para todas las variables. Cuando las relaciones son mayormente lineales (más edad → más EVAR, más FA → más EVAR), la Regresión Logística las captura directamente. El árbol puede capturarlas también, pero con menos eficiencia porque necesita muchas divisiones para aproximar una relación lineal.

**3. Los árboles tienen alta varianza.** Son sensibles a qué datos caen en el entrenamiento. Un cambio pequeño en el dataset puede cambiar completamente la estructura del árbol. La Regresión Logística es más estable.

**4. `max_depth=5` puede no ser el valor óptimo.** Un ajuste formal de hiperparámetros podría mejorar el resultado.

---

## Cómo se guarda y carga el modelo

```python
joblib.dump(model, 'modelo_arbol.pkl')
```

El árbol entrenado se serializa con joblib igual que el Pipeline de la regresión. El archivo `.pkl` guarda toda la estructura del árbol: qué variable se usa en cada nodo, qué umbral, qué clase predicen las hojas, etc.

Al cargarlo con `joblib.load('modelo_arbol.pkl')`, se recupera el árbol completo listo para hacer predicciones. A diferencia del Pipeline de la regresión, no lleva StandardScaler dentro, porque los árboles no lo necesitan.

---

## Resultados con validación cruzada real

Con la validación cruzada estratificada de 5 folds implementada en `validation.py`:

| Métrica | Media | Desviación |
|---|---|---|
| Accuracy | 64.22% | ±3.84% |
| Precision (weighted) | 64.12% | ±3.98% |
| Recall (weighted) | 64.22% | ±3.84% |
| F1 (weighted) | 64.14% | ±3.92% |

**Por fold individual:** 66.0% · 67.3% · 63.3% · 57.1% · 67.3%

La desviación entre folds (±3.84%) es menor que la de la Regresión Logística, lo que significa que el árbol es **más consistente** entre folds, aunque a un nivel de rendimiento inferior.

El modelo tiene dificultades similares con ambas clases (recall ~0.56 para Cirugía Abierta y ~0.70 para EVAR), lo que indica que no está aprendiendo características suficientemente discriminativas con la profundidad actual.

**Contraste importante:** antes de la validación cruzada, el árbol mostraba un 79.27% de accuracy aparente. Esa cifra era falsa porque el modelo fue evaluado sobre los mismos datos con los que se entrenó. La validación cruzada revela el rendimiento real: 64.22%, quince puntos menos. Ese hueco de 15 puntos es la medida del sobreajuste.
