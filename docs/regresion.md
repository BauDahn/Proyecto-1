# Regresión Logística — Explicación completa

## Qué problema resuelve

El objetivo es predecir qué tipo de tratamiento quirúrgico recibirá un paciente con aneurisma aórtico abdominal: **Cirugía Abierta (clase 0)** o **EVAR —reparación endovascular— (clase 1)**. Esto es un problema de **clasificación binaria**: cada paciente pertenece exactamente a una de dos categorías.

Se elige Regresión Logística como primer modelo porque es el punto de partida natural en clasificación binaria. Es interpretable (los pesos del modelo dicen directamente cuánto influye cada variable), es rápida de entrenar, y si funciona bien con datos linealmente separables, no tiene sentido usar algo más complejo. Si no funciona bien, al menos sirve como línea base con la que comparar.

---

## Cómo funciona la Regresión Logística

A pesar del nombre, la Regresión Logística **no hace regresión** en el sentido de predecir un número continuo. Hace clasificación. El nombre viene de la función matemática que usa internamente: la **función logística (o sigmoide)**.

El modelo aprende una combinación lineal de las variables de entrada:

```
z = w₀ + w₁·Edad + w₂·Fumador + w₃·FA + w₄·CANCER + w₅·ICC + w₆·EPOC + ...
```

Esa suma `z` puede dar cualquier número real (de -∞ a +∞). Para convertirla en una probabilidad (que debe estar entre 0 y 1), se aplica la función sigmoide:

```
P(EVAR) = 1 / (1 + e^(-z))
```

Si esa probabilidad supera 0.5, el modelo predice EVAR (clase 1). Si está por debajo, predice Cirugía Abierta (clase 0).

El entrenamiento consiste en ajustar los pesos `w₀, w₁, w₂...` para que las probabilidades predichas se acerquen lo máximo posible a las etiquetas reales del dataset. Esto se hace minimizando la **pérdida de entropía cruzada** (cross-entropy loss), que penaliza más cuanto más seguro está el modelo de una respuesta incorrecta.

---

## El dataset de entrada

El modelo trabaja con **246 pacientes** y **8 variables clínicas**:

| Variable | Tipo | Descripción |
|---|---|---|
| `Edad` | Numérica continua | Edad del paciente en años |
| `Fumador` | Categórica (0/1/2) | 0 = no fumador, 1 = exfumador, 2 = fumador activo |
| `FA` | Binaria (bool) | Fibrilación auricular |
| `CANCER` | Binaria (bool) | Antecedentes de neoplasia |
| `ICC` | Binaria (bool) | Insuficiencia cardíaca congestiva |
| `EPOC` | Binaria (bool) | Enfermedad pulmonar obstructiva crónica |
| `INTER_CANCER_ICC` | Binaria | Interacción: paciente con CANCER e ICC a la vez |
| `INTER_CANCER_EPOC` | Binaria | Interacción: paciente con CANCER y EPOC a la vez |

Las dos últimas son **variables de interacción** creadas manualmente. La idea es que tener cáncer y además insuficiencia cardíaca puede tener un efecto combinado sobre la decisión de tratamiento que no queda capturado sumando ambas variables por separado. Al multiplicarlas (CANCER × ICC) se crea una variable nueva que vale 1 solo si el paciente tiene ambas condiciones.

Las **correlaciones con la variable objetivo** `TTO` muestran que `Edad` es la variable más predictiva (r = 0.32), seguida de `FA` (r = 0.21) y `Fumador` (r = 0.21).

---

## Por qué se necesita StandardScaler

La Regresión Logística usa un optimizador numérico para encontrar los mejores pesos. Ese optimizador trabaja calculando gradientes y dando pasos proporcionales a la magnitud de cada variable. El problema es que las variables tienen escalas muy distintas:

- `Edad` varía entre 39 y 120 — un rango de ~80 unidades
- `FA`, `CANCER`, `ICC`, `EPOC` son binarias — solo valen 0 o 1

Si se usan así directamente, el optimizador tendría que dar pasos de tamaño muy diferente para cada peso, lo que hace que tarde mucho más en converger o directamente no converja bien.

El **StandardScaler** transforma cada variable numérica para que tenga:
- Media = 0
- Desviación estándar = 1

Matemáticamente: `X_escalada = (X - media) / desviación_estándar`

Esto pone todas las variables en la misma escala y el optimizador puede tratar todos los pesos de forma homogénea.

> Los árboles de decisión **no necesitan** este paso porque no usan gradientes — simplemente buscan el mejor punto de corte en cada variable, lo que es independiente de la escala.

---

## Por qué se usa un Pipeline

El código no usa StandardScaler y LogisticRegression por separado, sino que los encadena en un **Pipeline**:

```python
pipeline = Pipeline([
    ('escalado',     StandardScaler()),
    ('clasificador', LogisticRegression(...))
])
```

Un Pipeline une varios pasos en un único objeto que se comporta como un modelo normal: tiene `.fit()`, `.predict()`, y `.predict_proba()`.

La razón más importante de usar Pipeline no es la comodidad, sino **evitar data leakage** durante la validación cruzada. Si escalas los datos *antes* de hacer cross-validation, el escalador ve todos los datos (incluyendo los de test) al calcular la media y desviación. Eso filtra información de los folds de test hacia el entrenamiento. Al poner el escalado dentro del Pipeline, sklearn aplica el escalado *dentro* de cada fold: el escalador aprende la media y desviación solo con los datos de entrenamiento de ese fold, y luego aplica esa transformación (sin re-aprender) a los datos de test. Esto es lo correcto.

---

## Los hiperparámetros explicados

```python
LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver='lbfgs',
    random_state=42
)
```

### `C=1.0` — Regularización

`C` controla cuánto se penaliza al modelo por tener pesos grandes. Es el inverso de la fuerza de regularización: **C pequeño = mucha regularización, C grande = poca regularización**.

La regularización evita el sobreajuste. Si el modelo puede dar pesos arbitrariamente grandes, puede memorizar el dataset de entrenamiento perfectamente pero generalizar mal. Al penalizar los pesos grandes, se obliga al modelo a ser más conservador.

`C=1.0` es el valor por defecto de sklearn y es un buen punto de partida. Significa regularización moderada. Para ajustarlo formalmente se usaría una búsqueda de hiperparámetros (GridSearch o similar).

### `max_iter=1000` — Límite de iteraciones

El optimizador (`lbfgs`) funciona de forma iterativa: da pequeños pasos en la dirección que reduce el error. `max_iter=1000` establece cuántos pasos máximo puede dar antes de parar.

Si el modelo no ha convergido en 1000 iteraciones, sklearn lanza un `ConvergenceWarning`. El código captura ese aviso y lo muestra en español de forma legible en lugar del mensaje técnico por defecto.

En la práctica, para este dataset (246 muestras, 8 variables), el modelo converge bastante antes de las 1000 iteraciones.

### `solver='lbfgs'` — Algoritmo de optimización

`lbfgs` (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) es un algoritmo de optimización que usa información de la curvatura de la función de pérdida (el hessiano aproximado) para dar pasos más inteligentes que el descenso por gradiente simple. Converge más rápido en datasets pequeños y es el solver recomendado por sklearn para problemas de este tamaño.

Alternativas como `saga` son mejores para datasets muy grandes (millones de muestras) porque permiten mini-batches. Para 246 pacientes, `lbfgs` es la elección correcta.

### `random_state=42` — Reproducibilidad

Algunos pasos del optimizador tienen elementos aleatorios. Fijar `random_state=42` garantiza que los resultados sean idénticos en cada ejecución. El número 42 no tiene significado especial; cualquier entero sirve.

---

## Cómo se guarda y carga el modelo

Una vez entrenado, el Pipeline completo (escalador + clasificador, ya entrenados) se guarda en un archivo `.pkl` usando **joblib**:

```python
joblib.dump(modelo, 'modelo_regresion.pkl')
```

Joblib serializa el objeto Python entero — pesos del modelo, parámetros del escalador (media y desviación que aprendió), configuración del pipeline — en un único archivo binario.

Al cargarlo después con `joblib.load('modelo_regresion.pkl')`, se recupera el Pipeline exactamente como estaba, listo para hacer predicciones sin necesidad de volver a entrenar ni de escalar los datos manualmente, porque el escalador ya está dentro del Pipeline.

---

## Resultados con validación cruzada real

Con la validación cruzada estratificada de 5 folds implementada en `validation.py`, los resultados honestos son:

| Métrica | Media | Desviación |
|---|---|---|
| Accuracy | 71.13% | ±7.04% |
| Precision (weighted) | 70.86% | ±7.41% |
| Recall (weighted) | 71.13% | ±7.04% |
| F1 (weighted) | 70.48% | ±7.57% |

**Por fold individual:** 74.0% · 73.5% · 63.3% · 63.3% · 81.6%

La alta varianza entre folds (±7%) se explica principalmente por el tamaño pequeño del dataset: con 246 pacientes divididos en 5 folds, cada fold de test tiene ~49 pacientes, y pequeñas diferencias en su composición producen oscilaciones grandes en las métricas.

El modelo tiene más dificultad para identificar correctamente los casos de Cirugía Abierta (recall = 0.58) que los de EVAR (recall = 0.80). Esto es esperable: EVAR es la clase mayoritaria (143 casos vs 103), y los modelos tienden a estar sesgados hacia la clase con más ejemplos.
