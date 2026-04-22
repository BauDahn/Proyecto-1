# Validación de Modelos — Explicación completa

## Por qué existe un archivo de validación separado

Entrenar un modelo y evaluarlo son dos tareas con objetivos distintos. El entrenamiento ajusta los parámetros del modelo para que cometa el mínimo error posible sobre los datos disponibles. La validación responde una pregunta diferente: **¿cuánto error cometerá el modelo con pacientes que no ha visto nunca?**

Esa distinción es fundamental. Un modelo puede memorizar perfectamente los 246 pacientes del dataset (100% de accuracy en entrenamiento) y ser completamente inútil con un paciente nuevo, porque ha memorizado ruido en lugar de aprender patrones reales. La validación mide exactamente lo que importa en la práctica: la capacidad de generalización.

Tener la validación en un archivo separado también permite evaluar cualquier modelo guardado como `.pkl` con una sola llamada, sin necesitar los scripts de entrenamiento.

---

## El problema que resuelve: evaluación in-sample

Antes de la versión actual, `validation.py` evaluaba los modelos así:

```python
y_pred = model.predict(X)   # X = todos los 246 pacientes
accuracy_score(y, y_pred)
```

El problema es que el árbol de decisión se entrenó con esos mismos 246 pacientes. Preguntarle al modelo que prediga datos que ya vio durante el entrenamiento no dice nada sobre su capacidad real. Es como darle a un estudiante el examen con las mismas preguntas que ya estudió exactamente. Siempre sacará buena nota, pero no sabemos si entiende la materia.

Esto producía un accuracy aparente del **79.27%** para el árbol. Con validación cruzada correcta, el resultado real es **64.22%** — 15 puntos menos. Esos 15 puntos son la medida del sobreajuste que la evaluación in-sample ocultaba.

---

## La solución: Validación Cruzada Estratificada

La validación cruzada (cross-validation) divide el dataset en `k` grupos (llamados folds). En cada ronda:
- Se entrena el modelo con `k-1` folds
- Se evalúa sobre el fold restante, que el modelo nunca ha visto durante ese entrenamiento

Esto se repite `k` veces, rotando qué fold actúa como test. Al final, todos los pacientes han sido evaluados exactamente una vez como datos de test, siempre cuando el modelo no los había visto.

El código usa `k=5`:

```python
CV_FOLDS = 5
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
```

Con 246 pacientes y 5 folds: ~197 para entrenamiento y ~49 para test en cada ronda.

### Por qué `StratifiedKFold` y no `KFold` normal

El dataset tiene una distribución desigual de clases: 103 pacientes de Cirugía Abierta (42%) y 143 de EVAR (58%). Con `KFold` estándar, la distribución aleatoria podría crear folds desequilibrados — por ejemplo, un fold de test con 35 EVAR y 14 Cirugías Abiertas. Esto sesgaría las métricas de ese fold.

**`StratifiedKFold`** garantiza que cada fold respete la misma proporción de clases que el dataset completo (~42% / 58%). Así cada evaluación se hace en condiciones representativas.

### Por qué `shuffle=True`

Los datos en el CSV pueden estar ordenados (por hospital, por fecha, etc.). Si se divide sin mezclar, los folds podrían tener pacientes sistemáticamente diferentes (por ejemplo, todos los pacientes de un hospital en el mismo fold). `shuffle=True` mezcla aleatoriamente antes de dividir, evitando sesgos por ordenación.

### Por qué `random_state=42`

La mezcla aleatoria debe ser reproducible. `random_state=42` garantiza que los mismos folds se generan en cada ejecución, haciendo los resultados comparables y replicables.

---

## `clone()` — Por qué no se puede usar el modelo ya entrenado

```python
model     = joblib.load(model_path)
estimator = clone(model)
```

La función `clone()` crea una **copia sin entrenar** del modelo cargado, con exactamente los mismos hiperparámetros pero sin los pesos aprendidos.

Esto es necesario porque la validación cruzada necesita entrenar el modelo desde cero en cada fold. Si se pasara el modelo ya entrenado, sklearn no podría re-entrenarlo limpiamente desde un estado en blanco. `clone()` resuelve esto: extrae los hiperparámetros del modelo guardado y crea un estimador nuevo.

Esto también significa que `validation.py` evalúa la **arquitectura del modelo** (sus hiperparámetros: `max_depth=5`, `C=1.0`, etc.), no el modelo específico guardado en disco. Es la pregunta correcta: ¿esta configuración generaliza bien?

---

## `cross_validate` — Múltiples métricas a la vez

```python
SCORING = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
results = cross_validate(estimator, X, y, cv=cv, scoring=SCORING)
```

`cross_validate` ejecuta el proceso completo de cross-validation y devuelve un diccionario con los resultados de cada métrica en cada fold. Usando `scoring` con una lista, calcula todas las métricas en una sola pasada en lugar de tener que ejecutar el proceso varias veces.

El sufijo `_weighted` en precision, recall y F1 significa que se calcula cada métrica por clase y luego se promedia ponderando por el número de muestras de cada clase. Esto es más informativo que `_macro` (que trataría ambas clases igual) en datasets con clases desbalanceadas.

---

## `cross_val_predict` — Predicciones limpias para la matriz de confusión

```python
y_pred_cv = cross_val_predict(clone(model), X, y, cv=cv)
```

`cross_val_predict` devuelve la predicción de cada paciente en el momento en que ese paciente estaba en el fold de test (es decir, cuando el modelo no lo había visto). El resultado es un array de 246 predicciones, una por paciente, todas "limpias" sin data leakage.

Estas predicciones se usan para el **classification report** y la **matriz de confusión**. La alternativa (predecir con el modelo entrenado sobre todos los datos) daría métricas infladas, especialmente para el árbol.

Se pasa `clone(model)` de nuevo (no `estimator`) porque `cross_val_predict` también clona internamente, pero es más limpio ser explícito.

---

## Las métricas explicadas

### Accuracy

```
Accuracy = (predicciones correctas) / (total de predicciones)
```

Es la métrica más simple: el porcentaje de pacientes clasificados correctamente. Es fácil de interpretar pero puede ser engañosa cuando las clases están desbalanceadas. Si el 90% de los pacientes fueran EVAR, un modelo que predijera siempre EVAR tendría 90% de accuracy sin haber aprendido nada.

En este dataset el desbalance es moderado (42%/58%), así que la accuracy es razonablemente representativa.

### Precision

```
Precision (para EVAR) = TP_EVAR / (TP_EVAR + FP_EVAR)
```

De todos los pacientes que el modelo predijo como EVAR, ¿qué fracción realmente era EVAR? Mide la **fiabilidad de las predicciones positivas**. Precision baja significa que el modelo genera muchas falsas alarmas.

### Recall (Sensibilidad)

```
Recall (para EVAR) = TP_EVAR / (TP_EVAR + FN_EVAR)
```

De todos los pacientes que realmente eran EVAR, ¿qué fracción identificó el modelo? Mide la **capacidad de no perderse casos reales**. Recall bajo para Cirugía Abierta significa que el modelo está "perdiendo" pacientes que deberían operarse de forma abierta y clasificándolos como EVAR.

En contexto clínico, el recall de Cirugía Abierta (clase 0) es especialmente importante: un recall bajo significa que el modelo no detecta correctamente pacientes que requieren la intervención más compleja.

### F1-Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Es la media harmónica de Precision y Recall. Resume ambas en una sola cifra. Es útil cuando se quiere un equilibrio entre no perder casos reales y no generar falsas alarmas. La media harmónica penaliza más que la media aritmética cuando una de las dos métricas es muy baja.

---

## La Matriz de Confusión

```python
cm   = confusion_matrix(y, y_pred_cv)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Cirugía Abierta", "EVAR"])
```

La matriz de confusión muestra el desglose completo de aciertos y errores:

```
                    Predicho: Cirugía    Predicho: EVAR
Real: Cirugía          TN (acierto)       FP (error)
Real: EVAR             FN (error)         TP (acierto)
```

- **TN (Verdadero Negativo):** El modelo predijo Cirugía Abierta y el paciente realmente tuvo Cirugía Abierta. Correcto.
- **TP (Verdadero Positivo):** El modelo predijo EVAR y el paciente realmente tuvo EVAR. Correcto.
- **FP (Falso Positivo):** El modelo predijo EVAR pero el paciente tuvo Cirugía Abierta. Error: sobreestimación de EVAR.
- **FN (Falso Negativo):** El modelo predijo Cirugía Abierta pero el paciente tuvo EVAR. Error: infraestimación de EVAR.

La matriz se construye con las predicciones CV de `cross_val_predict`, por lo que refleja el rendimiento real del modelo, no el in-sample.

Se muestra como gráfico usando `ConfusionMatrixDisplay` de sklearn y `matplotlib.pyplot`, lo que genera una visualización con colores que facilita identificar dónde se acumula el error.

---

## Supresión del ConvergenceWarning

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    results   = cross_validate(estimator, X, y, cv=cv, scoring=SCORING)
    y_pred_cv = cross_val_predict(clone(model), X, y, cv=cv)
```

Durante la validación cruzada de la Regresión Logística, el optimizador puede no converger completamente en algunos folds (porque dispone de menos datos: ~197 en lugar de 246). Esto genera un `ConvergenceWarning` por fold, es decir, hasta 10 advertencias (5 de `cross_validate` y 5 de `cross_val_predict`).

Estas advertencias no indican un error grave, pero saturan la salida y dificultan leer los resultados. Se suprimen con el context manager `warnings.catch_warnings()` para que la salida sea limpia. Si se quisiera un modelo perfectamente convergido en cada fold, se aumentaría `max_iter` en la Regresión Logística.

---

## Las rutas de archivo y por qué son absolutas

```python
_SRC  = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.dirname(_SRC)

RUTA_DATOS     = os.path.join(_RAIZ, 'data', 'processed', 'dataset_clean.csv')
RUTA_ARBOL     = os.path.join(_SRC, 'modelo_arbol.pkl')
RUTA_REGRESION = os.path.join(_SRC, 'modelo_regresion.pkl')
```

`__file__` es una variable especial de Python que contiene la ruta del archivo que se está ejecutando. `os.path.abspath(__file__)` convierte esa ruta en una ruta absoluta (completa desde la raíz del sistema). `os.path.dirname()` sube un nivel en el árbol de directorios.

Esto garantiza que el script funcione sin importar desde qué directorio se ejecute. Si se usaran rutas relativas como `'../data/...'`, el script solo funcionaría si se ejecuta desde `src/`. Con rutas absolutas calculadas desde la posición del propio archivo, funciona desde cualquier directorio.

---

## Cómo se muestra el resultado por fold

```python
for nombre, clave in etiquetas:
    s = results[clave]
    print(f"  {nombre:<12} media={s.mean()*100:.2f}%  "
          f"±{s.std()*100:.2f}%  "
          f"[{s.min()*100:.2f}% – {s.max()*100:.2f}%]")
    fold_str = "  ".join(f"{v*100:.1f}%" for v in s)
    print(f"  {'':12} folds: {fold_str}")
```

Para cada métrica se muestra:
- **Media:** el valor esperado del modelo sobre datos nuevos
- **±Desviación estándar:** qué tan estable es ese resultado entre folds distintos. Desviación alta significa que el rendimiento depende mucho de qué pacientes caen en entrenamiento vs test
- **[min – max]:** el rango entre el peor y mejor fold, para ver los casos extremos
- **Folds individuales:** los 5 valores uno a uno, para detectar si hay algún fold muy anómalo

---

## Resumen del flujo completo

```
dataset_clean.csv
       │
       ▼
  Leer con pandas → convertir a numpy (X, y)
       │
       ▼
  Para cada modelo (.pkl):
    └─ joblib.load()          → modelo entrenado
    └─ clone()                → copia sin entrenar (mismos hiperparámetros)
    └─ StratifiedKFold(5)     → 5 particiones estratificadas
    └─ cross_validate()       → accuracy, precision, recall, F1 por fold
    └─ cross_val_predict()    → predicción de cada paciente en su fold de test
    └─ classification_report()→ tabla de métricas por clase
    └─ confusion_matrix()     → matriz de aciertos y errores
    └─ ConfusionMatrixDisplay → visualización gráfica
```
