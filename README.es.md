# Proyecto I: Predicción de Tratamiento para Pacientes Vasculares

🇬🇧 [Read in English](README.md)

Herramienta de apoyo a la decisión para cirujanos vasculares. A partir de la edad, el sexo, el estatus de fumador y las comorbilidades del paciente, estima la probabilidad de un resultado de riesgo (`TTO`) y muestra el razonamiento que hay detrás. Se entrenan y comparan dos modelos: una **regresión logística** (R) y un **árbol de decisión** (Python / scikit-learn).

## Arquitectura

Monorepositorio con dos partes independientes:

| Parte | Stack | Función |
|-------|-------|---------|
| [`frontend/`](frontend/README.md) | React 19, Vite, Tailwind CSS 4 | Formulario del paciente, resultado de la predicción, gráficas de explicabilidad |
| [`backend/`](backend/README.md) | FastAPI, scikit-learn, R | API REST, EDA, entrenamiento y validación de modelos |

```
Proyecto-1/
├── backend/
│   ├── EDA/               # datos (raw/processed) y notebooks de exploración en R / Python
│   └── src/
│       ├── api/           # aplicación FastAPI (POST /predictor)
│       ├── app/           # lógica de inferencia
│       └── models/        # entrenamiento, tuning, validación y modelos guardados
├── frontend/              # aplicación React
├── requirements.txt       # dependencias de Python
└── run_server.md          # guía de arranque paso a paso
```

## Datos

- Origen: historiales hospitalarios de pacientes con aneurisma de aorta abdominal (archivo original: `backend/EDA/data/raw/dataset_inicial.xlsx`).
- Tras la limpieza: **246 pacientes**, en `backend/EDA/data/processed/dataset_clean.csv`.
- Variables: `Edad`, `Sexo`, `Fumador` (0 = no fuma, 1 = activo, 2 = ex-fumador), `comorb_grupos` (carga de comorbilidades: Bajo / Medio / Alto).
- La variable objetivo `TTO` es binaria: 143 casos positivos frente a 103 negativos.
- Las comorbilidades (dislipemia, diabetes, HTA, IAM, ERC, EPOC, ACV, FA, cáncer, ICC, EAP) se agrupan según cuántas tiene el paciente: menos de 3 → Bajo, 3–5 → Medio, más de 5 → Alto.
- Se descartó la columna de diámetro del aneurisma porque falta en ~80 % de los registros.

## Modelos

### 1. Regresión logística (R → JSON → Python)

- Ajustada en R (`backend/EDA/notebooks/Exploración de datos.R`) tras comparar varios modelos candidatos e interacciones.
- Modelo final: `TTO ~ Edad + Sexo + Fumador + comorb_grupos`.
- Los coeficientes se exportan a [`backend/src/models/coeficientes.json`](backend/src/models/coeficientes.json). La API calcula `p = 1 / (1 + exp(-z))` con ellos en cada petición, así que no hace falta R para servir predicciones.
- La API traduce `p` en un nivel de riesgo: **Bajo** (≤ 0,3), **Medio** (≤ 0,6), **Alto** (> 0,6).

### 2. Árbol de decisión (Python / scikit-learn)

- `DecisionTreeClassifier` entrenado por `backend/src/models/train_model.py` con una partición 80/20 (`random_state=42`).
- Hiperparámetros ajustados con `GridSearchCV` (5 folds) en `tuning.py` sobre `max_depth`, `min_samples_split` y `criterion`. El modelo guardado usa `criterion='gini'`, `max_depth=4`.
- Se guarda en `decision_tree_model.pkl` (joblib). El entrenamiento también dibuja el árbol en `frontend/public/plots/diagrama_arbol.png` para la interfaz.

## Resultados

Ambos modelos se evalúan con validación cruzada de 5 folds sobre los 246 pacientes limpios.

| Modelo | AUC | Accuracy | F1 (ponderado) | Precisión (ponderada) | Recall (ponderado) |
|--------|-----|----------|----------------|-----------------------|--------------------|
| Regresión logística | **0,707** | – | – | – | – |
| Árbol de decisión | 0,617 | 0,622 ± 0,054 | 0,604 ± 0,054 | 0,628 ± 0,067 | 0,622 ± 0,054 |

Árbol de decisión, por clase (agregado de todos los folds):

| Clase | Precisión | Recall | F1 | Soporte |
|-------|-----------|--------|----|---------|
| `TTO = False` | 0,56 | 0,44 | 0,49 | 103 |
| `TTO = True` | 0,65 | 0,76 | 0,70 | 143 |

![Curva ROC de la regresión logística (CV 5 folds)](backend/EDA/notebooks/curva_roc.png)

**Cómo leer los resultados**

- La regresión logística separa mejor las clases (AUC 0,71 frente a 0,62) y es más fácil de explicar, por eso es el modelo que alimenta la API.
- Ambos modelos son modestos. Con ~250 pacientes y solo variables demográficas y de comorbilidad, es una ayuda a la decisión, **no** un sustituto del criterio clínico.
- El árbol favorece la clase mayoritaria: recall de 0,76 para `True` pero solo 0,44 para `False`.
- Del modelo en R solo se calculó el AUC; el resto de métricas salen de `validation.py` y corresponden al árbol.

## Cómo ejecutarlo

### Requisitos

- Python 3.10+ y Node.js 20+ (el frontend usa Vite 8)
- R (solo si quieres repetir el análisis de la regresión logística)

### Arrancar la aplicación

Arranca ambos servicios en terminales separadas. La guía completa paso a paso está en [`run_server.md`](run_server.md).

**Backend** (http://127.0.0.1:8000, documentación Swagger en `/docs`):

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt -r src/api/requirements.txt
export PYTHONPATH=$PYTHONPATH:.
fastapi dev src/api/main.py
```

**Frontend** (http://localhost:5173):

```bash
cd frontend
npm install
npm run dev
```

Ejemplo de petición:

```bash
curl -X POST http://127.0.0.1:8000/predictor -H "Content-Type: application/json" -d '{
  "edad": 72, "sexo": true, "fumador": "Ex-fumador",
  "dislipemia": true, "diabetes": false, "hta": true, "iam": false, "erc": false,
  "epoc": false, "acv": false, "fa": false, "cancer": false, "icc": false, "eap": false
}'
# → {"riesgo": "...", "probabilidad": ..., "nota": "..."}
```

### Validar el árbol de decisión

Desde `backend/` (con el entorno virtual activo):

```bash
python src/models/validation.py
```

Imprime las métricas de validación cruzada y muestra la matriz de confusión.

### Reentrenar el árbol de decisión

```bash
cd backend
python src/models/train_model.py
```

> Nota: `train_model.py` lee los datos con rutas relativas a `backend/` pero guarda el modelo y la gráfica con rutas relativas a la raíz del repositorio, por lo que tras reentrenar puede que debas mover `decision_tree_model.pkl` y `diagrama_arbol.png` a `backend/src/models/` y `frontend/public/plots/`.
> El `.pkl` incluido en el repositorio se guardó con scikit-learn 1.8.0; cargarlo con otra versión muestra una advertencia.

### Repetir la regresión logística

Abre `backend/EDA/notebooks/Exploración de datos.R` (o el `.Rmd`) en R / RStudio. Reescribe `dataset_clean.csv`, regenera la curva ROC, y los coeficientes ajustados deben copiarse a `coeficientes.json`.

## Más información

- [`run_server.md`](run_server.md): guía de arranque
- [`backend/README.md`](backend/README.md): estructura del backend y lógica de inferencia
- [`backend/src/models/README.md`](backend/src/models/README.md): detalles del modelado
- [`backend/EDA/data/README.md`](backend/EDA/data/README.md): carpeta de datos
- [`frontend/README.md`](frontend/README.md): frontend
