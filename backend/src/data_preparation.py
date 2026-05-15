import pandas as pd

# 1. CARGA DEL DATASET INICIAL
# Saltamos la primera fila de códigos para que no ensucie los nombres de columnas 
df_raw = pd.read_excel('../EDA/data/raw/dataset_inicial.xlsx', skiprows=[1])

# Esta línea elimina espacios invisibles al principio y al final de TODOS los nombres de columnas
df_raw.columns = df_raw.columns.str.strip()

# 2. JUSTIFICACIÓN CLÍNICA (Modificaciones del Word)
# Eliminamos variables que no influyen en la decisión técnica del tratamiento
# Se descartan: calidad de vida, ocio, social, laboral y fechas
# También se descartan diámetro y éxito por no ser variables de decisión


# 2. DICCIONARIO DE MAPEO ACTUALIZADO 
mapeo_columnas = {
    'AAADQoL Text': 'Hospital',
    'Edad': 'Edad',
    'Sexo': 'Sexo',
    'Fumador': 'Fumador',
    'Dislipemia': 'Dislipemia',
    'Diabetes': 'Diabetes',
    'HTA': 'HTA',
    'IAM': 'IAM',
    'ERC': 'ERC',
    'epoc': 'EPOC',
    'ACV': 'ACV', # Ahora lo buscamos sin el espacio extra
    'FA': 'FA',
    'Antecedentes neoplasia': 'CANCER',
    'insuficiencia cardiaca': 'ICC',
    'enfermedad arterial periferica': 'EAP',
    'tipo de tratamiento': 'TTO'
}

# 3. SELECCIÓN Y LIMPIEZA

columnas_disponibles = [c for c in mapeo_columnas.keys() if c in df_raw.columns]
df_final = df_raw[columnas_disponibles].rename(columns=mapeo_columnas)

df_final['Hospital'] = df_final['Hospital'].str.replace(r'\d+', '', regex=True)
df_final['Hospital'] = df_final['Hospital'].str.upper().str.strip()

# Eliminamos filas incompletas
df_final = df_final.dropna()

# 4. FORMATEO FINAL
def convertir_tipos(df):
    
    df["Hospital"] = df["Hospital"].astype('category').cat.codes
    df["Edad"] = df["Edad"].astype('Int64')
    df["Sexo"] = df["Sexo"].astype(bool)
    df["Fumador"] = df["Fumador"].astype('Int64').astype("category")
    df["Dislipemia"] = df["Dislipemia"].astype(bool)
    df["Diabetes"] = df["Diabetes"].astype(bool)
    df["HTA"] = df["HTA"].astype(bool)
    df["IAM"] = df["IAM"].astype(bool)
    df["ERC"] = df["ERC"].astype(bool)
    df["EPOC"] = df["EPOC"].astype(bool)
    df["ACV"] = df["ACV"].astype(bool)
    df["FA"] = df["FA"].astype(bool)
    df["CANCER"] = df["CANCER"].astype(bool)
    df["ICC"] = df["ICC"].astype(bool)
    df["EAP"] = df["EAP"].astype(bool)

    '''
    Con el objetivo de optimizar la capacidad predictiva del modelo sobre la toma de decisiones intervencionistas,
    se ha procedido a excluir del dataset a los pacientes bajo tratamiento conservador (Clase 0).
    Esta decisión se fundamenta en que dicho grupo representa a individuos que no alcanzan los criterios clínicos estandarizados para la reparación del aneurisma,
    definidos por un umbral de diámetro de 5,5 cm en hombres y 4,5 cm en mujeres. 
    Asimismo, este grupo incluye pacientes con una expectativa de vida muy limitada debido a comorbilidades extremas o neoplasias avanzadas,
    donde el riesgo quirúrgico supera el beneficio potencial.
    Al centrar el modelo exclusivamente en la elección entre cirugía abierta y tratamiento endovascular (EVAR),
    el algoritmo se especializa en predecir la estrategia terapéutica óptima para aquellos pacientes que ya tienen una indicación clara de intervención,
    eliminando el ruido estadístico que suponen los casos no quirúrgicos.
    '''

    # Como es un criterior definido en la medicina el hecho de intervenir o no, solo definiremos el tipo de intervencio (1 o 2)
    # Filtramos para quedarnos solo con los tratamientos 1 y 2
    df = df[df["TTO"] != 0]
    df["TTO"] = df["TTO"] - 1

    df["TTO"] = df["TTO"].astype('Int64').astype("category")

    return df # Limpieza final de otros NaNs menores

# 4. FORMATEO DE TIPOS
df_cleaned = convertir_tipos(df_final)

# 5. GUARDADO
df_cleaned.to_csv('../data/processed/dataset_clean.csv', index=False, sep=';', encoding='utf-8')


