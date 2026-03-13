import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

df = pd.read_csv("../data/raw/RawData.csv", sep=";")

columnas_faltanes = df.isnull().mean()
columnas_50 = columnas_faltanes[columnas_faltanes > 0.4]

columnas_a_eliminar = columnas_50.index.tolist()
df.drop(labels=columnas_a_eliminar, axis=1, inplace=True)

umbral_nulos = 0.4
porcentaje_nulos_por_fila = df.isnull().sum(axis=1) / len(df.columns)
df = df[porcentaje_nulos_por_fila < umbral_nulos]
df = df.reset_index(drop=True)

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
df["TTO"] = df["TTO"].astype('Int64').astype("category")
df["FORMA INTERV"] = df["FORMA INTERV"].astype(bool)

df.drop(columns=[df.columns[0]], inplace=True)


output_dir = "../data/processed/"

df.to_csv(os.path.join(output_dir, "processed_data.csv"), index=False)