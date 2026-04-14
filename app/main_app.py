import streamlit as st
import numpy as np
import sys
import os

# Para poder importar tree.py desde src/
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

st.set_page_config(page_title="Predictor Vascular", page_icon="🫀")

st.title("Predictor de Tratamiento para Pacientes Vasculares")
st.write("Introduce los datos del paciente para predecir el tratamiento más adecuado.")

st.divider()

# --- DATOS DEL PACIENTE ---
st.subheader("Datos del paciente")

col1, col2 = st.columns(2)

with col1:
    edad = st.number_input("Edad", min_value=0, max_value=120, value=70)
    sexo = st.selectbox("Sexo", ["Hombre", "Mujer"])
    fumador = st.selectbox("Fumador", ["No fumador (0)", "Ex fumador (1)", "Fumador activo (2)"])

with col2:
    diametro = st.number_input("Diámetro AAA (mm)", min_value=0.0, value=0.0, step=1.0)
    st.write("*Dejar en 0 si no se conoce*")

st.divider()

# --- COMORBILIDADES ---
st.subheader("Comorbilidades")

col3, col4 = st.columns(2)

with col3:
    dislipemia = st.checkbox("Dislipemia")
    diabetes = st.checkbox("Diabetes")
    hta = st.checkbox("HTA (Hipertensión)")
    iam = st.checkbox("IAM (Infarto de miocardio)")
    erc = st.checkbox("ERC (Enfermedad renal crónica)")
    epoc = st.checkbox("EPOC")

with col4:
    acv = st.checkbox("ACV (Accidente cerebrovascular)")
    fa = st.checkbox("FA (Fibrilación auricular)")
    cancer = st.checkbox("Cáncer")
    icc = st.checkbox("ICC (Insuficiencia cardíaca)")
    eap = st.checkbox("EAP (Enfermedad arterial periférica)")

st.divider()

# --- PREDICCIÓN ---
# Convertimos los inputs a números para pasárselos al modelo
def preparar_datos(edad, sexo, fumador, dislipemia, diabetes, hta, iam, erc, epoc, acv, fa, cancer, icc, eap, diametro):
    
    sexo_num = 0 if sexo == "Hombre" else 1
    fumador_num = int(fumador[0])  # Cogemos el número del principio del string

    # Convertimos los booleanos a 0 o 1
    comorbilidades = [
        int(dislipemia), int(diabetes), int(hta), int(iam),
        int(erc), int(epoc), int(acv), int(fa),
        int(cancer), int(icc), int(eap)
    ]

    # Si el diámetro es 0 lo ignoramos (no se conoce)
    if diametro > 0:
        datos = [edad, sexo_num, fumador_num] + comorbilidades + [diametro]
    else:
        datos = [edad, sexo_num, fumador_num] + comorbilidades

    return np.array(datos)


# Función temporal hasta que conectemos el árbol real
def predecir_temporal(datos):
    # TODO: reemplazar esto por tree.predict() cuando este compilado tree_core
    # De momento usamos reglas simples para que la interfaz funcione
    if diametro > 55:
        return 2  # EVAR
    elif iam or erc:
        return 0  # Conservador
    else:
        return 1  # Cirugia abierta


if st.button("Predecir tratamiento", type="primary"):

    datos = preparar_datos(edad, sexo, fumador, dislipemia, diabetes, hta, iam, erc, epoc, acv, fa, cancer, icc, eap, diametro)

    resultado = predecir_temporal(datos)

    st.divider()
    st.subheader("Resultado")

    if resultado == 0:
        st.success("✅ Manejo Conservador (0)")
        st.write("El modelo recomienda un seguimiento sin intervención quirúrgica.")
    elif resultado == 1:
        st.warning("⚠️ Cirugía Abierta (1)")
        st.write("El modelo recomienda una intervención de cirugía abierta.")
    elif resultado == 2:
        st.info("🔵 EVAR - Tratamiento Endovascular (2)")
        st.write("El modelo recomienda una reparación endovascular del aneurisma.")