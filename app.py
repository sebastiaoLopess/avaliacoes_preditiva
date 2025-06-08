import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Carregar modelo e encoders
model = joblib.load("modelo_avaliacao_carros.pkl")
le_dict = joblib.load("encoders.pkl")
df_marcas = pd.read_excel("dimensoes.xlsx", sheet_name="marcas")
df_modelos = pd.read_excel("dimensoes.xlsx", sheet_name="modelos")
df_versoes = pd.read_excel("dimensoes.xlsx", sheet_name="versao")

st.set_page_config(page_title="Avaliação de Carro", layout="centered")

st.title("🔍 Previsão de Valor Avaliado do Carro")

# Formulário de entrada
with st.form("form_carro"):
    KM = st.number_input("Quilometragem", min_value=0)
    ano_fab = st.number_input("Ano de Fabricação", min_value=1950, max_value=2100, value=2024)
    ano_mod = st.number_input("Ano do Modelo", min_value=1950, max_value=2100, value=2024)
    fipe = st.number_input("Valor FIPE", min_value=0.0, step=100.0)
    aav = st.number_input("ref. AAV", min_value=0.0, step=100.0)
    aav_b2c = st.number_input("ref. AAV B2C", min_value=0.0, step=100.0)
    aav_b2b = st.number_input("ref. AAV B2B", min_value=0.0, step=100.0)
    marca = st.selectbox("Marca", df_marcas['marcas'].dropna().unique())
    modelo = st.selectbox("Modelo", df_modelos['modelos'].dropna().unique())
    versao = st.selectbox("Versão", df_versoes['versao'].dropna().unique())
    
    classificacao = st.selectbox("Classificação", ['A', 'B', 'C', 'D', 'E'], index=0)
    finalidade = st.selectbox("Finalidade", ['SHOWROOM', 'REPASSE'], index=0)

    submitted = st.form_submit_button("🔎 Prever Valor")

# Se o botão for pressionado:
if submitted:
    # Montar dicionário
    novo_carro = {
        "KM": KM,
        "Ano Fabricação": ano_fab,
        "Ano Modelo": ano_mod,
        "FIPE": fipe,
        "ref. AAV": aav,
        "ref. AAV B2C": aav_b2c,
        "ref. AAV B2B": aav_b2b,
        "Marca": marca,
        "Modelo": modelo,
        "Versão": versao,
        "Classificação": classificacao,
        "Finalidade": finalidade
    }

    # Criar DataFrame
    df_novo = pd.DataFrame([novo_carro])

    # Codificar as variáveis categóricas
    for col in ['Marca', 'Modelo', 'Versão', 'Classificação', 'Finalidade']:
        le = le_dict[col]
        valor = df_novo.at[0, col]
        if valor in le.classes_:
            df_novo[col] = le.transform([valor])
        else:
            le_classes = np.append(le.classes_, valor)
            le.classes_ = le_classes
            df_novo[col] = le.transform([valor])

    # Reordenar colunas
    X_novo = df_novo[model.feature_names_in_]

    # Prever
    y_pred = model.predict(X_novo)[0]
    st.success(f"💰 Valor estimado da avaliação: R$ {y_pred:,.2f}")