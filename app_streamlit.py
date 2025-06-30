import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from openai import OpenAI      # ← nouveau client
import os

# 1) Récupère ta clé depuis les Secrets de Streamlit Cloud
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# 2) Instancie le client UNE SEULE FOIS ici
client = OpenAI(api_key=OPENAI_API_KEY)

@st.cache_data
def generer_donnees(n=200, seed=42):
    rng = np.random.default_rng(seed)
    temp = rng.normal(37, 0.5, n)
    fc   = rng.integers(60, 100, n)
    pa   = rng.integers(100, 130, n)
    urgence = ((temp > 38.2) | (fc > 95)).astype(int)
    return pd.DataFrame({
        "Temp (°C)": temp,
        "FC (bpm)" : fc,
        "PA (mmHg)": pa,
        "urgence"  : urgence,
    })

def entrainer_modele(data):
    X = data.drop("urgence", axis=1)
    y = data["urgence"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modele = RandomForestClassifier()
    modele.fit(X_train, y_train)
    return modele

def generer_rapport(temp, fc, pa, verdict):
    prompt = (
        f"Rédige un rapport médical pour un patient avec Température {temp}°C, "
        f"Fréquence cardiaque {fc} bpm, Pression artérielle {pa} mmHg. "
        f"Le verdict est {'urgence' if verdict else 'stable'}."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user", "content": prompt}],
        max_tokens=300
    )
    return resp.choices[0].message.content

def main():
    data = generer_donnees()
    modele = entrainer_modele(data)

    st.title("IA Médicale – Démo")
    st.sidebar.header("Test patient")
    temp = st.sidebar.number_input("Température (°C)", 34.0, 42.0, 37.0, 0.1)
    fc   = st.sidebar.number_input("Fréq. cardiaque (bpm)", 40, 180, 75, 1)
    pa   = st.sidebar.number_input("Pression (mmHg)", 80, 200, 120, 1)

    if st.sidebar.button("Prédire & Générer rapport"):
        verdict = modele.predict([[temp, fc, pa]])[0]
        st.markdown("**Verdict :** " + ("🆘 Urgence" if verdict else "✅ Stable"))
        rapport = generer_rapport(temp, fc, pa, verdict)
        st.write(rapport)

if __name__ == "__main__":
    main()


