import streamlit as st
import pandas as pd
import numpy as np

# essentiel :
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split   # ← assure-toi que cette ligne est bien là
from sklearn.metrics import classification_report

from openai import OpenAI
import os



# Récupère ta clé depuis les Secrets de Streamlit Cloud
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

@st.cache_data
def generer_donnees(n=200, seed=42):
    rng = np.random.default_rng(seed)
    temp = rng.normal(37, 0.5, n)
    fc   = rng.integers(60, 100, n)
    pa   = rng.integers(100, 130, n)
    urgence = ((temp > 38.2) | (fc > 95)).astype(int)
    return pd.DataFrame({
        'Température (°C)': temp,
        'Fréq. cardiaque (bpm)': fc,
        'Pression (mmHg)': pa,
        'urgence': urgence
    })

@st.cache_data
def entrainer_modele(data):
    X = data.drop('urgence', axis=1)
    y = data['urgence']
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    modele = RandomForestClassifier()
    modele.fit(X_train, y_train)
    return modele

def predire_patient(modele, temp, fc, pa):
    patient = pd.DataFrame([[temp, fc, pa]],
                           columns=['Température (°C)','Fréq. cardiaque (bpm)','Pression (mmHg)'])
    return int(modele.predict(patient)[0])

def generer_rapport(temp, fc, pa, verdict):
    prompt = (
        f"Rapport médical\n"
        f"- Température : {temp:.1f} °C\n"
        f"- Fréq. cardiaque : {fc} bpm\n"
        f"- Pression       : {pa} mmHg\n\n"
        f"Diagnostic : {'urgence' if verdict else 'stable'}.\n"
        f"Recommandations : surveiller, adapter traitement si besoin."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=300
    )
    return resp.choices[0].message.content

def main():
    st.title("IA Médicale – Démo Streamlit")

    # 1) Data & Modèle
    data = generer_donnees()
    modele = entrainer_modele(data)

    # 2) Répartition & perf
    st.subheader("Répartition")
    df_counts = data['urgence'].value_counts().rename_axis('urgence').reset_index(name='count')
    st.table(df_counts)

    st.subheader("Performance globale")
    perf = classification_report(
        data['urgence'],
        modele.predict(data.drop('urgence',axis=1)),
        target_names=['stable','urgence']
    )
    st.text(perf)

    # 3) Formulaire et verdict
    st.subheader("Test patient")
    temp = st.number_input("Température (°C)", 34.0, 42.0, 37.0, 0.1)
    fc   = st.number_input("Fréq. cardiaque (bpm)", 40, 180, 75, 1)
    pa   = st.number_input("Pression (mmHg)", 80, 200, 120, 1)

    if st.button("Prédire & Générer rapport"):
        verdict = predire_patient(modele, temp, fc, pa)
        st.markdown("**Verdict :** " + ("🆘 Urgence" if verdict else "✅ Stable"))
        rapport = generer_rapport(temp, fc, pa, verdict)
        st.markdown("**Rapport IA :**")
        st.markdown(rapport)

if __name__ == "__main__":
    main()

