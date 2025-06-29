# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])



# 1. Générer les données
@st.cache_data
def generer_donnees(n=200, seed=42):
    rng = np.random.default_rng(seed)
    temp = rng.normal(37, 0.5, n)
    fc   = rng.integers(60, 100, n)
    pa   = rng.integers(100, 130, n)
    urgence = ((temp > 38.2) | (fc > 95)).astype(int)
    return pd.DataFrame({
        'Temp (°C)': temp,
        'FC (bpm)' : fc,
        'PA (mmHg)': pa,
        'Urgence'  : urgence
    })

# 2. Entraîner le modèle
@st.cache_resource
def entrainer(data):
    X = data.drop('Urgence', axis=1)
    y = data['Urgence']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = RandomForestClassifier().fit(X_train, y_train)
    report = classification_report(y_test, mdl.predict(X_test), output_dict=False)
    return mdl, report

# 3. Générer rapport LLM
def generer_rapport(temp, fc, pa, verdict):
    prompt = (
        f"Patient – Température : {temp:.1f}°C, FC : {fc} bpm, PA : {pa} mmHg. "
        f"Diagnostic : {'urgence' if verdict else 'stable'}. Rédige un rapport court."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=300
    )
    return resp.choices[0].message.content.strip()

# --- Streamlit UI ---
st.title("IA Médicale – Démo Streamlit")

data = generer_donnees()
mdl, perf = entrainer(data)

st.subheader("Répartition & Performance")
st.write(data["Urgence"].value_counts())
st.text(perf)

st.subheader("Test patient")
col1, col2, col3 = st.columns(3)
with col1:
    t = st.number_input("Temp (°C)", value=37.0, step=0.1)
with col2:
    c = st.number_input("FC (bpm)", value=75, step=1)
with col3:
    p = st.number_input("PA (mmHg)", value=120, step=1)

if st.button("Prédire & Générer rapport"):
    verdict = int(mdl.predict([[t, c, p]])[0])
    st.markdown("**Verdict :** " + ("🆘 Urgence" if verdict else "✅ Stable"))
    with st.expander("Rapport IA"):
        st.write(generer_rapport(t, c, p, verdict))
