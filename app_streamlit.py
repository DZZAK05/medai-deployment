import streamlit as st
import pandas as pd
import numpy as np
import io
import os

import json
import pandas as pd
import numpy as np
import streamlit as st
import paho.mqtt.client as mqtt 

# ReportLab pour le PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# OpenAI
from openai import OpenAI

# Import de la fonction d'alerte mail
from notifier import send_alert

# Initialisation du client OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


DATA = []

# Callback : appelé à chaque nouveau message MQTT
def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    DATA.append(payload)
    # Ne garder que les 100 derniers points
    if len(DATA) > 100:
        DATA.pop(0)

# Configure et démarre le client MQTT
mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
mqtt_client.connect("localhost", 1883)       # ou l'adresse de ton broker
mqtt_client.subscribe("medai/capteurs")      # même topic que ton simulateur
mqtt_client.loop_start()                     # démarrage en arrière-plan


@st.cache_data
def generer_donnees(n=200, seed=42):
    rng = np.random.default_rng(seed)
    temp = rng.normal(37, 0.5, n)
    fc   = rng.integers(60, 100, n)
    pa   = rng.integers(100, 130, n)
    urgence = ((temp > 38.2) | (fc > 95)).astype(int)
    return pd.DataFrame({
        "Température (°C)": temp,
        "Fréq. cardiaque (bpm)": fc,
        "Pression (mmHg)": pa,
        "Urgence": urgence
    })

@st.cache_data
def entrainer_modele(df):
    X = df[["Température (°C)", "Fréq. cardiaque (bpm)", "Pression (mmHg)"]]
    y = df["Urgence"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modele = RandomForestClassifier()
    modele.fit(X_train, y_train)
    return modele

# Fonction de génération du rapport via OpenAI
# Correction : pas de saut de ligne dans la f-string

def generer_rapport_ia(temp, fc, pa, verdict):
    prompt = (
        f"Température : {temp}°C, Fréq. cardiaque : {fc} bpm, Pression : {pa} mmHg. "
        f"Diagnostic IA : {'urgence' if verdict else 'stable'}. "
        "Rédige un rapport médical détaillé en français, paragraphe par signe vital."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=800,
        temperature=0.3
    )
    return resp.choices[0].message.content.strip()

# Fonction de création du PDF utilisant la police intégrée Helvetica
def creer_pdf_unicode(texte: str) -> io.BytesIO:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setFont("Helvetica", 11)
    margin_x, margin_top = 40, 40
    y = letter[1] - margin_top
    line_h = 14
    for parag in texte.split("\n"):
        for chunk in [parag[i:i+90] for i in range(0, len(parag), 90)]:
            if y < margin_top:
                c.showPage()
                c.setFont("Helvetica", 11)
                y = letter[1] - margin_top
            c.drawString(margin_x, y, chunk)
            y -= line_h
    c.save()
    buf.seek(0)
    return buf

# Interface Streamlit
def main():
    # —⁋ Injection PWA ⁋————————————————————————————————————
    st.markdown(
        """ 
           <link rel="apple-touch-icon" sizes="180x180" href="/static/icon-192 (1).png">
           <link rel="apple-touch-icon" sizes="152x152" href="/static/icon-192 (2).png">
           <link rel="apple-touch-icon" sizes="120x120" href="/static/icon-192 (3).png">
           <link rel="apple-touch-icon" sizes="76x76"   href="/static/icon-192 (4).png">
        <link rel="manifest" href="/static/manifest.json">
        <meta name="theme-color" content="#2b6cb0">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        """,
        unsafe_allow_html=True
    )
    # —⁋ Fin Injection PWA ⁋—————————————————————————————

    st.title("IA Médicale – Démo")

    st.subheader("📈 Flux capteurs en temps réel")

if DATA:
    df_live = pd.DataFrame(DATA).set_index("timestamp")
    df_live.index = pd.to_datetime(df_live.index, unit="s")
    st.line_chart(df_live[["temperature","fc","pa"]])
else:
    st.info("En attente des premières données MQTT…")


    st.sidebar.header("Test patient")
    temp = st.sidebar.number_input("Température (°C)", 34.0, 42.0, 37.0, 0.1)
    fc   = st.sidebar.number_input("Fréq. cardiaque (bpm)", 40, 180, 75, 1)
    pa   = st.sidebar.number_input("Pression (mmHg)", 80, 200, 120, 1)

    if "history" not in st.session_state:
        st.session_state.history = []

    if st.sidebar.button("Prédire & Générer rapport"):
        df      = generer_donnees()
        modele  = entrainer_modele(df)
        verdict = modele.predict([[temp, fc, pa]])[0]

        st.markdown(f"**Verdict :** {'🆘 Urgence' if verdict else '✅ Stable'}")
        # Envoi de l’alerte e-mail si urgence
        if verdict == 1:
            sujet = "⚠️ Alerte URGENCE patient détectée"
            corps = (
                f"Une urgence IA a été détectée :\n"
                f"• Température : {temp} °C\n"
                f"• Fréquence cardiaque : {fc} bpm\n"
                f"• Pression artérielle : {pa} mmHg\n"
            )
            try:
                send_alert(sujet, corps)
                st.success("Alerte email envoyée avec succès !")
            except Exception as e:
                st.error(f"Erreur lors de l'envoi de l'alerte email : {e}")

        rapport = generer_rapport_ia(temp, fc, pa, verdict)
        st.markdown("**Rapport IA :**")
        st.write(rapport)

        # Téléchargement PDF
        pdf_buf = creer_pdf_unicode(rapport)
        st.download_button(
            "⬇️ Télécharger le rapport PDF",
            data=pdf_buf,
            file_name="rapport_medical.pdf",
            mime="application/pdf"
        )

        # Historique et export Excel
        st.session_state.history.append({
            "Température": temp,
            "FC": fc,
            "PA": pa,
            "Verdict": "Urgence" if verdict else "Stable",
            "Rapport": rapport
        })

    if st.session_state.history:
        st.header("📜 Historique des rapports")
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist)
        excel_buf = io.BytesIO()
        df_hist.to_excel(excel_buf, index=False)
        excel_buf.seek(0)
        st.download_button(
            "⬇️ Télécharger l'historique (Excel)",
            data=excel_buf.getvalue(),
            file_name="historique_patients.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()
