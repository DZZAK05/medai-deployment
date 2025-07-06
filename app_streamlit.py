import streamlit as st
import json
import pandas as pd
import numpy as np
import io
import paho.mqtt.client as mqtt
import time, json, random

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from openai import OpenAI
from notifier import send_alert

# ─── 2) OPENAI CLIENT ────────────────────────────────────────

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ─── 1) MQTT SETUP ───────────────────────────────────────────



BROKER_URL  = st.secrets.get("MQTT_BROKER_URL", "mqtt.eclipseprojects.io")
BROKER_PORT = int(st.secrets.get("MQTT_BROKER_PORT", 1883))

DATA = []

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    DATA.append(payload)
    if len(DATA) > 100:
        DATA.pop(0)

def main():
    st.title("IA Médicale – Démo")

    # ─── Initialise le client MQTT ─────────────────────────────
    mqtt_client = mqtt.Client()
    mqtt_client.on_message = on_message

    try:
        mqtt_client.connect(BROKER_URL, BROKER_PORT, keepalive=60)
        mqtt_client.subscribe("medai/capteurs")
        mqtt_client.loop_start()
    except Exception:
        DATA.clear()
        st.warning("⚠️ Impossible de se connecter au broker MQTT "
                   f"({BROKER_URL}:{BROKER_PORT}).")

# ─── 3) FONCTIONS ML & PDF ───────────────────────────────────

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
    mdl = RandomForestClassifier()
    mdl.fit(X_train, y_train)
    return mdl

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

def creer_pdf(texte: str) -> io.BytesIO:
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

# ─── 4) UI STREAMLIT ──────────────────────────────────────────

def main():
    st.title("IA Médicale – Démo")

    # 4.1 – Dashboard temps réel MQTT
    st.subheader("📈 Flux capteurs en temps réel")
    st.write("🔍 DATA length =", len(DATA))
    st.write("🔍 Extrait DATA[:3] =", DATA[:3])

    if DATA:
        df_live = pd.DataFrame(DATA).set_index("timestamp")
        df_live.index = pd.to_datetime(df_live.index, unit="s")
        st.line_chart(df_live[["temperature","fc","pa"]])
    else:
        st.info("En attente des premières données MQTT…")

    # 4.2 – Test patient & génération de rapport
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

        pdf_buf = creer_pdf(rapport)
        st.download_button(
            "⬇️ Télécharger le rapport PDF",
            data=pdf_buf,
            file_name="rapport_medical.pdf",
            mime="application/pdf"
        )

        st.session_state.history.append({
            "Température": temp,
            "FC": fc,
            "PA": pa,
            "Verdict": "Urgence" if verdict else "Stable",
            "Rapport": rapport
        })

    # 4.3 – Historique & Excel
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

