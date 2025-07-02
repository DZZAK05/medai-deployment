import streamlit as st
import pandas as pd
import numpy as np
import io
import os

# ReportLab pour le PDF Unicode
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ML & OpenAI
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from openai import OpenAI

# 1) Initialise le client OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 2) Enregistre la police Unicode (DejaVu Sans) au démarrage
FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")
pdfmetrics.registerFont(
    TTFont("DejaVuSans", os.path.join(FONTS_DIR, "DejaVuSans.ttf"))
)

@st.cache_data
def generer_donnees(n=200, seed=42):
    rng = np.random.default_rng(seed)
    temp = rng.normal(37, 0.5, n)
    fc   = rng.integers(60, 100, n)
    pa   = rng.integers(100, 130, n)
    urgence = ((temp > 38.2) | (fc > 95)).astype(int)
    return pd.DataFrame({"temp": temp, "fc": fc, "pa": pa, "urgence": urgence})

@st.cache_data
def entrainer_modele(df):
    X = df[["temp", "fc", "pa"]]
    y = df["urgence"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    mdl = RandomForestClassifier()
    mdl.fit(X_train, y_train)
    return mdl

def generer_rapport(t, c, p, verdict):
    prompt = (
        "Tu es un médecin rédigeant un rapport médical très détaillé.\n"
        f"• Examen : température={t}°C, fréquence cardiaque={c} bpm, pression={p} mmHg.\n"
        f"• Diagnostic : {'urgence' if verdict else 'stable'}.\n"
        "• Analyse des signes vitaux et recommandations cliniques.\n"
        "Présente chaque point en paragraphe séparé."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=1500,
        temperature=0.7
    )
    return resp.choices[0].message.content

def creer_pdf_unicode(texte: str) -> io.BytesIO:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # Utilise la police Unicode
    c.setFont("DejaVuSans", 11)

    x_margin = 40
    y = height - 40
    line_height = 14

    for paragraphe in texte.split("\n"):
        # découpe en morceaux d'environ 90 caractères
        morceaux = [paragraphe[i:i+90] for i in range(0, len(paragraphe), 90)]
        for ligne in morceaux:
            if y < 40:
                c.showPage()
                c.setFont("DejaVuSans", 11)
                y = height - 40
            c.drawString(x_margin, y, ligne)
            y -= line_height

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

def main():
    st.title("IA Médicale – Démo")

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

        rapport = generer_rapport(temp, fc, pa, verdict)
        st.markdown("**Rapport IA :**")
        st.write(rapport)

        # Téléchargement PDF Unicode
        pdf_buf = creer_pdf_unicode(rapport)
        st.download_button(
            "⬇️ Télécharger le rapport complet (PDF)",
            data=pdf_buf,
            file_name="rapport_medical.pdf",
            mime="application/pdf",
        )

        # Sauvegarde historique
        st.session_state.history.append({
            "temp": temp,
            "fc": fc,
            "pa": pa,
            "verdict": "Urgence" if verdict else "Stable",
            "rapport": rapport
        })

    # Affichage historique et export Excel
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
