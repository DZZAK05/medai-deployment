import streamlit as st
import pandas as pd
import numpy as np
import io

from fpdf import FPDF
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Initialise le client OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


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
        f"1) Examen : température={t}°C, fréquence cardiaque={c} bpm, pression={p} mmHg.\n"
        f"2) Diagnostic : {'urgence' if verdict else 'stable'}.\n"
        "3) Analyse des signes vitaux et recommandations cliniques.\n"
        "Présente chaque point en paragraphe séparé."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=1500,
        temperature=0.7
    )
    return resp.choices[0].message.content


import io
from fpdf import FPDF

def create_pdf_buffer(report_text: str) -> io.BytesIO:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    line_height  = pdf.font_size_pt * 0.35

    for line in report_text.split("\n"):
        pdf.multi_cell(usable_width, line_height, line)

    # Récupère le PDF sous forme de bytearray
    pdf_output = pdf.output(dest="S")

    # Convertit en bytes si nécessaire
    if isinstance(pdf_output, (bytes, bytearray)):
        pdf_bytes = bytes(pdf_output)
    else:
        # cas où fpdf renverrait du str (rare)
        pdf_bytes = pdf_output.encode("latin-1", "replace")

    buf = io.BytesIO(pdf_bytes)
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

        # Téléchargement PDF
        pdf_buf = create_pdf_buffer(rapport)
        st.download_button(
            "⬇️ Télécharger le rapport complet (PDF)",
            data=pdf_buf,
            file_name="rapport_medical.pdf",
            mime="application/pdf"
        )

        # Sauvegarde de l’historique
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
            file_name="historique.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


if __name__ == "__main__":
    main()

