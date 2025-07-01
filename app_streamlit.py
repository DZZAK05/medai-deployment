import streamlit as st
import pandas as pd
import numpy as np
import io
from fpdf import FPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

PDF_MIME_TYPE = "application/pdf"

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from openai import OpenAI

# 1) Initialise le client OpenAI UNE SEULE FOIS
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
        f"1) Contexte et antécédents du patient.\n"
        f"2) Examen clinique : température={t}°C, fréquence cardiaque={c} bpm, pression={p} mmHg.\n"
        f"3) Diagnostic : {'urgence' if verdict else 'stable'}.\n"
        "4) Analyse approfondie des signes vitaux.\n"
        "5) Conclusion et recommandations cliniques.\n"
        "Présente chaque section comme un paragraphe séparé."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.7,
    )
    return resp.choices[0].message.content


def split_long_lines(text, max_length=100):
    """Découpe les lignes trop longues pour le PDF."""
    import textwrap
    lines = []
    for line in text.split('\n'):
        lines.extend(textwrap.wrap(line, width=max_length) or [''])
    return lines



def create_pdf_buffer(report_text: str) -> io.BytesIO:
    # 0) normalisation des caractères « problématiques »
    # remplace l’apostrophe typographique par une apostrophe ASCII
    clean_text = report_text.replace("’", "'")\
                            .replace("–", "-")\
                            .replace("—", "-")\
                            # ajoute d’autres .replace() si besoin

    buf = io.BytesIO()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in clean_text.split("\n"):
        pdf.multi_cell(0, 10, line)

    pdf.output(buf)  # ici on passe le buffer BytesIO
    buf.seek(0)
    return buf




def main():
    st.title("IA Médicale – Démo")

    st.sidebar.header("Test patient")
    temp = st.sidebar.number_input("Température (°C)", 34.0, 42.0, 37.0, 0.1)
    fc   = st.sidebar.number_input("Fréq. cardiaque (bpm)", 40, 180, 75, 1)
    pa   = st.sidebar.number_input("Pression (mmHg)", 80, 200, 120, 1)

    # Historique en session
    if "history" not in st.session_state:
        st.session_state.history = []

    if st.sidebar.button("Prédire & Générer rapport"):
        df    = generer_donnees()
        mdl   = entrainer_modele(df)
        verdict = mdl.predict([[temp, fc, pa]])[0]
        rapport = generer_rapport(temp, fc, pa, verdict)
        st.session_state.last_rapport = rapport
        st.session_state.last_verdict = verdict
        st.session_state.last_temp = temp
        st.session_state.last_fc = fc
        st.session_state.last_pa = pa
        # Historique
        st.session_state.history.append({
            "temp": temp,
            "fc": fc,
            "pa": pa,
            "verdict": "Urgence" if verdict else "Stable",
            "rapport": rapport
        })

    # Affichage du rapport et du bouton PDF sous la conclusion
    if "last_rapport" in st.session_state:
        verdict = st.session_state.last_verdict
        st.markdown(f"**Verdict :** {'🆘 Urgence' if verdict else '✅ Stable'}")
        st.markdown("**Rapport IA :**")
        st.write(st.session_state.last_rapport)
        pdf_buf = create_pdf_buffer(st.session_state.last_rapport)
        st.download_button(
            "⬇️ Télécharger ce rapport en PDF",
            data=pdf_buf,
            file_name="rapport_medical.pdf",
            mime=PDF_MIME_TYPE
        )

    # Affichage de l'historique
    if st.session_state.history:
        st.header("📜 Historique des rapports")
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist)

        # Export Excel
        excel_bytes = df_hist.to_excel(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Télécharger l'historique Excel",
            data=excel_bytes,
            file_name="historique.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Téléchargement PDF du dernier rapport
        if len(df_hist) > 0:
            last_rapport = df_hist.iloc[-1]["rapport"]
            pdf_buffer = create_pdf_buffer(last_rapport)
            st.download_button(
                "⬇️ Télécharger le dernier rapport PDF",
                data=pdf_buffer,
                file_name="rapport_medical.pdf",
                mime=PDF_MIME_TYPE
            )

    # Section : Générer un PDF à partir d'un texte libre
    st.header("Générer un PDF à partir d'un texte libre")
    texte_libre = st.text_area("Votre texte à convertir en PDF", "", height=200)
    if st.button("Générer le PDF du texte saisi"):
        if texte_libre.strip():
            pdf_buf_libre = create_pdf_buffer(texte_libre)
            st.download_button(
                "⬇️ Télécharger le PDF du texte saisi",
                data=pdf_buf_libre,
                file_name="texte_libre.pdf",
                mime=PDF_MIME_TYPE
            )
        else:
            st.warning("Veuillez saisir du texte avant de générer le PDF.")

if __name__ == "__main__":
    main()


