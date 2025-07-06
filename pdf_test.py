from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

import streamlit as st
import io

import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def creer_pdf_unicode(texte: str) -> io.BytesIO:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # Utilise la police PostScript intégrée
    c.setFont("Helvetica", 11)

    x_margin = 40
    y = height - 40
    line_height = 14

    for paragraphe in texte.split("\n"):
        morceaux = [paragraphe[i:i+90] for i in range(0, len(paragraphe), 90)]
        for ligne in morceaux:
            if y < 40:
                c.showPage()
                c.setFont("Helvetica", 11)  # remets la police à chaque nouvelle page
                y = height - 40
            c.drawString(x_margin, y, ligne)
            y -= line_height

    c.showPage()
    c.save()
    buf.seek(0)
    return buf


st.title("🚀 Test PDF avec ReportLab")
sample = st.text_area("Texte à mettre dans le PDF", 
    "\n".join(f"Ligne {i+1}: ceci est un texte d'exemple assez long pour tester la césure." for i in range(50)),
    height=300)

# Correction : utiliser la fonction correcte pour générer le PDF à partir du texte saisi
if st.sidebar.button("Télécharger le rapport PDF"):
    # Utilise le texte de la zone de saisie, pas une variable de session potentiellement absente
    pdf_buf = creer_pdf_unicode(sample)
    st.download_button(
        "⬇️ Télécharger ce rapport en PDF",
        data=pdf_buf,
        file_name="rapport_medical.pdf",
        mime="application/pdf"
    )


