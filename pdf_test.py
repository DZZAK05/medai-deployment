from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

import streamlit as st
import io

def create_pdf_buffer(text: str) -> io.BytesIO:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    _, height = letter
    c.setFont("DejaVuSans", 11)
    x_margin = 40
    y = height - 40
    line_height = 14
    for paragraphe in text.split("\n"):
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

st.title("🚀 Test PDF avec ReportLab")
sample = st.text_area("Texte à mettre dans le PDF", 
    "\n".join(f"Ligne {i+1}: ceci est un texte d'exemple assez long pour tester la césure." for i in range(50)),
    height=300)

if st.sidebar.button("Télécharger le rapport PDF"):
    # on stocke le dernier rapport IA dans session_state au moment de la génération
    raw_report = st.session_state.last_rapport  
    pdf_buf = create_pdf_buffer(raw_report)
    st.download_button(
        "⬇️ Télécharger ce rapport en PDF",
        data=pdf_buf,
        file_name="rapport_medical.pdf",
        mime="application/pdf"
    )


