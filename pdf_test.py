from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

import streamlit as st
import io
from fpdf import FPDF

def create_pdf_buffer(text: str) -> io.BytesIO:
    buf = io.BytesIO()
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(buf)      # écrit dans le buffer
    buf.seek(0)
    return buf

st.title("🚀 Test PDF with FPDF2")
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


