import smtplib
from email.message import EmailMessage
import streamlit as st

# Récupère les secrets depuis Streamlit Cloud
EMAIL_HOST     = st.secrets["SMTP_HOST"]
EMAIL_PORT     = st.secrets["SMTP_PORT"]
EMAIL_USER     = st.secrets["SMTP_USER"]
EMAIL_PASSWORD = st.secrets["SMTP_PASSWORD"]
ALERT_TO       = st.secrets["ALERT_TO_EMAIL"]

def send_alert(subject: str, body: str):
    msg = EmailMessage()
    msg["From"]    = EMAIL_USER
    msg["To"]      = ALERT_TO
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT) as smtp:
        smtp.login(EMAIL_USER, EMAIL_PASSWORD)
        smtp.send_message(msg)
