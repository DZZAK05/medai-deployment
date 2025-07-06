# alerts.py

import os
import smtplib
from email.message import EmailMessage
# ou pour Twilio
# from twilio.rest import Client

def send_email_alert(to_email: str, subject: str, body: str):
    msg = EmailMessage()
    msg["From"]    = os.getenv("ALERT_FROM_EMAIL")
    msg["To"]      = to_email
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP_SSL(
        os.getenv("SMTP_SERVER"),
        int(os.getenv("SMTP_PORT"))
    ) as smtp:
        smtp.login(
            os.getenv("SMTP_USER"),
            os.getenv("SMTP_PASSWORD")
        )
        smtp.send_message(msg)

# def send_sms_alert(to_number: str, message: str):
#     tw_client = Client(
#         os.getenv("TWILIO_ACCOUNT_SID"),
#         os.getenv("TWILIO_AUTH_TOKEN")
#     )
#     tw_client.messages.create(
#         body=message,
#         from_=os.getenv("TWILIO_PHONE_NUMBER"),
#         to=to_number
#     )
