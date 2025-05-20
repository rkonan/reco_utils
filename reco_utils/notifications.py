import configparser
import smtplib
from email.mime.text import MIMEText

def send_email(subject="Fin d'exécution", body="Le job est terminé."):
    config = configparser.ConfigParser()
    try:
        config.read("config.ini")
        sender = config["EMAIL"]["SENDER"]
        receivers = config["EMAIL"]["RECEIVERS"].split(",")
        password = config["EMAIL"]["PASSWORD"]
    except Exception as e:
        print(f"Email non envoyé : {e}")
        return

    msg = MIMEText(body, _charset="utf-8")
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ", ".join(receivers)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        print("Email envoyé.")
    except Exception as e:
        print(f"Erreur d'envoi : {e}")

