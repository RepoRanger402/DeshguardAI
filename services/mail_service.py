import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SENDER_EMAIL = "cse12105007brur@gmail.com"
APP_PASSWORD = "vxxa uqpl czvv abcd"
RECEIVER_EMAIL = "annahian44@gmail.com"

def send_instant_notice(subject: str, details_html: str):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(details_html, 'html'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        # Enable active SMTP dispatch using App Password
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()

        print(f"[SMTP DISPATCH SUCCESS] Notice sent from {SENDER_EMAIL} to {RECEIVER_EMAIL}")
        return True
    except Exception as e:
        print(f"[SMTP ERROR] Failed to send email: {e}")
        return False