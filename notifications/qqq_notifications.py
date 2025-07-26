import smtplib
import yfinance as yf
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

def get_qqq_price():
    ticker = "QQQ"
    data = yf.download(ticker, period="90d", interval="1d")
    data["7d_high_ma"] = data["High"].rolling(window=7).mean()
    moving_avg_high = float(data["7d_high_ma"].max())
    current_price = float(data["Close"].iloc[-1])
    threshold = moving_avg_high * 0.85
    return current_price, threshold

def send_email(subject, body):
    sender_email = os.getenv("gmail_email")  # Replace with your Gmail
    receiver_email = os.getenv("gmail_email")
    password = os.getenv("gmail_app_pswd")  # Use App Password if 2FA is enabled
    
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Error sending email: {e}")

if __name__ == "__main__":
    current_price, threshold = get_qqq_price()
    if current_price < threshold:
        send_email("BrandonBot: QQQ Price Alert", f"QQQ price below threshold! Current: ${current_price:.2f}, Threshold: ${threshold:.2f}")
    else:
        print("Condition not met.")
