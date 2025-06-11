import json
from datetime import datetime, timedelta
from flask_mail import Mail, Message
from flask import Flask

app = Flask(__name__)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your_app_password'
mail = Mail(app)

def send_daily_notifications():
    with app.app_context():
        today = datetime.now().date()
        try:
            with open('high_stress_emails.json', 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            lines = []
        sent = set()
        for line in lines:
            entry = json.loads(line)
            email = entry['email']
            date = datetime.strptime(entry['date'], '%Y-%m-%d').date()
            # Send for 7 days after first high stress detection
            if 0 <= (today - date).days < 7 and (email, today) not in sent:
                msg = Message(
                    subject="Daily Stress Safety Tips",
                    sender=app.config['MAIL_USERNAME'],
                    recipients=[email],
                    body="Your stress level is high. Please take care! Here are some safety measures:\n- Get enough sleep\n- Spend time outdoors\n- Practice deep breathing exercises"
                )
                mail.send(msg)
                sent.add((email, today))

if __name__ == "__main__":
    send_daily_notifications()