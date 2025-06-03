# main.py

from flask import Flask
import threading
import time
import requests
import datetime

app = Flask(__name__)

# ========== CONFIGURATION ==========

TELEGRAM_BOT_TOKEN = "7772818817:AAF8pESrvi4Fo9kbyuxuu5BuXNjrduxyEcQ"
TELEGRAM_CHAT_ID = "-1002622594169"
SYMBOL = "GBP/USD"

# ========== TELEGRAM FUNCTION ==========

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    requests.post(url, data=payload)

# ========== SIGNAL GENERATION LOGIC ==========

def generate_signal():
    """
    Placeholder for your real signal logic.
    Currently just sends a timestamp every 5 minutes.
    """
    current_time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    signal = f"Signal generated at {current_time} for {SYMBOL}"
    send_telegram_message(signal)

# ========== BACKGROUND BOT LOOP ==========

def run_bot():
    while True:
        try:
            generate_signal()
            time.sleep(60 * 5)  # Run every 5 minutes
        except Exception as e:
            send_telegram_message(f"Bot error: {e}")
            time.sleep(60)

# ========== LAUNCH BACKGROUND THREAD ==========

@app.before_first_request
def activate_bot():
    thread = threading.Thread(target=run_bot)
    thread.daemon = True
    thread.start()

# ========== HOME ROUTE ==========

@app.route("/")
def home():
    return "Forex Signal Bot is Running ✅"

# ========== START SERVER ==========

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
