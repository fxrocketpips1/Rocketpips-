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
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Telegram error: {e}")

# ========== SIGNAL GENERATION LOGIC ==========

def generate_signal():
    current_time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    signal = f"Signal generated at {current_time} for {SYMBOL}"
    send_telegram_message(signal)

# ========== BACKGROUND THREAD ==========

def run_bot():
    while True:
        try:
            generate_signal()
            time.sleep(60 * 5)  # Every 5 minutes
        except Exception as e:
            send_telegram_message(f"Bot error: {e}")
            time.sleep(60)

def start_background_thread():
    thread = threading.Thread(target=run_bot)
    thread.daemon = True
    thread.start()

# ========== ROUTES ==========

@app.route("/")
def home():
    return "Forex Signal Bot is Running ✅"

# ========== SERVER ==========

if __name__ == "__main__":
    start_background_thread()
    app.run(host="0.0.0.0", port=10000)
