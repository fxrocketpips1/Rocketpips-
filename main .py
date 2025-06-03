
import requests
import time
from datetime import datetime
import threading
from flask import Flask
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Signal decision logic
def generate_signal(rsi, macd, signal, stoch_k, stoch_d, adx, ema_fast, ema_slow, close_price):
    if None in [rsi, macd, signal, stoch_k, stoch_d, adx, ema_fast, ema_slow, close_price]:
        return "NOT_ENOUGH_DATA"
    if (
        rsi < 30 and
        macd > signal and
        stoch_k < 20 and
        stoch_d < 20 and
        adx > 20 and
        ema_fast > ema_slow and
        close_price > ema_fast
    ):
        return "BUY"
    elif (
        rsi > 70 and
        macd < signal and
        stoch_k > 80 and
        stoch_d > 80 and
        adx > 20 and
        ema_fast < ema_slow and
        close_price < ema_fast
    ):
        return "SELL"
    else:
        return "HOLD"

# Send signal to Telegram
def send_telegram_message(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print("Telegram message sent successfully.")
    else:
        print(f"Failed to send Telegram message: {response.text}")

# Fetch and calculate indicators
def fetch_indicators(symbol, interval, api_key):
    base_url = "https://api.twelvedata.com"

    def get_indicator(indicator):
        url = f"{base_url}/{indicator}?symbol={symbol}&interval={interval}&apikey={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                data = response.json().get("values", [])
                return float(data[0].get("value")) if data else None
            except Exception as e:
                print(f"Error parsing {indicator}: {e}")
        else:
            print(f"Failed to fetch {indicator}: {response.text}")
        return None

    def get_macd():
        url = f"{base_url}/macd?symbol={symbol}&interval={interval}&apikey={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                data = response.json().get("values", [])
                if data:
                    return float(data[0]["macd"]), float(data[0]["signal"])
            except Exception as e:
                print(f"Error parsing MACD: {e}")
        else:
            print(f"Failed to fetch MACD: {response.text}")
        return None, None

    def get_stochastic():
        url = f"{base_url}/stochastic?symbol={symbol}&interval={interval}&apikey={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                data = response.json().get("values", [])
                if data:
                    return float(data[0]["stoch_k"]), float(data[0]["stoch_d"])
            except Exception as e:
                print(f"Error parsing Stochastic: {e}")
        else:
            print(f"Failed to fetch Stochastic: {response.text}")
        return None, None

    def get_ema(time_period):
        url = f"{base_url}/ema?symbol={symbol}&interval={interval}&time_period={time_period}&apikey={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                data = response.json().get("values", [])
                return float(data[0].get("ema")) if data else None
            except Exception as e:
                print(f"Error parsing EMA: {e}")
        else:
            print(f"Failed to fetch EMA: {response.text}")
        return None

    def get_close_price():
        url = f"{base_url}/time_series?symbol={symbol}&interval={interval}&apikey={api_key}&outputsize=1"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                data = response.json().get("values", [])
                return float(data[0].get("close")) if data else None
            except Exception as e:
                print(f"Error parsing close price: {e}")
        else:
            print(f"Failed to fetch close price: {response.text}")
        return None

    rsi = get_indicator("rsi")
    macd, signal = get_macd()
    stoch_k, stoch_d = get_stochastic()
    adx = get_indicator("adx")
    ema_fast = get_ema(9)
    ema_slow = get_ema(21)
    close_price = get_close_price()

    return rsi, macd, signal, stoch_k, stoch_d, adx, ema_fast, ema_slow, close_price

# Signal generation loop
def signal_generator(api_key, symbol, interval, telegram_token, chat_id):
    while True:
        rsi, macd, signal_line, stoch_k, stoch_d, adx, ema_fast, ema_slow, close_price = fetch_indicators(symbol, interval, api_key)
        signal = generate_signal(rsi, macd, signal_line, stoch_k, stoch_d, adx, ema_fast, ema_slow, close_price)

        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        message = (
            f"📈 Signal for {symbol} ({interval}) at {timestamp} UTC:\n"
            f"Signal: {signal}\n"
            f"RSI: {rsi}\nMACD: {macd}, Signal: {signal_line}\n"
            f"Stochastic: K={stoch_k}, D={stoch_d}\n"
            f"ADX: {adx}\nEMA(9): {ema_fast}, EMA(21): {ema_slow}\n"
            f"Close: {close_price}"
        )

        print(message)
        if signal in ["BUY", "SELL"]:
            send_telegram_message(telegram_token, chat_id, message)

        time.sleep(60)

# Run Flask server
def run_web_server():
    app.run(host='0.0.0.0', port=5000)

# Entry point
def main():
    # API credentials
    TWELVE_DATA_API = "051d77c4bd28446ca89e5d99380feb2c"
    TELEGRAM_TOKEN = "7772818817:AAF8pESrvi4Fo9kbyuxuu5BuXNjrduxyEcQ"
    CHAT_ID = "-1002622594169"

    # Start Flask server in background
    threading.Thread(target=run_web_server, daemon=True).start()

    # Start signal generation
    symbol = "GBP/USD"
    interval = "1min"
    signal_generator(TWELVE_DATA_API, symbol, interval, TELEGRAM_TOKEN, CHAT_ID)

# Run script
if __name__ == "__main__":
    main()
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Rocketpips FXBot is running 🚀"

if __name__ == '__main__':
    app.run()
