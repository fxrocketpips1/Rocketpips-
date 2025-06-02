
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import json

class ForexSignalSystem:
    def __init__(self, twelve_data_api_key, telegram_token, chat_id):
        self.api_key = twelve_data_api_key
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.symbol = "GBP/USD"
        self.interval = "5min"
        self.base_url = "https://api.twelvedata.com"
        
        # Indicator settings optimized for 5-min timeframe
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.bb_std = 2
        self.stoch_k_period = 14
        self.stoch_d_period = 3
        
        # Support/Resistance settings
        self.lookback_period = 100
        self.min_touches = 3
        self.zone_strength_threshold = 0.7
        
        # Signal tracking
        self.last_signal_time = None
        self.signal_cooldown = 1800  # 30 minutes cooldown between signals
        self.daily_signals_sent = 0
        self.max_daily_signals = 8
        
        print(f"🚀 Advanced GBP/USD Signal System Initialized")
        print(f"📊 Timeframe: {self.interval}")
        print(f"🎯 Symbol: {self.symbol}")
        print(f"🛡️ Signal Quality: High Accuracy Mode")
        print(f"📊 Max Daily Signals: {self.max_daily_signals}")

    def get_historical_data(self, outputsize=500):
        """Fetch historical OHLC data from Twelve Data API"""
        try:
            url = f"{self.base_url}/time_series"
            params = {
                'symbol': self.symbol,
                'interval': self.interval,
                'outputsize': outputsize,
                'apikey': self.api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'values' not in data:
                print(f"❌ Error fetching data: {data}")
                return None
                
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
            
            # Convert to numeric (handle missing volume)
            numeric_cols = ['open', 'high', 'low', 'close']
            if 'volume' in df.columns:
                numeric_cols.append('volume')
            
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])
                
            return df.reset_index(drop=True)
            
        except Exception as e:
            print(f"❌ Error fetching historical data: {e}")
            return None

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram

    def calculate_bollinger_bands(self, prices, period=20, std=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, sma, lower_band

    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent

    def find_support_resistance_zones(self, df):
        """Identify strong support and resistance zones"""
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        support_zones = []
        resistance_zones = []
        
        # Find pivot points
        for i in range(2, len(df) - 2):
            # Resistance levels (local highs)
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                
                level = highs[i]
                touches = self.count_touches(df, level, 'resistance')
                
                if touches >= self.min_touches:
                    strength = min(touches / 5, 1.0)  # Max strength = 1.0
                    resistance_zones.append({
                        'level': level,
                        'touches': touches,
                        'strength': strength,
                        'type': 'resistance'
                    })
            
            # Support levels (local lows)
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                
                level = lows[i]
                touches = self.count_touches(df, level, 'support')
                
                if touches >= self.min_touches:
                    strength = min(touches / 5, 1.0)
                    support_zones.append({
                        'level': level,
                        'touches': touches,
                        'strength': strength,
                        'type': 'support'
                    })
        
        # Sort by strength
        support_zones = sorted(support_zones, key=lambda x: x['strength'], reverse=True)[:3]
        resistance_zones = sorted(resistance_zones, key=lambda x: x['strength'], reverse=True)[:3]
        
        return support_zones, resistance_zones

    def count_touches(self, df, level, zone_type, tolerance=0.001):
        """Count how many times price touched a support/resistance level"""
        touches = 0
        
        if zone_type == 'support':
            for low in df['low']:
                if abs(low - level) / level <= tolerance:
                    touches += 1
        else:  # resistance
            for high in df['high']:
                if abs(high - level) / level <= tolerance:
                    touches += 1
                    
        return touches

    def analyze_market_structure(self, df, support_zones, resistance_zones):
        """Analyze overall market structure and trend"""
        current_price = df['close'].iloc[-1]
        
        # Trend analysis using 20 and 50 period EMAs
        ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
        ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
        
        if ema_20 > ema_50 and current_price > ema_20:
            trend = "BULLISH"
        elif ema_20 < ema_50 and current_price < ema_20:
            trend = "BEARISH"
        else:
            trend = "SIDEWAYS"
        
        # Find nearest support and resistance
        nearest_support = None
        nearest_resistance = None
        
        for zone in support_zones:
            if zone['level'] < current_price:
                nearest_support = zone
                break
                
        for zone in resistance_zones:
            if zone['level'] > current_price:
                nearest_resistance = zone
                break
        
        return {
            'trend': trend,
            'current_price': current_price,
            'ema_20': ema_20,
            'ema_50': ema_50,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance
        }

    def generate_signal(self, df):
        """Generate trading signal based on multiple indicators alignment"""
        if len(df) < 100:
            return None
            
        # Calculate all indicators
        df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
        macd, macd_signal, macd_hist = self.calculate_macd(df['close'], self.macd_fast, self.macd_slow, self.macd_signal)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['close'], self.bb_period, self.bb_std)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        
        stoch_k, stoch_d = self.calculate_stochastic(df['high'], df['low'], df['close'], self.stoch_k_period, self.stoch_d_period)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # Find support/resistance zones
        support_zones, resistance_zones = self.find_support_resistance_zones(df)
        
        # Analyze market structure
        market_analysis = self.analyze_market_structure(df, support_zones, resistance_zones)
        
        # Get current values
        current_price = df['close'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        current_macd = df['macd'].iloc[-1]
        current_macd_signal = df['macd_signal'].iloc[-1]
        current_stoch_k = df['stoch_k'].iloc[-1]
        current_stoch_d = df['stoch_d'].iloc[-1]
        
        # Signal generation logic
        bullish_signals = 0
        bearish_signals = 0
        signal_details = []
        
        # RSI Analysis
        if current_rsi < 30:
            bullish_signals += 1
            signal_details.append("✅ RSI Oversold (Bullish)")
        elif current_rsi > 70:
            bearish_signals += 1
            signal_details.append("❌ RSI Overbought (Bearish)")
        
        # MACD Analysis
        if current_macd > current_macd_signal and df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]:
            bullish_signals += 1
            signal_details.append("✅ MACD Bullish Crossover")
        elif current_macd < current_macd_signal and df['macd'].iloc[-2] >= df['macd_signal'].iloc[-2]:
            bearish_signals += 1
            signal_details.append("❌ MACD Bearish Crossover")
        
        # Bollinger Bands Analysis
        if current_price <= df['bb_lower'].iloc[-1]:
            bullish_signals += 1
            signal_details.append("✅ Price at Lower Bollinger Band (Bullish)")
        elif current_price >= df['bb_upper'].iloc[-1]:
            bearish_signals += 1
            signal_details.append("❌ Price at Upper Bollinger Band (Bearish)")
        
        # Stochastic Analysis
        if current_stoch_k < 20 and current_stoch_d < 20:
            bullish_signals += 1
            signal_details.append("✅ Stochastic Oversold (Bullish)")
        elif current_stoch_k > 80 and current_stoch_d > 80:
            bearish_signals += 1
            signal_details.append("❌ Stochastic Overbought (Bearish)")
        
        # Support/Resistance Analysis
        if market_analysis['nearest_support'] and abs(current_price - market_analysis['nearest_support']['level']) / current_price < 0.002:
            if market_analysis['nearest_support']['strength'] >= self.zone_strength_threshold:
                bullish_signals += 1
                signal_details.append(f"✅ Strong Support Zone ({market_analysis['nearest_support']['strength']:.2f})")
        
        if market_analysis['nearest_resistance'] and abs(current_price - market_analysis['nearest_resistance']['level']) / current_price < 0.002:
            if market_analysis['nearest_resistance']['strength'] >= self.zone_strength_threshold:
                bearish_signals += 1
                signal_details.append(f"❌ Strong Resistance Zone ({market_analysis['nearest_resistance']['strength']:.2f})")
        
        # Decision logic: Need at least 2-3 aligned signals
        signal = None
        if bullish_signals >= 2 and bullish_signals > bearish_signals:
            entry_price = current_price
            stop_loss = market_analysis['nearest_support']['level'] if market_analysis['nearest_support'] else current_price * 0.998
            take_profit = market_analysis['nearest_resistance']['level'] if market_analysis['nearest_resistance'] else current_price * 1.003
            
            signal = {
                'type': 'BUY',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': min(bullish_signals / 4 * 100, 95),
                'signals_count': bullish_signals,
                'details': signal_details,
                'market_analysis': market_analysis
            }
            
        elif bearish_signals >= 2 and bearish_signals > bullish_signals:
            entry_price = current_price
            stop_loss = market_analysis['nearest_resistance']['level'] if market_analysis['nearest_resistance'] else current_price * 1.002
            take_profit = market_analysis['nearest_support']['level'] if market_analysis['nearest_support'] else current_price * 0.997
            
            signal = {
                'type': 'SELL',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': min(bearish_signals / 4 * 100, 95),
                'signals_count': bearish_signals,
                'details': signal_details,
                'market_analysis': market_analysis
            }
        
        return signal

    def get_market_session(self):
        """Determine current market session"""
        current_hour = datetime.now().hour
        
        if 0 <= current_hour < 7:
            return "ASIAN SESSION 🌏"
        elif 7 <= current_hour < 15:
            return "LONDON SESSION 🇬🇧"
        elif 15 <= current_hour < 22:
            return "NEW YORK SESSION 🇺🇸"
        else:
            return "OVERLAP SESSION ⚡"
    
    def calculate_volatility(self, df, period=20):
        """Calculate average true range for volatility"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr.iloc[-1] if len(atr) > 0 else 0

    def send_test_signal(self):
        """Send a test signal to verify Telegram connection"""
        test_signal = {
            'type': 'BUY',
            'entry_price': 1.35500,
            'stop_loss': 1.35200,
            'take_profit': 1.35900,
            'confidence': 85.0,
            'signals_count': 3,
            'details': ['✅ TEST SIGNAL - RSI Oversold', '✅ TEST SIGNAL - MACD Bullish', '✅ TEST SIGNAL - Strong Support'],
            'market_analysis': {
                'trend': 'BULLISH',
                'current_price': 1.35500
            }
        }
        
        print("🧪 SENDING TEST SIGNAL...")
        self.send_telegram_signal(test_signal, is_test=True)

    def send_telegram_signal(self, signal, is_test=False):
        """Send signal to Telegram"""
        if not signal:
            return
            
        # Skip cooldown and daily limit checks for test signals
        if not is_test:
            # Check signal cooldown
            current_time = datetime.now()
            if (self.last_signal_time and 
                (current_time - self.last_signal_time).seconds < self.signal_cooldown):
                print("⏳ Signal cooldown active - Skipping duplicate signal")
                return
                
            # Check daily limit
            if self.daily_signals_sent >= self.max_daily_signals:
                print("📊 Daily signal limit reached - Quality over quantity!")
                return
            
        # Calculate additional metrics
        risk_reward = abs(signal['take_profit'] - signal['entry_price']) / abs(signal['entry_price'] - signal['stop_loss'])
        pip_value = abs(signal['entry_price'] - signal['stop_loss']) * 10000
        market_session = self.get_market_session()
        
        test_prefix = "🧪 TEST SIGNAL - " if is_test else ""
        
        message = f"""
🚀 {test_prefix}PREMIUM GBP/USD SIGNAL 🚀

📊 Signal: {signal['type']} 
💰 Entry: {signal['entry_price']:.5f}
🛑 Stop Loss: {signal['stop_loss']:.5f} ({pip_value:.1f} pips)
🎯 Take Profit: {signal['take_profit']:.5f}
📈 Confidence: {signal['confidence']:.1f}%
⚖️ Risk:Reward: 1:{risk_reward:.2f}

📊 Market Analysis:
📈 Trend: {signal['market_analysis']['trend']}
💲 Current: {signal['market_analysis']['current_price']:.5f}
🕐 Session: {market_session}

🎯 Signal Confirmations:
{chr(10).join(signal['details'])}

⏰ Execution Time: {datetime.now().strftime('%H:%M:%S')} UTC
📅 Date: {datetime.now().strftime('%Y-%m-%d')}
🕐 Timeframe: 5M Chart

⚠️ RISK MANAGEMENT:
🔹 Position Size: Max 2% risk
🔹 Use proper stop loss
🔹 Don't overtrade
🔹 Follow your plan

📊 Signal #{self.daily_signals_sent + 1 if not is_test else 'TEST'}/{self.max_daily_signals} Today
        """
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=data)
            if response.status_code == 200:
                print("✅ Signal sent to Telegram successfully!")
                if not is_test:
                    self.last_signal_time = datetime.now()
                    self.daily_signals_sent += 1
                    print(f"📊 Signals sent today: {self.daily_signals_sent}/{self.max_daily_signals}")
            else:
                print(f"❌ Failed to send Telegram message: {response.text}")
                print(f"🔧 Trying alternative chat ID format...")
                # Try with @ prefix for username format
                alt_data = {
                    'chat_id': f"@{self.chat_id}",
                    'text': message,
                    'parse_mode': 'HTML'
                }
                alt_response = requests.post(url, data=alt_data)
                if alt_response.status_code == 200:
                    print("✅ Signal sent with alternative format!")
                else:
                    print(f"❌ Alternative format also failed: {alt_response.text}")
        except Exception as e:
            print(f"❌ Error sending Telegram message: {e}")

    def run_analysis(self):
        """Run complete market analysis and generate signals"""
        print(f"\n{'='*60}")
        print(f"📊 GBP/USD ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Get historical data
        df = self.get_historical_data()
        if df is None:
            print("❌ Failed to fetch market data")
            return
        
        print(f"📈 Data fetched: {len(df)} candles")
        
        # Generate signal
        signal = self.generate_signal(df)
        
        # Calculate volatility
        volatility = self.calculate_volatility(df)
        market_session = self.get_market_session()
        
        if signal:
            risk_reward = abs(signal['take_profit'] - signal['entry_price']) / abs(signal['entry_price'] - signal['stop_loss'])
            print(f"🎯 **{signal['type']} SIGNAL GENERATED**")
            print(f"💰 Entry: {signal['entry_price']:.5f}")
            print(f"🛑 Stop Loss: {signal['stop_loss']:.5f}")
            print(f"🎯 Take Profit: {signal['take_profit']:.5f}")
            print(f"📊 Confidence: {signal['confidence']:.1f}%")
            print(f"⚖️ Risk:Reward: 1:{risk_reward:.2f}")
            print(f"📈 Market Trend: {signal['market_analysis']['trend']}")
            print(f"🕐 Session: {market_session}")
            print(f"📊 Volatility (ATR): {volatility:.5f}")
            
            # Send to Telegram
            self.send_telegram_signal(signal)
        else:
            print("⏳ No signal generated - Waiting for better setup")
            current_price = df['close'].iloc[-1]
            print(f"💲 Current GBP/USD: {current_price:.5f}")
            print(f"🕐 Session: {market_session}")
            print(f"📊 Volatility (ATR): {volatility:.5f}")
            print(f"📊 Signals sent today: {self.daily_signals_sent}/{self.max_daily_signals}")

def main():
    # API credentials
    TWELVE_DATA_API = "051d77c4bd28446ca89e5d99380feb2c"
    TELEGRAM_TOKEN = "7772818817:AAF8pESrvi4Fo9kbyuxuu5BuXNjrduxyEcQ"
    CHAT_ID = "-1002622594169"
    
    # Initialize signal system
    signal_system = ForexSignalSystem(TWELVE_DATA_API, TELEGRAM_TOKEN, CHAT_ID)
    
    print("🔥 GBP/USD PROFESSIONAL SIGNAL SYSTEM 🔥")
    print("📊 Advanced Technical Analysis with S/R Zones")
    print("⏰ 5-Minute Updates | 🎯 High Accuracy Signals")
    
    # Send test signal first
    print("\n🧪 SENDING TEST SIGNAL TO VERIFY TELEGRAM CONNECTION...")
    signal_system.send_test_signal()
    print("⏳ Waiting 10 seconds before starting main analysis...")
    time.sleep(10)
    
    try:
        while True:
            signal_system.run_analysis()
            print(f"\n⏰ Next analysis in 5 minutes...")
            print(f"{'='*60}")
            time.sleep(300)  # 5 minutes
            
    except KeyboardInterrupt:
        print("\n\n📊 Signal system stopped. Happy trading! 🚀")
    except Exception as e:
        print(f"\n❌ System Error: {e}")

if __name__ == "__main__":
    main()
