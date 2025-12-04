# INSTITUTIONAL PRE-MARKET ANALYSIS ENGINE
# AUTOMATIC 9 AM REPORT - BY INSTITUTIONAL TRADER

import os
import time
import requests
import pandas as pd
import yfinance as yf
import warnings
import json
from datetime import datetime, time as dtime, timedelta
import pytz
import numpy as np
from bs4 import BeautifulSoup
import re

warnings.filterwarnings("ignore")

# --------- TELEGRAM SETUP ---------
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
        requests.post(url, data=payload, timeout=10)
        return True
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

# --------- GET IST TIME ---------
def get_ist_time():
    utc_now = datetime.utcnow()
    ist = pytz.timezone('Asia/Kolkata')
    ist_now = utc_now.replace(tzinfo=pytz.utc).astimezone(ist)
    return ist_now

# 🚨 **1. SGX NIFTY DATA (MOST IMPORTANT)** 🚨
def get_sgx_nifty():
    """
    Institutional traders watch SGX Nifty for gap direction
    """
    try:
        # Method 1: Direct SGX ticker
        sgx = yf.download("NIFTY50", period="1d", interval="1m", progress=False)
        if not sgx.empty:
            sgx_close = sgx['Close'].iloc[-1]
            return round(float(sgx_close), 2)
        
        # Method 2: Alternative source
        url = "https://www.sgx.com/derivatives/delayed-prices?code=NIF"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Look for price pattern
            price_pattern = r'(\d{5,6}\.\d{2})'
            matches = re.findall(price_pattern, response.text)
            if matches:
                return round(float(matches[0]), 2)
        
        # Method 3: Use financial APIs
        try:
            url = "https://api.stockmarket.com/sgx/nifty"
            data = requests.get(url, timeout=5).json()
            if 'last_price' in data:
                return round(float(data['last_price']), 2)
        except:
            pass
            
    except Exception as e:
        print(f"SGX error: {e}")
    
    return None

# 🚨 **2. GLOBAL MARKET DATA** 🚨
def get_global_markets():
    """
    Get overnight global market performance
    """
    markets = {}
    
    try:
        # US Markets (previous night)
        dow = yf.download("^DJI", period="2d", interval="1d", progress=False)
        nasdaq = yf.download("^IXIC", period="2d", interval="1d", progress=False)
        sp500 = yf.download("^GSPC", period="2d", interval="1d", progress=False)
        
        if not dow.empty and len(dow) >= 2:
            dow_change = ((dow['Close'].iloc[-1] - dow['Close'].iloc[-2]) / dow['Close'].iloc[-2]) * 100
            markets['DOW'] = round(dow_change, 2)
        
        if not nasdaq.empty and len(nasdaq) >= 2:
            nasdaq_change = ((nasdaq['Close'].iloc[-1] - nasdaq['Close'].iloc[-2]) / nasdaq['Close'].iloc[-2]) * 100
            markets['NASDAQ'] = round(nasdaq_change, 2)
        
        if not sp500.empty and len(sp500) >= 2:
            sp500_change = ((sp500['Close'].iloc[-1] - sp500['Close'].iloc[-2]) / sp500['Close'].iloc[-2]) * 100
            markets['S&P500'] = round(sp500_change, 2)
        
        # Asian Markets (current morning)
        nikkei = yf.download("^N225", period="1d", interval="1m", progress=False)
        hang_seng = yf.download("^HSI", period="1d", interval="1m", progress=False)
        
        if not nikkei.empty:
            nikkei_change = ((nikkei['Close'].iloc[-1] - nikkei['Open'].iloc[0]) / nikkei['Open'].iloc[0]) * 100
            markets['NIKKEI'] = round(nikkei_change, 2)
        
        if not hang_seng.empty:
            hsi_change = ((hang_seng['Close'].iloc[-1] - hang_seng['Open'].iloc[0]) / hang_seng['Open'].iloc[0]) * 100
            markets['HSI'] = round(hsi_change, 2)
            
    except Exception as e:
        print(f"Global markets error: {e}")
    
    return markets

# 🚨 **3. PREVIOUS DAY NIFTY/BANKNIFTY DATA** 🚨
def get_previous_day_data():
    """
    Get previous day's high, low, close, and candle pattern
    """
    data = {}
    
    try:
        # NIFTY
        nifty = yf.download("^NSEI", period="5d", interval="1d", progress=False)
        if not nifty.empty and len(nifty) >= 2:
            prev = nifty.iloc[-2]
            data['NIFTY'] = {
                'OPEN': round(float(prev['Open']), 2),
                'HIGH': round(float(prev['High']), 2),
                'LOW': round(float(prev['Low']), 2),
                'CLOSE': round(float(prev['Close']), 2),
                'CHANGE': round(((prev['Close'] - prev['Open']) / prev['Open']) * 100, 2),
                'VOLUME': int(prev['Volume'])
            }
            
            # Candle pattern analysis
            body = abs(prev['Close'] - prev['Open'])
            upper_wick = prev['High'] - max(prev['Close'], prev['Open'])
            lower_wick = min(prev['Close'], prev['Open']) - prev['Low']
            
            if prev['Close'] > prev['Open']:
                if upper_wick < body * 0.1 and lower_wick < body * 0.1:
                    data['NIFTY']['PATTERN'] = "BULLISH MARUBOZU"
                elif body > 0:
                    data['NIFTY']['PATTERN'] = "BULLISH"
            else:
                if upper_wick < body * 0.1 and lower_wick < body * 0.1:
                    data['NIFTY']['PATTERN'] = "BEARISH MARUBOZU"
                elif body > 0:
                    data['NIFTY']['PATTERN'] = "BEARISH"
        
        # BANKNIFTY
        banknifty = yf.download("^NSEBANK", period="5d", interval="1d", progress=False)
        if not banknifty.empty and len(banknifty) >= 2:
            prev = banknifty.iloc[-2]
            data['BANKNIFTY'] = {
                'OPEN': round(float(prev['Open']), 2),
                'HIGH': round(float(prev['High']), 2),
                'LOW': round(float(prev['Low']), 2),
                'CLOSE': round(float(prev['Close']), 2),
                'CHANGE': round(((prev['Close'] - prev['Open']) / prev['Open']) * 100, 2)
            }
            
    except Exception as e:
        print(f"Previous day error: {e}")
    
    return data

# 🚨 **4. LAST 30 MINUTES OF PREVIOUS DAY** 🚨
def get_last_30min_data():
    """
    Get last 30 minutes (6x5min candles) of previous day
    """
    try:
        # Get last day's 5min data
        nifty_5min = yf.download("^NSEI", period="1d", interval="5m", progress=False)
        
        if not nifty_5min.empty and len(nifty_5min) >= 6:
            # Get last 6 candles (30 minutes)
            last_30min = nifty_5min.iloc[-6:]
            
            last_30min_high = last_30min['High'].max()
            last_30min_low = last_30min['Low'].min()
            last_5min_close = last_30min['Close'].iloc[-1]
            vwap = (last_30min['Close'] * last_30min['Volume']).sum() / last_30min['Volume'].sum()
            
            return {
                'HIGH_30M': round(float(last_30min_high), 2),
                'LOW_30M': round(float(last_30min_low), 2),
                'LAST_CLOSE': round(float(last_5min_close), 2),
                'VWAP': round(float(vwap), 2),
                'CLOSE_VS_VWAP': "ABOVE" if last_5min_close > vwap else "BELOW"
            }
            
    except Exception as e:
        print(f"Last 30min error: {e}")
    
    return None

# 🚨 **5. INDIA VIX (FEAR INDEX)** 🚨
def get_india_vix():
    """
    Get India VIX for volatility expectation
    """
    try:
        vix = yf.download("^INDIAVIX", period="1d", interval="1d", progress=False)
        if not vix.empty:
            vix_value = round(float(vix['Close'].iloc[-1]), 2)
            
            # Interpretation
            if vix_value < 12: vix_sentiment = "LOW FEAR (Rangebound)"
            elif vix_value < 18: vix_sentiment = "NORMAL"
            elif vix_value < 25: vix_sentiment = "HIGH FEAR (Volatile)"
            else: vix_sentiment = "EXTREME FEAR (High Volatility)"
            
            return vix_value, vix_sentiment
            
    except Exception as e:
        print(f"VIX error: {e}")
    
    return None, "UNAVAILABLE"

# 🚨 **6. FII/DII DATA (INSTITUTIONAL FLOW)** 🚨
def get_fii_dii_data():
    """
    Get FII/DII net buy/sell data from NSE
    """
    try:
        # NSE FII/DII data
        url = "https://www.nseindia.com/api/dashboardFII"
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json',
            'Referer': 'https://www.nseindia.com/'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'data' in data:
                for item in data['data']:
                    if 'fii' in item.get('key', '').lower():
                        fii_net = float(item.get('net', 0))
                    elif 'dii' in item.get('key', '').lower():
                        dii_net = float(item.get('net', 0))
                
                return {
                    'FII_NET': round(fii_net, 2),
                    'DII_NET': round(dii_net, 2),
                    'FII_SENTIMENT': 'BUYING' if fii_net > 0 else 'SELLING',
                    'DII_SENTIMENT': 'BUYING' if dii_net > 0 else 'SELLING'
                }
                
    except Exception as e:
        print(f"FII/DII error: {e}")
    
    # Fallback: Use Moneycontrol API
    try:
        url = "https://www.moneycontrol.com/technicals/fii_dii/fii_dii_data.json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Parse data
            return {
                'FII_NET': data.get('fii_net', 0),
                'DII_NET': data.get('dii_net', 0)
            }
    except:
        pass
    
    return None

# 🚨 **7. PUT-CALL RATIO (SENTIMENT INDICATOR)** 🚨
def get_put_call_ratio():
    """
    Get PCR from NSE options data
    """
    try:
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json',
            'Referer': 'https://www.nseindia.com/option-chain'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            total_ce_oi = 0
            total_pe_oi = 0
            
            if 'records' in data and 'data' in data['records']:
                for record in data['records']['data']:
                    if 'CE' in record:
                        total_ce_oi += record['CE'].get('openInterest', 0)
                    if 'PE' in record:
                        total_pe_oi += record['PE'].get('openInterest', 0)
            
            if total_pe_oi > 0:
                pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
                
                # Interpretation
                if pcr > 1.5: sentiment = "EXTREME FEAR (Oversold)"
                elif pcr > 1.2: sentiment = "FEAR (Bearish)"
                elif pcr > 0.8: sentiment = "NEUTRAL"
                elif pcr > 0.5: sentiment = "GREED (Bullish)"
                else: sentiment = "EXTREME GREED (Overbought)"
                
                return round(pcr, 2), sentiment, total_ce_oi, total_pe_oi
                
    except Exception as e:
        print(f"PCR error: {e}")
    
    return None, "UNAVAILABLE", 0, 0

# 🚨 **8. MAX PAIN THEORY** 🚨
def calculate_max_pain():
    """
    Calculate Max Pain level for NIFTY
    """
    try:
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json',
            'Referer': 'https://www.nseindia.com/option-chain'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            strikes = []
            pain_values = []
            
            if 'records' in data and 'data' in data['records']:
                # Get all strikes
                for record in data['records']['data']:
                    strike = record.get('strikePrice', 0)
                    if strike not in strikes:
                        strikes.append(strike)
                
                strikes.sort()
                
                # Calculate pain at each strike
                for strike in strikes:
                    total_pain = 0
                    
                    for record in data['records']['data']:
                        if record.get('strikePrice') == strike:
                            ce_oi = record.get('CE', {}).get('openInterest', 0)
                            pe_oi = record.get('PE', {}).get('openInterest', 0)
                            
                            # Calls ITM if strike < expiry price
                            # Puts ITM if strike > expiry price
                            for s in strikes:
                                if s < strike:
                                    total_pain += pe_oi * (strike - s)
                                elif s > strike:
                                    total_pain += ce_oi * (s - strike)
                    
                    pain_values.append(total_pain)
                
                # Find strike with minimum pain
                if pain_values:
                    min_pain_index = pain_values.index(min(pain_values))
                    max_pain_strike = strikes[min_pain_index]
                    
                    # Get current price for comparison
                    nifty = yf.download("^NSEI", period="1d", interval="1d", progress=False)
                    if not nifty.empty:
                        current_price = nifty['Close'].iloc[-1]
                        distance = abs(current_price - max_pain_strike)
                        distance_pct = (distance / current_price) * 100
                        
                        if current_price > max_pain_strike:
                            bias = "DOWNWARD PRESSURE"
                        else:
                            bias = "UPWARD PRESSURE"
                        
                        return {
                            'MAX_PAIN': max_pain_strike,
                            'CURRENT': round(float(current_price), 2),
                            'DISTANCE': round(float(distance), 2),
                            'DISTANCE_PCT': round(distance_pct, 2),
                            'BIAS': bias
                        }
                        
    except Exception as e:
        print(f"Max Pain error: {e}")
    
    return None

# 🚨 **9. KEY ECONOMIC EVENTS** 🚨
def get_economic_events():
    """
    Check for important economic events
    """
    events = []
    
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        
        # RBI events
        rbi_events = ["RBI MPC Meeting", "RBI Policy", "Repo Rate Decision"]
        
        # US events that affect Indian markets
        us_events = ["FOMC Meeting", "Fed Rate Decision", "US Non-Farm Payrolls", 
                    "US CPI Data", "US GDP Data"]
        
        # Global events
        global_events = ["OPEC Meeting", "G20 Summit", "Trade War News"]
        
        # Check if today is RBI MPC day (1st week of month)
        today_date = datetime.now()
        if today_date.day <= 7:  # First week
            events.append("RBI MPC MEETING THIS WEEK")
        
        # Check for US FOMC (usually 2nd week)
        if 8 <= today_date.day <= 14:
            events.append("US FOMC MEETING THIS WEEK")
            
    except Exception as e:
        print(f"Events error: {e}")
    
    return events

# 🚨 **10. TECHNICAL LEVELS** 🚨
def get_technical_levels():
    """
    Calculate key technical levels
    """
    try:
        nifty = yf.download("^NSEI", period="20d", interval="1d", progress=False)
        
        if not nifty.empty and len(nifty) >= 20:
            closes = nifty['Close']
            
            # Moving Averages
            ma20 = closes.rolling(20).mean().iloc[-1]
            ma50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else ma20
            
            # Support/Resistance
            recent_high = nifty['High'].iloc[-5:].max()
            recent_low = nifty['Low'].iloc[-5:].min()
            
            # Fibonacci levels (if we have swing high/low)
            swing_high = nifty['High'].iloc[-10:].max()
            swing_low = nifty['Low'].iloc[-10:].min()
            swing_range = swing_high - swing_low
            
            fib_levels = {
                '0.236': swing_low + swing_range * 0.236,
                '0.382': swing_low + swing_range * 0.382,
                '0.500': swing_low + swing_range * 0.500,
                '0.618': swing_low + swing_range * 0.618,
                '0.786': swing_low + swing_range * 0.786
            }
            
            return {
                'MA20': round(float(ma20), 2),
                'MA50': round(float(ma50), 2),
                'RESISTANCE': round(float(recent_high), 2),
                'SUPPORT': round(float(recent_low), 2),
                'FIB_LEVELS': {k: round(float(v), 2) for k, v in fib_levels.items()}
            }
            
    except Exception as e:
        print(f"Technical levels error: {e}")
    
    return None

# 🚨 **11. OPENING GAP PREDICTION ALGORITHM** 🚨
def predict_opening_gap():
    """
    Institutional algorithm to predict opening gap
    """
    score = 0
    factors = []
    
    try:
        # Get all data
        sgx_nifty = get_sgx_nifty()
        prev_data = get_previous_day_data()
        global_mkts = get_global_markets()
        vix_value, vix_sentiment = get_india_vix()
        
        if 'NIFTY' in prev_data and sgx_nifty:
            prev_close = prev_data['NIFTY']['CLOSE']
            gap_pct = ((sgx_nifty - prev_close) / prev_close) * 100
            
            # Factor 1: SGX Gap (40% weight)
            if gap_pct > 0.3:
                score += 40
                factors.append(f"SGX GAP UP: +{gap_pct:.2f}%")
            elif gap_pct < -0.3:
                score -= 40
                factors.append(f"SGX GAP DOWN: {gap_pct:.2f}%")
            else:
                factors.append(f"SGX FLAT: {gap_pct:.2f}%")
            
            # Factor 2: Global Markets (20% weight)
            global_score = 0
            for market, change in global_mkts.items():
                if change > 0.5:
                    global_score += 5
                elif change < -0.5:
                    global_score -= 5
            
            if global_score > 10:
                score += 20
                factors.append("GLOBAL MARKETS STRONGLY POSITIVE")
            elif global_score > 0:
                score += 10
                factors.append("GLOBAL MARKETS POSITIVE")
            elif global_score < -10:
                score -= 20
                factors.append("GLOBAL MARKETS STRONGLY NEGATIVE")
            elif global_score < 0:
                score -= 10
                factors.append("GLOBAL MARKETS NEGATIVE")
            
            # Factor 3: Previous Day Close (15% weight)
            prev_close_pos = (prev_data['NIFTY']['CLOSE'] - prev_data['NIFTY']['LOW']) / (prev_data['NIFTY']['HIGH'] - prev_data['NIFTY']['LOW'])
            if prev_close_pos > 0.6:
                score += 15
                factors.append("PREV CLOSE IN UPPER RANGE")
            elif prev_close_pos < 0.4:
                score -= 15
                factors.append("PREV CLOSE IN LOWER RANGE")
            else:
                factors.append("PREV CLOSE IN MIDDLE")
            
            # Factor 4: VIX (10% weight)
            if vix_value and vix_value > 18:
                score -= 10  # High VIX negative for gap up
                factors.append(f"HIGH VIX: {vix_value} ({vix_sentiment})")
            elif vix_value and vix_value < 12:
                score += 5   # Low VIX positive
                factors.append(f"LOW VIX: {vix_value} ({vix_sentiment})")
            else:
                factors.append(f"VIX NORMAL: {vix_value}")
            
            # Factor 5: FII/DII Flow (15% weight)
            fii_dii = get_fii_dii_data()
            if fii_dii:
                if fii_dii['FII_NET'] > 500:
                    score += 15
                    factors.append(f"FII NET BUY: ₹{fii_dii['FII_NET']}Cr")
                elif fii_dii['FII_NET'] < -500:
                    score -= 15
                    factors.append(f"FII NET SELL: ₹{abs(fii_dii['FII_NET'])}Cr")
            
            # Prediction
            if score >= 50:
                prediction = "STRONG GAP UP OPENING"
                bias = "BULLISH"
            elif score >= 20:
                prediction = "MODERATE GAP UP OPENING"
                bias = "MILD BULLISH"
            elif score <= -50:
                prediction = "STRONG GAP DOWN OPENING"
                bias = "BEARISH"
            elif score <= -20:
                prediction = "MODERATE GAP DOWN OPENING"
                bias = "MILD BEARISH"
            else:
                prediction = "FLAT TO MIXED OPENING"
                bias = "NEUTRAL"
            
            return {
                'SCORE': score,
                'PREDICTION': prediction,
                'BIAS': bias,
                'GAP_PCT': round(gap_pct, 2),
                'SGX_PRICE': sgx_nifty,
                'PREV_CLOSE': prev_close,
                'FACTORS': factors
            }
            
    except Exception as e:
        print(f"Gap prediction error: {e}")
    
    return None

# 🚨 **12. GENERATE INSTITUTIONAL PRE-MARKET REPORT** 🚨
def generate_premarket_report():
    """
    Generate comprehensive 9 AM institutional report
    """
    ist_now = get_ist_time()
    report_date = ist_now.strftime("%d %b %Y, %A")
    report_time = ist_now.strftime("%H:%M IST")
    
    report = []
    report.append(f"<b>📊 INSTITUTIONAL PRE-MARKET ANALYSIS</b>")
    report.append(f"<b>📅 {report_date}</b>")
    report.append(f"<b>⏰ {report_time}</b>")
    report.append("")
    report.append("<b>══════════════════════════════</b>")
    report.append("")
    
    try:
        # 1. SGX NIFTY
        sgx_nifty = get_sgx_nifty()
        if sgx_nifty:
            report.append(f"<b>🌏 SGX NIFTY:</b> <code>{sgx_nifty}</code>")
        
        # 2. GLOBAL MARKETS
        global_mkts = get_global_markets()
        if global_mkts:
            report.append(f"<b>🌍 GLOBAL MARKETS:</b>")
            for market, change in global_mkts.items():
                if change > 0:
                    report.append(f"  {market}: <code>🟢 +{change}%</code>")
                else:
                    report.append(f"  {market}: <code>🔴 {change}%</code>")
        
        report.append("")
        
        # 3. PREVIOUS DAY DATA
        prev_data = get_previous_day_data()
        if 'NIFTY' in prev_data:
            n = prev_data['NIFTY']
            change_icon = "🟢" if n['CHANGE'] > 0 else "🔴"
            report.append(f"<b>📈 PREVIOUS DAY (NIFTY):</b>")
            report.append(f"  Open: <code>{n['OPEN']}</code>")
            report.append(f"  High: <code>{n['HIGH']}</code>")
            report.append(f"  Low: <code>{n['LOW']}</code>")
            report.append(f"  Close: <code>{n['CLOSE']}</code> {change_icon} {n['CHANGE']}%")
            if 'PATTERN' in n:
                report.append(f"  Pattern: <code>{n['PATTERN']}</code>")
        
        if 'BANKNIFTY' in prev_data:
            bn = prev_data['BANKNIFTY']
            change_icon = "🟢" if bn['CHANGE'] > 0 else "🔴"
            report.append(f"<b>🏦 PREVIOUS DAY (BANKNIFTY):</b>")
            report.append(f"  Close: <code>{bn['CLOSE']}</code> {change_icon} {bn['CHANGE']}%")
        
        report.append("")
        
        # 4. LAST 30 MINUTES
        last_30min = get_last_30min_data()
        if last_30min:
            report.append(f"<b>⏱️ LAST 30 MINUTES:</b>")
            report.append(f"  High: <code>{last_30min['HIGH_30M']}</code>")
            report.append(f"  Low: <code>{last_30min['LOW_30M']}</code>")
            report.append(f"  Last Close: <code>{last_30min['LAST_CLOSE']}</code>")
            report.append(f"  Vs VWAP: <code>{last_30min['CLOSE_VS_VWAP']}</code>")
        
        report.append("")
        
        # 5. INDIA VIX
        vix_value, vix_sentiment = get_india_vix()
        if vix_value:
            report.append(f"<b>😨 INDIA VIX:</b> <code>{vix_value}</code>")
            report.append(f"  Sentiment: <code>{vix_sentiment}</code>")
        
        # 6. FII/DII DATA
        fii_dii = get_fii_dii_data()
        if fii_dii:
            fii_icon = "🟢" if fii_dii['FII_NET'] > 0 else "🔴"
            dii_icon = "🟢" if fii_dii['DII_NET'] > 0 else "🔴"
            report.append(f"<b>💰 INSTITUTIONAL FLOW:</b>")
            report.append(f"  FII: {fii_icon} <code>₹{fii_dii['FII_NET']}Cr</code>")
            report.append(f"  DII: {dii_icon} <code>₹{fii_dii['DII_NET']}Cr</code>")
        
        # 7. PUT-CALL RATIO
        pcr, pcr_sentiment, ce_oi, pe_oi = get_put_call_ratio()
        if pcr:
            report.append(f"<b>⚖️ PUT-CALL RATIO:</b> <code>{pcr}</code>")
            report.append(f"  Sentiment: <code>{pcr_sentiment}</code>")
        
        report.append("")
        
        # 8. MAX PAIN
        max_pain = calculate_max_pain()
        if max_pain:
            report.append(f"<b>🎯 MAX PAIN THEORY:</b>")
            report.append(f"  Max Pain: <code>{max_pain['MAX_PAIN']}</code>")
            report.append(f"  Current: <code>{max_pain['CURRENT']}</code>")
            report.append(f"  Distance: <code>{max_pain['DISTANCE']} ({max_pain['DISTANCE_PCT']}%)</code>")
            report.append(f"  Bias: <code>{max_pain['BIAS']}</code>")
        
        report.append("")
        
        # 9. TECHNICAL LEVELS
        tech_levels = get_technical_levels()
        if tech_levels:
            report.append(f"<b>📊 TECHNICAL LEVELS:</b>")
            report.append(f"  MA20: <code>{tech_levels['MA20']}</code>")
            report.append(f"  MA50: <code>{tech_levels['MA50']}</code>")
            report.append(f"  Support: <code>{tech_levels['SUPPORT']}</code>")
            report.append(f"  Resistance: <code>{tech_levels['RESISTANCE']}</code>")
        
        report.append("")
        report.append("<b>══════════════════════════════</b>")
        report.append("")
        
        # 10. OPENING GAP PREDICTION
        gap_prediction = predict_opening_gap()
        if gap_prediction:
            report.append(f"<b>🎯 OPENING GAP PREDICTION:</b>")
            report.append(f"  <b>{gap_prediction['PREDICTION']}</b>")
            report.append(f"  Score: <code>{gap_prediction['SCORE']}/100</code>")
            report.append(f"  Bias: <code>{gap_prediction['BIAS']}</code>")
            
            if sgx_nifty and 'PREV_CLOSE' in gap_prediction:
                gap_value = sgx_nifty - gap_prediction['PREV_CLOSE']
                report.append(f"  Expected Gap: <code>{gap_value:.2f} points ({gap_prediction['GAP_PCT']:.2f}%)</code>")
            
            report.append("")
            report.append(f"<b>📋 KEY FACTORS:</b>")
            for factor in gap_prediction['FACTORS']:
                report.append(f"  • {factor}")
        
        report.append("")
        report.append("<b>══════════════════════════════</b>")
        report.append("")
        
        # 11. INSTITUTIONAL TRADING PLAN
        report.append(f"<b>🎯 INSTITUTIONAL TRADING PLAN:</b>")
        
        if gap_prediction:
            bias = gap_prediction['BIAS']
            if "BULLISH" in bias:
                report.append(f"  • <b>Gap Up Play:</b> Wait for pullback to buy")
                report.append(f"  • <b>Resistance:</b> {prev_data['NIFTY']['HIGH'] + 50 if prev_data else 'N/A'}")
                report.append(f"  • <b>Strategy:</b> Buy on dip with SL below opening low")
            elif "BEARISH" in bias:
                report.append(f"  • <b>Gap Down Play:</b> Sell on rise")
                report.append(f"  • <b>Support:</b> {prev_data['NIFTY']['LOW'] - 50 if prev_data else 'N/A'}")
                report.append(f"  • <b>Strategy:</b> Sell rallies with SL above opening high")
            else:
                report.append(f"  • <b>Rangebound Play:</b> Buy support, Sell resistance")
                report.append(f"  • <b>Range:</b> {prev_data['NIFTY']['LOW'] if prev_data else 'N/A'} - {prev_data['NIFTY']['HIGH'] if prev_data else 'N/A'}")
                report.append(f"  • <b>Strategy:</b> Fade extremes")
        
        report.append("")
        report.append("<b>⚠️ RISK DISCLAIMER:</b>")
        report.append("This is institutional analysis for educational purposes.")
        report.append("Trade at your own risk. Past performance ≠ future results.")
        report.append("")
        report.append("<b>✅ REPORT GENERATED: Institutional Trading Desk</b>")
        
        return "\n".join(report)
        
    except Exception as e:
        error_msg = f"<b>⚠️ REPORT GENERATION ERROR:</b>\n{str(e)[:200]}"
        return error_msg

# 🚨 **13. MAIN SCHEDULER** 🚨
def main():
    """
    Main function to run at 9 AM IST
    """
    print("🚀 Institutional Pre-Market Analysis Engine Started...")
    
    last_report_sent = False
    
    while True:
        try:
            ist_now = get_ist_time()
            current_time = ist_now.time()
            current_date = ist_now.date()
            
            # Check if it's 9:00 AM IST
            if (dtime(8, 59) <= current_time <= dtime(9, 2) and 
                not last_report_sent):
                
                print(f"⏰ {ist_now.strftime('%H:%M:%S')} - Generating 9 AM Report...")
                
                # Generate report
                report = generate_premarket_report()
                
                # Split report if too long (Telegram limit)
                max_length = 4000
                if len(report) > max_length:
                    parts = [report[i:i+max_length] for i in range(0, len(report), max_length)]
                    for part in parts:
                        send_telegram(part)
                        time.sleep(1)
                else:
                    send_telegram(report)
                
                print("✅ 9 AM Report Sent to Telegram!")
                last_report_sent = True
            
            # Reset at 9:30 AM to allow next day's report
            elif current_time >= dtime(9, 30):
                last_report_sent = False
            
            # Sleep for 30 seconds
            time.sleep(30)
            
        except Exception as e:
            print(f"❌ Main loop error: {e}")
            time.sleep(60)

# 🚨 **RUN THE ENGINE** 🚨
if __name__ == "__main__":
    # Send startup message
    ist_now = get_ist_time()
    startup_msg = f"🚀 <b>Institutional Pre-Market Analysis Engine</b>\n"
    startup_msg += f"⏰ Started at: {ist_now.strftime('%H:%M:%S IST')}\n"
    startup_msg += f"📅 Date: {ist_now.strftime('%d %b %Y')}\n"
    startup_msg += f"📊 Next report at: 9:00 AM IST\n"
    startup_msg += f"✅ Engine running..."
    send_telegram(startup_msg)
    
    # Run main scheduler
    main()
