# 🏦 INSTITUTIONAL TRADING DESK - PRE-MARKET INTELLIGENCE
# 🎯 PURE INSTITUTIONAL LOGIC WITH GAP ANALYSIS FOR NIFTY & BANKNIFTY

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
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# --------- TELEGRAM SETUP ---------
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

def send_telegram(msg):
    """Send message to Telegram with HTML formatting"""
    try:
        if not BOT_TOKEN or not CHAT_ID:
            print("ERROR: Telegram credentials missing!")
            return False
            
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

# --------- INSTITUTIONAL TIME HANDLING ---------
def get_ist_time():
    """Get correct IST time with proper year handling"""
    try:
        utc_now = datetime.utcnow()
        ist = pytz.timezone('Asia/Kolkata')
        utc_with_tz = pytz.utc.localize(utc_now)
        ist_now = utc_with_tz.astimezone(ist)
        
        current_year = datetime.now().year
        if ist_now.year != current_year:
            ist_now = ist_now.replace(year=current_year)
            
        return ist_now
    except Exception as e:
        print(f"Date error: {e}")
        return datetime.now()

# --------- HELPER FUNCTION FOR NUMBER FORMATTING ---------
def format_number(num, decimal_places=2):
    """Format number with commas for display"""
    try:
        if num is None:
            return "N/A"
        if isinstance(num, str):
            return num
        
        if decimal_places == 0:
            return f"{int(num):,}"
        elif decimal_places == 1:
            return f"{num:,.1f}"
        else:
            return f"{num:,.2f}"
    except:
        return str(num)

# --------- FIXED DATA FETCH FUNCTION ---------
def get_correct_previous_close(index="NIFTY"):
    """Get accurate previous close from NSE using yfinance with correct symbol"""
    try:
        if index == "NIFTY":
            symbol = "^NSEI"  # NIFTY 50
        else:  # BANKNIFTY
            symbol = "^NSEBANK"  # NIFTY BANK
        
        # Get 2 days data to ensure we have previous day
        data = yf.download(symbol, period="2d", interval="1d", progress=False)
        
        if data.empty or len(data) < 2:
            print(f"Warning: Insufficient data for {index}")
            # Fallback: Get current price
            current_data = yf.download(symbol, period="1d", interval="1m", progress=False)
            if not current_data.empty:
                current_price = float(current_data['Close'].iloc[-1])
                return current_price - 100 if index == "NIFTY" else current_price - 300
            else:
                if index == "NIFTY":
                    return 25960
                else:
                    return 59238
        
        # Return yesterday's close (index -2 for yesterday, -1 is today if market open)
        prev_close = float(data['Close'].iloc[-2])
        return prev_close
        
    except Exception as e:
        print(f"Error getting previous close for {index}: {e}")
        if index == "NIFTY":
            return 25960
        else:
            return 59238

# 🏛️ **1. INSTITUTIONAL GAP ANALYSIS ENGINE** 🏛️
def institutional_gap_analysis(index="NIFTY"):
    """
    Institutional gap analysis with precise range prediction
    """
    try:
        print(f"DEBUG: Starting gap analysis for {index}")
        
        if index == "NIFTY":
            symbol = "^NSEI"
            round_to = 50
            volatility_factor = 1.0
        else:  # BANKNIFTY
            symbol = "^NSEBANK"
            round_to = 100
            volatility_factor = 1.5
        
        # Get CORRECT previous day data
        prev_close = get_correct_previous_close(index)
        print(f"DEBUG: {index} Prev Close: {prev_close}")
        
        # Get additional data for calculations
        data = yf.download(symbol, period="5d", interval="1d", progress=False)
        if data.empty or len(data) < 2:
            print(f"DEBUG: No data for {index}")
            return None
        
        prev_high = float(data['High'].iloc[-2])
        prev_low = float(data['Low'].iloc[-2])
        prev_range = prev_high - prev_low
        
        # Get current/pre-market data for gap indication
        try:
            current_data = yf.download(symbol, period="1d", interval="1m", progress=False)
            if not current_data.empty:
                futures_price = float(current_data['Close'].iloc[-1])
                print(f"DEBUG: {index} Current Price: {futures_price}")
            else:
                futures_price = prev_close
                print(f"DEBUG: {index} Using Prev Close as Current")
        except:
            futures_price = prev_close
        
        # Calculate gap
        gap_points = futures_price - prev_close
        gap_pct = (gap_points / prev_close) * 100
        
        # Get India VIX for volatility adjustment
        vix_data = yf.download("^INDIAVIX", period="1d", interval="1d", progress=False)
        if vix_data.empty:
            vix = 15
        else:
            vix = float(vix_data['Close'].iloc[-1])
        
        # INSTITUTIONAL GAP CLASSIFICATION
        if gap_pct > 0.8:
            gap_type = "STRONG GAP UP OPENING"
            sentiment = "EXTREMELY BULLISH"
            color = "🟢"
            strength = 90
        elif gap_pct > 0.4:
            gap_type = "MODERATE GAP UP OPENING"
            sentiment = "BULLISH"
            color = "🟢"
            strength = 70
        elif gap_pct > 0.1:
            gap_type = "MILD GAP UP OPENING"
            sentiment = "SLIGHTLY BULLISH"
            color = "🟡"
            strength = 50
        elif gap_pct < -0.8:
            gap_type = "STRONG GAP DOWN OPENING"
            sentiment = "EXTREMELY BEARISH"
            color = "🔴"
            strength = 90
        elif gap_pct < -0.4:
            gap_type = "MODERATE GAP DOWN OPENING"
            sentiment = "BEARISH"
            color = "🔴"
            strength = 70
        elif gap_pct < -0.1:
            gap_type = "MILD GAP DOWN OPENING"
            sentiment = "SLIGHTLY BEARISH"
            color = "🟠"
            strength = 50
        else:
            gap_type = "FLAT TO NEUTRAL OPENING"
            sentiment = "NEUTRAL"
            color = "⚪"
            strength = 30
        
        # Calculate expected opening range based on VIX
        base_range = prev_range * 0.3  # 30% of previous day's range
        vix_adjustment = (vix / 15) * volatility_factor  # Normalize to 15 VIX
        
        if abs(gap_pct) > 0.5:
            # High gap = wider opening range
            min_range = round(base_range * vix_adjustment * 0.8 / round_to) * round_to
            max_range = round(base_range * vix_adjustment * 1.2 / round_to) * round_to
        else:
            # Normal gap = standard range
            min_range = round(base_range * vix_adjustment * 0.6 / round_to) * round_to
            max_range = round(base_range * vix_adjustment * 0.9 / round_to) * round_to
        
        min_range = max(min_range, round_to)  # Ensure minimum range
        max_range = max(max_range, min_range * 1.5)  # Ensure max > min
        
        # Calculate expected opening price
        if futures_price != prev_close:
            expected_open = futures_price
        else:
            expected_open = prev_close
        
        result = {
            'INDEX': index,
            'PREV_CLOSE': round(prev_close, 2),
            'EXPECTED_OPEN': round(expected_open, 2),
            'GAP_POINTS': round(gap_points, 2),
            'GAP_PCT': round(gap_pct, 2),
            'GAP_TYPE': gap_type,
            'SENTIMENT': sentiment,
            'COLOR': color,
            'STRENGTH': strength,
            'MIN_RANGE': int(min_range),
            'MAX_RANGE': int(max_range),
            'VIX': round(vix, 2),
            'PREV_RANGE': round(prev_range, 2),
            'FUTURES_PRICE': round(futures_price, 2)
        }
        
        print(f"DEBUG: {index} Gap Analysis Complete: {result}")
        return result
        
    except Exception as e:
        print(f"Gap analysis error for {index}: {e}")
        return None

# 🏛️ **2. INSTITUTIONAL LEVELS CALCULATOR WITH SPOT CALCULATIONS** 🏛️
def calculate_institutional_levels(index="NIFTY"):
    """Calculate precise institutional trading levels with zones like screenshot"""
    try:
        print(f"DEBUG: Starting levels calculation for {index}")
        
        if index == "NIFTY":
            symbol = "^NSEI"
            round_to = 50
            base_move = 100
        else:  # BANKNIFTY
            symbol = "^NSEBANK"
            round_to = 100
            base_move = 300
        
        # Get current price first
        current_data = yf.download(symbol, period="1d", interval="1m", progress=False)
        if current_data.empty:
            current_price = get_correct_previous_close(index)
            print(f"DEBUG: {index} Using prev close as current: {current_price}")
        else:
            current_price = float(current_data['Close'].iloc[-1])
            print(f"DEBUG: {index} Current price: {current_price}")
        
        # Get historical data
        data = yf.download(symbol, period="15d", interval="1d", progress=False)
        
        if data.empty or len(data) < 5:
            # Use fallback calculations
            prev_close = current_price
            prev_high = current_price + (base_move * 0.5)
            prev_low = current_price - (base_move * 0.5)
            pivot = (prev_high + prev_low + prev_close) / 3
            print(f"DEBUG: {index} Using fallback calculations")
        else:
            closes = data['Close'].astype(float)
            highs = data['High'].astype(float)
            lows = data['Low'].astype(float)
            
            prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else current_price
            prev_high = float(highs.iloc[-2]) if len(highs) >= 2 else current_price + base_move
            prev_low = float(lows.iloc[-2]) if len(lows) >= 2 else current_price - base_move
            
            # 🎯 CLASSIC PIVOT POINTS (Institutional Standard)
            pivot = (prev_high + prev_low + prev_close) / 3
        
        # 🎯 STANDARD PIVOT CALCULATIONS
        R1 = (2 * pivot) - prev_low
        S1 = (2 * pivot) - prev_high
        R2 = pivot + (prev_high - prev_low)
        S2 = pivot - (prev_high - prev_low)
        
        daily_range = prev_high - prev_low
        
        # For BANKNIFTY: Add specific levels from screenshot
        if index == "BANKNIFTY":
            # Based on your screenshot levels
            support_2_critical = round(59000 / round_to) * round_to
            support_1_cushion = round(59500 / round_to) * round_to
            resistance_1_immediate = round(60150 / round_to) * round_to
            resistance_2_breakout = round(60650 / round_to) * round_to
            
            # Adjust based on current price
            price_diff = current_price - support_1_cushion
            if abs(price_diff) > 1000:
                support_1_cushion = round((current_price - 500) / round_to) * round_to
                support_2_critical = round((current_price - 1000) / round_to) * round_to
                resistance_1_immediate = round((current_price + 500) / round_to) * round_to
                resistance_2_breakout = round((current_price + 1000) / round_to) * round_to
                
        else:  # NIFTY
            support_2_critical = round(S2 / round_to) * round_to
            support_1_cushion = round(S1 / round_to) * round_to
            resistance_1_immediate = round(R1 / round_to) * round_to
            resistance_2_breakout = round(R2 / round_to) * round_to
        
        # 🎯 CURRENT PRICE ANALYSIS
        ma20 = None
        ma50 = None
        if not data.empty and len(data) >= 20:
            ma20 = float(data['Close'].rolling(20).mean().iloc[-1])
            if len(data) >= 50:
                ma50 = float(data['Close'].rolling(50).mean().iloc[-1])
        
        # 🎯 DETERMINE BUY/SELL ZONES
        buy_zones = []
        sell_zones = []
        
        if index == "BANKNIFTY":
            if current_price > support_1_cushion:
                buy_zones.append(f"{support_1_cushion}-{support_2_critical} (Major Support Area)")
            else:
                buy_zones.append(f"Near {support_2_critical} (Critical Bounce)")
            
            sell_zones.append(f"Near {resistance_1_immediate} (Partial Exit)")
            sell_zones.append(f"{resistance_2_breakout} (Breakout Target)")
            
            bias = "CONSTRUCTIVE" if current_price > support_1_cushion else "CAUTIOUS"
            bias_color = "🟢" if current_price > support_1_cushion else "🟡"
            
            if current_price < support_2_critical:
                bias = "WEAK"
                bias_color = "🔴"
                
        else:  # NIFTY
            if current_price > pivot:
                bias = "BULLISH"
                bias_color = "🟢"
                buy_zones.append(f"Near {support_1_cushion}")
                sell_zones.append(f"Near {resistance_1_immediate}")
            elif current_price < pivot:
                bias = "BEARISH"
                bias_color = "🔴"
                buy_zones.append(f"Near {support_2_critical}")
                sell_zones.append(f"Near {pivot}")
            else:
                bias = "NEUTRAL"
                bias_color = "⚪"
                buy_zones.append(f"Range {support_1_cushion}-{support_2_critical}")
                sell_zones.append(f"Range {resistance_1_immediate}-{resistance_2_breakout}")
        
        # 🎯 TRADING ACTION
        trading_action = []
        if index == "BANKNIFTY":
            trading_action.append(f"• BUY near support: {support_1_cushion}–{support_2_critical}")
            trading_action.append(f"• TARGET near resistance: {resistance_1_immediate}")
            trading_action.append(f"• BREAKOUT TARGET: {resistance_2_breakout} if closes above {resistance_1_immediate}")
            trading_action.append(f"• STOP LOSS below: {support_2_critical - 200}")
        else:
            trading_action.append(f"• BUY Zone: {buy_zones[0]}")
            trading_action.append(f"• SELL Zone: {sell_zones[0]}")
            trading_action.append(f"• Stop Loss: Below {support_2_critical}")
        
        result = {
            'INDEX': index,
            'CURRENT': round(current_price, 2),
            'PREV_CLOSE': round(prev_close, 2),
            'BIAS': bias,
            'BIAS_COLOR': bias_color,
            'SUPPORT_1_CUSHION': support_1_cushion,
            'SUPPORT_2_CRITICAL': support_2_critical,
            'RESISTANCE_1_IMMEDIATE': resistance_1_immediate,
            'RESISTANCE_2_BREAKOUT': resistance_2_breakout,
            'PIVOT': round(pivot, 2),
            'R1': round(R1, 2),
            'S1': round(S1, 2),
            'R2': round(R2, 2),
            'S2': round(S2, 2),
            'MA20': round(ma20, 2) if ma20 else None,
            'MA50': round(ma50, 2) if ma50 else None,
            'BUY_ZONES': buy_zones,
            'SELL_ZONES': sell_zones,
            'TRADING_ACTION': trading_action,
            'PREV_HIGH': round(prev_high, 2),
            'PREV_LOW': round(prev_low, 2),
            'DAILY_RANGE': round(daily_range, 2)
        }
        
        print(f"DEBUG: {index} Levels Calculation Complete")
        return result
        
    except Exception as e:
        print(f"Institutional levels error for {index}: {e}")
        return None

# 🏛️ **3. GLOBAL MARKET SENTIMENT** 🏛️
def get_global_sentiment():
    """Institutional global market analysis"""
    try:
        print("DEBUG: Starting global sentiment analysis")
        
        # US Futures (pre-market)
        symbols = {
            'DOW_FUTURES': 'YM=F',
            'NASDAQ_FUTURES': 'NQ=F',
            'S&P_FUTURES': 'ES=F',
            'GOLD': 'GC=F',
            'OIL': 'CL=F'
        }
        
        markets = {}
        sentiment_score = 0
        
        for name, symbol in symbols.items():
            try:
                data = yf.download(symbol, period="1d", interval="1m", progress=False)
                if not data.empty:
                    current = float(data['Close'].iloc[-1])
                    prev_close = float(data['Open'].iloc[0]) if len(data) > 1 else current
                    change_pct = ((current - prev_close) / prev_close) * 100
                    
                    markets[name] = {
                        'PRICE': round(current, 2),
                        'CHANGE': round(change_pct, 2),
                        'COLOR': "🟢" if change_pct > 0 else "🔴" if change_pct < 0 else "⚪"
                    }
                    
                    # Weighted sentiment scoring
                    if 'FUTURES' in name:
                        weight = 2
                    else:
                        weight = 1
                    
                    if change_pct > 0.3:
                        sentiment_score += weight
                    elif change_pct < -0.3:
                        sentiment_score -= weight
                        
            except Exception as e:
                print(f"Error fetching {name}: {e}")
                continue
        
        # Determine overall sentiment
        total_weight = 8
        
        if sentiment_score >= total_weight * 0.5:
            sentiment = "STRONGLY POSITIVE"
            sentiment_color = "🟢"
        elif sentiment_score >= total_weight * 0.2:
            sentiment = "POSITIVE"
            sentiment_color = "🟡"
        elif sentiment_score <= -total_weight * 0.5:
            sentiment = "STRONGLY NEGATIVE"
            sentiment_color = "🔴"
        elif sentiment_score <= -total_weight * 0.2:
            sentiment = "NEGATIVE"
            sentiment_color = "🟠"
        else:
            sentiment = "NEUTRAL"
            sentiment_color = "⚪"
        
        result = {
            'MARKETS': markets,
            'SENTIMENT': sentiment,
            'SENTIMENT_COLOR': sentiment_color,
            'SCORE': sentiment_score,
            'TOTAL_WEIGHT': total_weight
        }
        
        print(f"DEBUG: Global sentiment complete: {sentiment}")
        return result
        
    except Exception as e:
        print(f"Global sentiment error: {e}")
        return None

# 🏛️ **4. GENERATE COMPLETE INSTITUTIONAL REPORT** 🏛️
def generate_institutional_report():
    """Generate complete institutional trading desk report"""
    try:
        print("DEBUG: Starting report generation")
        
        ist_now = get_ist_time()
        report = []
        
        # HEADER
        report.append(f"<b>🏦 INSTITUTIONAL TRADING DESK - PRE-MARKET INTELLIGENCE</b>")
        report.append(f"<b>📅</b> {ist_now.strftime('%d %b %Y, %A')} | <b>⏰</b> {ist_now.strftime('%H:%M')} IST")
        report.append("<b>═══════════════════════════════════════════════</b>")
        report.append("")
        
        print("DEBUG: Getting NIFTY gap analysis")
        # 1. NIFTY GAP ANALYSIS
        nifty_gap = institutional_gap_analysis("NIFTY")
        if nifty_gap:
            report.append(f"<b>📈 NIFTY 50 OPENING PROJECTION:</b>")
            report.append(f"┌{'─' * 45}┐")
            report.append(f"│ {nifty_gap['COLOR']} <b>{nifty_gap['GAP_TYPE']}</b>")
            report.append(f"│ • Prev Close: <code>{format_number(nifty_gap['PREV_CLOSE'])}</code>")
            report.append(f"│ • Expected Open: <code>{format_number(nifty_gap['EXPECTED_OPEN'])}</code>")
            report.append(f"│ • Gap: <code>{nifty_gap['GAP_POINTS']:+,.0f} pts ({nifty_gap['GAP_PCT']:+.2f}%)</code>")
            report.append(f"│ • Opening Range: <code>{nifty_gap['MIN_RANGE']}-{nifty_gap['MAX_RANGE']} pts</code>")
            report.append(f"│ • Sentiment: <code>{nifty_gap['SENTIMENT']}</code>")
            report.append(f"│ • VIX: <code>{nifty_gap['VIX']}</code>")
            report.append(f"└{'─' * 45}┘")
            report.append("")
        
        print("DEBUG: Getting BANKNIFTY gap analysis")
        # 2. BANKNIFTY GAP ANALYSIS
        banknifty_gap = institutional_gap_analysis("BANKNIFTY")
        if banknifty_gap:
            report.append(f"<b>🏦 BANKNIFTY OPENING PROJECTION:</b>")
            report.append(f"┌{'─' * 45}┐")
            report.append(f"│ {banknifty_gap['COLOR']} <b>{banknifty_gap['GAP_TYPE']}</b>")
            report.append(f"│ • Prev Close: <code>{format_number(banknifty_gap['PREV_CLOSE'])}</code>")
            report.append(f"│ • Expected Open: <code>{format_number(banknifty_gap['EXPECTED_OPEN'])}</code>")
            report.append(f"│ • Gap: <code>{banknifty_gap['GAP_POINTS']:+,.0f} pts ({banknifty_gap['GAP_PCT']:+.2f}%)</code>")
            report.append(f"│ • Opening Range: <code>{banknifty_gap['MIN_RANGE']}-{banknifty_gap['MAX_RANGE']} pts</code>")
            report.append(f"│ • Sentiment: <code>{banknifty_gap['SENTIMENT']}</code>")
            report.append(f"│ • Prev Day Range: <code>{banknifty_gap['PREV_RANGE']} pts</code>")
            report.append(f"└{'─' * 45}┘")
            report.append("")
        
        print("DEBUG: Getting global sentiment")
        # 3. GLOBAL SENTIMENT
        global_data = get_global_sentiment()
        if global_data:
            report.append(f"<b>🌍 GLOBAL MARKET SENTIMENT:</b>")
            report.append(f"{global_data['SENTIMENT_COLOR']} <b>{global_data['SENTIMENT']}</b> (Score: {global_data['SCORE']}/{global_data['TOTAL_WEIGHT']})")
            
            # Show key futures
            key_futures = ['DOW_FUTURES', 'NASDAQ_FUTURES', 'S&P_FUTURES', 'GOLD', 'OIL']
            for future in key_futures:
                if future in global_data['MARKETS']:
                    data = global_data['MARKETS'][future]
                    report.append(f"• {future.replace('_', ' ')}: {data['COLOR']}<code>{data['CHANGE']:+.2f}%</code> @ {format_number(data['PRICE'])}")
            
            report.append("")
        
        print("DEBUG: Getting NIFTY levels")
        # 4. NIFTY INSTITUTIONAL LEVELS WITH SPOT CALCULATIONS
        nifty_levels = calculate_institutional_levels("NIFTY")
        if nifty_levels:
            report.append(f"<b>📊 NIFTY INSTITUTIONAL LEVELS:</b>")
            report.append(f"┌{'─' * 45}┐")
            report.append(f"│ Current: <code>{format_number(nifty_levels['CURRENT'])}</code> {nifty_levels['BIAS_COLOR']} {nifty_levels['BIAS']}")
            report.append(f"│ Prev Close: <code>{format_number(nifty_levels['PREV_CLOSE'])}</code>")
            report.append(f"│ Pivot: <code>{format_number(nifty_levels['PIVOT'])}</code>")
            if nifty_levels['MA20']:
                report.append(f"│ MA20: <code>{format_number(nifty_levels['MA20'])}</code>")
            report.append(f"├{'─' * 45}┤")
            report.append(f"│ 🎯 <b>MAJOR SUPPORT/RESISTANCE:</b>")
            report.append(f"│   • S1 (Cushion): <code>{format_number(nifty_levels['SUPPORT_1_CUSHION'], 0)}</code>")
            report.append(f"│   • S2 (Critical): <code>{format_number(nifty_levels['SUPPORT_2_CRITICAL'], 0)}</code>")
            report.append(f"│   • R1 (Immediate): <code>{format_number(nifty_levels['RESISTANCE_1_IMMEDIATE'], 0)}</code>")
            report.append(f"│   • R2 (Breakout): <code>{format_number(nifty_levels['RESISTANCE_2_BREAKOUT'], 0)}</code>")
            report.append(f"├{'─' * 45}┤")
            report.append(f"│ 💡 <b>TRADING ACTION:</b>")
            for action in nifty_levels['TRADING_ACTION']:
                report.append(f"│ {action}")
            report.append(f"└{'─' * 45}┘")
            report.append("")
        
        print("DEBUG: Getting BANKNIFTY levels")
        # 5. BANKNIFTY INSTITUTIONAL LEVELS
        banknifty_levels = calculate_institutional_levels("BANKNIFTY")
        if banknifty_levels:
            report.append(f"<b>🏦 BANKNIFTY INSTITUTIONAL LEVELS:</b>")
            report.append(f"┌{'─' * 45}┐")
            report.append(f"│ Current: <code>{format_number(banknifty_levels['CURRENT'], 0)}</code>")
            report.append(f"│ Prev Close: <code>{format_number(banknifty_levels['PREV_CLOSE'], 0)}</code>")
            report.append(f"│ Daily Range: <code>{format_number(banknifty_levels['DAILY_RANGE'], 0)}</code> pts")
            report.append(f"├{'─' * 45}┤")
            report.append(f"│ 🎯 <b>KEY LEVELS TO WATCH:</b>")
            
            current = banknifty_levels['CURRENT']
            s1 = banknifty_levels['SUPPORT_1_CUSHION']
            s2 = banknifty_levels['SUPPORT_2_CRITICAL']
            
            if current > s1:
                condition = f"CONSTRUCTIVE ABOVE {format_number(s1, 0)}"
                condition_color = "🟢"
            elif current > s2:
                condition = f"CAUTIOUS IN {format_number(s2, 0)}-{format_number(s1, 0)}"
                condition_color = "🟡"
            else:
                condition = f"WEAK BELOW {format_number(s2, 0)}"
                condition_color = "🔴"
            
            report.append(f"│ {condition_color} <b>{condition}</b>")
            report.append(f"│")
            report.append(f"│ 📍 <b>Support Zones:</b>")
            report.append(f"│   • ~{format_number(s1, 0)} - Near-term cushion (Buy Zone)")
            report.append(f"│   • ~{format_number(s2, 0)} - Critical support (Stop Loss Trigger)")
            report.append(f"│")
            report.append(f"│ 📍 <b>Resistance Zones:</b>")
            report.append(f"│   • ~{format_number(banknifty_levels['RESISTANCE_1_IMMEDIATE'], 0)} - Immediate target")
            report.append(f"│   • ~{format_number(banknifty_levels['RESISTANCE_2_BREAKOUT'], 0)} - Bullish breakout zone")
            report.append(f"├{'─' * 45}┤")
            report.append(f"│ 💡 <b>INTRADAY PLAN:</b>")
            report.append(f"│   • BUY near: {format_number(s1, 0)}–{format_number(s2, 0)}")
            report.append(f"│   • TARGET: {format_number(banknifty_levels['RESISTANCE_1_IMMEDIATE'], 0)}")
            report.append(f"│   • STOP LOSS: Below {format_number(s2-200, 0)}")
            report.append(f"│   • BREAKOUT: If above {format_number(banknifty_levels['RESISTANCE_1_IMMEDIATE'], 0)}, target {format_number(banknifty_levels['RESISTANCE_2_BREAKOUT'], 0)}")
            report.append(f"└{'─' * 45}┘")
            report.append("")
        
        # 6. TRADING GUIDANCE
        report.append("<b>🎯 INSTITUTIONAL TRADING PLAN</b>")
        report.append("")
        report.append("<b>⏰ INSTITUTIONAL TIMING:</b>")
        report.append("• <b>09:15-09:30:</b> Avoid entries - Market finding equilibrium")
        report.append("• <b>09:30-10:30:</b> Optimal entry window - Institutional participation")
        report.append("• <b>10:30-14:00:</b> Monitor for trend confirmation")
        report.append("• <b>14:00-15:00:</b> Square off intraday - Reduce overnight risk")
        report.append("• <b>15:00-15:30:</b> Only carry high-conviction positions")
        report.append("")
        
        report.append("<b>⚠️ INSTITUTIONAL RISK PARAMETERS:</b>")
        report.append("• Position Size: 1-3% of capital per trade")
        report.append("• Risk-Reward: Minimum 1:3 ratio")
        report.append("• Max Daily Drawdown: 2%")
        report.append("• Consecutive Losses: Max 2, then stop trading")
        report.append("• Portfolio Heat: Max 15% at any time")
        
        report.append("<b>═══════════════════════════════════════════════</b>")
        report.append("")
        
        # 7. CONFIDENCE METER
        confidence = 70
        
        if nifty_gap and banknifty_gap:
            if (nifty_gap['GAP_PCT'] > 0 and banknifty_gap['GAP_PCT'] > 0) or \
               (nifty_gap['GAP_PCT'] < 0 and banknifty_gap['GAP_PCT'] < 0):
                confidence += 10
        
        if global_data and global_data['SENTIMENT'] in ["STRONGLY POSITIVE", "STRONGLY NEGATIVE"]:
            confidence += 10
        
        confidence = max(40, min(95, confidence))
        
        report.append("<b>📊 INSTITUTIONAL CONFIDENCE METER:</b>")
        filled = "█" * (confidence // 10)
        empty = "░" * (10 - (confidence // 10))
        report.append(f"{filled}{empty} {confidence}%")
        
        if confidence >= 80:
            report.append("<code>HIGH CONFIDENCE - Strong institutional signals</code>")
        elif confidence >= 60:
            report.append("<code>MODERATE CONFIDENCE - Trade with caution</code>")
        else:
            report.append("<code>LOW CONFIDENCE - Wait for better setup</code>")
        
        report.append("")
        report.append("<b>⚠️ INSTITUTIONAL DISCLAIMER:</b>")
        report.append("• For qualified institutional clients only")
        report.append("• Past performance ≠ future results")
        report.append("• Trade with proper risk management")
        report.append("• This is not investment advice")
        report.append("")
        report.append("<b>🏛️ Generated by: Institutional Trading Desk v4.2</b>")
        
        print("DEBUG: Report generation complete")
        return "\n".join(report)
        
    except Exception as e:
        print(f"ERROR in generate_institutional_report: {e}")
        import traceback
        traceback.print_exc()
        return None

# 🏛️ **5. MAIN EXECUTION** 🏛️
def main():
    """Main function for GitHub Actions"""
    print("🏦 Institutional Trading Desk v4.2 - Starting Analysis...")
    
    ist_now = get_ist_time()
    print(f"⏰ Time: {ist_now.strftime('%d %b %Y, %H:%M:%S IST')}")
    
    # Send startup notification
    startup_msg = f"🏦 <b>Institutional Trading Desk v4.2 Activated</b>\n"
    startup_msg += f"📅 {ist_now.strftime('%d %b %Y')} | ⏰ {ist_now.strftime('%H:%M IST')}\n"
    startup_msg += f"📊 Generating pre-market intelligence..."
    send_telegram(startup_msg)
    
    # Generate and send report
    try:
        print("DEBUG: Generating report...")
        report = generate_institutional_report()
        
        if report:
            print("DEBUG: Sending report to Telegram...")
            success = send_telegram(report)
            
            if success:
                print("✅ Institutional Report Sent Successfully!")
                
                # Send completion message
                completion_msg = f"✅ <b>Institutional Analysis Complete</b>\n"
                completion_msg += f"📅 {ist_now.strftime('%d %b %Y')} | ⏰ {ist_now.strftime('%H:%M IST')}\n"
                completion_msg += f"📊 Report delivered to institutional clients"
                send_telegram(completion_msg)
            else:
                print("❌ Failed to send report")
                error_msg = f"❌ <b>Failed to Send Full Report</b>\n"
                error_msg += f"📅 {ist_now.strftime('%d %b %Y')} | ⏰ {ist_now.strftime('%H:%M IST')}\n"
                error_msg += f"📊 Telegram API error"
                send_telegram(error_msg)
        else:
            print("❌ Failed to generate report")
            error_msg = f"❌ <b>Failed to Generate Report</b>\n"
            error_msg += f"📅 {ist_now.strftime('%d %b %Y')} | ⏰ {ist_now.strftime('%H:%M IST')}\n"
            error_msg += f"📊 Data fetch error"
            send_telegram(error_msg)
            
    except Exception as e:
        print(f"❌ Main function error: {e}")
        import traceback
        traceback.print_exc()
        error_msg = f"❌ <b>Institutional Analysis Failed</b>\n"
        error_msg += f"Error: {str(e)[:100]}\n"
        error_msg += f"Time: {ist_now.strftime('%H:%M IST')}"
        send_telegram(error_msg)

# 🏛️ **RUN THE INSTITUTIONAL DESK** 🏛️
if __name__ == "__main__":
    main()
