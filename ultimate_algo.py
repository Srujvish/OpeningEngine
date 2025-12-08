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
        
        # Force correct year (2024)
        current_year = datetime.now().year
        if ist_now.year != current_year:
            ist_now = ist_now.replace(year=current_year)
            
        return ist_now
    except Exception as e:
        print(f"Date error: {e}")
        return datetime.now()

# 🏛️ **1. INSTITUTIONAL GAP ANALYSIS ENGINE** 🏛️
def institutional_gap_analysis(index="NIFTY"):
    """
    Institutional gap analysis with precise range prediction
    """
    try:
        if index == "NIFTY":
            symbol = "^NSEI"
            futures_symbol = "NIFTY_50.NS"
            round_to = 50
            volatility_factor = 1.0
        else:  # BANKNIFTY
            symbol = "^NSEBANK"
            futures_symbol = "BANKNIFTY.NS"
            round_to = 100
            volatility_factor = 1.5
        
        # Get previous day data
        data = yf.download(symbol, period="5d", interval="1d", progress=False)
        if data.empty or len(data) < 2:
            return None
        
        prev_close = float(data['Close'].iloc[-2])
        prev_high = float(data['High'].iloc[-2])
        prev_low = float(data['Low'].iloc[-2])
        prev_range = prev_high - prev_low
        
        # Get SGX/Futures data for gap indication
        futures_data = yf.download(futures_symbol, period="1d", interval="1m", progress=False)
        if futures_data.empty:
            futures_price = prev_close
        else:
            futures_price = float(futures_data['Close'].iloc[-1])
        
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
        
        return {
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
        
    except Exception as e:
        print(f"Gap analysis error for {index}: {e}")
        return None

# 🏛️ **2. INSTITUTIONAL LEVELS CALCULATOR** 🏛️
def calculate_institutional_levels(index="NIFTY"):
    """Calculate precise institutional trading levels with zones"""
    try:
        if index == "NIFTY":
            symbol = "^NSEI"
            round_to = 50
            base_move = 100
        else:  # BANKNIFTY
            symbol = "^NSEBANK"
            round_to = 100
            base_move = 300
        
        # Get 15 days data for institutional analysis
        data = yf.download(symbol, period="15d", interval="1d", progress=False)
        
        if data.empty or len(data) < 10:
            return None
        
        closes = data['Close'].astype(float)
        highs = data['High'].astype(float)
        lows = data['Low'].astype(float)
        volumes = data['Volume'].astype(float)
        
        current_price = float(closes.iloc[-1])
        
        # 🎯 PIVOT POINTS (Institutional Standard)
        pivot = (highs.iloc[-1] + lows.iloc[-1] + closes.iloc[-1]) / 3
        r1 = (2 * pivot) - lows.iloc[-1]
        s1 = (2 * pivot) - highs.iloc[-1]
        r2 = pivot + (highs.iloc[-1] - lows.iloc[-1])
        s2 = pivot - (highs.iloc[-1] - lows.iloc[-1])
        
        # 🎯 VOLUME PROFILE AREAS
        recent_volume = volumes.iloc[-5:].mean()
        volume_avg = volumes.iloc[-20:].mean() if len(volumes) >= 20 else recent_volume
        volume_ratio = recent_volume / volume_avg if volume_avg > 0 else 1
        
        # 🎯 PRICE ACTION ZONES
        support_zone = []
        resistance_zone = []
        
        # Previous day's high/low as immediate levels
        prev_high = float(highs.iloc[-2]) if len(highs) >= 2 else highs.iloc[-1]
        prev_low = float(lows.iloc[-2]) if len(lows) >= 2 else lows.iloc[-1]
        
        support_zone.append(round(prev_low / round_to) * round_to)
        resistance_zone.append(round(prev_high / round_to) * round_to)
        
        # Add previous swing points
        for i in range(-5, -10, -1):
            if i < -len(highs):
                break
            swing_high = float(highs.iloc[i])
            swing_low = float(lows.iloc[i])
            
            if swing_high < current_price * 1.05:  # Within 5%
                resistance_zone.append(round(swing_high / round_to) * round_to)
            if swing_low > current_price * 0.95:  # Within 5%
                support_zone.append(round(swing_low / round_to) * round_to)
        
        # Remove duplicates and sort
        support_zone = sorted(list(set(support_zone)))
        resistance_zone = sorted(list(set(resistance_zone)))
        
        # Take 2 closest levels
        immediate_support = support_zone[-1] if support_zone else round(s1 / round_to) * round_to
        critical_support = support_zone[-2] if len(support_zone) >= 2 else round(s2 / round_to) * round_to
        
        immediate_resistance = resistance_zone[0] if resistance_zone else round(r1 / round_to) * round_to
        critical_resistance = resistance_zone[1] if len(resistance_zone) >= 2 else round(r2 / round_to) * round_to
        
        # 🎯 MARKET BIAS DETERMINATION
        ma20 = float(closes.rolling(20).mean().iloc[-1]) if len(closes) >= 20 else current_price
        ma50 = float(closes.rolling(50).mean().iloc[-1]) if len(closes) >= 50 else ma20
        
        price_vs_ma20 = ((current_price - ma20) / ma20) * 100
        price_vs_ma50 = ((current_price - ma50) / ma50) * 100
        
        if price_vs_ma20 > 1 and price_vs_ma50 > 0.5:
            bias = "STRONGLY BULLISH"
            bias_color = "🟢"
            bias_score = 80
        elif price_vs_ma20 > 0:
            bias = "BULLISH"
            bias_color = "🟢"
            bias_score = 60
        elif price_vs_ma20 < -1 and price_vs_ma50 < -0.5:
            bias = "STRONGLY BEARISH"
            bias_color = "🔴"
            bias_score = 80
        elif price_vs_ma20 < 0:
            bias = "BEARISH"
            bias_color = "🔴"
            bias_score = 60
        else:
            bias = "NEUTRAL/RANGEBOUND"
            bias_color = "🟡"
            bias_score = 40
        
        # Adjust bias based on volume
        if volume_ratio > 1.2 and bias_score < 70:
            bias = f"{bias} (HIGH VOLUME)"
            bias_score += 10
        
        return {
            'INDEX': index,
            'CURRENT': round(current_price, 2),
            'BIAS': bias,
            'BIAS_COLOR': bias_color,
            'BIAS_SCORE': bias_score,
            'IMMEDIATE_SUPPORT': immediate_support,
            'CRITICAL_SUPPORT': critical_support,
            'IMMEDIATE_RESISTANCE': immediate_resistance,
            'CRITICAL_RESISTANCE': critical_resistance,
            'MA20': round(ma20, 2),
            'MA50': round(ma50, 2),
            'PIVOT': round(pivot, 2),
            'R1': round(r1, 2),
            'S1': round(s1, 2),
            'VOLUME_RATIO': round(volume_ratio, 2),
            'PREV_HIGH': round(prev_high, 2),
            'PREV_LOW': round(prev_low, 2)
        }
        
    except Exception as e:
        print(f"Institutional levels error for {index}: {e}")
        return None

# 🏛️ **3. GLOBAL MARKET SENTIMENT** 🏛️
def get_global_sentiment():
    """Institutional global market analysis"""
    try:
        # US Futures (pre-market)
        symbols = {
            'DOW_FUTURES': 'YM=F',
            'NASDAQ_FUTURES': 'NQ=F',
            'S&P_FUTURES': 'ES=F',
            'GOLD': 'GC=F',
            'USD/INR': 'INR=X',
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
                        
            except Exception:
                continue
        
        # Determine overall sentiment
        total_weight = 8  # Approximate total weight
        
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
        
        return {
            'MARKETS': markets,
            'SENTIMENT': sentiment,
            'SENTIMENT_COLOR': sentiment_color,
            'SCORE': sentiment_score,
            'TOTAL_WEIGHT': total_weight
        }
        
    except Exception as e:
        print(f"Global sentiment error: {e}")
        return None

# 🏛️ **4. INSTITUTIONAL TRADING PLAN** 🏛️
def generate_institutional_trading_plan():
    """Generate specific institutional trading plan with levels"""
    
    plan = []
    
    # Get analyses for both indices
    nifty_gap = institutional_gap_analysis("NIFTY")
    banknifty_gap = institutional_gap_analysis("BANKNIFTY")
    nifty_levels = calculate_institutional_levels("NIFTY")
    banknifty_levels = calculate_institutional_levels("BANKNIFTY")
    
    plan.append("<b>🎯 INSTITUTIONAL TRADING PLAN</b>")
    plan.append("")
    
    # NIFTY STRATEGY
    if nifty_gap and nifty_levels:
        plan.append(f"<b>📈 NIFTY 50 STRATEGY:</b>")
        plan.append(f"┌{'─' * 45}┐")
        
        # Opening scenario
        if nifty_gap['GAP_PCT'] > 0.4:
            plan.append(f"│ {nifty_gap['COLOR']} <b>{nifty_gap['GAP_TYPE']}</b>")
            plan.append(f"│ • Expected Gap: <b>+{abs(nifty_gap['GAP_POINTS']):.0f}</b> pts")
            plan.append(f"│ • Opening Range: <b>{nifty_gap['MIN_RANGE']}-{nifty_gap['MAX_RANGE']}</b> pts")
            plan.append(f"│ • Strategy: <b>Buy on dip</b>")
            plan.append(f"│ • Entry Zone: {nifty_levels['IMMEDIATE_SUPPORT']:,}-{nifty_gap['EXPECTED_OPEN']:.0f}")
            plan.append(f"│ • Target 1: {nifty_levels['IMMEDIATE_RESISTANCE']:,}")
            plan.append(f"│ • Target 2: {nifty_levels['CRITICAL_RESISTANCE']:,}")
            plan.append(f"│ • Stop Loss: Below {nifty_levels['CRITICAL_SUPPORT']:,}")
            
        elif nifty_gap['GAP_PCT'] < -0.4:
            plan.append(f"│ {nifty_gap['COLOR']} <b>{nifty_gap['GAP_TYPE']}</b>")
            plan.append(f"│ • Expected Gap: <b>-{abs(nifty_gap['GAP_POINTS']):.0f}</b> pts")
            plan.append(f"│ • Opening Range: <b>{nifty_gap['MIN_RANGE']}-{nifty_gap['MAX_RANGE']}</b> pts")
            plan.append(f"│ • Strategy: <b>Sell on rise</b>")
            plan.append(f"│ • Entry Zone: {nifty_gap['EXPECTED_OPEN']:.0f}-{nifty_levels['IMMEDIATE_RESISTANCE']:,}")
            plan.append(f"│ • Target 1: {nifty_levels['IMMEDIATE_SUPPORT']:,}")
            plan.append(f"│ • Target 2: {nifty_levels['CRITICAL_SUPPORT']:,}")
            plan.append(f"│ • Stop Loss: Above {nifty_levels['CRITICAL_RESISTANCE']:,}")
            
        else:
            plan.append(f"│ {nifty_gap['COLOR']} <b>{nifty_gap['GAP_TYPE']}</b>")
            plan.append(f"│ • Expected Gap: <b>{nifty_gap['GAP_POINTS']:+.0f}</b> pts")
            plan.append(f"│ • Opening Range: <b>{nifty_gap['MIN_RANGE']}-{nifty_gap['MAX_RANGE']}</b> pts")
            plan.append(f"│ • Strategy: <b>Rangebound - Fade extremes</b>")
            plan.append(f"│ • Buy Zone: Near {nifty_levels['IMMEDIATE_SUPPORT']:,}")
            plan.append(f"│ • Sell Zone: Near {nifty_levels['IMMEDIATE_RESISTANCE']:,}")
            plan.append(f"│ • Range: {nifty_levels['CRITICAL_SUPPORT']:,}-{nifty_levels['CRITICAL_RESISTANCE']:,}")
        
        plan.append(f"└{'─' * 45}┘")
        plan.append("")
    
    # BANKNIFTY STRATEGY
    if banknifty_gap and banknifty_levels:
        plan.append(f"<b>🏦 BANKNIFTY STRATEGY:</b>")
        plan.append(f"┌{'─' * 45}┐")
        
        # BankNifty specific logic
        current_bn = banknifty_levels['CURRENT']
        sup1_bn = banknifty_levels['IMMEDIATE_SUPPORT']
        sup2_bn = banknifty_levels['CRITICAL_SUPPORT']
        res1_bn = banknifty_levels['IMMEDIATE_RESISTANCE']
        res2_bn = banknifty_levels['CRITICAL_RESISTANCE']
        
        # Determine BankNifty tone
        if current_bn > sup1_bn + 200:
            tone = "CONSTRUCTIVE"
            tone_color = "🟢"
            condition = f"Above {sup1_bn:,}"
        elif current_bn > sup2_bn:
            tone = "CAUTIOUS"
            tone_color = "🟡"
            condition = f"{sup2_bn:,}-{sup1_bn:,} zone"
        else:
            tone = "WEAK"
            tone_color = "🔴"
            condition = f"Below {sup2_bn:,}"
        
        plan.append(f"│ {tone_color} <b>{tone}:</b> {condition:<35} │")
        plan.append(f"│ {banknifty_gap['COLOR']} <b>{banknifty_gap['GAP_TYPE']}</b>")
        plan.append(f"│ • Expected Gap: <b>{banknifty_gap['GAP_POINTS']:+.0f}</b> pts")
        plan.append(f"│ • Opening Range: <b>{banknifty_gap['MIN_RANGE']}-{banknifty_gap['MAX_RANGE']}</b> pts")
        plan.append(f"│ {' ' * 45}│")
        plan.append(f"│ 📊 <b>Institutional Levels:</b>")
        plan.append(f"│   • Support-1 (Cushion): {sup1_bn:,}")
        plan.append(f"│   • Support-2 (Critical): {sup2_bn:,}")
        plan.append(f"│   • Resistance-1: {res1_bn:,}")
        plan.append(f"│   • Resistance-2: {res2_bn:,}")
        plan.append(f"│ {' ' * 45}│")
        
        # Trading plan based on gap
        if banknifty_gap['GAP_PCT'] > 0:
            plan.append(f"│ 💡 <b>Gap Up Plan:</b>")
            plan.append(f"│   • Wait for pullback to {sup1_bn:,}")
            plan.append(f"│   • Buy with SL: {sup2_bn:,}")
            plan.append(f"│   • Target 1: {res1_bn:,} (50% Exit)")
            plan.append(f"│   • Target 2: {res2_bn:,} (If breaks {res1_bn:,})")
        else:
            plan.append(f"│ 💡 <b>Gap Down Plan:</b>")
            plan.append(f"│   • Sell rallies to {res1_bn:,}")
            plan.append(f"│   • SL: Above {res2_bn:,}")
            plan.append(f"│   • Target 1: {sup1_bn:,}")
            plan.append(f"│   • Target 2: {sup2_bn:,}")
        
        plan.append(f"└{'─' * 45}┘")
        plan.append("")
    
    # TIME-BASED GUIDANCE
    plan.append("<b>⏰ INSTITUTIONAL TIMING:</b>")
    plan.append("• <b>09:15-09:30:</b> Avoid entries - Market finding equilibrium")
    plan.append("• <b>09:30-10:30:</b> Optimal entry window - Institutional participation")
    plan.append("• <b>10:30-14:00:</b> Monitor for trend confirmation")
    plan.append("• <b>14:00-15:00:</b> Square off intraday - Reduce overnight risk")
    plan.append("• <b>15:00-15:30:</b> Only carry high-conviction positions")
    plan.append("")
    
    # RISK PARAMETERS
    plan.append("<b>⚠️ INSTITUTIONAL RISK PARAMETERS:</b>")
    plan.append("• Position Size: 1-3% of capital per trade")
    plan.append("• Risk-Reward: Minimum 1:3 ratio")
    plan.append("• Max Daily Drawdown: 2%")
    plan.append("• Consecutive Losses: Max 2, then stop trading")
    plan.append("• Portfolio Heat: Max 15% at any time")
    
    return "\n".join(plan)

# 🏛️ **5. GENERATE COMPLETE INSTITUTIONAL REPORT** 🏛️
def generate_institutional_report():
    """Generate complete institutional trading desk report"""
    
    ist_now = get_ist_time()
    report = []
    
    # HEADER
    report.append(f"<b>🏦 INSTITUTIONAL TRADING DESK - PRE-MARKET INTELLIGENCE</b>")
    report.append(f"<b>📅</b> {ist_now.strftime('%d %b %Y, %A')} | <b>⏰</b> {ist_now.strftime('%H:%M')} IST")
    report.append("<b>═══════════════════════════════════════════════</b>")
    report.append("")
    
    # 1. NIFTY GAP ANALYSIS
    nifty_gap = institutional_gap_analysis("NIFTY")
    if nifty_gap:
        report.append(f"<b>📈 NIFTY 50 OPENING PROJECTION:</b>")
        report.append(f"┌{'─' * 45}┐")
        report.append(f"│ {nifty_gap['COLOR']} <b>{nifty_gap['GAP_TYPE']}</b>")
        report.append(f"│ • Prev Close: <code>{nifty_gap['PREV_CLOSE']:,}</code>")
        report.append(f"│ • Expected Open: <code>{nifty_gap['EXPECTED_OPEN']:,}</code>")
        report.append(f"│ • Gap: <code>{nifty_gap['GAP_POINTS']:+,.0f} pts ({nifty_gap['GAP_PCT']:+.2f}%)</code>")
        report.append(f"│ • Opening Range: <code>{nifty_gap['MIN_RANGE']}-{nifty_gap['MAX_RANGE']} pts</code>")
        report.append(f"│ • Sentiment: <code>{nifty_gap['SENTIMENT']}</code>")
        report.append(f"│ • VIX: <code>{nifty_gap['VIX']}</code>")
        report.append(f"└{'─' * 45}┘")
        report.append("")
    
    # 2. BANKNIFTY GAP ANALYSIS
    banknifty_gap = institutional_gap_analysis("BANKNIFTY")
    if banknifty_gap:
        report.append(f"<b>🏦 BANKNIFTY OPENING PROJECTION:</b>")
        report.append(f"┌{'─' * 45}┐")
        report.append(f"│ {banknifty_gap['COLOR']} <b>{banknifty_gap['GAP_TYPE']}</b>")
        report.append(f"│ • Prev Close: <code>{banknifty_gap['PREV_CLOSE']:,}</code>")
        report.append(f"│ • Expected Open: <code>{banknifty_gap['EXPECTED_OPEN']:,}</code>")
        report.append(f"│ • Gap: <code>{banknifty_gap['GAP_POINTS']:+,.0f} pts ({banknifty_gap['GAP_PCT']:+.2f}%)</code>")
        report.append(f"│ • Opening Range: <code>{banknifty_gap['MIN_RANGE']}-{banknifty_gap['MAX_RANGE']} pts</code>")
        report.append(f"│ • Sentiment: <code>{banknifty_gap['SENTIMENT']}</code>")
        report.append(f"│ • Prev Day Range: <code>{banknifty_gap['PREV_RANGE']} pts</code>")
        report.append(f"└{'─' * 45}┘")
        report.append("")
    
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
                report.append(f"• {future.replace('_', ' ')}: {data['COLOR']}<code>{data['CHANGE']:+.2f}%</code> @ {data['PRICE']}")
        
        report.append("")
    
    # 4. NIFTY INSTITUTIONAL LEVELS
    nifty_levels = calculate_institutional_levels("NIFTY")
    if nifty_levels:
        report.append(f"<b>📊 NIFTY INSTITUTIONAL LEVELS:</b>")
        report.append(f"┌{'─' * 45}┐")
        report.append(f"│ Current: <code>{nifty_levels['CURRENT']:,}</code> {nifty_levels['BIAS_COLOR']} {nifty_levels['BIAS']}")
        report.append(f"│ MA20: <code>{nifty_levels['MA20']:,}</code> | MA50: <code>{nifty_levels['MA50']:,}</code>")
        report.append(f"│ Volume Ratio: <code>{nifty_levels['VOLUME_RATIO']:.1f}x</code> | Pivot: <code>{nifty_levels['PIVOT']:,}</code>")
        report.append(f"├{'─' * 45}┤")
        report.append(f"│ 🎯 <b>TRADING ZONES:</b>")
        report.append(f"│   • Support-1: <code>{nifty_levels['IMMEDIATE_SUPPORT']:,}</code> (Buy Zone)")
        report.append(f"│   • Support-2: <code>{nifty_levels['CRITICAL_SUPPORT']:,}</code> (SL Trigger)")
        report.append(f"│   • Resistance-1: <code>{nifty_levels['IMMEDIATE_RESISTANCE']:,}</code> (Take Profit)")
        report.append(f"│   • Resistance-2: <code>{nifty_levels['CRITICAL_RESISTANCE']:,}</code> (Breakout)")
        report.append(f"└{'─' * 45}┘")
        report.append("")
    
    # 5. BANKNIFTY INSTITUTIONAL LEVELS
    banknifty_levels = calculate_institutional_levels("BANKNIFTY")
    if banknifty_levels:
        report.append(f"<b>🏦 BANKNIFTY INSTITUTIONAL LEVELS:</b>")
        report.append(f"┌{'─' * 45}┐")
        report.append(f"│ Current: <code>{banknifty_levels['CURRENT']:,.0f}</code> {banknifty_levels['BIAS_COLOR']} {banknifty_levels['BIAS']}")
        report.append(f"│ MA20: <code>{banknifty_levels['MA20']:,.0f}</code> | MA50: <code>{banknifty_levels['MA50']:,.0f}</code>")
        report.append(f"│ Prev High: <code>{banknifty_levels['PREV_HIGH']:,.0f}</code> | Prev Low: <code>{banknifty_levels['PREV_LOW']:,.0f}</code>")
        report.append(f"├{'─' * 45}┤")
        report.append(f"│ 🎯 <b>CRITICAL ZONES:</b>")
        
        # Determine BankNifty condition
        current = banknifty_levels['CURRENT']
        sup1 = banknifty_levels['IMMEDIATE_SUPPORT']
        sup2 = banknifty_levels['CRITICAL_SUPPORT']
        
        if current > sup1:
            condition = f"CONSTRUCTIVE ABOVE {sup1:,}"
            condition_color = "🟢"
        elif current > sup2:
            condition = f"CAUTIOUS IN {sup2:,}-{sup1:,}"
            condition_color = "🟡"
        else:
            condition = f"WEAK BELOW {sup2:,}"
            condition_color = "🔴"
        
        report.append(f"│ {condition_color} <b>{condition}</b>")
        report.append(f"│   • Support-1 (Cushion): <code>{sup1:,}</code>")
        report.append(f"│   • Support-2 (Critical): <code>{sup2:,}</code>")
        report.append(f"│   • Resistance-1: <code>{banknifty_levels['IMMEDIATE_RESISTANCE']:,}</code>")
        report.append(f"│   • Resistance-2: <code>{banknifty_levels['CRITICAL_RESISTANCE']:,}</code>")
        report.append(f"└{'─' * 45}┘")
        report.append("")
    
    # 6. TRADING PLAN
    trading_plan = generate_institutional_trading_plan()
    report.append(trading_plan)
    
    # 7. CONFIDENCE METER
    report.append("<b>═══════════════════════════════════════════════</b>")
    report.append("")
    
    # Calculate overall confidence
    confidence = 70  # Base
    
    if nifty_gap and banknifty_gap:
        # Both gaps agree = higher confidence
        if (nifty_gap['GAP_PCT'] > 0 and banknifty_gap['GAP_PCT'] > 0) or \
           (nifty_gap['GAP_PCT'] < 0 and banknifty_gap['GAP_PCT'] < 0):
            confidence += 10
    
    if global_data and global_data['SENTIMENT'] in ["STRONGLY POSITIVE", "STRONGLY NEGATIVE"]:
        confidence += 10
    
    if nifty_levels and nifty_levels['BIAS_SCORE'] > 70:
        confidence += 5
    
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
    report.append("<b>🏛️ Generated by: Institutional Trading Desk v3.0</b>")
    
    return "\n".join(report)

# 🏛️ **6. MAIN EXECUTION** 🏛️
def main():
    """Main function for GitHub Actions"""
    print("🏦 Institutional Trading Desk v3.0 - Starting Analysis...")
    
    ist_now = get_ist_time()
    print(f"⏰ Time: {ist_now.strftime('%d %b %Y, %H:%M:%S IST')}")
    
    # Send startup notification
    startup_msg = f"🏦 <b>Institutional Trading Desk Activated</b>\n"
    startup_msg += f"📅 {ist_now.strftime('%d %b %Y')} | ⏰ {ist_now.strftime('%H:%M IST')}\n"
    startup_msg += f"📊 Generating pre-market intelligence..."
    send_telegram(startup_msg)
    
    # Generate and send report
    try:
        report = generate_institutional_report()
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
            
    except Exception as e:
        error_msg = f"❌ <b>Institutional Analysis Failed</b>\n"
        error_msg += f"Error: {str(e)[:100]}\n"
        error_msg += f"Time: {ist_now.strftime('%H:%M IST')}"
        send_telegram(error_msg)
        print(f"❌ Error: {e}")

# 🏛️ **RUN THE INSTITUTIONAL DESK** 🏛️
if __name__ == "__main__":
    main()
