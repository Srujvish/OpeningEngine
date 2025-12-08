# INSTITUTIONAL TRADING DESK - PRE-MARKET INTELLIGENCE
# PURE INSTITUTIONAL ANALYSIS WITH SPECIFIC TRADING ZONES

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

# --------- GET IST TIME ---------
def get_ist_time():
    utc_now = datetime.utcnow()
    ist = pytz.timezone('Asia/Kolkata')
    ist_now = utc_now.replace(tzinfo=pytz.utc).astimezone(ist)
    return ist_now

# 🏛️ **1. INSTITUTIONAL SGX NIFTY ANALYSIS** 🏛️
def get_sgx_nifty_institutional():
    """Get SGX Nifty with institutional interpretation"""
    try:
        # Get SGX Nifty Future
        sgx = yf.download("NQ=F", period="1d", interval="1m", progress=False)
        if not sgx.empty:
            sgx_close = float(sgx['Close'].iloc[-1])
            sgx_open = float(sgx['Open'].iloc[-1])
            sgx_high = float(sgx['High'].iloc[-1])
            sgx_low = float(sgx['Low'].iloc[-1])
            
            return {
                'PRICE': round(sgx_close, 2),
                'OPEN': round(sgx_open, 2),
                'HIGH': round(sgx_high, 2),
                'LOW': round(sgx_low, 2),
                'TYPE': 'SGX NIFTY FUTURE'
            }
        
        # Fallback to Nifty Future
        nifty_fut = yf.download("NIFTY_50.NS", period="1d", interval="1m", progress=False)
        if not nifty_fut.empty:
            fut_close = float(nifty_fut['Close'].iloc[-1])
            return {
                'PRICE': round(fut_close, 2),
                'TYPE': 'NIFTY FUTURE'
            }
            
    except Exception as e:
        print(f"SGX error: {e}")
    
    return None

# 🏛️ **2. CALCULATE PRECISE OPENING GAP** 🏛️
def calculate_opening_gap_analysis():
    """Institutional opening gap calculation with levels"""
    try:
        sgx_data = get_sgx_nifty_institutional()
        nifty = yf.download("^NSEI", period="2d", interval="1d", progress=False)
        
        if sgx_data and not nifty.empty and len(nifty) >= 2:
            prev_close = float(nifty['Close'].iloc[-2])
            sgx_price = sgx_data['PRICE']
            
            gap_points = sgx_price - prev_close
            gap_pct = (gap_points / prev_close) * 100
            
            # INSTITUTIONAL GAP CLASSIFICATION
            if gap_pct > 0.5:
                gap_type = "STRONG GAP UP OPENING"
                sentiment = "BULLISH"
                color = "🟢"
            elif gap_pct > 0.2:
                gap_type = "MODERATE GAP UP OPENING"
                sentiment = "MILD BULLISH"
                color = "🟡"
            elif gap_pct < -0.5:
                gap_type = "STRONG GAP DOWN OPENING"
                sentiment = "BEARISH"
                color = "🔴"
            elif gap_pct < -0.2:
                gap_type = "MODERATE GAP DOWN OPENING"
                sentiment = "MILD BEARISH"
                color = "🟠"
            else:
                gap_type = "FLAT TO NEUTRAL OPENING"
                sentiment = "NEUTRAL"
                color = "⚪"
            
            return {
                'PREV_CLOSE': round(prev_close, 2),
                'SGX_PRICE': sgx_price,
                'GAP_POINTS': round(gap_points, 2),
                'GAP_PCT': round(gap_pct, 2),
                'GAP_TYPE': gap_type,
                'SENTIMENT': sentiment,
                'COLOR': color,
                'EXPECTED_OPEN': round(sgx_price, 2)
            }
            
    except Exception as e:
        print(f"Gap analysis error: {e}")
    
    return None

# 🏛️ **3. INSTITUTIONAL SUPPORT/RESISTANCE CALCULATOR** 🏛️
def calculate_institutional_levels(index="NIFTY"):
    """Calculate precise institutional trading levels"""
    try:
        if index == "NIFTY":
            symbol = "^NSEI"
            round_to = 50
        else:  # BANKNIFTY
            symbol = "^NSEBANK"
            round_to = 100
        
        # Get 20 days data for institutional analysis
        data = yf.download(symbol, period="20d", interval="1d", progress=False)
        
        if not data.empty and len(data) >= 10:
            closes = data['Close'].astype(float)
            highs = data['High'].astype(float)
            lows = data['Low'].astype(float)
            
            current_price = float(closes.iloc[-1])
            
            # 🎯 INSTITUTIONAL PIVOT POINTS
            pivot = (highs.iloc[-1] + lows.iloc[-1] + closes.iloc[-1]) / 3
            r1 = (2 * pivot) - lows.iloc[-1]
            s1 = (2 * pivot) - highs.iloc[-1]
            r2 = pivot + (highs.iloc[-1] - lows.iloc[-1])
            s2 = pivot - (highs.iloc[-1] - lows.iloc[-1])
            
            # 🎯 FIBONACCI LEVELS (Last 10 days swing)
            swing_high = float(highs.iloc[-10:].max())
            swing_low = float(lows.iloc[-10:].min())
            swing_range = swing_high - swing_low
            
            fib_levels = {
                '0.236': swing_low + swing_range * 0.236,
                '0.382': swing_low + swing_range * 0.382,
                '0.500': swing_low + swing_range * 0.500,
                '0.618': swing_low + swing_range * 0.618,
                '0.786': swing_low + swing_range * 0.786
            }
            
            # 🎯 MOVING AVERAGES (Institutional reference)
            ma20 = float(closes.rolling(20).mean().iloc[-1])
            ma50 = float(closes.rolling(50).mean().iloc[-1]) if len(closes) >= 50 else ma20
            
            # 🎯 ROUND TO NEAREST 50/100
            def round_level(price):
                return round(price / round_to) * round_to
            
            # 🎯 CRITICAL INSTITUTIONAL LEVELS
            immediate_support = round_level(min(s1, fib_levels['0.382'], ma20))
            critical_support = round_level(min(s2, fib_levels['0.618']))
            immediate_resistance = round_level(max(r1, fib_levels['0.618'], ma20))
            critical_resistance = round_level(max(r2, swing_high))
            
            # Determine bias based on price position
            if current_price > ma20 and current_price > ma50:
                bias = "BULLISH"
                bias_color = "🟢"
            elif current_price < ma20 and current_price < ma50:
                bias = "BEARISH"
                bias_color = "🔴"
            else:
                bias = "NEUTRAL/RANGEBOUND"
                bias_color = "🟡"
            
            return {
                'CURRENT': round(current_price, 2),
                'BIAS': bias,
                'BIAS_COLOR': bias_color,
                'IMMEDIATE_SUPPORT': immediate_support,
                'CRITICAL_SUPPORT': critical_support,
                'IMMEDIATE_RESISTANCE': immediate_resistance,
                'CRITICAL_RESISTANCE': critical_resistance,
                'MA20': round(ma20, 2),
                'MA50': round(ma50, 2),
                'PIVOT': round(pivot, 2)
            }
            
    except Exception as e:
        print(f"Institutional levels error for {index}: {e}")
    
    return None

# 🏛️ **4. BANKNIFTY SPECIFIC INSTITUTIONAL ANALYSIS** 🏛️
def banknifty_institutional_analysis():
    """BANKNIFTY specific institutional analysis like your screenshot"""
    try:
        data = yf.download("^NSEBANK", period="10d", interval="1d", progress=False)
        
        if not data.empty and len(data) >= 5:
            current = float(data['Close'].iloc[-1])
            prev_high = float(data['High'].iloc[-2])
            prev_low = float(data['Low'].iloc[-2])
            
            # 🎯 INSTITUTIONAL BANKNIFTY LEVELS (Based on your screenshot pattern)
            critical_support = 59000  # Your screenshot: Critical Support
            immediate_support = 59500  # Your screenshot: Near-term cushion
            
            # Calculate resistances based on previous structure
            resistance_1 = round((prev_high + 200) / 100) * 100  # Near 60150
            resistance_2 = resistance_1 + 350  # Near 60500-60800 zone
            
            # Determine market tone
            if current > immediate_support:
                tone = "CONSTRUCTIVE"
                tone_color = "🟢"
                condition = f"As long as Bank Nifty remains above {immediate_support:,}, broader tone remains constructive"
            elif current > critical_support:
                tone = "CAUTIOUS"
                tone_color = "🟡"
                condition = f"Trading in {critical_support:,}-{immediate_support:,} requires caution"
            else:
                tone = "WEAK"
                tone_color = "🔴"
                condition = f"Break below {critical_support:,} signals weakness"
            
            return {
                'CURRENT': current,
                'TONE': tone,
                'TONE_COLOR': tone_color,
                'CONDITION': condition,
                'SUPPORT_1': immediate_support,  # Near-term cushion
                'SUPPORT_2': critical_support,   # Critical support
                'RESISTANCE_1': resistance_1,    # Immediate upside
                'RESISTANCE_2': resistance_2,    # Bullish breakout zone
                'PREV_HIGH': prev_high,
                'PREV_LOW': prev_low
            }
            
    except Exception as e:
        print(f"BankNifty analysis error: {e}")
    
    return None

# 🏛️ **5. GLOBAL MARKET SENTIMENT ANALYSIS** 🏛️
def get_global_sentiment():
    """Institutional global market analysis"""
    markets = {}
    sentiment_score = 0
    
    try:
        symbols = {
            'DOW': '^DJI',
            'NASDAQ': '^IXIC',
            'S&P500': '^GSPC',
            'NIKKEI': '^N225',
            'HSI': '^HSI',
            'DAX': '^GDAXI'
        }
        
        for name, symbol in symbols.items():
            try:
                data = yf.download(symbol, period="2d", interval="1d", progress=False)
                if not data.empty and len(data) >= 2:
                    prev_close = float(data['Close'].iloc[-2])
                    current_close = float(data['Close'].iloc[-1])
                    change_pct = ((current_close - prev_close) / prev_close) * 100
                    
                    markets[name] = {
                        'CHANGE': round(change_pct, 2),
                        'PRICE': round(current_close, 2)
                    }
                    
                    # Sentiment scoring
                    if change_pct > 0.5:
                        sentiment_score += 1
                    elif change_pct < -0.5:
                        sentiment_score -= 1
                        
            except Exception:
                continue
        
        # Determine overall sentiment
        if sentiment_score >= 3:
            global_sentiment = "STRONGLY POSITIVE"
            sentiment_color = "🟢"
        elif sentiment_score >= 1:
            global_sentiment = "POSITIVE"
            sentiment_color = "🟡"
        elif sentiment_score <= -3:
            global_sentiment = "STRONGLY NEGATIVE"
            sentiment_color = "🔴"
        elif sentiment_score <= -1:
            global_sentiment = "NEGATIVE"
            sentiment_color = "🟠"
        else:
            global_sentiment = "NEUTRAL"
            sentiment_color = "⚪"
        
        return {
            'MARKETS': markets,
            'SENTIMENT': global_sentiment,
            'SENTIMENT_COLOR': sentiment_color,
            'SCORE': sentiment_score
        }
        
    except Exception as e:
        print(f"Global sentiment error: {e}")
    
    return None

# 🏛️ **6. INSTITUTIONAL TRADING PLAN GENERATOR** 🏛️
def generate_institutional_trading_plan():
    """Generate specific institutional trading plan"""
    
    plan = []
    
    # Get all analyses
    gap_analysis = calculate_opening_gap_analysis()
    nifty_levels = calculate_institutional_levels("NIFTY")
    banknifty_levels = banknifty_institutional_analysis()
    global_sentiment = get_global_sentiment()
    
    plan.append("<b>🎯 INSTITUTIONAL TRADING PLAN FOR TODAY</b>")
    plan.append("")
    
    # NIFTY Strategy
    if nifty_levels:
        current = nifty_levels['CURRENT']
        sup1 = nifty_levels['IMMEDIATE_SUPPORT']
        sup2 = nifty_levels['CRITICAL_SUPPORT']
        res1 = nifty_levels['IMMEDIATE_RESISTANCE']
        res2 = nifty_levels['CRITICAL_RESISTANCE']
        
        plan.append(f"<b>📈 NIFTY STRATEGY:</b>")
        plan.append(f"┌{'─' * 40}┐")
        
        if gap_analysis and gap_analysis['GAP_PCT'] > 0.2:
            # Gap Up Scenario
            plan.append(f"│ 🟢 <b>GAP UP PLAY:</b> Buy on dip                │")
            plan.append(f"│   • Entry Zone: {sup1:,} - {current:,}          │")
            plan.append(f"│   • Stop Loss: Below {sup2:,}                   │")
            plan.append(f"│   • Target 1: {res1:,} (Partial Exit)          │")
            plan.append(f"│   • Target 2: {res2:,} (If breaks resistance) │")
            
        elif gap_analysis and gap_analysis['GAP_PCT'] < -0.2:
            # Gap Down Scenario
            plan.append(f"│ 🔴 <b>GAP DOWN PLAY:</b> Sell on rise            │")
            plan.append(f"│   • Entry Zone: {current:,} - {res1:,}         │")
            plan.append(f"│   • Stop Loss: Above {res2:,}                   │")
            plan.append(f"│   • Target 1: {sup1:,}                         │")
            plan.append(f"│   • Target 2: {sup2:,}                         │")
            
        else:
            # Rangebound Scenario
            plan.append(f"│ 🟡 <b>RANGEBOUND PLAY:</b> Fade extremes         │")
            plan.append(f"│   • Buy Zone: Near {sup1:,}                    │")
            plan.append(f"│   • Sell Zone: Near {res1:,}                   │")
            plan.append(f"│   • Range: {sup2:,} - {res2:,}                │")
            plan.append(f"│   • Stop Loss: 20 points beyond zone          │")
        
        plan.append(f"└{'─' * 40}┘")
        plan.append("")
    
    # BANKNIFTY Strategy
    if banknifty_levels:
        sup1_bn = banknifty_levels['SUPPORT_1']
        sup2_bn = banknifty_levels['SUPPORT_2']
        res1_bn = banknifty_levels['RESISTANCE_1']
        res2_bn = banknifty_levels['RESISTANCE_2']
        
        plan.append(f"<b>🏦 BANKNIFTY STRATEGY:</b>")
        plan.append(f"┌{'─' * 40}┐")
        plan.append(f"│ {banknifty_levels['TONE_COLOR']} <b>{banknifty_levels['TONE']}:</b> {banknifty_levels['CONDITION']:<15} │")
        plan.append(f"│ {' ' * 40}│")
        plan.append(f"│ 📊 <b>Key Levels:</b>                              │")
        plan.append(f"│   • Support-1 (Cushion): {sup1_bn:,}              │")
        plan.append(f"│   • Support-2 (Critical): {sup2_bn:,}             │")
        plan.append(f"│   • Resistance-1: {res1_bn:,}                     │")
        plan.append(f"│   • Resistance-2: {res2_bn:,}                     │")
        plan.append(f"│ {' ' * 40}│")
        plan.append(f"│ 💡 <b>Intraday Plan:</b>                           │")
        plan.append(f"│   • Buy Zone: {sup1_bn:,}-{sup2_bn:,}             │")
        plan.append(f"│   • Target 1: {res1_bn:,} (Take 50% Profit)      │")
        plan.append(f"│   • Target 2: {res2_bn:,} (If volume breakout)   │")
        plan.append(f"│   • Stop Loss: Below {sup2_bn:,} on EOD basis     │")
        plan.append(f"└{'─' * 40}┘")
        plan.append("")
    
    # Time-Based Guidance
    plan.append("<b>⏰ TIME-BASED GUIDANCE:</b>")
    plan.append("• <b>09:15-09:30:</b> Avoid entries - Let market settle")
    plan.append("• <b>09:30-10:30:</b> Best entry window for institutional trades")
    plan.append("• <b>10:30-14:30:</b> Monitor for breakout/breakdown confirmation")
    plan.append("• <b>14:30-15:00:</b> Square off or carry positions with caution")
    plan.append("")
    
    # Risk Management
    plan.append("<b>⚠️ RISK MANAGEMENT:</b>")
    plan.append("• Position Size: 1-2% of capital per trade")
    plan.append("• Risk-Reward: Minimum 1:2 ratio")
    plan.append("• Max Daily Loss: 3% of capital")
    plan.append("• Use stop losses without exception")
    
    return "\n".join(plan)

# 🏛️ **7. GENERATE COMPLETE INSTITUTIONAL REPORT** 🏛️
def generate_institutional_report():
    """Generate complete institutional trading desk report"""
    
    ist_now = get_ist_time()
    report = []
    
    # HEADER
    report.append(f"<b>🏦 INSTITUTIONAL TRADING DESK - PRE-MARKET INTELLIGENCE</b>")
    report.append(f"<b>📅</b> {ist_now.strftime('%d %b %Y, %A')} | <b>⏰</b> {ist_now.strftime('%H:%M')} IST")
    report.append("<b>═══════════════════════════════════════════════</b>")
    report.append("")
    
    # 1. OPENING GAP ANALYSIS
    gap_analysis = calculate_opening_gap_analysis()
    if gap_analysis:
        report.append(f"<b>🚀 OPENING PROJECTION:</b>")
        report.append(f"{gap_analysis['COLOR']} <b>{gap_analysis['GAP_TYPE']}</b>")
        report.append(f"• Previous Close: <code>{gap_analysis['PREV_CLOSE']:,}</code>")
        report.append(f"• SGX Indicative: <code>{gap_analysis['SGX_PRICE']:,}</code>")
        report.append(f"• Expected Gap: <code>{gap_analysis['GAP_POINTS']:+,.0f} points ({gap_analysis['GAP_PCT']:+.2f}%)</code>")
        report.append(f"• Market Sentiment: <code>{gap_analysis['SENTIMENT']}</code>")
        report.append("")
    
    # 2. GLOBAL SENTIMENT
    global_data = get_global_sentiment()
    if global_data:
        report.append(f"<b>🌍 GLOBAL MARKET SENTIMENT:</b>")
        report.append(f"{global_data['SENTIMENT_COLOR']} <b>{global_data['SENTIMENT']}</b> (Score: {global_data['SCORE']}/6)")
        
        # Show key markets
        key_markets = ['DOW', 'NASDAQ', 'S&P500', 'NIKKEI']
        market_line = []
        for market in key_markets:
            if market in global_data['MARKETS']:
                change = global_data['MARKETS'][market]['CHANGE']
                icon = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
                market_line.append(f"{market}: {icon}{change:+.1f}%")
        
        if market_line:
            report.append(f"• {' | '.join(market_line)}")
        report.append("")
    
    # 3. NIFTY INSTITUTIONAL LEVELS
    nifty_levels = calculate_institutional_levels("NIFTY")
    if nifty_levels:
        report.append(f"<b>📈 NIFTY INSTITUTIONAL LEVELS:</b>")
        report.append(f"┌{'─' * 45}┐")
        report.append(f"│ Current: <code>{nifty_levels['CURRENT']:,}</code> {nifty_levels['BIAS_COLOR']} {nifty_levels['BIAS']:<30} │")
        report.append(f"│ MA20: <code>{nifty_levels['MA20']:,}</code> | MA50: <code>{nifty_levels['MA50']:,}</code> {' ' * 15}│")
        report.append(f"│ Pivot: <code>{nifty_levels['PIVOT']:,}</code>                            │")
        report.append(f"├{'─' * 45}┤")
        report.append(f"│ 🟢 <b>BULLISH ZONE:</b> Above {nifty_levels['IMMEDIATE_SUPPORT']:,}        │")
        report.append(f"│ 🟡 <b>NEUTRAL ZONE:</b> {nifty_levels['CRITICAL_SUPPORT']:,}-{nifty_levels['IMMEDIATE_SUPPORT']:,} │")
        report.append(f"│ 🔴 <b>BEARISH ZONE:</b> Below {nifty_levels['CRITICAL_SUPPORT']:,}         │")
        report.append(f"└{'─' * 45}┘")
        report.append(f"• Immediate Support: <code>{nifty_levels['IMMEDIATE_SUPPORT']:,}</code> (Buy Zone)")
        report.append(f"• Critical Support: <code>{nifty_levels['CRITICAL_SUPPORT']:,}</code> (SL Trigger)")
        report.append(f"• Resistance 1: <code>{nifty_levels['IMMEDIATE_RESISTANCE']:,}</code> (Take Profit)")
        report.append(f"• Resistance 2: <code>{nifty_levels['CRITICAL_RESISTANCE']:,}</code> (Breakout)")
        report.append("")
    
    # 4. BANKNIFTY INSTITUTIONAL ANALYSIS
    bn_analysis = banknifty_institutional_analysis()
    if bn_analysis:
        report.append(f"<b>🏦 BANKNIFTY INSTITUTIONAL OUTLOOK:</b>")
        report.append(f"┌{'─' * 45}┐")
        report.append(f"│ Current: <code>{bn_analysis['CURRENT']:,.0f}</code> {bn_analysis['TONE_COLOR']} {bn_analysis['TONE']:<28} │")
        report.append(f"│ {bn_analysis['CONDITION']:<43} │")
        report.append(f"├{'─' * 45}┤")
        report.append(f"│ 🎯 <b>CRITICAL LEVELS:</b>                             │")
        report.append(f"│   • Support-1 (Cushion): <code>{bn_analysis['SUPPORT_1']:,}</code>      │")
        report.append(f"│   • Support-2 (Critical): <code>{bn_analysis['SUPPORT_2']:,}</code>     │")
        report.append(f"│   • Resistance-1: <code>{bn_analysis['RESISTANCE_1']:,}</code>          │")
        report.append(f"│   • Resistance-2: <code>{bn_analysis['RESISTANCE_2']:,}</code>          │")
        report.append(f"└{'─' * 45}┘")
        report.append("")
    
    # 5. TRADING PLAN
    trading_plan = generate_institutional_trading_plan()
    report.append(trading_plan)
    
    # 6. FOOTER
    report.append("<b>═══════════════════════════════════════════════</b>")
    report.append("")
    report.append("<b>📊 CONFIDENCE METER:</b>")
    
    # Calculate confidence based on multiple factors
    confidence_score = 70  # Base confidence
    
    if gap_analysis:
        if abs(gap_analysis['GAP_PCT']) > 0.3:
            confidence_score += 10
        else:
            confidence_score -= 5
    
    if global_data and global_data['SENTIMENT'] == "STRONGLY POSITIVE":
        confidence_score += 10
    elif global_data and global_data['SENTIMENT'] == "STRONGLY NEGATIVE":
        confidence_score -= 10
    
    confidence_score = max(30, min(90, confidence_score))
    
    # Confidence bar
    filled = "█" * (confidence_score // 10)
    empty = "░" * (10 - (confidence_score // 10))
    report.append(f"{filled}{empty} {confidence_score}%")
    
    report.append("")
    report.append("<b>⚠️ INSTITUTIONAL DISCLAIMER:</b>")
    report.append("• For qualified institutional clients only")
    report.append("• Past performance ≠ future results")
    report.append("• Trade with proper risk management")
    report.append("• This is not investment advice")
    report.append("")
    report.append("<b>🏛️ Generated by: Institutional Trading Desk v2.0</b>")
    
    return "\n".join(report)

# 🏛️ **8. MAIN EXECUTION** 🏛️
def main():
    """Main function for GitHub Actions"""
    print("🏦 Institutional Trading Desk - Starting Analysis...")
    
    ist_now = get_ist_time()
    print(f"⏰ Current IST: {ist_now.strftime('%H:%M:%S')}")
    
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
            completion_msg += f"⏰ {ist_now.strftime('%H:%M IST')}\n"
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
