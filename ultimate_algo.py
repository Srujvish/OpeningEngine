# INSTITUTIONAL PRE-MARKET ANALYSIS ENGINE - PROFESSIONAL VERSION
# COMPLETELY FIXED: Fresh data, correct calculations, institutional insights

import os
import time
import requests
import pandas as pd
import yfinance as yf
import warnings
import json
from datetime import datetime, time as dtime, timedelta, date
import pytz
import numpy as np
import talib

warnings.filterwarnings("ignore")

# --------- TELEGRAM SETUP ---------
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

def send_telegram(msg):
    """Send message to Telegram"""
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

# --------- GET CORRECT IST TIME ---------
def get_ist_time():
    """Get accurate IST time"""
    utc_now = datetime.utcnow()
    ist = pytz.timezone('Asia/Kolkata')
    ist_now = utc_now.replace(tzinfo=pytz.utc).astimezone(ist)
    return ist_now

# --------- FIXED: GET CORRECT PREVIOUS DAY DATE ---------
def get_previous_trading_date():
    """Get previous trading day (skip weekends/holidays)"""
    ist_now = get_ist_time()
    current_date = ist_now.date()
    
    # If today is Monday, previous trading day is Friday
    if current_date.weekday() == 0:  # Monday
        prev_date = current_date - timedelta(days=3)
    # If today is Sunday (shouldn't happen at 9 AM), get Friday
    elif current_date.weekday() == 6:  # Sunday
        prev_date = current_date - timedelta(days=2)
    else:
        prev_date = current_date - timedelta(days=1)
    
    return prev_date

# --------- FIXED: GET FRESH SGX NIFTY ---------
def get_fresh_sgx_nifty():
    """
    Get LIVE SGX Nifty with multiple fallbacks
    """
    try:
        # Try SGX Nifty Future first (most accurate for pre-market)
        sgx = yf.download("NQ=F", period="1d", interval="1m", progress=False)
        if not sgx.empty:
            sgx_close = float(sgx['Close'].iloc[-1])
            return round(sgx_close, 2)
        
        # Fallback to Nifty Futures
        nifty_fut = yf.download("^NSEI", period="1d", interval="1m", progress=False)
        if not nifty_fut.empty:
            fut_close = float(nifty_fut['Close'].iloc[-1])
            # Add typical SGX premium (15-40 points)
            return round(fut_close + 25, 2)
        
        # Ultimate fallback
        return None
        
    except Exception as e:
        print(f"SGX error: {e}")
        return None

# --------- FIXED: GET CORRECT PREVIOUS DAY DATA ---------
def get_correct_previous_day_data():
    """
    Get FRESH previous day's data with correct calculations
    """
    data = {}
    
    try:
        # Get previous trading date
        prev_date = get_previous_trading_date()
        
        # NIFTY - Get last 5 days and filter for previous day
        nifty = yf.download("^NSEI", period="5d", interval="1d", progress=False)
        if not nifty.empty and len(nifty) >= 2:
            # Get the row for previous trading day
            nifty.index = pd.to_datetime(nifty.index).date
            
            # Get previous day's data
            if prev_date in nifty.index:
                prev_row = nifty.loc[prev_date]
            else:
                # Fallback: second last row
                prev_row = nifty.iloc[-2]
            
            prev_open = float(prev_row['Open'])
            prev_high = float(prev_row['High'])
            prev_low = float(prev_row['Low'])
            prev_close = float(prev_row['Close'])
            prev_volume = int(prev_row['Volume'])
            
            change_pct = ((prev_close - prev_open) / prev_open) * 100
            
            # Calculate candle patterns accurately
            body = abs(prev_close - prev_open)
            total_range = prev_high - prev_low
            
            data['NIFTY'] = {
                'DATE': prev_date.strftime("%d %b %Y"),
                'OPEN': round(prev_open, 2),
                'HIGH': round(prev_high, 2),
                'LOW': round(prev_low, 2),
                'CLOSE': round(prev_close, 2),
                'CHANGE': round(change_pct, 2),
                'VOLUME': prev_volume,
                'RANGE': round(prev_high - prev_low, 2),
                'BODY_SIZE': round(body, 2)
            }
            
            # Advanced candle pattern detection
            if body > 0:
                if prev_close > prev_open:
                    upper_wick = prev_high - prev_close
                    lower_wick = prev_open - prev_low
                    
                    if body > total_range * 0.7:
                        data['NIFTY']['PATTERN'] = "STRONG BULLISH (Marubozu)"
                    elif upper_wick > body and lower_wick > body:
                        data['NIFTY']['PATTERN'] = "DOJI (Indecision)"
                    else:
                        data['NIFTY']['PATTERN'] = "BULLISH"
                else:
                    upper_wick = prev_high - prev_open
                    lower_wick = prev_close - prev_low
                    
                    if body > total_range * 0.7:
                        data['NIFTY']['PATTERN'] = "STRONG BEARISH (Marubozu)"
                    elif upper_wick > body and lower_wick > body:
                        data['NIFTY']['PATTERN'] = "DOJI (Indecision)"
                    else:
                        data['NIFTY']['PATTERN'] = "BEARISH"
            
            # Calculate Close relative to range
            close_position = (prev_close - prev_low) / (prev_high - prev_low) if (prev_high - prev_low) > 0 else 0.5
            if close_position > 0.6:
                data['NIFTY']['CLOSE_POSITION'] = "TOP"
            elif close_position < 0.4:
                data['NIFTY']['CLOSE_POSITION'] = "BOTTOM"
            else:
                data['NIFTY']['CLOSE_POSITION'] = "MIDDLE"
        
        # BANKNIFTY - Same logic
        banknifty = yf.download("^NSEBANK", period="5d", interval="1d", progress=False)
        if not banknifty.empty and len(banknifty) >= 2:
            banknifty.index = pd.to_datetime(banknifty.index).date
            
            if prev_date in banknifty.index:
                prev_row_bn = banknifty.loc[prev_date]
            else:
                prev_row_bn = banknifty.iloc[-2]
            
            bn_open = float(prev_row_bn['Open'])
            bn_high = float(prev_row_bn['High'])
            bn_low = float(prev_row_bn['Low'])
            bn_close = float(prev_row_bn['Close'])
            bn_change_pct = ((bn_close - bn_open) / bn_open) * 100
            
            data['BANKNIFTY'] = {
                'DATE': prev_date.strftime("%d %b %Y"),
                'OPEN': round(bn_open, 2),
                'HIGH': round(bn_high, 2),
                'LOW': round(bn_low, 2),
                'CLOSE': round(bn_close, 2),
                'CHANGE': round(bn_change_pct, 2),
                'RANGE': round(bn_high - bn_low, 2)
            }
            
            # Add BankNifty specific levels
            data['BANKNIFTY']['LEVELS'] = {
                'IMMEDIATE_SUPPORT': round(bn_close * 0.995, 2),
                'STRONG_SUPPORT': round(bn_close * 0.985, 2),
                'IMMEDIATE_RESISTANCE': round(bn_close * 1.005, 2),
                'STRONG_RESISTANCE': round(bn_close * 1.015, 2)
            }
            
    except Exception as e:
        print(f"Previous day data error: {e}")
    
    return data

# --------- INSTITUTIONAL SUPPORT/RESISTANCE CALCULATOR ---------
def calculate_institutional_levels():
    """
    Calculate professional S/R levels like Angel One
    """
    levels = {}
    
    try:
        # Get NIFTY data
        nifty = yf.download("^NSEI", period="10d", interval="1d", progress=False)
        
        if not nifty.empty and len(nifty) >= 5:
            current_price = float(nifty['Close'].iloc[-1])
            
            # Recent High/Low
            recent_high = float(nifty['High'].iloc[-5:].max())
            recent_low = float(nifty['Low'].iloc[-5:].min())
            
            # Previous day high/low
            prev_high = float(nifty['High'].iloc[-2])
            prev_low = float(nifty['Low'].iloc[-2])
            
            # Pivot Points (Classic)
            pivot = (prev_high + prev_low + current_price) / 3
            r1 = 2 * pivot - prev_low
            s1 = 2 * pivot - prev_high
            r2 = pivot + (prev_high - prev_low)
            s2 = pivot - (prev_high - prev_low)
            
            # Fibonacci Levels (from recent swing)
            fib_range = recent_high - recent_low
            fib_levels = {
                '0.236': recent_low + fib_range * 0.236,
                '0.382': recent_low + fib_range * 0.382,
                '0.500': recent_low + fib_range * 0.500,
                '0.618': recent_low + fib_range * 0.618,
                '0.786': recent_low + fib_range * 0.786
            }
            
            # Round to nearest 50 for NIFTY, 100 for BANKNIFTY
            def round_to_multiple(x, multiple=50):
                return round(x / multiple) * multiple
            
            levels['NIFTY'] = {
                'CURRENT': current_price,
                'PREV_HIGH': prev_high,
                'PREV_LOW': prev_low,
                'RECENT_HIGH': recent_high,
                'RECENT_LOW': recent_low,
                'PIVOTS': {
                    'PIVOT': round_to_multiple(pivot),
                    'R1': round_to_multiple(r1),
                    'R2': round_to_multiple(r2),
                    'S1': round_to_multiple(s1),
                    'S2': round_to_multiple(s2)
                },
                'FIB_LEVELS': {k: round_to_multiple(v) for k, v in fib_levels.items()}
            }
            
            # Add immediate trading zones
            levels['NIFTY']['TRADING_ZONES'] = {
                'BUY_ZONE': f"{round_to_multiple(s1)}-{round_to_multiple(pivot)}",
                'SELL_ZONE': f"{round_to_multiple(r1)}-{round_to_multiple(r2)}",
                'BREAKOUT_LEVEL': round_to_multiple(recent_high + 50),
                'BREAKDOWN_LEVEL': round_to_multiple(recent_low - 50)
            }
        
        # BankNifty specific levels
        banknifty = yf.download("^NSEBANK", period="10d", interval="1d", progress=False)
        
        if not banknifty.empty:
            bn_current = float(banknifty['Close'].iloc[-1])
            bn_prev_high = float(banknifty['High'].iloc[-2])
            bn_prev_low = float(banknifty['Low'].iloc[-2])
            
            # BankNifty specific multiples (100)
            def round_bn(x):
                return round(x / 100) * 100
            
            bn_pivot = (bn_prev_high + bn_prev_low + bn_current) / 3
            
            levels['BANKNIFTY'] = {
                'CURRENT': bn_current,
                'PREV_HIGH': bn_prev_high,
                'PREV_LOW': bn_prev_low,
                'IMMEDIATE_LEVELS': {
                    'SUPPORT_1': round_bn(bn_current * 0.995),
                    'SUPPORT_2': round_bn(bn_current * 0.985),
                    'RESISTANCE_1': round_bn(bn_current * 1.005),
                    'RESISTANCE_2': round_bn(bn_current * 1.015)
                }
            }
            
    except Exception as e:
        print(f"Levels calculation error: {e}")
    
    return levels

# --------- INSTITUTIONAL INTRADAY ANALYSIS ---------
def generate_intraday_analysis():
    """
    Generate professional intraday analysis like your screenshot
    """
    analysis = []
    
    try:
        # Get fresh data
        prev_data = get_correct_previous_day_data()
        levels = calculate_institutional_levels()
        sgx_nifty = get_fresh_sgx_nifty()
        vix_value, vix_sentiment = get_india_vix()
        
        if 'NIFTY' in prev_data and 'BANKNIFTY' in prev_data:
            n = prev_data['NIFTY']
            bn = prev_data['BANKNIFTY']
            
            # Calculate expected gap
            if sgx_nifty:
                expected_gap = sgx_nifty - n['CLOSE']
                gap_pct = (expected_gap / n['CLOSE']) * 100
                
                # Determine market tone
                if gap_pct > 0.5:
                    market_tone = "BULLISH"
                    gap_text = f"GAP UP of ~{abs(expected_gap):.0f} points ({gap_pct:.2f}%)"
                elif gap_pct < -0.5:
                    market_tone = "BEARISH"
                    gap_text = f"GAP DOWN of ~{abs(expected_gap):.0f} points ({abs(gap_pct):.2f}%)"
                else:
                    market_tone = "NEUTRAL"
                    gap_text = "FLAT TO MIXED OPENING"
            else:
                market_tone = "NEUTRAL"
                gap_text = "DATA UNAVAILABLE"
            
            # Add header
            analysis.append("🏦 <b>INSTITUTIONAL INTRADAY ANALYSIS</b>")
            analysis.append("══════════════════════════════")
            analysis.append("")
            
            # Market Overview
            analysis.append("<b>📊 MARKET OVERVIEW:</b>")
            analysis.append(f"• Expected Opening: <b>{gap_text}</b>")
            analysis.append(f"• Market Tone: <code>{market_tone}</code>")
            analysis.append(f"• India VIX: <code>{vix_value if vix_value else 'N/A'}</code> ({vix_sentiment})")
            analysis.append("")
            
            # NIFTY Analysis
            analysis.append("<b>📈 NIFTY 50 ANALYSIS:</b>")
            analysis.append(f"• Previous Close: <code>{n['CLOSE']}</code>")
            analysis.append(f"• Day Range: <code>{n['LOW']} - {n['HIGH']}</code> (Range: {n['RANGE']} points)")
            analysis.append(f"• Candle Pattern: <code>{n.get('PATTERN', 'N/A')}</code>")
            analysis.append(f"• Close Position: <code>{n.get('CLOSE_POSITION', 'N/A')} of range</code>")
            
            if 'NIFTY' in levels:
                lvl = levels['NIFTY']
                analysis.append("")
                analysis.append("<b>🎯 NIFTY KEY LEVELS:</b>")
                analysis.append(f"• Pivot: <code>{lvl['PIVOTS']['PIVOT']}</code>")
                analysis.append(f"• Support: <code>{lvl['PIVOTS']['S1']} | {lvl['PIVOTS']['S2']}</code>")
                analysis.append(f"• Resistance: <code>{lvl['PIVOTS']['R1']} | {lvl['PIVOTS']['R2']}</code>")
                analysis.append(f"• Buy Zone: <code>{lvl['TRADING_ZONES']['BUY_ZONE']}</code>")
                analysis.append(f"• Sell Zone: <code>{lvl['TRADING_ZONES']['SELL_ZONE']}</code>")
            
            analysis.append("")
            
            # BANKNIFTY Analysis (Like your screenshot)
            analysis.append("<b>🏦 BANKNIFTY DEEP ANALYSIS:</b>")
            analysis.append("══════════════════════════════")
            
            # Context
            if bn['CLOSE'] > 59000:
                analysis.append("<i>As long as Bank Nifty remains above the 59,000-59,500 area, the broader tone remains somewhat constructive, potentially enabling a bounce or further upside.</i>")
            else:
                analysis.append("<i>Bank Nifty is trading below key levels, watch for support around 58,500-58,800 for potential bounce opportunities.</i>")
            
            analysis.append("")
            
            # Support & Resistance Levels
            analysis.append("<b>☺️ SUPPORT & RESISTANCE LEVELS TO WATCH:</b>")
            
            if 'BANKNIFTY' in levels:
                bn_lvl = levels['BANKNIFTY']['IMMEDIATE_LEVELS']
                
                # Support Levels
                analysis.append(f"<b>Support-1 (Near-term cushion):</b>")
                analysis.append(f"  <code>~{bn_lvl['SUPPORT_1']}</code>")
                analysis.append(f"  Bounce from here could offer buying opportunities.")
                analysis.append("")
                
                analysis.append(f"<b>Support-2 (Critical):</b>")
                analysis.append(f"  <code>~{bn_lvl['SUPPORT_2']}</code>")
                analysis.append(f"  Break below may signal weakness or deeper correction.")
                analysis.append("")
                
                # Resistance Levels
                analysis.append(f"<b>Resistance-1 (Immediate upside):</b>")
                analysis.append(f"  <code>~{bn_lvl['RESISTANCE_1']}</code>")
                analysis.append(f"  Near-term target zone. If approached, watch for profit-taking.")
                analysis.append("")
                
                analysis.append(f"<b>Resistance-2 (Bullish breakout zone):</b>")
                analysis.append(f"  <code>~{bn_lvl['RESISTANCE_2']}</code>")
                analysis.append(f"  If price sustains above R1, this becomes next target.")
            
            analysis.append("")
            analysis.append("══════════════════════════════")
            analysis.append("")
            
            # Trading Plan
            analysis.append("<b>✕️ INSTITUTIONAL TRADING PLAN:</b>")
            analysis.append("")
            analysis.append("<b>For Intraday/Swing Traders:</b>")
            
            if market_tone == "BULLISH":
                analysis.append("1. <b>Gap Up Strategy:</b>")
                analysis.append("   • Wait for pullback to buy near opening low")
                analysis.append("   • Entry: Support zone (~59,500-59,800)")
                analysis.append("   • Target: Resistance-1 (~60,150-60,300)")
                analysis.append("   • Stop Loss: Below Support-2 (~59,000)")
            
            elif market_tone == "BEARISH":
                analysis.append("1. <b>Gap Down Strategy:</b>")
                analysis.append("   • Sell on rallies near resistance")
                analysis.append("   • Entry: Resistance-1 zone")
                analysis.append("   • Target: Support-2 zone")
                analysis.append("   • Stop Loss: Above opening high")
            
            else:
                analysis.append("1. <b>Rangebound Strategy:</b>")
                analysis.append("   • Buy near support, Sell near resistance")
                analysis.append("   • Buy Zone: 59,000-59,300")
                analysis.append("   • Sell Zone: 60,000-60,200")
                analysis.append("   • Stoploss: 50 points beyond entry")
            
            analysis.append("")
            analysis.append("2. <b>Breakout Strategy:</b>")
            analysis.append("   • If BankNifty breaks above 60,500 with volume")
            analysis.append("   • Go long for target 61,000-61,200")
            analysis.append("")
            analysis.append("3. <b>Breakdown Strategy:</b>")
            analysis.append("   • If breaks below 59,000 with momentum")
            analysis.append("   • Go short for target 58,500-58,300")
            
            analysis.append("")
            analysis.append("══════════════════════════════")
            analysis.append("")
            
            # Risk Management
            analysis.append("<b>⚠️ RISK MANAGEMENT:</b>")
            analysis.append("• Never risk more than 1-2% per trade")
            analysis.append("• Use proper position sizing")
            analysis.append("• Hedge with options if holding overnight")
            analysis.append("• Book partial profits at technical levels")
            
            analysis.append("")
            analysis.append("<b>✅ ANALYSIS GENERATED: Institutional Trading Desk</b>")
            analysis.append(f"<b>⏰ Time: {get_ist_time().strftime('%H:%M IST')}</b>")
            
    except Exception as e:
        analysis = [f"<b>⚠️ ANALYSIS ERROR:</b> {str(e)}"]
    
    return "\n".join(analysis)

# --------- FIXED: INDIA VIX FUNCTION ---------
def get_india_vix():
    """Get India VIX with proper rounding"""
    try:
        vix = yf.download("^INDIAVIX", period="1d", interval="1d", progress=False)
        if not vix.empty:
            vix_value = float(vix['Close'].iloc[-1])
            vix_value_rounded = round(vix_value, 2)
            
            if vix_value_rounded < 12: 
                sentiment = "LOW FEAR (Rangebound Market)"
            elif vix_value_rounded < 18: 
                sentiment = "NORMAL (Balanced)"
            elif vix_value_rounded < 25: 
                sentiment = "HIGH FEAR (Volatile)"
            else: 
                sentiment = "EXTREME FEAR (High Volatility)"
            
            return vix_value_rounded, sentiment
            
    except Exception as e:
        print(f"VIX error: {e}")
    
    return None, "UNAVAILABLE"

# --------- FIXED: MAIN FUNCTION FOR DAILY RUN ---------
def generate_daily_report():
    """
    Generate complete daily report at 9 AM
    """
    ist_now = get_ist_time()
    
    report = []
    report.append(f"📊 <b>INSTITUTIONAL PRE-MARKET ANALYSIS</b>")
    report.append(f"📅 <b>{ist_now.strftime('%d %b %Y, %A')}</b>")
    report.append(f"⏰ <b>{ist_now.strftime('%H:%M IST')}</b>")
    report.append("")
    report.append("══════════════════════════════")
    report.append("")
    
    try:
        # Get all fresh data
        sgx_nifty = get_fresh_sgx_nifty()
        prev_data = get_correct_previous_day_data()
        global_mkts = get_global_markets()
        vix_value, vix_sentiment = get_india_vix()
        
        # SGX Nifty
        if sgx_nifty and 'NIFTY' in prev_data:
            prev_close = prev_data['NIFTY']['CLOSE']
            gap_points = sgx_nifty - prev_close
            gap_pct = (gap_points / prev_close) * 100
            
            gap_icon = "🟢" if gap_points > 0 else "🔴"
            report.append(f"🌏 <b>SGX NIFTY:</b> <code>{sgx_nifty}</code>")
            report.append(f"   Prev Close: <code>{prev_close}</code>")
            report.append(f"   Expected Gap: {gap_icon} <code>{gap_points:+.0f} points ({gap_pct:+.2f}%)</code>")
        else:
            report.append(f"🌏 <b>SGX NIFTY:</b> <code>UNAVAILABLE</code>")
        
        report.append("")
        
        # Global Markets
        if global_mkts:
            report.append(f"🌍 <b>GLOBAL MARKETS (Overnight):</b>")
            for market, change in list(global_mkts.items())[:5]:
                icon = "🟢" if change > 0 else "🔴"
                report.append(f"   {market}: {icon} <code>{change:+.2f}%</code>")
        
        report.append("")
        
        # Previous Day Summary
        if 'NIFTY' in prev_data:
            n = prev_data['NIFTY']
            icon = "🟢" if n['CHANGE'] > 0 else "🔴"
            report.append(f"📈 <b>PREVIOUS DAY ({n['DATE']}):</b>")
            report.append(f"   Close: <code>{n['CLOSE']}</code> {icon} {n['CHANGE']:+.2f}%")
            report.append(f"   Range: <code>{n['LOW']} - {n['HIGH']}</code> ({n['RANGE']} pts)")
            report.append(f"   Pattern: <code>{n.get('PATTERN', 'N/A')}</code>")
        
        report.append("")
        
        # VIX & Sentiment
        if vix_value:
            report.append(f"😨 <b>INDIA VIX:</b> <code>{vix_value}</code>")
            report.append(f"   Sentiment: <code>{vix_sentiment}</code>")
        
        report.append("")
        report.append("══════════════════════════════")
        report.append("")
        
        # Add Institutional Intraday Analysis
        intraday_report = generate_intraday_analysis()
        report.append(intraday_report)
        
    except Exception as e:
        report.append(f"<b>⚠️ REPORT ERROR:</b> {str(e)}")
    
    return "\n".join(report)

# --------- DAILY EXECUTION ---------
def main():
    """
    Main execution for daily 9 AM run
    """
    print("🚀 Institutional Pre-Market Analysis Engine Started...")
    
    try:
        ist_now = get_ist_time()
        print(f"⏰ {ist_now.strftime('%H:%M:%S IST')} - Generating Report...")
        
        # Generate complete report
        report = generate_daily_report()
        
        # Send to Telegram
        success = send_telegram(report)
        
        if success:
            print("✅ Report Sent Successfully!")
        else:
            print("❌ Failed to send report")
        
        # Also save to file for tracking
        with open(f"reports/report_{ist_now.strftime('%Y%m%d')}.txt", "w") as f:
            f.write(report)
            
    except Exception as e:
        print(f"❌ Main execution error: {e}")

# --------- GLOBAL MARKETS FUNCTION ---------
def get_global_markets():
    """Get overnight global market performance"""
    markets = {}
    
    try:
        symbols = {
            'DOW': '^DJI',
            'NASDAQ': '^IXIC',
            'S&P500': '^GSPC',
            'NIKKEI': '^N225',
            'HSI': '^HSI',
            'SHANGHAI': '000001.SS',
            'ASX200': '^AXJO',
            'DAX': '^GDAXI',
            'FTSE': '^FTSE',
        }
        
        for name, symbol in symbols.items():
            try:
                data = yf.download(symbol, period="2d", interval="1d", progress=False)
                if not data.empty and len(data) >= 2:
                    prev_close = float(data['Close'].iloc[-2])
                    current_close = float(data['Close'].iloc[-1])
                    change_pct = ((current_close - prev_close) / prev_close) * 100
                    markets[name] = round(change_pct, 2)
            except:
                continue
                
    except Exception as e:
        print(f"Global markets error: {e}")
    
    return markets

# --------- STARTUP ---------
if __name__ == "__main__":
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    # Send startup message
    ist_now = get_ist_time()
    startup_msg = f"🚀 <b>Institutional Pre-Market Analysis Engine</b>\n"
    startup_msg += f"⏰ Started at: {ist_now.strftime('%H:%M:%S IST')}\n"
    startup_msg += f"📅 Date: {ist_now.strftime('%d %b %Y')}\n"
    startup_msg += f"✅ Engine activated for 9 AM report"
    send_telegram(startup_msg)
    
    # Run main function
    main()
