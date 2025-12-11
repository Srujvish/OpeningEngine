# INSTITUTIONAL PRE-MARKET ANALYSIS ENGINE - ULTIMATE INTRADAY VERSION
# FIXED: Real intraday levels based on TODAY'S opening, not yesterday's range

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

# --------- IMPROVED SGX NIFTY ---------
def get_fresh_sgx_nifty():
    """Get accurate SGX Nifty"""
    try:
        # Primary source
        data = yf.download("NQ=F", period="1d", interval="5m", progress=False)
        if not data.empty:
            return round(float(data['Close'].iloc[-1]), 2)
        
        # Fallback
        data = yf.download("^NSEI", period="1d", interval="5m", progress=False)
        if not data.empty:
            nifty_last = float(data['Close'].iloc[-1])
            return round(nifty_last * 0.998, 2)  # Adjust for SGX premium
        
    except Exception as e:
        print(f"SGX error: {e}")
    
    return None

# --------- REAL INTRADAY LEVELS FOR TODAY ---------
def calculate_intraday_levels_today(sgx_price, is_banknifty=False):
    """
    Calculate REAL levels for TODAY based on:
    1. Today's expected opening (SGX)
    2. Recent volatility (last 5 days)
    3. Average daily range
    4. Psychological levels
    """
    try:
        if not sgx_price:
            return None
        
        # Get recent data for volatility calculation
        if is_banknifty:
            symbol = "^NSEBANK"
            data = yf.download(symbol, period="5d", interval="15m", progress=False)
        else:
            symbol = "^NSEI"
            data = yf.download(symbol, period="5d", interval="15m", progress=False)
        
        today_open = sgx_price
        
        if data.empty:
            # Default ranges if no data
            if is_banknifty:
                daily_range = 300  # BankNifty average range
                support_distance = 100
                resistance_distance = 100
            else:
                daily_range = 100  # Nifty average range
                support_distance = 50
                resistance_distance = 50
        else:
            # Calculate REAL average daily range from last 5 days
            data['Date'] = data.index.date
            unique_days = data['Date'].unique()
            
            daily_ranges = []
            for day in unique_days[-5:]:  # Last 5 days
                day_data = data[data['Date'] == day]
                if not day_data.empty:
                    day_high = day_data['High'].max()
                    day_low = day_data['Low'].min()
                    daily_ranges.append(day_high - day_low)
            
            if daily_ranges:
                avg_daily_range = np.mean(daily_ranges)
            else:
                avg_daily_range = 100 if not is_banknifty else 300
            
            # Calculate today's expected range based on gap
            # If gap > 100 points, expect larger range
            gap_size = 0  # We'll calculate this separately
            
            if is_banknifty:
                # BankNifty levels
                if avg_daily_range > 400:
                    support_distance = 150
                    resistance_distance = 150
                elif avg_daily_range > 250:
                    support_distance = 100
                    resistance_distance = 100
                else:
                    support_distance = 80
                    resistance_distance = 80
            else:
                # Nifty levels
                if avg_daily_range > 150:
                    support_distance = 60
                    resistance_distance = 60
                elif avg_daily_range > 100:
                    support_distance = 50
                    resistance_distance = 50
                else:
                    support_distance = 40
                    resistance_distance = 40
        
        # Get psychological levels (round numbers)
        if is_banknifty:
            # Round to nearest 50
            today_open = round(today_open / 50) * 50
            support_distance = round(support_distance / 50) * 50
            resistance_distance = round(resistance_distance / 50) * 50
        else:
            # Round to nearest 20
            today_open = round(today_open / 20) * 20
            support_distance = round(support_distance / 20) * 20
            resistance_distance = round(resistance_distance / 20) * 20
        
        # Calculate levels
        immediate_support = today_open - (support_distance * 0.5)
        strong_support = today_open - support_distance
        immediate_resistance = today_open + (resistance_distance * 0.5)
        strong_resistance = today_open + resistance_distance
        
        # Ensure levels are realistic (not too close)
        min_distance = 20 if not is_banknifty else 50
        if (immediate_resistance - immediate_support) < min_distance:
            immediate_resistance = today_open + min_distance
            immediate_support = today_open - min_distance
        
        # Get yesterday's close for reference
        daily_data = yf.download(symbol, period="2d", interval="1d", progress=False)
        if not daily_data.empty and len(daily_data) >= 2:
            prev_close = float(daily_data['Close'].iloc[-2])
            prev_high = float(daily_data['High'].iloc[-2])
            prev_low = float(daily_data['Low'].iloc[-2])
        else:
            prev_close = prev_high = prev_low = today_open
        
        # Calculate gap from previous close
        gap_points = today_open - prev_close
        gap_percent = (gap_points / prev_close) * 100
        
        return {
            'TODAY_OPEN': round(today_open, 2),
            'PREV_CLOSE': round(prev_close, 2),
            'PREV_HIGH': round(prev_high, 2),
            'PREV_LOW': round(prev_low, 2),
            'GAP_POINTS': round(gap_points, 2),
            'GAP_PERCENT': round(gap_percent, 2),
            'LEVELS': {
                'IMMEDIATE_SUPPORT': round(immediate_support, 2),
                'STRONG_SUPPORT': round(strong_support, 2),
                'IMMEDIATE_RESISTANCE': round(immediate_resistance, 2),
                'STRONG_RESISTANCE': round(strong_resistance, 2),
                'INTRADAY_RANGE': f"{round(strong_support, 2)} - {round(strong_resistance, 2)}",
                'TRADING_ZONE': f"{round(immediate_support, 2)} - {round(immediate_resistance, 2)}"
            },
            'PSYCHOLOGICAL_LEVELS': {
                'NEAREST_100': round(today_open / 100) * 100 if is_banknifty else None,
                'NEAREST_50': round(today_open / 50) * 50,
                'NEAREST_20': round(today_open / 20) * 20 if not is_banknifty else None
            }
        }
        
    except Exception as e:
        print(f"Today's levels error: {e}")
        return None

# --------- SIMPLIFIED GAP ANALYSIS ---------
def get_index_gap_analysis():
    """Simple and accurate gap analysis"""
    try:
        # Get SGX
        sgx_nifty = get_fresh_sgx_nifty()
        
        if not sgx_nifty:
            return None
        
        # Get Nifty previous close
        nifty = yf.download("^NSEI", period="2d", interval="1d", progress=False)
        if nifty.empty:
            return None
        
        nifty_prev_close = float(nifty['Close'].iloc[-2])
        nifty_gap_points = sgx_nifty - nifty_prev_close
        nifty_gap_pct = (nifty_gap_points / nifty_prev_close) * 100
        
        # Get BankNifty previous close
        banknifty = yf.download("^NSEBANK", period="2d", interval="1d", progress=False)
        if banknifty.empty:
            return None
        
        bn_prev_close = float(banknifty['Close'].iloc[-2])
        
        # BankNifty gap (simplified - 1.1x to 1.3x of Nifty gap)
        if abs(nifty_gap_pct) > 1.0:
            bn_multiplier = 1.3
        elif abs(nifty_gap_pct) > 0.5:
            bn_multiplier = 1.2
        else:
            bn_multiplier = 1.1
        
        bn_gap_points = nifty_gap_points * bn_multiplier
        bn_gap_pct = nifty_gap_pct * bn_multiplier
        bn_expected_open = bn_prev_close + bn_gap_points
        
        # Gap classification
        def classify_gap(gap_pct):
            if gap_pct > 0.8:
                return "STRONG GAP UP", "VERY BULLISH"
            elif gap_pct > 0.3:
                return "MODERATE GAP UP", "BULLISH"
            elif gap_pct < -0.8:
                return "STRONG GAP DOWN", "VERY BEARISH"
            elif gap_pct < -0.3:
                return "MODERATE GAP DOWN", "BEARISH"
            else:
                return "MINOR GAP", "NEUTRAL"
        
        nifty_gap_type, nifty_strength = classify_gap(nifty_gap_pct)
        bn_gap_type, bn_strength = classify_gap(bn_gap_pct)
        
        # Calculate TODAY'S levels
        nifty_levels = calculate_intraday_levels_today(sgx_nifty, is_banknifty=False)
        bn_levels = calculate_intraday_levels_today(bn_expected_open, is_banknifty=True)
        
        return {
            'NIFTY': {
                'PREV_CLOSE': round(nifty_prev_close, 2),
                'SGX_PRICE': round(sgx_nifty, 2),
                'GAP_POINTS': round(nifty_gap_points, 2),
                'GAP_PERCENT': round(nifty_gap_pct, 2),
                'GAP_TYPE': nifty_gap_type,
                'GAP_STRENGTH': nifty_strength,
                'TODAY_LEVELS': nifty_levels
            },
            'BANKNIFTY': {
                'PREV_CLOSE': round(bn_prev_close, 2),
                'EXPECTED_OPEN': round(bn_expected_open, 2),
                'GAP_POINTS': round(bn_gap_points, 2),
                'GAP_PERCENT': round(bn_gap_pct, 2),
                'GAP_TYPE': bn_gap_type,
                'GAP_STRENGTH': bn_strength,
                'TODAY_LEVELS': bn_levels,
                'BETA_USED': round(bn_multiplier, 2)
            }
        }
        
    except Exception as e:
        print(f"Gap analysis error: {e}")
        return None

# --------- SIMPLIFIED PREVIOUS DAY DATA ---------
def get_previous_day_summary():
    """Get only essential previous day data"""
    try:
        # Nifty
        nifty = yf.download("^NSEI", period="2d", interval="1d", progress=False)
        if nifty.empty or len(nifty) < 2:
            return None
        
        prev_close = float(nifty['Close'].iloc[-2])
        prev_open = float(nifty['Open'].iloc[-2])
        prev_high = float(nifty['High'].iloc[-2])
        prev_low = float(nifty['Low'].iloc[-2])
        change_pct = ((prev_close - prev_open) / prev_open) * 100
        
        # Simple pattern
        if prev_close > prev_open:
            pattern = "BULLISH"
        elif prev_close < prev_open:
            pattern = "BEARISH"
        else:
            pattern = "NEUTRAL"
        
        return {
            'CLOSE': round(prev_close, 2),
            'HIGH': round(prev_high, 2),
            'LOW': round(prev_low, 2),
            'CHANGE': round(change_pct, 2),
            'RANGE': round(prev_high - prev_low, 2),
            'PATTERN': pattern
        }
        
    except Exception as e:
        print(f"Previous day error: {e}")
        return None

# --------- GLOBAL MARKETS ---------
def get_global_markets():
    """Get global markets"""
    markets = {}
    try:
        symbols = {'DOW': '^DJI', 'NASDAQ': '^IXIC', 'S&P500': '^GSPC'}
        for name, symbol in symbols.items():
            data = yf.download(symbol, period="2d", interval="1d", progress=False)
            if not data.empty and len(data) >= 2:
                change = ((float(data['Close'].iloc[-1]) - float(data['Close'].iloc[-2])) / float(data['Close'].iloc[-2])) * 100
                markets[name] = round(change, 2)
    except Exception as e:
        print(f"Global markets error: {e}")
    return markets

# --------- VIX ---------
def get_india_vix():
    """Get India VIX"""
    try:
        vix = yf.download("^INDIAVIX", period="1d", interval="1d", progress=False)
        if not vix.empty:
            vix_value = round(float(vix['Close'].iloc[-1]), 2)
            sentiment = "LOW FEAR" if vix_value < 13 else "NORMAL" if vix_value < 18 else "HIGH FEAR"
            return vix_value, sentiment
    except:
        pass
    return None, "UNAVAILABLE"

# --------- MAIN REPORT ---------
def generate_daily_report():
    """Generate ultra-simplified actionable report"""
    ist_now = get_ist_time()
    
    report = []
    report.append(f"📊 <b>INSTITUTIONAL PRE-MARKET ANALYSIS</b>")
    report.append(f"📅 <b>{ist_now.strftime('%d %b %Y, %A')}</b>")
    report.append(f"⏰ <b>{ist_now.strftime('%H:%M IST')}</b>")
    report.append("")
    report.append("══════════════════════════════")
    report.append("")
    
    try:
        gap_data = get_index_gap_analysis()
        
        if gap_data:
            nifty = gap_data['NIFTY']
            banknifty = gap_data['BANKNIFTY']
            
            # OPENING PROJECTIONS
            report.append("🎯 <b>OPENING PROJECTIONS</b>")
            report.append("")
            
            nifty_icon = "🟢" if nifty['GAP_POINTS'] > 0 else "🔴"
            report.append(f"{nifty_icon} <b>NIFTY 50:</b>")
            report.append(f"   Prev: <code>{nifty['PREV_CLOSE']}</code>")
            report.append(f"   SGX: <code>{nifty['SGX_PRICE']}</code>")
            report.append(f"   Gap: <code>{nifty['GAP_POINTS']:+.0f} pts ({nifty['GAP_PERCENT']:+.2f}%)</code>")
            report.append(f"   Type: <b>{nifty['GAP_TYPE']}</b>")
            report.append("")
            
            bn_icon = "🟢" if banknifty['GAP_POINTS'] > 0 else "🔴"
            report.append(f"{bn_icon} <b>BANKNIFTY:</b>")
            report.append(f"   Prev: <code>{banknifty['PREV_CLOSE']}</code>")
            report.append(f"   Expected: <code>{banknifty['EXPECTED_OPEN']}</code>")
            report.append(f"   Gap: <code>{banknifty['GAP_POINTS']:+.0f} pts ({banknifty['GAP_PERCENT']:+.2f}%)</code>")
            report.append(f"   Type: <b>{banknifty['GAP_TYPE']}</b>")
        
        report.append("")
        report.append("══════════════════════════════")
        report.append("")
        
        # TODAY'S ACTIONABLE LEVELS
        report.append("🎯 <b>TODAY'S ACTIONABLE LEVELS</b>")
        report.append("══════════════════════════════")
        report.append("")
        
        if gap_data and 'TODAY_LEVELS' in gap_data['NIFTY']:
            nifty_levels = gap_data['NIFTY']['TODAY_LEVELS']
            if nifty_levels and 'LEVELS' in nifty_levels:
                levels = nifty_levels['LEVELS']
                
                report.append(f"<b>📊 NIFTY 50 - TODAY'S TRADING ZONES:</b>")
                report.append(f"• Expected Open: <code>{nifty_levels['TODAY_OPEN']}</code>")
                report.append(f"• Gap: <code>{nifty_levels['GAP_POINTS']:+.0f} pts ({nifty_levels['GAP_PERCENT']:+.2f}%)</code>")
                report.append("")
                report.append(f"<b>Support (Buy Zone):</b>")
                report.append(f"  Immediate: <code>{levels['IMMEDIATE_SUPPORT']}</code>")
                report.append(f"  Strong: <code>{levels['STRONG_SUPPORT']}</code>")
                report.append("")
                report.append(f"<b>Resistance (Sell Zone):</b>")
                report.append(f"  Immediate: <code>{levels['IMMEDIATE_RESISTANCE']}</code>")
                report.append(f"  Strong: <code>{levels['STRONG_RESISTANCE']}</code>")
                report.append("")
                report.append(f"<b>Trading Range:</b> <code>{levels['TRADING_ZONE']}</code>")
                report.append(f"<b>Full Range:</b> <code>{levels['INTRADAY_RANGE']}</code>")
        
        report.append("")
        
        if gap_data and 'TODAY_LEVELS' in gap_data['BANKNIFTY']:
            bn_levels = gap_data['BANKNIFTY']['TODAY_LEVELS']
            if bn_levels and 'LEVELS' in bn_levels:
                levels = bn_levels['LEVELS']
                
                report.append(f"<b>🏦 BANKNIFTY - TODAY'S TRADING ZONES:</b>")
                report.append(f"• Expected Open: <code>{bn_levels['TODAY_OPEN']}</code>")
                report.append(f"• Gap: <code>{bn_levels['GAP_POINTS']:+.0f} pts ({bn_levels['GAP_PERCENT']:+.2f}%)</code>")
                report.append("")
                report.append(f"<b>Support (Buy Zone):</b>")
                report.append(f"  Immediate: <code>{levels['IMMEDIATE_SUPPORT']}</code>")
                report.append(f"  Strong: <code>{levels['STRONG_SUPPORT']}</code>")
                report.append("")
                report.append(f"<b>Resistance (Sell Zone):</b>")
                report.append(f"  Immediate: <code>{levels['IMMEDIATE_RESISTANCE']}</code>")
                report.append(f"  Strong: <code>{levels['STRONG_RESISTANCE']}</code>")
                report.append("")
                report.append(f"<b>Trading Range:</b> <code>{levels['TRADING_ZONE']}</code>")
                report.append(f"<b>Full Range:</b> <code>{levels['INTRADAY_RANGE']}</code>")
        
        report.append("")
        report.append("══════════════════════════════")
        report.append("")
        
        # MARKET CONTEXT
        report.append("<b>📊 MARKET CONTEXT:</b>")
        report.append("")
        
        # Previous day
        prev_day = get_previous_day_summary()
        if prev_day:
            icon = "🟢" if prev_day['CHANGE'] > 0 else "🔴"
            report.append(f"• Yesterday Close: <code>{prev_day['CLOSE']}</code> {icon} {prev_day['CHANGE']:+.2f}%")
            report.append(f"• Yesterday Range: <code>{prev_day['LOW']} - {prev_day['HIGH']}</code>")
            report.append(f"• Pattern: <code>{prev_day['PATTERN']}</code>")
        
        # VIX
        vix_value, vix_sentiment = get_india_vix()
        if vix_value:
            report.append(f"• India VIX: <code>{vix_value}</code> ({vix_sentiment})")
        
        # Global
        global_mkts = get_global_markets()
        if global_mkts:
            report.append(f"• Global: ", end="")
            for idx, (market, change) in enumerate(global_mkts.items()):
                icon = "🟢" if change > 0 else "🔴"
                if idx > 0:
                    report[-1] += f", {market} {icon} {change:+.1f}%"
                else:
                    report.append(f"  {market} {icon} {change:+.1f}%", end="")
        
        report.append("")
        report.append("══════════════════════════════")
        report.append("")
        
        # TRADING PLAN BASED ON GAP
        if gap_data:
            nifty_gap = gap_data['NIFTY']['GAP_PERCENT']
            
            report.append("<b>✕️ TODAY'S TRADING PLAN:</b>")
            report.append("")
            
            if nifty_gap < -0.5:  # Gap Down
                report.append("<b>GAP DOWN SCENARIO:</b>")
                report.append("1. <b>Short on rallies</b> to fill 30-50% of gap")
                if gap_data['NIFTY']['TODAY_LEVELS']:
                    res = gap_data['NIFTY']['TODAY_LEVELS']['LEVELS']['IMMEDIATE_RESISTANCE']
                    report.append(f"   • Sell Zone: <code>{res}</code>")
                report.append("2. <b>Targets:</b> Yesterday's low or next support")
                report.append("3. <b>Stop Loss:</b> Above opening high")
                report.append("4. <b>Alternative:</b> Wait for gap fill then reverse")
                
            elif nifty_gap > 0.5:  # Gap Up
                report.append("<b>GAP UP SCENARIO:</b>")
                report.append("1. <b>Buy on dips</b> to 30-50% of gap")
                if gap_data['NIFTY']['TODAY_LEVELS']:
                    sup = gap_data['NIFTY']['TODAY_LEVELS']['LEVELS']['IMMEDIATE_SUPPORT']
                    report.append(f"   • Buy Zone: <code>{sup}</code>")
                report.append("2. <b>Targets:</b> Yesterday's high or next resistance")
                report.append("3. <b>Stop Loss:</b> Below opening low")
                
            else:  # Flat
                report.append("<b>RANGEBOUND SCENARIO:</b>")
                report.append("1. <b>Buy low, Sell high</b> within range")
                report.append("2. Use support/resistance levels for entries")
                report.append("3. Tight stops (20-30 points Nifty)")
                report.append("4. Look for breakout with volume")
        
        report.append("")
        report.append("══════════════════════════════")
        report.append("")
        
        # KEY LEVELS TO WATCH
        report.append("<b>🔑 KEY LEVELS TO WATCH TODAY:</b>")
        report.append("")
        
        if gap_data:
            nifty_data = gap_data['NIFTY']
            bn_data = gap_data['BANKNIFTY']
            
            report.append(f"<b>NIFTY:</b>")
            report.append(f"• Opening: <code>{nifty_data['SGX_PRICE']}</code>")
            report.append(f"• Previous Close: <code>{nifty_data['PREV_CLOSE']}</code>")
            if nifty_data['TODAY_LEVELS']:
                levels = nifty_data['TODAY_LEVELS']['LEVELS']
                report.append(f"• Immediate S/R: <code>{levels['IMMEDIATE_SUPPORT']} / {levels['IMMEDIATE_RESISTANCE']}</code>")
            
            report.append("")
            report.append(f"<b>BANKNIFTY:</b>")
            report.append(f"• Opening: <code>{bn_data['EXPECTED_OPEN']}</code>")
            report.append(f"• Previous Close: <code>{bn_data['PREV_CLOSE']}</code>")
            if bn_data['TODAY_LEVELS']:
                levels = bn_data['TODAY_LEVELS']['LEVELS']
                report.append(f"• Immediate S/R: <code>{levels['IMMEDIATE_SUPPORT']} / {levels['IMMEDIATE_RESISTANCE']}</code>")
        
        report.append("")
        report.append("<b>⚠️ RISK RULES:</b>")
        report.append("• Max 1% risk per trade")
        report.append("• No revenge trading")
        report.append("• Book partial profits")
        report.append("• Stop loss is MANDATORY")
        
        report.append("")
        report.append(f"<b>✅ GENERATED: {ist_now.strftime('%H:%M IST')}</b>")
        
    except Exception as e:
        report.append(f"<b>⚠️ ERROR:</b> {str(e)}")
    
    return "\n".join(report)

# --------- MAIN ---------
def main():
    """Main function"""
    print("🚀 Ultimate Intraday Analysis Started...")
    
    try:
        ist_now = get_ist_time()
        print(f"⏰ {ist_now.strftime('%H:%M:%S IST')}")
        
        report = generate_daily_report()
        
        # Send to Telegram
        success = send_telegram(report)
        
        if success:
            print("✅ Report Sent!")
        else:
            print("❌ Telegram failed")
        
        # Save
        os.makedirs("reports", exist_ok=True)
        with open(f"reports/today_{ist_now.strftime('%Y%m%d_%H%M')}.txt", "w") as f:
            f.write(report)
            
    except Exception as e:
        print(f"❌ Error: {e}")

# --------- START ---------
if __name__ == "__main__":
    # Startup message
    ist_now = get_ist_time()
    startup_msg = f"🚀 <b>Ultimate Intraday Analysis v3.0</b>\n"
    startup_msg += f"⏰ {ist_now.strftime('%H:%M:%S IST')}\n"
    startup_msg += f"📅 {ist_now.strftime('%d %b %Y')}\n"
    startup_msg += f"✅ Generating TODAY'S levels..."
    send_telegram(startup_msg)
    
    # Run
    main()
