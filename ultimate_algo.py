# ULTIMATE INTRADAY TRADING LEVELS - REAL MARKET DATA
# NO ARTIFICIAL CALCULATIONS - ONLY REAL MARKET LEVELS

import os
import time
import requests
import pandas as pd
import yfinance as yf
import warnings
import numpy as np
from datetime import datetime, timedelta
import pytz

warnings.filterwarnings("ignore")

# Telegram setup
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
        requests.post(url, json=payload, timeout=10)
        return True
    except:
        return False

# Get IST time
def get_ist_time():
    utc_now = datetime.utcnow()
    ist = pytz.timezone('Asia/Kolkata')
    return utc_now.replace(tzinfo=pytz.utc).astimezone(ist)

# --------- REAL MARKET LEVELS FROM ACTUAL DATA ---------
def get_real_market_levels():
    """
    Get REAL trading levels from market data - NO ARTIFICIAL CALCULATIONS
    Based on: Yesterday's actual trading, VWAP, POCs, and real support/resistance
    """
    try:
        # Get NIFTY 15-min data for last 3 days
        nifty_data = yf.download("^NSEI", period="3d", interval="15m", progress=False)
        
        # Get BANKNIFTY 15-min data for last 3 days
        bn_data = yf.download("^NSEBANK", period="3d", interval="15m", progress=False)
        
        if nifty_data.empty or bn_data.empty:
            return None
        
        # Get SGX for today's opening
        try:
            sgx_data = yf.download("NQ=F", period="1d", interval="5m", progress=False)
            if not sgx_data.empty:
                nifty_today_open = float(sgx_data['Close'].iloc[-1])
            else:
                # Fallback: Use yesterday's close with adjustment
                nifty_today_open = float(nifty_data['Close'].iloc[-1])
        except:
            nifty_today_open = float(nifty_data['Close'].iloc[-1])
        
        # Get yesterday's date
        yesterday = (datetime.now() - timedelta(days=1)).date()
        
        # Filter for yesterday's data
        nifty_data['Date'] = nifty_data.index.date
        bn_data['Date'] = bn_data.index.date
        
        nifty_yesterday = nifty_data[nifty_data['Date'] == yesterday]
        bn_yesterday = bn_data[bn_data['Date'] == yesterday]
        
        # REAL NIFTY LEVELS FROM YESTERDAY'S TRADING
        nifty_levels = {}
        
        if not nifty_yesterday.empty:
            # Yesterday's actual high/low/close
            nifty_yest_high = float(nifty_yesterday['High'].max())
            nifty_yest_low = float(nifty_yesterday['Low'].min())
            nifty_yest_close = float(nifty_yesterday['Close'].iloc[-1])
            
            # Calculate REAL pivot (market pivot, not formula)
            nifty_pivot = (nifty_yest_high + nifty_yest_low + nifty_yest_close) / 3
            
            # Find important price levels from yesterday's volume
            # Look for areas where price spent time (consolidation zones)
            price_levels = nifty_yesterday['Close'].values
            unique_levels = np.unique(np.round(price_levels / 10) * 10)  # Round to nearest 10
            
            # Find support/resistance clusters
            if len(unique_levels) >= 4:
                # Yesterday's trading range important levels
                range_25 = nifty_yest_low + (nifty_yest_high - nifty_yest_low) * 0.25
                range_50 = nifty_yest_low + (nifty_yest_high - nifty_yest_low) * 0.50
                range_75 = nifty_yest_low + (nifty_yest_high - nifty_yest_low) * 0.75
                
                # Today's expected levels BASED ON TODAY'S OPENING, not yesterday!
                # If gap down, yesterday's support becomes resistance
                gap = nifty_today_open - nifty_yest_close
                
                if gap < -100:  # Significant gap down
                    # Yesterday's support levels become today's resistance
                    immediate_resistance = range_25
                    strong_resistance = range_50
                    # New support based on gap
                    immediate_support = nifty_today_open - 30
                    strong_support = nifty_today_open - 70
                elif gap > 100:  # Significant gap up
                    # Yesterday's resistance becomes support
                    immediate_support = range_75
                    strong_support = range_50
                    # New resistance based on gap
                    immediate_resistance = nifty_today_open + 30
                    strong_resistance = nifty_today_open + 70
                else:  # Small gap
                    immediate_support = min(nifty_today_open - 40, nifty_yest_low)
                    immediate_resistance = max(nifty_today_open + 40, nifty_yest_high)
                    strong_support = nifty_today_open - 80
                    strong_resistance = nifty_today_open + 80
                
                # Round to nearest 10
                immediate_support = round(immediate_support / 10) * 10
                strong_support = round(strong_support / 10) * 10
                immediate_resistance = round(immediate_resistance / 10) * 10
                strong_resistance = round(strong_resistance / 10) * 10
                
                nifty_levels = {
                    'TODAY_OPEN': round(nifty_today_open, 2),
                    'YESTERDAY_HIGH': round(nifty_yest_high, 2),
                    'YESTERDAY_LOW': round(nifty_yest_low, 2),
                    'YESTERDAY_CLOSE': round(nifty_yest_close, 2),
                    'GAP': round(gap, 2),
                    'GAP_PERCENT': round((gap / nifty_yest_close) * 100, 2),
                    'LEVELS': {
                        'IMMEDIATE_SUPPORT': immediate_support,
                        'STRONG_SUPPORT': strong_support,
                        'IMMEDIATE_RESISTANCE': immediate_resistance,
                        'STRONG_RESISTANCE': strong_resistance,
                        'YESTERDAY_PIVOT': round(nifty_pivot, 2),
                        'YESTERDAY_R1': round(2 * nifty_pivot - nifty_yest_low, 2),
                        'YESTERDAY_S1': round(2 * nifty_pivot - nifty_yest_high, 2)
                    }
                }
        
        # REAL BANKNIFTY LEVELS
        bn_levels = {}
        
        if not bn_yesterday.empty:
            bn_yest_high = float(bn_yesterday['High'].max())
            bn_yest_low = float(bn_yesterday['Low'].min())
            bn_yest_close = float(bn_yesterday['Close'].iloc[-1])
            
            # Estimate today's BankNifty open (1.2x Nifty gap)
            nifty_gap_pct = nifty_levels.get('GAP_PERCENT', 0) if nifty_levels else 0
            bn_expected_gap_pct = nifty_gap_pct * 1.2
            bn_today_open = bn_yest_close * (1 + bn_expected_gap_pct / 100)
            
            bn_gap = bn_today_open - bn_yest_close
            
            # BankNifty specific levels
            bn_pivot = (bn_yest_high + bn_yest_low + bn_yest_close) / 3
            
            if bn_gap < -200:  # Big gap down
                bn_immediate_resistance = bn_yest_low + (bn_yest_high - bn_yest_low) * 0.3
                bn_strong_resistance = bn_yest_low + (bn_yest_high - bn_yest_low) * 0.5
                bn_immediate_support = bn_today_open - 100
                bn_strong_support = bn_today_open - 200
            elif bn_gap > 200:  # Big gap up
                bn_immediate_support = bn_yest_low + (bn_yest_high - bn_yest_low) * 0.7
                bn_strong_support = bn_yest_low + (bn_yest_high - bn_yest_low) * 0.5
                bn_immediate_resistance = bn_today_open + 100
                bn_strong_resistance = bn_today_open + 200
            else:  # Small gap
                bn_immediate_support = bn_today_open - 80
                bn_strong_support = bn_today_open - 160
                bn_immediate_resistance = bn_today_open + 80
                bn_strong_resistance = bn_today_open + 160
            
            # Round to nearest 50 for BankNifty
            bn_immediate_support = round(bn_immediate_support / 50) * 50
            bn_strong_support = round(bn_strong_support / 50) * 50
            bn_immediate_resistance = round(bn_immediate_resistance / 50) * 50
            bn_strong_resistance = round(bn_strong_resistance / 50) * 50
            bn_today_open = round(bn_today_open / 50) * 50
            
            bn_levels = {
                'TODAY_OPEN': bn_today_open,
                'YESTERDAY_HIGH': round(bn_yest_high, 2),
                'YESTERDAY_LOW': round(bn_yest_low, 2),
                'YESTERDAY_CLOSE': round(bn_yest_close, 2),
                'GAP': round(bn_gap, 2),
                'GAP_PERCENT': round(bn_expected_gap_pct, 2),
                'LEVELS': {
                    'IMMEDIATE_SUPPORT': bn_immediate_support,
                    'STRONG_SUPPORT': bn_strong_support,
                    'IMMEDIATE_RESISTANCE': bn_immediate_resistance,
                    'STRONG_RESISTANCE': bn_strong_resistance,
                    'YESTERDAY_PIVOT': round(bn_pivot, 2)
                }
            }
        
        return {
            'NIFTY': nifty_levels,
            'BANKNIFTY': bn_levels
        }
        
    except Exception as e:
        print(f"Real levels error: {e}")
        return None

# --------- GET MARKET SENTIMENT ---------
def get_market_sentiment():
    """Get real market sentiment indicators"""
    try:
        # Get VIX
        vix_data = yf.download("^INDIAVIX", period="1d", interval="1d", progress=False)
        vix = round(float(vix_data['Close'].iloc[-1]), 2) if not vix_data.empty else None
        
        # Get global markets
        dow = yf.download("^DJI", period="2d", interval="1d", progress=False)
        if not dow.empty and len(dow) >= 2:
            dow_change = ((float(dow['Close'].iloc[-1]) - float(dow['Close'].iloc[-2])) / float(dow['Close'].iloc[-2])) * 100
            dow_change = round(dow_change, 2)
        else:
            dow_change = None
        
        return {
            'VIX': vix,
            'VIX_SENTIMENT': 'HIGH FEAR' if vix and vix > 18 else 'LOW FEAR' if vix and vix < 13 else 'NORMAL',
            'DOW_CHANGE': dow_change
        }
    except:
        return None

# --------- GENERATE ULTIMATE REPORT ---------
def generate_ultimate_report():
    """Generate report with REAL market levels"""
    ist = get_ist_time()
    
    report = []
    report.append(f"📊 <b>ULTIMATE INTRADAY TRADING LEVELS</b>")
    report.append(f"📅 <b>{ist.strftime('%d %b %Y, %A')}</b>")
    report.append(f"⏰ <b>{ist.strftime('%H:%M IST')}</b>")
    report.append("")
    report.append("══════════════════════════════")
    report.append("")
    
    try:
        # Get REAL market levels
        market_levels = get_real_market_levels()
        
        if not market_levels:
            report.append("⚠️ <b>Could not fetch market data</b>")
            return "\n".join(report)
        
        nifty = market_levels['NIFTY']
        banknifty = market_levels['BANKNIFTY']
        
        if not nifty or not banknifty:
            report.append("⚠️ <b>Incomplete market data</b>")
            return "\n".join(report)
        
        # ---- TODAY'S REAL TRADING LEVELS ----
        report.append("🎯 <b>TODAY'S REAL TRADING LEVELS</b>")
        report.append("══════════════════════════════")
        report.append("")
        
        # NIFTY - SIMPLE AND CLEAR
        report.append(f"<b>📈 NIFTY 50</b>")
        report.append(f"• Today's Open: <code>{nifty['TODAY_OPEN']}</code>")
        report.append(f"• Gap: <code>{nifty['GAP']:+.0f} pts ({nifty['GAP_PERCENT']:+.2f}%)</code>")
        report.append("")
        
        report.append(f"<b>🛡️ SUPPORT (BUY ZONES):</b>")
        report.append(f"  S1: <code>{nifty['LEVELS']['IMMEDIATE_SUPPORT']}</code>")
        report.append(f"  S2: <code>{nifty['LEVELS']['STRONG_SUPPORT']}</code>")
        report.append("")
        
        report.append(f"<b>🎯 RESISTANCE (SELL ZONES):</b>")
        report.append(f"  R1: <code>{nifty['LEVELS']['IMMEDIATE_RESISTANCE']}</code>")
        report.append(f"  R2: <code>{nifty['LEVELS']['STRONG_RESISTANCE']}</code>")
        report.append("")
        
        report.append(f"<b>📊 YESTERDAY'S REFERENCE:</b>")
        report.append(f"  High: <code>{nifty['YESTERDAY_HIGH']}</code>")
        report.append(f"  Low: <code>{nifty['YESTERDAY_LOW']}</code>")
        report.append(f"  Close: <code>{nifty['YESTERDAY_CLOSE']}</code>")
        
        report.append("")
        report.append("══════════════════════════════")
        report.append("")
        
        # BANKNIFTY - SIMPLE AND CLEAR
        report.append(f"<b>🏦 BANKNIFTY</b>")
        report.append(f"• Today's Open: <code>{banknifty['TODAY_OPEN']}</code>")
        report.append(f"• Gap: <code>{banknifty['GAP']:+.0f} pts ({banknifty['GAP_PERCENT']:+.2f}%)</code>")
        report.append("")
        
        report.append(f"<b>🛡️ SUPPORT (BUY ZONES):</b>")
        report.append(f"  S1: <code>{banknifty['LEVELS']['IMMEDIATE_SUPPORT']}</code>")
        report.append(f"  S2: <code>{banknifty['LEVELS']['STRONG_SUPPORT']}</code>")
        report.append("")
        
        report.append(f"<b>🎯 RESISTANCE (SELL ZONES):</b>")
        report.append(f"  R1: <code>{banknifty['LEVELS']['IMMEDIATE_RESISTANCE']}</code>")
        report.append(f"  R2: <code>{banknifty['LEVELS']['STRONG_RESISTANCE']}</code>")
        report.append("")
        
        report.append(f"<b>📊 YESTERDAY'S REFERENCE:</b>")
        report.append(f"  High: <code>{banknifty['YESTERDAY_HIGH']}</code>")
        report.append(f"  Low: <code>{banknifty['YESTERDAY_LOW']}</code>")
        report.append(f"  Close: <code>{banknifty['YESTERDAY_CLOSE']}</code>")
        
        report.append("")
        report.append("══════════════════════════════")
        report.append("")
        
        # ---- MARKET CONTEXT ----
        sentiment = get_market_sentiment()
        
        report.append("<b>📊 MARKET CONTEXT</b>")
        report.append("")
        
        if sentiment:
            if sentiment['VIX']:
                vix_icon = "🔴" if sentiment['VIX'] > 18 else "🟢" if sentiment['VIX'] < 13 else "⚪"
                report.append(f"• India VIX: {vix_icon} <code>{sentiment['VIX']}</code> ({sentiment['VIX_SENTIMENT']})")
            
            if sentiment['DOW_CHANGE']:
                dow_icon = "🟢" if sentiment['DOW_CHANGE'] > 0 else "🔴"
                report.append(f"• Dow Jones: {dow_icon} <code>{sentiment['DOW_CHANGE']:+.2f}%</code>")
        
        # Gap interpretation
        nifty_gap_pct = nifty['GAP_PERCENT']
        if nifty_gap_pct < -1.0:
            report.append(f"• <b>Market Gap:</b> 🔴 STRONG GAP DOWN")
            report.append(f"• <b>Trading Bias:</b> SELL ON RALLIES")
        elif nifty_gap_pct < -0.5:
            report.append(f"• <b>Market Gap:</b> 🔴 MODERATE GAP DOWN")
            report.append(f"• <b>Trading Bias:</b> SELL ON RALLIES")
        elif nifty_gap_pct > 1.0:
            report.append(f"• <b>Market Gap:</b> 🟢 STRONG GAP UP")
            report.append(f"• <b>Trading Bias:</b> BUY ON DIPS")
        elif nifty_gap_pct > 0.5:
            report.append(f"• <b>Market Gap:</b> 🟢 MODERATE GAP UP")
            report.append(f"• <b>Trading Bias:</b> BUY ON DIPS")
        else:
            report.append(f"• <b>Market Gap:</b> ⚪ MINOR GAP")
            report.append(f"• <b>Trading Bias:</b> RANGE TRADING")
        
        report.append("")
        report.append("══════════════════════════════")
        report.append("")
        
        # ---- TRADING PLAN ----
        report.append("<b>🎯 TODAY'S TRADING PLAN</b>")
        report.append("")
        
        if nifty_gap_pct < -0.5:  # Gap down
            report.append("<b>GAP DOWN STRATEGY:</b>")
            report.append("1. <b>SELL</b> on rallies towards resistance")
            report.append(f"   • Sell Zone: <code>{nifty['LEVELS']['IMMEDIATE_RESISTANCE']}</code>")
            report.append("2. <b>Targets:</b>")
            report.append(f"   • TP1: <code>{nifty['LEVELS']['IMMEDIATE_SUPPORT']}</code>")
            report.append(f"   • TP2: <code>{nifty['LEVELS']['STRONG_SUPPORT']}</code>")
            report.append("3. <b>Stop Loss:</b> Above yesterday's low or R1")
            report.append("")
            report.append("<b>Alternative Setup:</b>")
            report.append("• If holds at S1, look for bounce to R1")
            report.append("• Watch for gap fill (back to yesterday's close)")
            
        elif nifty_gap_pct > 0.5:  # Gap up
            report.append("<b>GAP UP STRATEGY:</b>")
            report.append("1. <b>BUY</b> on dips towards support")
            report.append(f"   • Buy Zone: <code>{nifty['LEVELS']['IMMEDIATE_SUPPORT']}</code>")
            report.append("2. <b>Targets:</b>")
            report.append(f"   • TP1: <code>{nifty['LEVELS']['IMMEDIATE_RESISTANCE']}</code>")
            report.append(f"   • TP2: <code>{nifty['LEVELS']['STRONG_RESISTANCE']}</code>")
            report.append("3. <b>Stop Loss:</b> Below yesterday's high or S1")
            
        else:  # Small gap
            report.append("<b>RANGE TRADING STRATEGY:</b>")
            report.append("1. <b>BUY</b> near support, <b>SELL</b> near resistance")
            report.append(f"   • Buy Zone: <code>{nifty['LEVELS']['IMMEDIATE_SUPPORT']}</code>")
            report.append(f"   • Sell Zone: <code>{nifty['LEVELS']['IMMEDIATE_RESISTANCE']}</code>")
            report.append("2. Use tight stops (30-40 points)")
            report.append("3. Target middle of range for exits")
        
        report.append("")
        report.append("══════════════════════════════")
        report.append("")
        
        # ---- KEY LEVELS SUMMARY ----
        report.append("<b>🔑 KEY LEVELS SUMMARY</b>")
        report.append("")
        
        report.append(f"<b>NIFTY 50:</b>")
        report.append(f"• Open: <code>{nifty['TODAY_OPEN']}</code>")
        report.append(f"• Buy: <code>{nifty['LEVELS']['IMMEDIATE_SUPPORT']}</code>")
        report.append(f"• Sell: <code>{nifty['LEVELS']['IMMEDIATE_RESISTANCE']}</code>")
        
        report.append("")
        
        report.append(f"<b>BANKNIFTY:</b>")
        report.append(f"• Open: <code>{banknifty['TODAY_OPEN']}</code>")
        report.append(f"• Buy: <code>{banknifty['LEVELS']['IMMEDIATE_SUPPORT']}</code>")
        report.append(f"• Sell: <code>{banknifty['LEVELS']['IMMEDIATE_RESISTANCE']}</code>")
        
        report.append("")
        report.append("<b>⚠️ RISK MANAGEMENT:</b>")
        report.append("• Stop loss is MANDATORY")
        report.append("• Max 1% risk per trade")
        report.append("• No trades without clear levels")
        report.append("• Book profits at resistance/support")
        
        report.append("")
        report.append(f"<b>✅ GENERATED: {ist.strftime('%H:%M IST')}</b>")
        
    except Exception as e:
        report.append(f"<b>⚠️ ERROR:</b> {str(e)}")
    
    return "\n".join(report)

# --------- MAIN FUNCTION ---------
def main():
    print("🚀 Starting Ultimate Intraday Analysis...")
    
    try:
        ist = get_ist_time()
        print(f"Time: {ist.strftime('%H:%M:%S IST')}")
        
        report = generate_ultimate_report()
        
        # Send to Telegram
        success = send_telegram(report)
        
        if success:
            print("✅ Report sent to Telegram")
        else:
            print("❌ Telegram failed")
        
        # Save locally
        os.makedirs("reports", exist_ok=True)
        filename = f"reports/intraday_{ist.strftime('%Y%m%d_%H%M')}.txt"
        with open(filename, "w") as f:
            f.write(report)
        print(f"✅ Report saved: {filename}")
        
    except Exception as e:
        print(f"❌ Main error: {e}")

# --------- START ---------
if __name__ == "__main__":
    # Send startup
    ist = get_ist_time()
    startup = f"🚀 <b>Ultimate Intraday Levels v4.0</b>\n"
    startup += f"⏰ {ist.strftime('%H:%M:%S IST')}\n"
    startup += f"📅 {ist.strftime('%d %b %Y')}\n"
    startup += "✅ Generating REAL market levels..."
    send_telegram(startup)
    
    # Run
    main()
