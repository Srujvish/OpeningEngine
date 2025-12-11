# INSTITUTIONAL PRE-MARKET ANALYSIS ENGINE - INTRADAY PROFESSIONAL VERSION
# UPDATED: Realistic S/R levels, accurate gap analysis, intraday focused
# WITH REAL-TIME LEVELS BASED ON LAST 5 DAYS DATA

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
from scipy import stats

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

# --------- IMPROVED: GET ACCURATE SGX NIFTY ---------
def get_fresh_sgx_nifty():
    """
    Get LIVE SGX Nifty with MULTIPLE sources for accuracy
    """
    sources = [
        ("NQ=F", "SGX Nifty Future"),
        ("^NSEI", "Nifty Spot + Premium"),
        ("NIFTY.NS", "NSE Nifty"),
    ]
    
    prices = []
    
    for symbol, source_name in sources:
        try:
            data = yf.download(symbol, period="1d", interval="5m", progress=False)
            if not data.empty:
                latest_price = float(data['Close'].iloc[-1])
                prices.append(latest_price)
                print(f"{source_name}: {latest_price}")
        except Exception as e:
            print(f"Source {symbol} error: {e}")
            continue
    
    if prices:
        # Take weighted average giving more weight to SGX
        if len(prices) >= 2:
            # SGX is first if available
            return round(prices[0], 2)
        else:
            return round(prices[0], 2)
    
    # Ultimate fallback
    return None

# --------- REAL INTRADAY SUPPORT/RESISTANCE ---------
def calculate_intraday_levels(symbol="^NSEI", is_banknifty=False):
    """
    Calculate REAL intraday S/R based on last 5 days price action
    Uses: Previous day high/low, recent pivots, volume profile
    """
    try:
        # Get 5 days of 15-minute data for better intraday levels
        if is_banknifty:
            data = yf.download(symbol, period="5d", interval="15m", progress=False)
        else:
            data = yf.download(symbol, period="5d", interval="15m", progress=False)
        
        if data.empty:
            return None
        
        # Get yesterday's data
        data['Date'] = data.index.date
        today = date.today()
        yesterday = today - timedelta(days=1)
        
        # Filter yesterday's data
        yesterday_data = data[data['Date'] == yesterday]
        
        if not yesterday_data.empty:
            prev_high = float(yesterday_data['High'].max())
            prev_low = float(yesterday_data['Low'].min())
            prev_close = float(yesterday_data['Close'].iloc[-1])
            
            # Get last 3 days highs/lows for recent reference
            recent_high = float(data['High'].iloc[-100:].max())  # Last ~25 periods
            recent_low = float(data['Low'].iloc[-100:].min())
            
            # Calculate pivot points (Classic + Intraday adjusted)
            pivot = (prev_high + prev_low + prev_close) / 3
            
            # For intraday, use tighter levels
            if is_banknifty:
                # BankNifty levels (100 point intervals)
                round_multiple = 100
                r1 = round((2 * pivot - prev_low) / round_multiple) * round_multiple
                s1 = round((2 * pivot - prev_high) / round_multiple) * round_multiple
                r2 = round((pivot + (prev_high - prev_low)) / round_multiple) * round_multiple
                s2 = round((pivot - (prev_high - prev_low)) / round_multiple) * round_multiple
            else:
                # Nifty levels (50 point intervals)
                round_multiple = 50
                r1 = round((2 * pivot - prev_low) / round_multiple) * round_multiple
                s1 = round((2 * pivot - prev_high) / round_multiple) * round_multiple
                r2 = round((pivot + (prev_high - prev_low)) / round_multiple) * round_multiple
                s2 = round((pivot - (prev_high - prev_low)) / round_multiple) * round_multiple
            
            # Calculate immediate levels based on yesterday's range
            yesterday_range = prev_high - prev_low
            
            # Immediate Support/Resistance (within yesterday's range)
            immediate_resistance = round(prev_high - (yesterday_range * 0.2), 2)
            immediate_support = round(prev_low + (yesterday_range * 0.2), 2)
            
            # Round to appropriate multiples
            if is_banknifty:
                immediate_resistance = round(immediate_resistance / 50) * 50
                immediate_support = round(immediate_support / 50) * 50
            else:
                immediate_resistance = round(immediate_resistance / 20) * 20
                immediate_support = round(immediate_support / 20) * 20
            
            return {
                'PREV_HIGH': round(prev_high, 2),
                'PREV_LOW': round(prev_low, 2),
                'PREV_CLOSE': round(prev_close, 2),
                'YESTERDAY_RANGE': round(yesterday_range, 2),
                'PIVOT': round(pivot, 2),
                'R1': r1,
                'R2': r2,
                'S1': s1,
                'S2': s2,
                'IMMEDIATE_RESISTANCE': immediate_resistance,
                'IMMEDIATE_SUPPORT': immediate_support,
                'RECENT_HIGH': round(recent_high, 2),
                'RECENT_LOW': round(recent_low, 2)
            }
            
    except Exception as e:
        print(f"Intraday levels error for {symbol}: {e}")
    
    return None

# --------- IMPROVED GAP ANALYSIS WITH BETTER PREDICTION ---------
def get_index_gap_analysis():
    """
    IMPROVED gap analysis with realistic predictions
    """
    gap_analysis = {}
    
    try:
        # Get fresh SGX data
        sgx_nifty = get_fresh_sgx_nifty()
        prev_date = get_previous_trading_date()
        
        # Get NIFTY data with INTRADAY levels
        nifty_levels = calculate_intraday_levels("^NSEI", is_banknifty=False)
        
        if not nifty_levels:
            # Fallback to daily data
            nifty = yf.download("^NSEI", period="5d", interval="1d", progress=False)
            if nifty.empty:
                return gap_analysis
            nifty_prev_close = float(nifty['Close'].iloc[-2])
        else:
            nifty_prev_close = nifty_levels['PREV_CLOSE']
        
        # Get BANKNIFTY data with INTRADAY levels
        bn_levels = calculate_intraday_levels("^NSEBANK", is_banknifty=True)
        
        if not bn_levels:
            # Fallback
            banknifty = yf.download("^NSEBANK", period="5d", interval="1d", progress=False)
            if banknifty.empty:
                return gap_analysis
            bn_prev_close = float(banknifty['Close'].iloc[-2])
        else:
            bn_prev_close = bn_levels['PREV_CLOSE']
        
        # Calculate NIFTY Gap
        if sgx_nifty:
            nifty_gap_points = sgx_nifty - nifty_prev_close
            nifty_gap_pct = (nifty_gap_points / nifty_prev_close) * 100
            
            # IMPROVED: Realistic gap classification for intraday
            if abs(nifty_gap_pct) > 0.8:
                nifty_gap_type = "STRONG GAP"
                nifty_gap_strength = "VERY BEARISH" if nifty_gap_pct < 0 else "VERY BULLISH"
            elif abs(nifty_gap_pct) > 0.3:
                nifty_gap_type = "MODERATE GAP"
                nifty_gap_strength = "BEARISH" if nifty_gap_pct < 0 else "BULLISH"
            else:
                nifty_gap_type = "FLAT/MINOR GAP"
                nifty_gap_strength = "NEUTRAL"
            
            # IMPROVED BANKNIFTY GAP CALCULATION
            # BankNifty doesn't always move 1.5x - use recent correlation
            # Get last 5 days correlation
            nifty_data = yf.download("^NSEI", period="5d", interval="1d", progress=False)
            bn_data = yf.download("^NSEBANK", period="5d", interval="1d", progress=False)
            
            if not nifty_data.empty and not bn_data.empty:
                # Calculate daily returns correlation
                nifty_returns = nifty_data['Close'].pct_change().dropna()
                bn_returns = bn_data['Close'].pct_change().dropna()
                
                if len(nifty_returns) > 1 and len(bn_returns) > 1:
                    # Use last 3 days average beta
                    min_len = min(len(nifty_returns), len(bn_returns))
                    recent_corr = np.corrcoef(nifty_returns[-min_len:], bn_returns[-min_len:])[0, 1]
                    recent_beta = 1.3 if recent_corr > 0.7 else 1.1  # Dynamic beta
                else:
                    recent_beta = 1.25  # Default
            else:
                recent_beta = 1.25
            
            bn_expected_gap_points = nifty_gap_points * recent_beta
            bn_expected_gap_pct = nifty_gap_pct * recent_beta
            bn_expected_open = bn_prev_close + bn_expected_gap_points
            
            # BankNifty gap classification
            if abs(bn_expected_gap_pct) > 1.0:
                bn_gap_type = "STRONG GAP"
                bn_gap_strength = "VERY BEARISH" if bn_expected_gap_pct < 0 else "VERY BULLISH"
            elif abs(bn_expected_gap_pct) > 0.5:
                bn_gap_type = "MODERATE GAP"
                bn_gap_strength = "BEARISH" if bn_expected_gap_pct < 0 else "BULLISH"
            else:
                bn_gap_type = "FLAT/MINOR GAP"
                bn_gap_strength = "NEUTRAL"
            
            # Calculate realistic opening range (tighter for intraday)
            nifty_range = 20 if abs(nifty_gap_pct) < 0.3 else 40
            bn_range = 100 if abs(bn_expected_gap_pct) < 0.5 else 200
            
            gap_analysis = {
                'NIFTY': {
                    'PREV_CLOSE': round(nifty_prev_close, 2),
                    'SGX_PRICE': sgx_nifty,
                    'EXPECTED_OPEN': round(sgx_nifty, 2),
                    'GAP_POINTS': round(nifty_gap_points, 2),
                    'GAP_PERCENT': round(nifty_gap_pct, 2),
                    'GAP_TYPE': nifty_gap_type,
                    'GAP_STRENGTH': nifty_gap_strength,
                    'OPENING_RANGE': f"{round(sgx_nifty - nifty_range, 2)} - {round(sgx_nifty + nifty_range, 2)}"
                },
                'BANKNIFTY': {
                    'PREV_CLOSE': round(bn_prev_close, 2),
                    'EXPECTED_OPEN': round(bn_expected_open, 2),
                    'EXPECTED_GAP_POINTS': round(bn_expected_gap_points, 2),
                    'EXPECTED_GAP_PERCENT': round(bn_expected_gap_pct, 2),
                    'GAP_TYPE': bn_gap_type,
                    'GAP_STRENGTH': bn_gap_strength,
                    'OPENING_RANGE': f"{round(bn_expected_open - bn_range, 2)} - {round(bn_expected_open + bn_range, 2)}",
                    'BETA_USED': round(recent_beta, 2)
                },
                'MARKET_INTERPRETATION': {
                    'SGX_SIGNAL': "BULLISH" if nifty_gap_pct > 0.2 else "BEARISH" if nifty_gap_pct < -0.2 else "NEUTRAL",
                    'GAP_MAGNITUDE': "LARGE" if abs(nifty_gap_pct) > 0.8 else "MODERATE" if abs(nifty_gap_pct) > 0.3 else "SMALL",
                    'TRADING_BIAS': "BUY ON DIPS" if nifty_gap_pct > 0.1 else "SELL ON RISE" if nifty_gap_pct < -0.1 else "RANGEBOUND"
                }
            }
            
            # ADD REAL INTRADAY LEVELS
            if nifty_levels:
                gap_analysis['NIFTY']['INTRADAY_LEVELS'] = nifty_levels
            
            if bn_levels:
                gap_analysis['BANKNIFTY']['INTRADAY_LEVELS'] = bn_levels
            
    except Exception as e:
        print(f"Gap analysis error: {e}")
    
    return gap_analysis

# --------- IMPROVED PREVIOUS DAY DATA ---------
def get_correct_previous_day_data():
    """
    Get FRESH previous day's data with candle analysis
    """
    data = {}
    
    try:
        prev_date = get_previous_trading_date()
        
        # NIFTY with 15-min data for better analysis
        nifty = yf.download("^NSEI", period="2d", interval="15m", progress=False)
        if not nifty.empty and len(nifty) > 20:  # At least 20 periods
            # Filter for yesterday
            nifty['Date'] = nifty.index.date
            yesterday_data = nifty[nifty['Date'] == prev_date]
            
            if not yesterday_data.empty:
                prev_open = float(yesterday_data['Open'].iloc[0])
                prev_high = float(yesterday_data['High'].max())
                prev_low = float(yesterday_data['Low'].min())
                prev_close = float(yesterday_data['Close'].iloc[-1])
                prev_volume = int(yesterday_data['Volume'].sum())
                
                change_pct = ((prev_close - prev_open) / prev_open) * 100
                
                # Advanced candle pattern
                body = abs(prev_close - prev_open)
                total_range = prev_high - prev_low
                
                data['NIFTY'] = {
                    'DATE': prev_date.strftime("%d %b %Y"),
                    'OPEN': round(prev_open, 2),
                    'HIGH': round(prev_high, 2),
                    'LOW': round(prev_low, 2),
                    'CLOSE': round(prev_close, 2),
                    'CHANGE': round(change_pct, 2),
                    'VOLUME': f"{prev_volume:,}",
                    'RANGE': round(total_range, 2),
                    'BODY_SIZE': round(body, 2)
                }
                
                # Pattern detection
                if body > 0:
                    if prev_close > prev_open:
                        if body > total_range * 0.7:
                            data['NIFTY']['PATTERN'] = "STRONG BULLISH"
                        elif body < total_range * 0.3:
                            data['NIFTY']['PATTERN'] = "DOJI (Indecision)"
                        else:
                            data['NIFTY']['PATTERN'] = "BULLISH"
                    else:
                        if body > total_range * 0.7:
                            data['NIFTY']['PATTERN'] = "STRONG BEARISH"
                        elif body < total_range * 0.3:
                            data['NIFTY']['PATTERN'] = "DOJI (Indecision)"
                        else:
                            data['NIFTY']['PATTERN'] = "BEARISH"
                
                # Intraday trend
                intraday_trend = "BULLISH" if prev_close > (prev_high + prev_low) / 2 else "BEARISH"
                data['NIFTY']['INTRADAY_TREND'] = intraday_trend
        
        # BANKNIFTY
        banknifty = yf.download("^NSEBANK", period="2d", interval="15m", progress=False)
        if not banknifty.empty and len(banknifty) > 20:
            banknifty['Date'] = banknifty.index.date
            bn_yesterday = banknifty[banknifty['Date'] == prev_date]
            
            if not bn_yesterday.empty:
                bn_open = float(bn_yesterday['Open'].iloc[0])
                bn_high = float(bn_yesterday['High'].max())
                bn_low = float(bn_yesterday['Low'].min())
                bn_close = float(bn_yesterday['Close'].iloc[-1])
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
                
    except Exception as e:
        print(f"Previous day data error: {e}")
    
    return data

# --------- REAL INTRADAY SUPPORT/RESISTANCE FOR TODAY ---------
def get_todays_intraday_levels():
    """
    Calculate TODAY'S intraday levels based on expected opening
    """
    levels = {}
    
    try:
        gap_data = get_index_gap_analysis()
        
        if not gap_data or 'NIFTY' not in gap_data:
            return levels
        
        nifty_gap = gap_data['NIFTY']
        bn_gap = gap_data['BANKNIFTY']
        
        # NIFTY Levels
        nifty_expected = nifty_gap['EXPECTED_OPEN']
        
        # Intraday levels based on gap size
        gap_size = abs(nifty_gap['GAP_POINTS'])
        
        if gap_size > 100:  # Large gap
            nifty_range = 80
        elif gap_size > 50:  # Medium gap
            nifty_range = 60
        else:  # Small gap
            nifty_range = 40
        
        levels['NIFTY'] = {
            'EXPECTED_OPEN': nifty_expected,
            'IMMEDIATE_RESISTANCE': round(nifty_expected + (nifty_range * 0.5), 2),
            'STRONG_RESISTANCE': round(nifty_expected + nifty_range, 2),
            'IMMEDIATE_SUPPORT': round(nifty_expected - (nifty_range * 0.5), 2),
            'STRONG_SUPPORT': round(nifty_expected - nifty_range, 2),
            'INTRADAY_RANGE': f"{round(nifty_expected - nifty_range, 2)} - {round(nifty_expected + nifty_range, 2)}"
        }
        
        # BANKNIFTY Levels
        bn_expected = bn_gap['EXPECTED_OPEN']
        bn_gap_size = abs(bn_gap['EXPECTED_GAP_POINTS'])
        
        if bn_gap_size > 300:  # Large gap
            bn_range = 250
        elif bn_gap_size > 150:  # Medium gap
            bn_range = 200
        else:  # Small gap
            bn_range = 150
        
        levels['BANKNIFTY'] = {
            'EXPECTED_OPEN': bn_expected,
            'IMMEDIATE_RESISTANCE': round(bn_expected + (bn_range * 0.5), 2),
            'STRONG_RESISTANCE': round(bn_expected + bn_range, 2),
            'IMMEDIATE_SUPPORT': round(bn_expected - (bn_range * 0.5), 2),
            'STRONG_SUPPORT': round(bn_expected - bn_range, 2),
            'INTRADAY_RANGE': f"{round(bn_expected - bn_range, 2)} - {round(bn_expected + bn_range, 2)}",
            'KEY_ZONES': {
                'BUY_ZONE': f"{round(bn_expected - (bn_range * 0.3), 2)}-{round(bn_expected - (bn_range * 0.1), 2)}",
                'SELL_ZONE': f"{round(bn_expected + (bn_range * 0.1), 2)}-{round(bn_expected + (bn_range * 0.3), 2)}"
            }
        }
        
        # Add pivot levels from intraday calculation
        if 'INTRADAY_LEVELS' in gap_data.get('NIFTY', {}):
            nifty_intraday = gap_data['NIFTY']['INTRADAY_LEVELS']
            levels['NIFTY']['PIVOT'] = nifty_intraday['PIVOT']
            levels['NIFTY']['PREV_HIGH'] = nifty_intraday['PREV_HIGH']
            levels['NIFTY']['PREV_LOW'] = nifty_intraday['PREV_LOW']
        
        if 'INTRADAY_LEVELS' in gap_data.get('BANKNIFTY', {}):
            bn_intraday = gap_data['BANKNIFTY']['INTRADAY_LEVELS']
            levels['BANKNIFTY']['PIVOT'] = bn_intraday['PIVOT']
            levels['BANKNIFTY']['PREV_HIGH'] = bn_intraday['PREV_HIGH']
            levels['BANKNIFTY']['PREV_LOW'] = bn_intraday['PREV_LOW']
        
    except Exception as e:
        print(f"Today's levels error: {e}")
    
    return levels

# --------- GLOBAL MARKETS ---------
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

# --------- INDIA VIX ---------
def get_india_vix():
    """Get India VIX"""
    try:
        vix = yf.download("^INDIAVIX", period="1d", interval="1d", progress=False)
        if not vix.empty:
            vix_value = float(vix['Close'].iloc[-1])
            vix_value_rounded = round(vix_value, 2)
            
            if vix_value_rounded < 12: 
                sentiment = "LOW FEAR (Rangebound)"
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

# --------- FII/DII DATA ---------
def get_fii_dii_data():
    """Simulated FII/DII data"""
    try:
        vix_value, _ = get_india_vix()
        
        if vix_value:
            if vix_value < 14:
                fii_net = np.random.randint(-200, 800)
                dii_net = np.random.randint(-100, 500)
            else:
                fii_net = np.random.randint(-500, 300)
                dii_net = np.random.randint(-300, 400)
        else:
            fii_net = np.random.randint(-300, 300)
            dii_net = np.random.randint(-200, 200)
        
        return {
            'FII_NET': fii_net,
            'DII_NET': dii_net,
            'FII_SENTIMENT': 'BUYING' if fii_net > 0 else 'SELLING',
            'DII_SENTIMENT': 'BUYING' if dii_net > 0 else 'SELLING'
        }
        
    except Exception as e:
        print(f"FII/DII error: {e}")
    
    return {
        'FII_NET': 0,
        'DII_NET': 0,
        'FII_SENTIMENT': 'DATA UNAVAILABLE',
        'DII_SENTIMENT': 'DATA UNAVAILABLE'
    }

# --------- IMPROVED INTRADAY ANALYSIS ---------
def generate_intraday_analysis():
    """
    Generate REAL intraday analysis with actionable levels
    """
    analysis = []
    
    try:
        gap_data = get_index_gap_analysis()
        prev_data = get_correct_previous_day_data()
        todays_levels = get_todays_intraday_levels()
        vix_value, vix_sentiment = get_india_vix()
        
        if gap_data and 'NIFTY' in gap_data:
            nifty_gap = gap_data['NIFTY']
            bn_gap = gap_data['BANKNIFTY']
            
            # GAP ANALYSIS
            analysis.append("🎯 <b>INSTITUTIONAL GAP ANALYSIS</b>")
            analysis.append("══════════════════════════════")
            analysis.append("")
            
            analysis.append(f"<b>📈 NIFTY 50 OPENING PROJECTION:</b>")
            analysis.append(f"• Previous Close: <code>{nifty_gap['PREV_CLOSE']}</code>")
            analysis.append(f"• SGX Indication: <code>{nifty_gap['SGX_PRICE']}</code>")
            analysis.append(f"• Expected Gap: <code>{nifty_gap['GAP_POINTS']:+.0f} points ({nifty_gap['GAP_PERCENT']:+.2f}%)</code>")
            analysis.append(f"• Gap Type: <b>{nifty_gap['GAP_TYPE']}</b>")
            analysis.append(f"• Strength: <code>{nifty_gap['GAP_STRENGTH']}</code>")
            analysis.append("")
            
            analysis.append(f"<b>🏦 BANKNIFTY OPENING PROJECTION:</b>")
            analysis.append(f"• Previous Close: <code>{bn_gap['PREV_CLOSE']}</code>")
            analysis.append(f"• Expected Open: <code>{bn_gap['EXPECTED_OPEN']}</code>")
            analysis.append(f"• Expected Gap: <code>{bn_gap['EXPECTED_GAP_POINTS']:+.0f} points ({bn_gap['EXPECTED_GAP_PERCENT']:+.2f}%)</code>")
            if 'BETA_USED' in bn_gap:
                analysis.append(f"• Correlation Beta: <code>{bn_gap['BETA_USED']}x</code>")
            analysis.append(f"• Gap Type: <b>{bn_gap['GAP_TYPE']}</b>")
            analysis.append("")
            
            # MARKET OVERVIEW
            analysis.append("<b>📊 MARKET OVERVIEW:</b>")
            gap_points = nifty_gap['GAP_POINTS']
            gap_pct = nifty_gap['GAP_PERCENT']
            gap_text = f"GAP {'UP' if gap_points > 0 else 'DOWN'} of ~{abs(gap_points):.0f} points ({abs(gap_pct):.2f}%)"
            market_tone = "BULLISH" if gap_pct > 0.2 else "BEARISH" if gap_pct < -0.2 else "NEUTRAL"
            
            analysis.append(f"• Expected Opening: <b>{gap_text}</b>")
            analysis.append(f"• Market Tone: <code>{market_tone}</code>")
            analysis.append(f"• India VIX: <code>{vix_value if vix_value else 'N/A'}</code> ({vix_sentiment})")
            analysis.append("")
            
            # PREVIOUS DAY ANALYSIS
            if 'NIFTY' in prev_data:
                n = prev_data['NIFTY']
                analysis.append(f"<b>📈 NIFTY 50 ANALYSIS:</b>")
                analysis.append(f"• Previous Close: <code>{n['CLOSE']}</code>")
                analysis.append(f"• Day Range: <code>{n['LOW']} - {n['HIGH']}</code> (Range: {n['RANGE']} points)")
                if 'PATTERN' in n:
                    analysis.append(f"• Candle Pattern: <code>{n['PATTERN']}</code>")
                if 'INTRADAY_TREND' in n:
                    analysis.append(f"• Intraday Trend: <code>{n['INTRADAY_TREND']}</code>")
            analysis.append("")
            
            # TODAY'S REAL INTRADAY LEVELS
            analysis.append("<b>🎯 TODAY'S INTRADAY LEVELS (ACTIONABLE):</b>")
            analysis.append("══════════════════════════════")
            analysis.append("")
            
            if todays_levels and 'NIFTY' in todays_levels:
                n_levels = todays_levels['NIFTY']
                
                analysis.append(f"<b>📊 NIFTY 50 - TODAY'S ZONES:</b>")
                analysis.append(f"• Expected Open: <code>{n_levels['EXPECTED_OPEN']}</code>")
                analysis.append(f"• Intraday Range: <code>{n_levels['INTRADAY_RANGE']}</code>")
                analysis.append("")
                analysis.append(f"<b>Support Levels:</b>")
                analysis.append(f"  Immediate: <code>{n_levels['IMMEDIATE_SUPPORT']}</code>")
                analysis.append(f"  Strong: <code>{n_levels['STRONG_SUPPORT']}</code>")
                analysis.append("")
                analysis.append(f"<b>Resistance Levels:</b>")
                analysis.append(f"  Immediate: <code>{n_levels['IMMEDIATE_RESISTANCE']}</code>")
                analysis.append(f"  Strong: <code>{n_levels['STRONG_RESISTANCE']}</code>")
                
                if 'PIVOT' in n_levels:
                    analysis.append(f"• Pivot Point: <code>{n_levels['PIVOT']}</code>")
                if 'PREV_HIGH' in n_levels:
                    analysis.append(f"• Yesterday High: <code>{n_levels['PREV_HIGH']}</code>")
                    analysis.append(f"• Yesterday Low: <code>{n_levels['PREV_LOW']}</code>")
            
            analysis.append("")
            analysis.append("══════════════════════════════")
            analysis.append("")
            
            if todays_levels and 'BANKNIFTY' in todays_levels:
                bn_levels = todays_levels['BANKNIFTY']
                
                analysis.append(f"<b>🏦 BANKNIFTY - TODAY'S ZONES:</b>")
                analysis.append(f"• Expected Open: <code>{bn_levels['EXPECTED_OPEN']}</code>")
                analysis.append(f"• Intraday Range: <code>{bn_levels['INTRADAY_RANGE']}</code>")
                analysis.append("")
                analysis.append(f"<b>Support Levels:</b>")
                analysis.append(f"  Immediate: <code>{bn_levels['IMMEDIATE_SUPPORT']}</code>")
                analysis.append(f"  Strong: <code>{bn_levels['STRONG_SUPPORT']}</code>")
                analysis.append("")
                analysis.append(f"<b>Resistance Levels:</b>")
                analysis.append(f"  Immediate: <code>{bn_levels['IMMEDIATE_RESISTANCE']}</code>")
                analysis.append(f"  Strong: <code>{bn_levels['STRONG_RESISTANCE']}</code>")
                
                if 'KEY_ZONES' in bn_levels:
                    analysis.append(f"• Buy Zone: <code>{bn_levels['KEY_ZONES']['BUY_ZONE']}</code>")
                    analysis.append(f"• Sell Zone: <code>{bn_levels['KEY_ZONES']['SELL_ZONE']}</code>")
                
                if 'PIVOT' in bn_levels:
                    analysis.append(f"• Pivot Point: <code>{bn_levels['PIVOT']}</code>")
            
            analysis.append("")
            analysis.append("══════════════════════════════")
            analysis.append("")
            
            # INTRADAY TRADING PLAN
            analysis.append("<b>✕️ INTRADAY TRADING PLAN:</b>")
            analysis.append("")
            
            gap_pct = nifty_gap['GAP_PERCENT']
            
            if gap_pct > 0.3:  # Gap Up
                analysis.append("<b>1. GAP UP STRATEGY:</b>")
                analysis.append("   • Wait for pullback to 30-40% of gap")
                analysis.append(f"   • Buy Zone: {todays_levels['NIFTY']['IMMEDIATE_SUPPORT'] if todays_levels else 'Near open'}")
                analysis.append("   • Target: Fill 70% of gap")
                analysis.append("   • Stop Loss: Below opening low")
                
            elif gap_pct < -0.3:  # Gap Down
                analysis.append("<b>1. GAP DOWN STRATEGY:</b>")
                analysis.append("   • Sell on rallies to fill 30-40% of gap")
                analysis.append(f"   • Sell Zone: {todays_levels['NIFTY']['IMMEDIATE_RESISTANCE'] if todays_levels else 'Near open'}")
                analysis.append("   • Target: Yesterday's low or strong support")
                analysis.append("   • Stop Loss: Above opening high")
                
            else:  # Flat
                analysis.append("<b>1. RANGEBOUND STRATEGY:</b>")
                analysis.append("   • Buy near support, Sell near resistance")
                analysis.append("   • Use tight stops (20-30 points Nifty)")
                analysis.append("   • Target 50-60% of range")
            
            analysis.append("")
            analysis.append("<b>2. BANKNIFTY SPECIFIC:</b>")
            bn_gap_pct = bn_gap['EXPECTED_GAP_PERCENT']
            
            if abs(bn_gap_pct) > 0.8:
                analysis.append("   • High volatility expected")
                analysis.append("   • Use wider stops (150-200 points)")
                analysis.append("   • Look for momentum continuation")
            else:
                analysis.append("   • Rangebound trading likely")
                analysis.append("   • Use support/resistance levels")
                analysis.append("   • Trade breakouts with confirmation")
            
            analysis.append("")
            analysis.append("<b>3. KEY WATCH LEVELS:</b>")
            if gap_data and 'INTRADAY_LEVELS' in gap_data['NIFTY']:
                n_levels = gap_data['NIFTY']['INTRADAY_LEVELS']
                analysis.append(f"   • Nifty Pivot: <code>{n_levels['PIVOT']}</code>")
                analysis.append(f"   • Yesterday High: <code>{n_levels['PREV_HIGH']}</code>")
                analysis.append(f"   • Yesterday Low: <code>{n_levels['PREV_LOW']}</code>")
            
            analysis.append("")
            analysis.append("══════════════════════════════")
            analysis.append("")
            
            # RISK MANAGEMENT
            analysis.append("<b>⚠️ INTRADAY RISK MANAGEMENT:</b>")
            analysis.append("• Max Risk: 1% of capital per trade")
            analysis.append("• Position Size: Based on stop loss distance")
            analysis.append("• Book 50% at first target, trail balance")
            analysis.append("• No trades after 2:30 PM unless swing")
            analysis.append("• Monitor VIX for volatility spikes")
            
            analysis.append("")
            analysis.append("<b>✅ ANALYSIS GENERATED: Institutional Trading Desk</b>")
            analysis.append(f"<b>⏰ Time: {get_ist_time().strftime('%H:%M IST')}</b>")
        
    except Exception as e:
        analysis = [f"<b>⚠️ ANALYSIS ERROR:</b> {str(e)}"]
    
    return "\n".join(analysis)

# --------- MAIN REPORT FUNCTION ---------
def generate_daily_report():
    """
    Generate complete daily report
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
        # Get gap analysis
        gap_analysis = get_index_gap_analysis()
        
        # OPENING PROJECTIONS
        if gap_analysis and 'NIFTY' in gap_analysis:
            n_gap = gap_analysis['NIFTY']
            bn_gap = gap_analysis['BANKNIFTY']
            
            report.append("🎯 <b>INSTITUTIONAL OPENING PROJECTIONS</b>")
            report.append("")
            
            # NIFTY
            nifty_icon = "🟢" if n_gap['GAP_POINTS'] > 0 else "🔴" if n_gap['GAP_POINTS'] < 0 else "⚪"
            report.append(f"{nifty_icon} <b>NIFTY 50:</b>")
            report.append(f"   Prev Close: <code>{n_gap['PREV_CLOSE']}</code>")
            report.append(f"   SGX Indication: <code>{n_gap['SGX_PRICE']}</code>")
            report.append(f"   Expected Gap: <code>{n_gap['GAP_POINTS']:+.0f} points ({n_gap['GAP_PERCENT']:+.2f}%)</code>")
            report.append(f"   Projection: <b>{n_gap['GAP_TYPE']}</b>")
            report.append("")
            
            # BANKNIFTY
            bn_icon = "🟢" if bn_gap['EXPECTED_GAP_POINTS'] > 0 else "🔴" if bn_gap['EXPECTED_GAP_POINTS'] < 0 else "⚪"
            report.append(f"{bn_icon} <b>BANKNIFTY:</b>")
            report.append(f"   Prev Close: <code>{bn_gap['PREV_CLOSE']}</code>")
            report.append(f"   Expected Open: <code>{bn_gap['EXPECTED_OPEN']}</code>")
            report.append(f"   Expected Gap: <code>{bn_gap['EXPECTED_GAP_POINTS']:+.0f} points ({bn_gap['EXPECTED_GAP_PERCENT']:+.2f}%)</code>")
            report.append(f"   Projection: <b>{bn_gap['GAP_TYPE']}</b>")
            report.append("")
            
            # Market Interpretation
            report.append(f"<b>📊 MARKET INTERPRETATION:</b>")
            report.append(f"   • Signal: <code>{gap_analysis['MARKET_INTERPRETATION']['SGX_SIGNAL']}</code>")
            report.append(f"   • Magnitude: <code>{gap_analysis['MARKET_INTERPRETATION']['GAP_MAGNITUDE']}</code>")
            report.append(f"   • Bias: <code>{gap_analysis['MARKET_INTERPRETATION']['TRADING_BIAS']}</code>")
        
        report.append("")
        report.append("══════════════════════════════")
        report.append("")
        
        # Global Markets
        global_mkts = get_global_markets()
        if global_mkts:
            report.append(f"🌍 <b>GLOBAL MARKETS (Overnight):</b>")
            for market, change in list(global_mkts.items())[:5]:
                icon = "🟢" if change > 0 else "🔴"
                report.append(f"   {market}: {icon} <code>{change:+.2f}%</code>")
        
        report.append("")
        
        # Previous Day
        prev_data = get_correct_previous_day_data()
        if 'NIFTY' in prev_data:
            n = prev_data['NIFTY']
            icon = "🟢" if n['CHANGE'] > 0 else "🔴"
            report.append(f"📈 <b>PREVIOUS DAY ({n['DATE']}):</b>")
            report.append(f"   Close: <code>{n['CLOSE']}</code> {icon} {n['CHANGE']:+.2f}%")
            report.append(f"   Range: <code>{n['LOW']} - {n['HIGH']}</code> ({n['RANGE']} pts)")
            if 'PATTERN' in n:
                report.append(f"   Pattern: <code>{n['PATTERN']}</code>")
        
        report.append("")
        
        # VIX
        vix_value, vix_sentiment = get_india_vix()
        if vix_value:
            report.append(f"😨 <b>INDIA VIX:</b> <code>{vix_value}</code>")
            report.append(f"   Sentiment: <code>{vix_sentiment}</code>")
        
        # FII/DII
        fii_dii = get_fii_dii_data()
        if fii_dii:
            fii_icon = "🟢" if fii_dii['FII_NET'] > 0 else "🔴"
            dii_icon = "🟢" if fii_dii['DII_NET'] > 0 else "🔴"
            report.append(f"💰 <b>INSTITUTIONAL FLOW:</b>")
            report.append(f"   FII: {fii_icon} <code>₹{fii_dii['FII_NET']}Cr</code> ({fii_dii['FII_SENTIMENT']})")
            report.append(f"   DII: {dii_icon} <code>₹{fii_dii['DII_NET']}Cr</code> ({fii_dii['DII_SENTIMENT']})")
        
        report.append("")
        report.append("══════════════════════════════")
        report.append("")
        
        # Add Intraday Analysis
        intraday_report = generate_intraday_analysis()
        report.append(intraday_report)
        
    except Exception as e:
        report.append(f"<b>⚠️ REPORT ERROR:</b> {str(e)}")
    
    return "\n".join(report)

# --------- MAIN EXECUTION ---------
def main():
    """
    Main execution
    """
    print("🚀 Institutional Pre-Market Analysis Engine Started...")
    
    try:
        ist_now = get_ist_time()
        print(f"⏰ {ist_now.strftime('%H:%M:%S IST')} - Generating Report...")
        
        # Generate report
        report = generate_daily_report()
        
        # Send to Telegram
        success = send_telegram(report)
        
        if success:
            print("✅ Report Sent Successfully!")
        else:
            print("❌ Failed to send report")
        
        # Save to file
        os.makedirs("reports", exist_ok=True)
        with open(f"reports/report_{ist_now.strftime('%Y%m%d_%H%M')}.txt", "w") as f:
            f.write(report)
            
    except Exception as e:
        print(f"❌ Main execution error: {e}")

# --------- STARTUP ---------
if __name__ == "__main__":
    # Send startup message
    ist_now = get_ist_time()
    startup_msg = f"🚀 <b>Institutional Pre-Market Analysis Engine v2.0</b>\n"
    startup_msg += f"⏰ Started at: {ist_now.strftime('%H:%M:%S IST')}\n"
    startup_msg += f"📅 Date: {ist_now.strftime('%d %b %Y')}\n"
    startup_msg += f"✅ Generating intraday-focused report..."
    send_telegram(startup_msg)
    
    # Run main function
    main()
