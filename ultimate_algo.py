#!/usr/bin/env python3
"""
INSTITUTIONAL PRE-MARKET ANALYSIS ENGINE
AUTOMATIC 9 AM REPORT - BY INSTITUTIONAL TRADER
Version 2.0 - Complete with working data sources
"""

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
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import schedule
from functools import lru_cache
import hashlib

# ========== CONFIGURATION ==========
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('premarket_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Telegram Configuration
BOT_TOKEN = os.getenv("BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
CHAT_ID = os.getenv("CHAT_ID", "YOUR_CHAT_ID_HERE")
ADMIN_ID = os.getenv("ADMIN_ID", "")

# NSE Configuration
NSE_BASE_URL = "https://www.nseindia.com"
NSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Host': 'www.nseindia.com',
    'Origin': NSE_BASE_URL,
    'Referer': f'{NSE_BASE_URL}/',
    'sec-ch-ua': '"Chromium";v="118", "Google Chrome";v="118", "Not=A?Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin'
}

# ========== DATA CLASSES ==========
@dataclass
class MarketData:
    """Data class for market information"""
    symbol: str
    open_price: float
    high: float
    low: float
    close: float
    volume: int
    change_pct: float
    timestamp: datetime

@dataclass
class GlobalMarket:
    """Data class for global market data"""
    name: str
    last_price: float
    change_pct: float
    status: str  # OPEN, CLOSED

# ========== TELEGRAM FUNCTIONS ==========
def send_telegram(msg: str, parse_mode: str = "HTML") -> bool:
    """Send message to Telegram with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": CHAT_ID,
                "text": msg,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }
            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()
            logger.info("Telegram message sent successfully")
            return True
        except requests.exceptions.Timeout:
            logger.warning(f"Telegram timeout (attempt {attempt + 1}/{max_retries})")
            time.sleep(2)
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            if attempt == max_retries - 1:
                logger.error("Failed to send Telegram message after all retries")
            time.sleep(1)
    return False

def send_admin_alert(msg: str):
    """Send alert to admin"""
    if ADMIN_ID:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": ADMIN_ID,
            "text": f"🚨 ALERT: {msg}",
            "parse_mode": "HTML"
        }
        try:
            requests.post(url, json=payload, timeout=10)
        except:
            pass

# ========== NSE SESSION MANAGEMENT ==========
class NSESession:
    """Manage NSE session with cookies"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)
        self.last_refresh = None
        
    def initialize(self) -> bool:
        """Initialize NSE session with cookies"""
        try:
            # Get initial cookies
            response = self.session.get(
                f"{NSE_BASE_URL}/",
                timeout=15,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Get market data to get more cookies
            self.session.get(
                f"{NSE_BASE_URL}/api/marketStatus",
                timeout=15
            )
            
            self.last_refresh = datetime.now()
            logger.info("NSE session initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize NSE session: {e}")
            return False
    
    def refresh_if_needed(self):
        """Refresh session if it's old"""
        if not self.last_refresh or (datetime.now() - self.last_refresh).seconds > 1800:  # 30 minutes
            self.initialize()
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """Get request with session refresh"""
        self.refresh_if_needed()
        return self.session.get(url, **kwargs)

# Initialize global NSE session
nse_session = NSESession()
nse_session.initialize()

# ========== TIME FUNCTIONS ==========
def get_ist_time() -> datetime:
    """Get current IST time"""
    utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)
    ist = pytz.timezone('Asia/Kolkata')
    return utc_now.astimezone(ist)

def is_market_open() -> bool:
    """Check if Indian market is open (9:15 AM - 3:30 PM IST)"""
    ist_now = get_ist_time()
    market_open = dtime(9, 15)
    market_close = dtime(15, 30)
    return market_open <= ist_now.time() <= market_close

# ========== CACHE FUNCTIONS ==========
@lru_cache(maxsize=32)
def cached_yfinance(symbol: str, period: str = "1d", interval: str = "1d") -> pd.DataFrame:
    """Cached yfinance data fetching"""
    cache_key = f"{symbol}_{period}_{interval}_{datetime.now().strftime('%Y%m%d%H')}"
    logger.debug(f"Fetching yfinance data for {symbol}")
    return yf.download(symbol, period=period, interval=interval, progress=False, group_by='ticker')

# ========== MARKET DATA FUNCTIONS ==========

# 🚨 **1. SGX NIFTY DATA - WORKING SOURCES** 🚨
def get_sgx_nifty() -> Optional[float]:
    """
    Get SGX Nifty from reliable sources
    """
    sources = [
        # Source 1: Investing.com SGX Nifty
        {
            "name": "Investing.com",
            "url": "https://api.investing.com/api/financialdata/table/list/1055713",
            "parser": lambda data: data.get('last_close', None) if data else None
        },
        # Source 2: TradingView data
        {
            "name": "TradingView",
            "url": "https://scanner.tradingview.com/india/scan",
            "parser": lambda data: parse_tradingview_sgx(data)
        },
        # Source 3: Yahoo Finance SGX
        {
            "name": "Yahoo Finance",
            "url": None,
            "parser": lambda _: get_sgx_from_yahoo()
        }
    ]
    
    for source in sources:
        try:
            logger.info(f"Trying SGX source: {source['name']}")
            
            if source['url']:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json'
                }
                response = requests.get(source['url'], headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    price = source['parser'](data)
                    if price:
                        logger.info(f"SGX Nifty from {source['name']}: {price}")
                        return round(float(price), 2)
            
            else:
                price = source['parser'](None)
                if price:
                    logger.info(f"SGX Nifty from {source['name']}: {price}")
                    return round(float(price), 2)
                    
        except Exception as e:
            logger.warning(f"SGX source {source['name']} failed: {e}")
            continue
    
    # Fallback: Manual scraping from reliable websites
    try:
        # Try Moneycontrol
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(
            "https://www.moneycontrol.com/indian-indices/sgx-nifty-50-250.html",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            price_div = soup.find('div', {'class': 'inprice1'})
            if price_div:
                price_text = price_div.text.strip()
                price_match = re.search(r'(\d{4,5}\.\d{2})', price_text)
                if price_match:
                    return round(float(price_match.group(1)), 2)
    except Exception as e:
        logger.error(f"SGX fallback failed: {e}")
    
    return None

def parse_tradingview_sgx(data: dict) -> Optional[float]:
    """Parse TradingView data for SGX"""
    try:
        if 'data' in data and len(data['data']) > 0:
            for item in data['data']:
                if 's' in item and 'SGX' in item.get('s', ''):
                    return item.get('close', None)
    except:
        pass
    return None

def get_sgx_from_yahoo() -> Optional[float]:
    """Get SGX from Yahoo Finance"""
    try:
        # SGX Nifty futures symbol
        sgx = yf.download("NQ=F", period="1d", interval="1m", progress=False)
        if not sgx.empty:
            return float(sgx['Close'].iloc[-1])
    except:
        pass
    return None

# 🚨 **2. GLOBAL MARKETS - UPDATED SOURCES** 🚨
def get_global_markets() -> Dict[str, float]:
    """Get global market performance"""
    markets = {}
    
    try:
        # US Markets (S&P 500, Dow, Nasdaq)
        us_symbols = {
            'S&P500': '^GSPC',
            'DOW': '^DJI', 
            'NASDAQ': '^IXIC'
        }
        
        for name, symbol in us_symbols.items():
            try:
                data = cached_yfinance(symbol, period="2d", interval="1d")
                if not data.empty and len(data) >= 2:
                    prev_close = data['Close'].iloc[-2]
                    current_close = data['Close'].iloc[-1]
                    change_pct = ((current_close - prev_close) / prev_close) * 100
                    markets[name] = round(change_pct, 2)
            except Exception as e:
                logger.warning(f"Failed to get {name}: {e}")
        
        # Asian Markets
        asian_symbols = {
            'NIKKEI': '^N225',
            'HSI': '^HSI',
            'SHANGHAI': '000001.SS',
            'ASX200': '^AXJO'
        }
        
        for name, symbol in asian_symbols.items():
            try:
                data = cached_yfinance(symbol, period="1d", interval="1d")
                if not data.empty:
                    if len(data) >= 2:
                        prev_close = data['Close'].iloc[-2]
                        current_close = data['Close'].iloc[-1]
                    else:
                        prev_close = data['Open'].iloc[0]
                        current_close = data['Close'].iloc[-1]
                    
                    change_pct = ((current_close - prev_close) / prev_close) * 100
                    markets[name] = round(change_pct, 2)
            except Exception as e:
                logger.warning(f"Failed to get {name}: {e}")
        
        # European Markets (for reference)
        europe_symbols = {
            'DAX': '^GDAXI',
            'FTSE': '^FTSE',
            'CAC40': '^FCHI'
        }
        
        for name, symbol in europe_symbols.items():
            try:
                data = cached_yfinance(symbol, period="1d", interval="1d")
                if not data.empty and len(data) >= 2:
                    prev_close = data['Close'].iloc[-2]
                    current_close = data['Close'].iloc[-1]
                    change_pct = ((current_close - prev_close) / prev_close) * 100
                    markets[name] = round(change_pct, 2)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Global markets error: {e}")
    
    return markets

# 🚨 **3. PREVIOUS DAY DATA - ENHANCED** 🚨
def get_previous_day_data() -> Dict[str, Dict]:
    """Get previous day's market data with candle patterns"""
    data = {}
    
    try:
        symbols = {
            'NIFTY': '^NSEI',
            'BANKNIFTY': '^NSEBANK',
            'FINNIFTY': '^NSFINNIFTY',
            'MIDCAP': '^NSEMDCP50'
        }
        
        for name, symbol in symbols.items():
            try:
                df = cached_yfinance(symbol, period="5d", interval="1d")
                
                if not df.empty and len(df) >= 2:
                    prev_day = df.iloc[-2]
                    current_day = df.iloc[-1] if len(df) > 1 else prev_day
                    
                    # Basic data
                    day_data = {
                        'OPEN': round(float(prev_day['Open']), 2),
                        'HIGH': round(float(prev_day['High']), 2),
                        'LOW': round(float(prev_day['Low']), 2),
                        'CLOSE': round(float(prev_day['Close']), 2),
                        'VOLUME': int(prev_day['Volume']),
                        'CHANGE': round(((prev_day['Close'] - prev_day['Open']) / prev_day['Open']) * 100, 2),
                        'RANGE': round(prev_day['High'] - prev_day['Low'], 2)
                    }
                    
                    # Candle pattern analysis
                    day_data['PATTERN'] = analyze_candle_pattern(prev_day)
                    
                    # Trend analysis
                    if len(df) >= 5:
                        short_avg = df['Close'].iloc[-5:].mean()
                        long_avg = df['Close'].iloc[-10:].mean() if len(df) >= 10 else short_avg
                        day_data['TREND'] = 'BULLISH' if short_avg > long_avg else 'BEARISH'
                    
                    data[name] = day_data
                    
            except Exception as e:
                logger.warning(f"Failed to get {name} data: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Previous day data error: {e}")
    
    return data

def analyze_candle_pattern(candle: pd.Series) -> str:
    """Analyze candle pattern"""
    try:
        body = abs(candle['Close'] - candle['Open'])
        upper_wick = candle['High'] - max(candle['Close'], candle['Open'])
        lower_wick = min(candle['Close'], candle['Open']) - candle['Low']
        total_range = candle['High'] - candle['Low']
        
        if total_range == 0:
            return "DOJI"
        
        body_ratio = body / total_range
        upper_ratio = upper_wick / total_range
        lower_ratio = lower_wick / total_range
        
        is_bullish = candle['Close'] > candle['Open']
        
        # Marubozu (no wicks)
        if upper_ratio < 0.05 and lower_ratio < 0.05:
            return "BULLISH MARUBOZU" if is_bullish else "BEARISH MARUBOZU"
        
        # Doji (small body)
        elif body_ratio < 0.1:
            return "DOJI"
        
        # Hammer/Hanging man
        elif lower_ratio > 0.6 and body_ratio < 0.3:
            return "HAMMER" if is_bullish else "HANGING MAN"
        
        # Shooting star/Inverted hammer
        elif upper_ratio > 0.6 and body_ratio < 0.3:
            return "INVERTED HAMMER" if is_bullish else "SHOOTING STAR"
        
        # Normal candle
        else:
            return "BULLISH" if is_bullish else "BEARISH"
            
    except:
        return "UNKNOWN"

# 🚨 **4. INDIA VIX - WORKING** 🚨
def get_india_vix() -> Tuple[Optional[float], str]:
    """Get India VIX data"""
    try:
        vix_data = cached_yfinance("^INDIAVIX", period="1d", interval="1d")
        
        if not vix_data.empty:
            vix_value = round(float(vix_data['Close'].iloc[-1]), 2)
            
            # Interpretation
            if vix_value < 12:
                sentiment = "LOW FEAR (Rangebound Market)"
            elif vix_value < 18:
                sentiment = "NORMAL VOLATILITY"
            elif vix_value < 25:
                sentiment = "HIGH FEAR (Volatile)"
            else:
                sentiment = "EXTREME FEAR (High Volatility Expected)"
            
            return vix_value, sentiment
            
    except Exception as e:
        logger.error(f"VIX error: {e}")
    
    return None, "UNAVAILABLE"

# 🚨 **5. FII/DII DATA - WORKING SOURCE** 🚨
def get_fii_dii_data() -> Optional[Dict]:
    """Get FII/DII data from reliable source"""
    try:
        # Use Moneycontrol API (more reliable)
        today = datetime.now().strftime("%d-%m-%Y")
        url = f"https://www.moneycontrol.com/technicals/fii_dii/fii_dii_data.json?date={today}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json',
            'Referer': 'https://www.moneycontrol.com/'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            # Parse the response
            if isinstance(data, list) and len(data) > 0:
                latest = data[0]  # Most recent data
                
                result = {
                    'FII_NET': float(latest.get('fii_net', 0)),
                    'DII_NET': float(latest.get('dii_net', 0)),
                    'DATE': latest.get('date', today),
                    'FII_SENTIMENT': 'BUYING' if float(latest.get('fii_net', 0)) > 0 else 'SELLING',
                    'DII_SENTIMENT': 'BUYING' if float(latest.get('dii_net', 0)) > 0 else 'SELLING'
                }
                
                # Calculate net institutional flow
                result['TOTAL_NET'] = result['FII_NET'] + result['DII_NET']
                result['NET_SENTIMENT'] = 'NET BUYING' if result['TOTAL_NET'] > 0 else 'NET SELLING'
                
                return result
                
    except Exception as e:
        logger.error(f"FII/DII error: {e}")
        
        # Fallback to NSE if Moneycontrol fails
        try:
            return get_fii_dii_nse_fallback()
        except:
            pass
    
    return None

def get_fii_dii_nse_fallback() -> Optional[Dict]:
    """Fallback to NSE for FII/DII data"""
    try:
        # NSE provides this data in their market activity
        url = f"{NSE_BASE_URL}/api/reportFII"
        
        response = nse_session.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'data' in data:
                # Parse the complex NSE response
                fii_buy = 0
                fii_sell = 0
                dii_buy = 0
                dii_sell = 0
                
                for item in data['data']:
                    category = item.get('category', '').upper()
                    buy = float(item.get('buyValue', 0))
                    sell = float(item.get('sellValue', 0))
                    
                    if 'FII' in category or 'FOREIGN' in category:
                        fii_buy += buy
                        fii_sell += sell
                    elif 'DII' in category or 'DOMESTIC' in category:
                        dii_buy += buy
                        dii_sell += sell
                
                return {
                    'FII_NET': round(fii_buy - fii_sell, 2),
                    'DII_NET': round(dii_buy - dii_sell, 2),
                    'FII_BUY': round(fii_buy, 2),
                    'FII_SELL': round(fii_sell, 2),
                    'DII_BUY': round(dii_buy, 2),
                    'DII_SELL': round(dii_sell, 2)
                }
                
    except Exception as e:
        logger.error(f"NSE FII/DII fallback error: {e}")
    
    return None

# 🚨 **6. PUT-CALL RATIO - WORKING** 🚨
def get_put_call_ratio() -> Tuple[Optional[float], str, float, float]:
    """Get PCR from NSE options data"""
    try:
        # Use NSE API
        url = f"{NSE_BASE_URL}/api/option-chain-indices?symbol=NIFTY"
        
        response = nse_session.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            total_ce_oi = 0
            total_pe_oi = 0
            
            if 'records' in data and 'data' in data['records']:
                for record in data['records']['data']:
                    if 'CE' in record and 'openInterest' in record['CE']:
                        total_ce_oi += record['CE']['openInterest']
                    if 'PE' in record and 'openInterest' in record['PE']:
                        total_pe_oi += record['PE']['openInterest']
            
            if total_ce_oi > 0 and total_pe_oi > 0:
                pcr = total_pe_oi / total_ce_oi
                
                # Interpretation
                if pcr > 1.5:
                    sentiment = "EXTREME FEAR (Oversold)"
                elif pcr > 1.2:
                    sentiment = "FEAR (Bearish Bias)"
                elif pcr > 0.8:
                    sentiment = "NEUTRAL"
                elif pcr > 0.5:
                    sentiment = "GREED (Bullish Bias)"
                else:
                    sentiment = "EXTREME GREED (Overbought)"
                
                return (
                    round(pcr, 2),
                    sentiment,
                    total_ce_oi,
                    total_pe_oi
                )
                
    except Exception as e:
        logger.error(f"PCR error: {e}")
    
    return None, "UNAVAILABLE", 0, 0

# 🚨 **7. MAX PAIN - WORKING** 🚨
def calculate_max_pain() -> Optional[Dict]:
    """Calculate Max Pain level for NIFTY"""
    try:
        url = f"{NSE_BASE_URL}/api/option-chain-indices?symbol=NIFTY"
        
        response = nse_session.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'records' in data and 'data' in data['records']:
                # Extract strikes and OI
                strikes = []
                ce_oi_dict = {}
                pe_oi_dict = {}
                
                for record in data['records']['data']:
                    strike = record.get('strikePrice', 0)
                    if strike not in strikes:
                        strikes.append(strike)
                    
                    if 'CE' in record:
                        ce_oi_dict[strike] = record['CE'].get('openInterest', 0)
                    if 'PE' in record:
                        pe_oi_dict[strike] = record['PE'].get('openInterest', 0)
                
                strikes.sort()
                
                # Calculate pain at each strike
                min_pain = float('inf')
                max_pain_strike = strikes[0]
                
                for strike in strikes:
                    total_pain = 0
                    
                    for s in strikes:
                        ce_oi = ce_oi_dict.get(s, 0)
                        pe_oi = pe_oi_dict.get(s, 0)
                        
                        if s < strike:
                            total_pain += pe_oi * (strike - s)
                        elif s > strike:
                            total_pain += ce_oi * (s - strike)
                    
                    if total_pain < min_pain:
                        min_pain = total_pain
                        max_pain_strike = strike
                
                # Get current NIFTY price
                nifty_data = cached_yfinance("^NSEI", period="1d", interval="1d")
                if not nifty_data.empty:
                    current_price = nifty_data['Close'].iloc[-1]
                    distance = current_price - max_pain_strike
                    distance_pct = (distance / current_price) * 100
                    
                    if distance > 0:
                        bias = "DOWNWARD PRESSURE (Price above Max Pain)"
                    else:
                        bias = "UPWARD PRESSURE (Price below Max Pain)"
                    
                    return {
                        'MAX_PAIN': max_pain_strike,
                        'CURRENT': round(float(current_price), 2),
                        'DISTANCE': round(float(distance), 2),
                        'DISTANCE_PCT': round(distance_pct, 2),
                        'BIAS': bias,
                        'MIN_PAIN_VALUE': min_pain
                    }
                    
    except Exception as e:
        logger.error(f"Max Pain error: {e}")
    
    return None

# 🚨 **8. TECHNICAL LEVELS - ENHANCED** 🚨
def get_technical_levels() -> Optional[Dict]:
    """Calculate key technical levels"""
    try:
        nifty_data = cached_yfinance("^NSEI", period="50d", interval="1d")
        
        if not nifty_data.empty and len(nifty_data) >= 20:
            closes = nifty_data['Close']
            highs = nifty_data['High']
            lows = nifty_data['Low']
            
            # Moving Averages
            ma20 = closes.rolling(20).mean().iloc[-1]
            ma50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else ma20
            ma200 = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else ma50
            
            # Pivot Points (Classic)
            prev_high = highs.iloc[-2]
            prev_low = lows.iloc[-2]
            prev_close = closes.iloc[-2]
            
            pivot = (prev_high + prev_low + prev_close) / 3
            r1 = 2 * pivot - prev_low
            s1 = 2 * pivot - prev_high
            r2 = pivot + (prev_high - prev_low)
            s2 = pivot - (prev_high - prev_low)
            
            # Support/Resistance
            recent_high = highs.iloc[-10:].max()
            recent_low = lows.iloc[-10:].min()
            
            # Fibonacci levels
            swing_high = highs.iloc[-20:].max()
            swing_low = lows.iloc[-20:].min()
            swing_range = swing_high - swing_low
            
            fib_levels = {
                '0.236': swing_low + swing_range * 0.236,
                '0.382': swing_low + swing_range * 0.382,
                '0.500': swing_low + swing_range * 0.500,
                '0.618': swing_low + swing_range * 0.618,
                '0.786': swing_low + swing_range * 0.786
            }
            
            # RSI
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            return {
                'MA20': round(float(ma20), 2),
                'MA50': round(float(ma50), 2),
                'MA200': round(float(ma200), 2),
                'PIVOT': round(float(pivot), 2),
                'R1': round(float(r1), 2),
                'S1': round(float(s1), 2),
                'R2': round(float(r2), 2),
                'S2': round(float(s2), 2),
                'RESISTANCE': round(float(recent_high), 2),
                'SUPPORT': round(float(recent_low), 2),
                'RSI': round(float(current_rsi), 2),
                'FIB_LEVELS': {k: round(float(v), 2) for k, v in fib_levels.items()},
                'TREND': 'BULLISH' if ma20 > ma50 > ma200 else 'BEARISH' if ma20 < ma50 < ma200 else 'SIDEWAYS'
            }
            
    except Exception as e:
        logger.error(f"Technical levels error: {e}")
    
    return None

# 🚨 **9. ECONOMIC EVENTS** 🚨
def get_economic_events() -> List[str]:
    """Get upcoming economic events"""
    events = []
    
    try:
        today = datetime.now()
        
        # Check for RBI MPC meetings (usually first week of month)
        if today.day <= 7:
            events.append("📅 RBI MPC MEETING THIS WEEK")
        
        # Check for US FOMC meetings (usually 2nd week)
        if 8 <= today.day <= 14:
            events.append("🇺🇸 US FOMC MEETING THIS WEEK")
        
        # Check for monthly expiry (Last Thursday)
        last_thursday = find_last_thursday(today.year, today.month)
        if abs((today - last_thursday).days) <= 2:
            events.append("📆 MONTHLY EXPIRY THIS WEEK")
        
        # Check for Budget sessions (Feb)
        if today.month == 2:
            events.append("💰 UNION BUDGET SESSION")
        
        # Check for Quarterly results season
        month = today.month
        if month in [1, 4, 7, 10]:
            events.append("📊 QUARTERLY RESULTS SEASON")
            
    except Exception as e:
        logger.error(f"Events error: {e}")
    
    return events

def find_last_thursday(year: int, month: int) -> datetime:
    """Find last Thursday of the month"""
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    
    # Go back from first day of next month
    last_day = next_month - timedelta(days=1)
    
    while last_day.weekday() != 3:  # Thursday is 3
        last_day -= timedelta(days=1)
    
    return last_day

# 🚨 **10. OPENING GAP PREDICTION ALGORITHM** 🚨
def predict_opening_gap() -> Optional[Dict]:
    """Predict opening gap with institutional algorithm"""
    try:
        # Collect all data
        sgx_nifty = get_sgx_nifty()
        prev_data = get_previous_day_data()
        global_mkts = get_global_markets()
        vix_value, vix_sentiment = get_india_vix()
        fii_dii = get_fii_dii_data()
        
        if 'NIFTY' not in prev_data or not sgx_nifty:
            return None
        
        prev_close = prev_data['NIFTY']['CLOSE']
        gap_pct = ((sgx_nifty - prev_close) / prev_close) * 100
        
        # Initialize scoring
        score = 0
        factors = []
        weights = {
            'SGX_GAP': 0.35,
            'GLOBAL_MARKETS': 0.20,
            'PREV_CLOSE_POSITION': 0.15,
            'VIX': 0.10,
            'FII_FLOW': 0.15,
            'PCR': 0.05
        }
        
        # Factor 1: SGX Gap
        if gap_pct > 0.3:
            score += weights['SGX_GAP'] * 100
            factors.append(f"📈 SGX GAP UP: +{gap_pct:.2f}%")
        elif gap_pct < -0.3:
            score += weights['SGX_GAP'] * -100
            factors.append(f"📉 SGX GAP DOWN: {gap_pct:.2f}%")
        else:
            factors.append(f"➡️ SGX NEAR FLAT: {gap_pct:.2f}%")
        
        # Factor 2: Global Markets
        global_score = 0
        for market, change in global_mkts.items():
            if change > 0.5:
                global_score += 1
            elif change < -0.5:
                global_score -= 1
        
        if global_score >= 3:
            score += weights['GLOBAL_MARKETS'] * 100
            factors.append("🌍 GLOBAL MARKETS STRONGLY POSITIVE")
        elif global_score >= 1:
            score += weights['GLOBAL_MARKETS'] * 50
            factors.append("🌍 GLOBAL MARKETS POSITIVE")
        elif global_score <= -3:
            score += weights['GLOBAL_MARKETS'] * -100
            factors.append("🌍 GLOBAL MARKETS STRONGLY NEGATIVE")
        elif global_score <= -1:
            score += weights['GLOBAL_MARKETS'] * -50
            factors.append("🌍 GLOBAL MARKETS NEGATIVE")
        
        # Factor 3: Previous Day Close Position
        prev_range = prev_data['NIFTY']['HIGH'] - prev_data['NIFTY']['LOW']
        if prev_range > 0:
            close_position = (prev_data['NIFTY']['CLOSE'] - prev_data['NIFTY']['LOW']) / prev_range
            
            if close_position > 0.6:
                score += weights['PREV_CLOSE_POSITION'] * 100
                factors.append("📊 PREV CLOSE IN UPPER RANGE (Bullish)")
            elif close_position < 0.4:
                score += weights['PREV_CLOSE_POSITION'] * -100
                factors.append("📊 PREV CLOSE IN LOWER RANGE (Bearish)")
        
        # Factor 4: VIX
        if vix_value:
            if vix_value > 20:
                score += weights['VIX'] * -100  # High VIX negative for gap up
                factors.append(f"😨 HIGH VIX: {vix_value} ({vix_sentiment})")
            elif vix_value < 12:
                score += weights['VIX'] * 50    # Low VIX mildly positive
                factors.append(f"😊 LOW VIX: {vix_value} ({vix_sentiment})")
        
        # Factor 5: FII/DII Flow
        if fii_dii and fii_dii['FII_NET'] != 0:
            if fii_dii['FII_NET'] > 1000:
                score += weights['FII_FLOW'] * 100
                factors.append(f"💰 STRONG FII BUYING: ₹{fii_dii['FII_NET']:.0f}Cr")
            elif fii_dii['FII_NET'] > 500:
                score += weights['FII_FLOW'] * 50
                factors.append(f"💰 MODERATE FII BUYING: ₹{fii_dii['FII_NET']:.0f}Cr")
            elif fii_dii['FII_NET'] < -1000:
                score += weights['FII_FLOW'] * -100
                factors.append(f"💰 STRONG FII SELLING: ₹{abs(fii_dii['FII_NET']):.0f}Cr")
            elif fii_dii['FII_NET'] < -500:
                score += weights['FII_FLOW'] * -50
                factors.append(f"💰 MODERATE FII SELLING: ₹{abs(fii_dii['FII_NET']):.0f}Cr")
        
        # Factor 6: PCR
        pcr, pcr_sentiment, _, _ = get_put_call_ratio()
        if pcr:
            if pcr > 1.3:
                score += weights['PCR'] * -50  # High PCR (fear) negative for gap up
                factors.append(f"⚖️ HIGH PCR: {pcr} ({pcr_sentiment})")
            elif pcr < 0.7:
                score += weights['PCR'] * 50   # Low PCR (greed) positive
                factors.append(f"⚖️ LOW PCR: {pcr} ({pcr_sentiment})")
        
        # Final prediction
        if score >= 60:
            prediction = "STRONG GAP UP OPENING"
            bias = "STRONGLY BULLISH"
            color = "🟢"
        elif score >= 30:
            prediction = "MODERATE GAP UP OPENING"
            bias = "BULLISH"
            color = "🟡"
        elif score <= -60:
            prediction = "STRONG GAP DOWN OPENING"
            bias = "STRONGLY BEARISH"
            color = "🔴"
        elif score <= -30:
            prediction = "MODERATE GAP DOWN OPENING"
            bias = "BEARISH"
            color = "🟠"
        else:
            prediction = "FLAT TO MIXED OPENING"
            bias = "NEUTRAL"
            color = "⚪"
        
        return {
            'SCORE': round(score, 1),
            'PREDICTION': prediction,
            'BIAS': bias,
            'COLOR': color,
            'GAP_PCT': round(gap_pct, 2),
            'SGX_PRICE': sgx_nifty,
            'PREV_CLOSE': prev_close,
            'FACTORS': factors,
            'RAW_SCORE': score
        }
        
    except Exception as e:
        logger.error(f"Gap prediction error: {e}")
    
    return None

# 🚨 **11. GENERATE COMPREHENSIVE REPORT** 🚨
def generate_premarket_report() -> str:
    """Generate institutional pre-market report"""
    ist_now = get_ist_time()
    report_date = ist_now.strftime("%d %b %Y, %A")
    report_time = ist_now.strftime("%H:%M IST")
    
    report_lines = []
    
    # Header
    report_lines.append(f"<b>📊 INSTITUTIONAL PRE-MARKET ANALYSIS</b>")
    report_lines.append(f"<b>📅 {report_date}</b>")
    report_lines.append(f"<b>⏰ {report_time}</b>")
    report_lines.append("")
    report_lines.append("<b>══════════════════════════════</b>")
    report_lines.append("")
    
    try:
        # 1. SGX NIFTY
        sgx_nifty = get_sgx_nifty()
        if sgx_nifty:
            prev_data = get_previous_day_data()
            if 'NIFTY' in prev_data:
                prev_close = prev_data['NIFTY']['CLOSE']
                gap = sgx_nifty - prev_close
                gap_pct = (gap / prev_close) * 100
                
                report_lines.append(f"<b>🌏 SGX NIFTY:</b> <code>{sgx_nifty}</code>")
                report_lines.append(f"  Prev Close: <code>{prev_close}</code>")
                report_lines.append(f"  Expected Gap: <code>{gap:+.2f} ({gap_pct:+.2f}%)</code>")
        
        report_lines.append("")
        
        # 2. GLOBAL MARKETS
        global_mkts = get_global_markets()
        if global_mkts:
            report_lines.append(f"<b>🌍 GLOBAL MARKETS:</b>")
            for market, change in list(global_mkts.items())[:6]:  # Show top 6
                if change > 0:
                    report_lines.append(f"  {market}: <code>🟢 +{change}%</code>")
                else:
                    report_lines.append(f"  {market}: <code>🔴 {change}%</code>")
        
        report_lines.append("")
        
        # 3. PREVIOUS DAY
        prev_data = get_previous_day_data()
        if 'NIFTY' in prev_data:
            n = prev_data['NIFTY']
            change_icon = "🟢" if n['CHANGE'] > 0 else "🔴"
            report_lines.append(f"<b>📈 PREVIOUS DAY (NIFTY):</b>")
            report_lines.append(f"  O: <code>{n['OPEN']}</code> | H: <code>{n['HIGH']}</code>")
            report_lines.append(f"  L: <code>{n['LOW']}</code> | C: <code>{n['CLOSE']}</code>")
            report_lines.append(f"  Change: {change_icon} <code>{n['CHANGE']}%</code>")
            report_lines.append(f"  Pattern: <code>{n.get('PATTERN', 'N/A')}</code>")
        
        report_lines.append("")
        
        # 4. INDIA VIX
        vix_value, vix_sentiment = get_india_vix()
        if vix_value:
            report_lines.append(f"<b>😨 INDIA VIX:</b> <code>{vix_value}</code>")
            report_lines.append(f"  Sentiment: <code>{vix_sentiment}</code>")
        
        # 5. FII/DII
        fii_dii = get_fii_dii_data()
        if fii_dii:
            fii_icon = "🟢" if fii_dii['FII_NET'] > 0 else "🔴"
            dii_icon = "🟢" if fii_dii['DII_NET'] > 0 else "🔴"
            report_lines.append(f"<b>💰 INSTITUTIONAL FLOW:</b>")
            report_lines.append(f"  FII Net: {fii_icon} <code>₹{fii_dii['FII_NET']:.0f}Cr</code>")
            report_lines.append(f"  DII Net: {dii_icon} <code>₹{fii_dii['DII_NET']:.0f}Cr</code>")
        
        # 6. PCR
        pcr, pcr_sentiment, ce_oi, pe_oi = get_put_call_ratio()
        if pcr:
            pcr_icon = "🔴" if pcr > 1.2 else "🟢" if pcr < 0.8 else "🟡"
            report_lines.append(f"<b>⚖️ PUT-CALL RATIO:</b> {pcr_icon} <code>{pcr}</code>")
            report_lines.append(f"  Sentiment: <code>{pcr_sentiment}</code>")
        
        report_lines.append("")
        
        # 7. MAX PAIN
        max_pain = calculate_max_pain()
        if max_pain:
            report_lines.append(f"<b>🎯 MAX PAIN THEORY:</b>")
            report_lines.append(f"  Level: <code>{max_pain['MAX_PAIN']}</code>")
            report_lines.append(f"  Current: <code>{max_pain['CURRENT']}</code>")
            report_lines.append(f"  Bias: <code>{max_pain['BIAS']}</code>")
        
        report_lines.append("")
        
        # 8. TECHNICAL LEVELS
        tech_levels = get_technical_levels()
        if tech_levels:
            report_lines.append(f"<b>📊 KEY TECHNICALS:</b>")
            report_lines.append(f"  MA20: <code>{tech_levels['MA20']}</code> | MA50: <code>{tech_levels['MA50']}</code>")
            report_lines.append(f"  Support: <code>{tech_levels['SUPPORT']}</code>")
            report_lines.append(f"  Resistance: <code>{tech_levels['RESISTANCE']}</code>")
            report_lines.append(f"  RSI: <code>{tech_levels['RSI']}</code> | Trend: <code>{tech_levels['TREND']}</code>")
        
        report_lines.append("")
        report_lines.append("<b>══════════════════════════════</b>")
        report_lines.append("")
        
        # 9. GAP PREDICTION
        gap_prediction = predict_opening_gap()
        if gap_prediction:
            report_lines.append(f"<b>🎯 OPENING GAP PREDICTION:</b>")
            report_lines.append(f"  {gap_prediction['COLOR']} <b>{gap_prediction['PREDICTION']}</b>")
            report_lines.append(f"  Score: <code>{gap_prediction['SCORE']}/100</code>")
            report_lines.append(f"  Bias: <code>{gap_prediction['BIAS']}</code>")
            
            report_lines.append("")
            report_lines.append(f"<b>📋 KEY FACTORS:</b>")
            for factor in gap_prediction['FACTORS'][:6]:  # Show top 6 factors
                report_lines.append(f"  • {factor}")
        
        report_lines.append("")
        report_lines.append("<b>══════════════════════════════</b>")
        report_lines.append("")
        
        # 10. TRADING PLAN
        report_lines.append(f"<b>🎯 INSTITUTIONAL TRADING PLAN:</b>")
        
        if gap_prediction:
            bias = gap_prediction['BIAS']
            
            if "BULLISH" in bias:
                report_lines.append("  • <b>Strategy:</b> Look for buying opportunities")
                report_lines.append("  • <b>Entry:</b> Wait for pullback to support")
                report_lines.append("  • <b>Stop Loss:</b> Below yesterday's low")
                report_lines.append("  • <b>Target:</b> Yesterday's high + extension")
            
            elif "BEARISH" in bias:
                report_lines.append("  • <b>Strategy:</b> Look for selling opportunities")
                report_lines.append("  • <b>Entry:</b> Sell on rallies to resistance")
                report_lines.append("  • <b>Stop Loss:</b> Above yesterday's high")
                report_lines.append("  • <b>Target:</b> Yesterday's low - extension")
            
            else:
                report_lines.append("  • <b>Strategy:</b> Rangebound trading")
                report_lines.append("  • <b>Entry:</b> Buy near support, sell near resistance")
                report_lines.append("  • <b>Stop Loss:</b> Outside range boundaries")
        
        # 11. ECONOMIC EVENTS
        events = get_economic_events()
        if events:
            report_lines.append("")
            report_lines.append(f"<b>📅 KEY EVENTS THIS WEEK:</b>")
            for event in events:
                report_lines.append(f"  • {event}")
        
        report_lines.append("")
        report_lines.append("<b>⚠️ RISK DISCLAIMER:</b>")
        report_lines.append("This analysis is for educational purposes only.")
        report_lines.append("Past performance ≠ future results. Trade responsibly.")
        report_lines.append("")
        report_lines.append("<b>✅ Generated by: Institutional Analysis Engine v2.0</b>")
        
        return "\n".join(report_lines)
        
    except Exception as e:
        logger.error(f"Report generation error: {e}", exc_info=True)
        error_msg = f"<b>⚠️ REPORT GENERATION ERROR</b>\n\n"
        error_msg += f"Error: {str(e)[:200]}\n"
        error_msg += f"Time: {report_time}\n"
        error_msg += f"Please check logs for details."
        return error_msg

# 🚨 **12. MAIN SCHEDULER** 🚨
def send_daily_report():
    """Send the daily pre-market report"""
    try:
        logger.info("Starting daily report generation...")
        
        # Refresh NSE session
        nse_session.initialize()
        
        # Generate report
        report = generate_premarket_report()
        
        # Send report
        if len(report) > 4000:
            # Split long messages
            parts = [report[i:i+4000] for i in range(0, len(report), 4000)]
            for i, part in enumerate(parts):
                time.sleep(1)
                success = send_telegram(part)
                if not success and i == 0:
                    logger.error("Failed to send report")
                    send_admin_alert("Failed to send daily report")
        else:
            success = send_telegram(report)
            if not success:
                logger.error("Failed to send report")
                send_admin_alert("Failed to send daily report")
        
        logger.info("Daily report sent successfully")
        
    except Exception as e:
        logger.error(f"Failed to send daily report: {e}", exc_info=True)
        send_admin_alert(f"Report generation failed: {str(e)[:100]}")

def main():
    """Main scheduler"""
    logger.info("🚀 Institutional Pre-Market Analysis Engine Started")
    
    # Schedule daily report at 9:00 AM IST
    schedule.every().day.at("09:00").do(send_daily_report)
    
    # Also schedule at 8:55 AM as backup
    schedule.every().day.at("08:55").do(send_daily_report)
    
    # Send startup message
    ist_now = get_ist_time()
    startup_msg = (
        f"🚀 <b>Institutional Pre-Market Analysis Engine v2.0</b>\n"
        f"⏰ Started: {ist_now.strftime('%H:%M:%S IST')}\n"
        f"📅 Date: {ist_now.strftime('%d %b %Y')}\n"
        f"📊 Next report: 9:00 AM IST\n"
        f"✅ Engine running smoothly..."
    )
    send_telegram(startup_msg)
    
    # Health check every hour
    def health_check():
        try:
            # Test key data sources
            sgx = get_sgx_nifty()
            vix, _ = get_india_vix()
            
            status = "HEALTHY" if sgx and vix else "DEGRADED"
            send_admin_alert(f"System {status} | SGX: {sgx}, VIX: {vix}")
        except:
            send_admin_alert("System UNHEALTHY - Check immediately")
    
    schedule.every().hour.do(health_check)
    
    logger.info("Scheduler started. Waiting for scheduled tasks...")
    
    # Main loop
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            shutdown_msg = "🔴 Institutional Analysis Engine Stopped"
            send_telegram(shutdown_msg)
            break
            
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            time.sleep(300)  # Wait 5 minutes on error

# 🚨 **13. MANUAL TRIGGER** 🚨
def manual_trigger():
    """Manually trigger report generation"""
    print("🔧 Manual trigger activated...")
    send_daily_report()
    print("✅ Report sent!")

# 🚨 **RUN THE ENGINE** 🚨
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        manual_trigger()
    else:
        main()
