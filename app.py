# -*- coding: utf-8 -*-
"""
美股崩盘预警系统 - 21因子 V10.051 (Secure Secrets Edition)
【安全与修复更新】
1. 安全升级：移除所有硬编码 API Key，改为从 Streamlit Secrets 读取。
   (GENAI_API_KEY, FRED_KEY, FIRECRAWL_KEY 均需在后台配置)
2. 缓存优化：保留 @st.cache_data 机制，防止 Yahoo Finance 限流。
"""
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup 
from datetime import datetime, timedelta
from pathlib import Path
import os
import platform
import webbrowser
import warnings
import time
import random
import re
import traceback 
import sys
import json 
import io
import subprocess 
from firecrawl import Firecrawl 
from PIL import Image 

# --- FRED API 库 ---
try:
    from fredapi import Fred
except ImportError:
    print(">>> 提示：未找到 fredapi 库，建议运行 pip install fredapi 以启用完整功能。")

# --- Google Gemini AI 库 ---
try:
    from google import genai
except ImportError:
    print(">>> 严重错误：未找到 google-genai 库。请运行 pip install google-genai")
    sys.exit(1)

# ==========================================
# 【API 配置区 - 安全版】
# ==========================================
# 自动从 Streamlit Secrets 读取 Key，防止上传 GitHub 后被封
# 请确保在 Streamlit Cloud -> App Settings -> Secrets 中配置了以下三个变量
try:
    GENAI_API_KEY = st.secrets["GENAI_API_KEY"]
    USER_FRED_KEY = st.secrets["FRED_KEY"]
    FIRECRAWL_KEY = st.secrets["FIRECRAWL_KEY"]
except FileNotFoundError:
    st.error("❌ 错误：未检测到 Secrets 配置！请在 Streamlit Cloud 后台 Settings -> Secrets 中填入 API Key。")
    st.stop()
except KeyError as k:
    st.error(f"❌ 错误：Secrets 中缺少键值 {k}。请检查配置文件名拼写是否正确。")
    st.stop()

# 初始化 Google AI
client = genai.Client(api_key=GENAI_API_KEY)

warnings.filterwarnings("ignore")

# ==========================================
# 【UI 美化工具类】
# ==========================================
class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_h(msg): print(f"\n{C.HEADER}{C.BOLD}━━━ {msg} ━━━{C.ENDC}")
def print_step(msg): print(f"{C.CYAN}\U0001f539 {msg}{C.ENDC}")
def print_ok(msg): print(f"{C.GREEN}\u2705 {msg}{C.ENDC}")
def print_warn(msg): print(f"{C.YELLOW}\u26a0\ufe0f  {msg}{C.ENDC}")
def print_err(msg): print(f"{C.RED}\u274c {msg}{C.ENDC}")
def print_info(msg): print(f"{C.BLUE}\u2139\ufe0f  {msg}{C.ENDC}")


# ==========================================
# 【缓存加速层 (解决 Yahoo 限流问题)】
# ==========================================
@st.cache_data(ttl=86400) # 列表缓存 24 小时
def get_cached_tickers():
    """缓存获取标普500成分股名单"""
    print_step("获取标普500成分股名单 (Cached)...")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15).text)
        for t in tables:
            if 'Symbol' in t.columns:
                return t['Symbol'].str.replace('.', '-', regex=False).tolist()
    except: return []

@st.cache_data(ttl=3600) # 数据缓存 1 小时
def get_cached_sp500_data(tickers):
    """缓存下载 500 只股票数据"""
    if not tickers: return pd.DataFrame()
    print_step(f"下载 {len(tickers)} 只成分股数据 (5年) [Cached]...")
    print_info("提示: 首次运行需联网下载，后续 1 小时内将直接读取缓存...")
    
    closes = []
    for i in range(0, len(tickers), 80):
        batch = tickers[i:i+80]
        try:
            data = yf.download(batch, period="5y", auto_adjust=True, progress=False, threads=True, timeout=30)
            if isinstance(data.columns, pd.MultiIndex):
                try: close = data['Close']
                except: close = data
            else: close = data
            closes.append(close)
            print(f"   进度: {min(i+80, len(tickers))}/{len(tickers)}")
        except: pass
        
    if not closes: return pd.DataFrame()
    return pd.concat(closes, axis=1).dropna(axis=1, how='all')

@st.cache_data(ttl=3600) # 数据缓存 1 小时
def get_cached_sector_data(tickers, start_date):
    """缓存下载板块轮动数据"""
    print_step(f"下载 {len(tickers)} 个板块数据 ({start_date} ~ Now) [Cached]...")
    raw_data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)
    return raw_data

@st.cache_data(ttl=3600) # 数据缓存 1 小时
def get_cached_smt_data(tickers, period):
    """缓存下载 SMT 分析数据"""
    print_step("下载全量数据 (含期货/等权ETF) [Cached]...")
    data = yf.download(tickers, period=period, auto_adjust=False, progress=False)
    return data


# ==========================================
# 【WebScraper: 纯净 Firecrawl 版】
# ==========================================
class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        # [安全修复] 直接使用从 secrets 读取到的变量，不再硬编码
        self.firecrawl_key = FIRECRAWL_KEY 
        self.app = Firecrawl(api_key=self.firecrawl_key)
        self.fred_key = USER_FRED_KEY
        self.cached_gdp = None 

    # --- 1. Shiller PE ---
    def fetch_shiller_pe(self):
        print_step("[Shiller PE] 启动 Firecrawl 抓取 (Multpl)...")
        url = "https://www.multpl.com/shiller-pe"
        try:
            response = self.app.scrape(url, formats=['markdown'])
            md = getattr(response, 'markdown', '')
            if md:
                match = re.search(r'Shiller PE Ratio.*?(\d{2}\.\d{1,2})', md, re.S | re.I)
                if match:
                    val = float(match.group(1))
                    print_ok(f"AI 识别成功! Shiller PE: {val}")
                    return val
        except Exception as e:
            print_err(f"Shiller PE 抓取异常: {e}")
        return None

    # --- 2. Fear & Greed ---
    def fetch_fear_greed(self):
        print_step("[Fear & Greed] 方案 A: 调用 Python 库 (fear_and_greed)...")
        try:
            import fear_and_greed
            index_data = fear_and_greed.get()
            score = int(index_data.value)
            rating = index_data.description
            if isinstance(rating, str): rating = rating.capitalize()
            print_ok(f"[Fear & Greed] Python 库调用成功: {score} ({rating})")
            return score, rating
        except Exception:
            print_warn("Python 库调用出错，切换至 API 直连...")

        print_step("[Fear & Greed] 方案 B: 启动 API 直连模式...")
        api_url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            r = requests.get(api_url, headers=headers, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if 'fear_and_greed' in data and 'score' in data['fear_and_greed']:
                    score = int(data['fear_and_greed']['score'])
                    rating = data['fear_and_greed']['rating']
                    print_ok(f"[Fear & Greed] API 直连成功: {score} ({rating})")
                    return score, rating
        except Exception as e:
            print_err(f"F&G 获取失败: {e}")
        return None, "获取失败"

    # --- 4. GDP ---
    def fetch_us_gdp(self):
        if self.cached_gdp: return self.cached_gdp
        print_h("[US GDP] 启动数据获取 (FRED API 直连)...")
        try:
            fred = Fred(api_key=self.fred_key)
            gdp_series = fred.get_series('GDP', sort_order='desc', limit=1)
            if not gdp_series.empty:
                val = gdp_series.iloc[0] 
                gdp_trillion = val / 1000.0
                date_str = gdp_series.index[0].strftime('%Y-%m-%d')
                print_ok(f"[US GDP] 成功: {gdp_trillion:.3f}T (日期: {date_str})")
                self.cached_gdp = gdp_trillion 
                return gdp_trillion
        except Exception as e:
            print_err(f"FRED GDP 获取异常: {e}")
        return None

    # --- 3. Buffett Indicator ---
    def fetch_buffett_indicator(self):
        print_step("[Buffett Indicator] 启动计算模式 (Market Cap / GDP)...")
        gdp_tril = self.fetch_us_gdp()
        if not gdp_tril: return None
        try:
            w5000 = yf.Ticker("^W5000")
            hist = w5000.history(period="5d")
            if not hist.empty:
                market_cap_proxy = hist['Close'].iloc[-1] 
                gdp_billions = gdp_tril * 1000.0          
                ratio = (market_cap_proxy / gdp_billions) * 100
                print_ok(f"[巴菲特指标] 计算成功: {ratio:.2f}%")
                return ratio
        except Exception as e:
             print_err(f"Buffett Indicator 计算异常: {e}")
        return None

    # --- 5. Margin Debt ---
    def fetch_margin_debt(self):
        print_h("[Margin Debt] 启动 Firecrawl 抓取 (FINRA)...")
        url = "https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics"
        gdp_val = self.fetch_us_gdp()
        try:
            response = self.app.scrape(url, formats=['markdown'])
            md = getattr(response, 'markdown', '')
            if md:
                matches = re.findall(r'([A-Z][a-z]{2}-\d{2})\s*\|\s*([\d,]+)', md, re.S | re.I)
                if matches and len(matches) > 0:
                    latest_date, latest_val_str = matches[0]
                    absolute_debt_trillion = float(latest_val_str.replace(',', '')) / 1_000_000
                    gdp_ratio = None
                    if gdp_val: gdp_ratio = (absolute_debt_trillion / gdp_val) * 100
                    
                    yoy_val = None
                    if len(matches) >= 13: 
                        prev_val = float(matches[12][1].replace(',', ''))
                        current_val = float(latest_val_str.replace(',', ''))
                        yoy_val = ((current_val - prev_val) / prev_val) * 100
                    
                    print_ok(f"Margin数据: {absolute_debt_trillion:.3f}T, GDP比: {gdp_ratio if gdp_ratio else 0:.2f}%")
                    return yoy_val, absolute_debt_trillion, gdp_ratio
        except Exception as e:
            print_err(f"Margin Debt 抓取异常: {e}")
        return None, None, None

    # --- 6. Sahm Rule ---
    def fetch_sahm_rule(self):
        print_step("[Sahm Rule] 启动 Firecrawl 抓取 (FRED)...")
        url = "https://fred.stlouisfed.org/series/SAHMREALTIME"
        try:
            response = self.app.scrape(url, formats=['markdown'])
            md = getattr(response, 'markdown', '')
            if md:
                match = re.search(r'([A-Z][a-z]{2}\s+\d{4}):\s*([\d\.]+)', md, re.S | re.I)
                if match:
                    val = float(match.group(2))
                    print_ok(f"[Sahm Rule] 抓取成功: {val}%")
                    return val
        except Exception as e:
            print_err(f"Sahm Rule 抓取异常: {e}")
        return None

    # --- 7. LEI (Hybrid Vision - Smart Restore) ---
    def fetch_lei(self):
        print_h("[LEI 3Ds] 启动混合视觉模式 (Firecrawl + Gemini)...")
        depth, diffusion = None, None
        url = "https://www.conference-board.org/topics/us-leading-indicators"
        try:
            print_step("正在解析页面结构 (寻找 Summary Table 图片)...")
            response = self.app.scrape(url, formats=['markdown'])
            md = getattr(response, 'markdown', '')
            img_url = None
            
            if md:
                # [Smart Restore] 智能锚点定位
                anchor_idx = md.find("Summary Table")
                if anchor_idx == -1: anchor_idx = md.find("Composite Economic Indexes")
                
                if anchor_idx != -1:
                    # 只看锚点附近 1500 字符
                    snippet = md[anchor_idx : anchor_idx + 1500]
                    # 寻找图片链接
                    img_match = re.search(r'\((https://.*?lei.*?\.png)\)', snippet, re.I)
                    if img_match:
                        img_url = img_match.group(1)
                        print_ok(f"定位到数据图片: {img_url.split('/')[-1]}")
                
                # 兜底: 如果锚点没找到，才使用全局搜索
                if not img_url:
                    all_imgs = re.findall(r'\((https://.*?lei.*?\.png)\)', md, re.I)
                    if all_imgs: 
                        img_url = all_imgs[0]
                        print_warn(f"锚点未命中，使用首张 LEI 图片: {img_url}")

            if img_url:
                print_step("下载图片并进行 AI 分析...")
                img_resp = requests.get(img_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                if img_resp.status_code == 200:
                    img_data = Image.open(io.BytesIO(img_resp.content))
                    prompt = """
                    Analyze this LEI Summary Table image.
                    Extract two values:
                    1. "6-Month % Change" (last column, e.g., -2.1). Key: "depth"
                    2. "Diffusion" (value 0-100, e.g., 35.0). Key: "diffusion"
                    Return ONLY JSON. Example: {"depth": -2.1, "diffusion": 35.0}
                    """
                    ai_resp = client.models.generate_content(
                        model='gemini-2.0-flash',
                        contents=[prompt, img_data]
                    )
                    
                    if ai_resp and ai_resp.text:
                        json_match = re.search(r'\{.*\}', ai_resp.text, re.DOTALL)
                        if json_match:
                            js = json.loads(json_match.group(0))
                            depth = js.get('depth')
                            diffusion = js.get('diffusion')
                            if depth is not None:
                                print_ok(f"Gemini 视觉读取成功: Depth={depth}%, Diffusion={diffusion}")
                                return float(depth), float(diffusion)
                        else:
                            print_err(f"AI 返回非 JSON 格式: {ai_resp.text[:50]}...")
                    else:
                        print_err("AI 响应为空")

        except Exception as e:
            print_err(f"LEI 流程异常: {e}")
        return None, None

     # --- 8. HO Internals (Firecrawl + Gemini + 强力 Prompt) ---
    def fetch_nyse_internals_robust(self):
        print_step("启动 Firecrawl 访问 WSJ (PCR 模式)...")
        nyse_data = None
        nasdaq_data = None
        
        target_url = "https://www.wsj.com/market-data/stocks/marketsdiary"
        headers = {"Authorization": f"Bearer {self.firecrawl_key}", "Content-Type": "application/json"}
        # 请求 markdown 和 screenshot
        payload = {"url": target_url, "formats": ["markdown", "screenshot"], "waitFor": 12000, "mobile": False}
        
        try:
            print_step("发送 API 请求 (获取云端 Markdown + 截图)...")
            response = requests.post("https://api.firecrawl.dev/v1/scrape", headers=headers, json=payload, timeout=90)
            
            if response.status_code == 200:
                data = response.json()
                md = data.get('data', {}).get('markdown', '')
                scr_url = data.get('data', {}).get('screenshot', '')
                
                # --- 1. Markdown 文本分析 (Prompt 强化) ---
                if md:
                    print_step("正在进行 Markdown 结构化分析 (Gemini)...")
                    # 【强制 Prompt】
                    prompt = f"""
                    Analyze the Markdown content scraped from WSJ Market Diary.
                    
                    MISSION:
                    Extract Market Breadth data for "NYSE" and "NASDAQ".
                    
                    CRITICAL RULES:
                    1. Ignore "Weekly" or "Week Ago" columns. I ONLY want "Latest Close" / DAILY data.
                    2. Look for "Advances", "Declines", "Unchanged", "New Highs", "New Lows".
                    3. For Volume ("Adv. Volume", "Decl. Volume"):
                       **IMPORTANT**: The table has two sections. You MUST extract data from the "Composite Trading" section (usually at the bottom), NOT the "Trading Activity" section.
                       The correct Volume numbers should be in the BILLIONS (e.g., 3,000,000,000+), whereas the wrong ones are in millions.
                    
                    RETURN JSON FORMAT:
                    {{
                      "NYSE": {{ "adv": 1234, "dec": 567, "unch": 89, "high": 50, "low": 10, "adv_vol": 3000000000, "dec_vol": 2000000000 }},
                      "NASDAQ": {{ "adv": 2345, "dec": 678, ... }}
                    }}
                    
                    MARKDOWN CONTENT:
                    {md[:28000]} 
                    """
                    
                    try:
                        ai_resp = client.models.generate_content(model='gemini-2.0-flash', contents=[prompt])
                        if ai_resp and ai_resp.text:
                            clean_text = re.sub(r'```json|```', '', ai_resp.text).strip()
                            result = json.loads(re.search(r'\{.*\}', clean_text, re.DOTALL).group(0))
                            nyse_data = result.get('NYSE')
                            nasdaq_data = result.get('NASDAQ')
                            if nyse_data: print_ok(f"WSJ Text 分析成功: {nyse_data}")
                    except Exception as e:
                        print_warn(f"WSJ Text 分析微恙: {e}")

                # --- 2. Vision 视觉兜底 ---
                if not nyse_data and scr_url:
                    print_step("启用 Vision 视觉补救 (Gemini)...")
                    try:
                        img_bytes = requests.get(scr_url, timeout=30).content
                        img = Image.open(io.BytesIO(img_bytes))
                        prompt_v = "Analyze image. Extract Daily data for NYSE & NASDAQ. Ignore Weekly. For Volume, use the larger 'Composite' numbers (Billions). Return JSON."
                        ai_resp_v = client.models.generate_content(model='gemini-2.0-flash', contents=[prompt_v, img])
                        if ai_resp_v.text:
                            res_v = json.loads(re.search(r'\{.*\}', ai_resp_v.text, re.DOTALL).group(0))
                            nyse_data = res_v.get('NYSE')
                            nasdaq_data = res_v.get('NASDAQ')
                            print_ok("WSJ Vision 补救成功")
                    except: pass
            
            if nasdaq_data:
                self.cached_nasdaq = nasdaq_data

        except Exception as e:
            print_err(f"WSJ Firecrawl 异常: {e}")
        
        return nyse_data

    # --- 8.5 [重构] NYMO Vision Fetch (StockCharts Source) ---
    def fetch_nymo_vision(self):
        print_step("启动 Firecrawl 视觉抓取 StockCharts ($NYMO)...")
        target_url = "https://stockcharts.com/h-sc/ui?s=$NYMO"
        nymo_val = None
        
        headers = {"Authorization": f"Bearer {self.firecrawl_key}", "Content-Type": "application/json"}
        # StockCharts 加载需要时间，等待 8秒 以确保图表和图例渲染完成
        payload = {"url": target_url, "formats": ["screenshot"], "waitFor": 8000, "mobile": False}
        
        try:
            print_step("请求云端截图...")
            resp = requests.post("https://api.firecrawl.dev/v1/scrape", headers=headers, json=payload, timeout=60)
            
            if resp.status_code == 200:
                data = resp.json()
                scr_url = data.get('data', {}).get('screenshot', '')
                
                if scr_url:
                    print_step("截图获取成功，正在进行 AI 读数...")
                    try:
                        img_bytes = requests.get(scr_url, timeout=30).content
                        img = Image.open(io.BytesIO(img_bytes))
                        
                        prompt = """
                        Analyze this StockCharts image for "$NYMO".
                        Locate the data legend (usually top left).
                        Extract the value labeled "Last", "Close", or the final number in the OHLC sequence.
                        The value can be negative (e.g., -15.40).
                        Return ONLY JSON: {"value": -12.34}
                        """
                        
                        ai_resp = client.models.generate_content(
                            model='gemini-2.0-flash',
                            contents=[prompt, img]
                        )
                        
                        if ai_resp.text:
                            clean_text = re.sub(r'```json|```', '', ai_resp.text).strip()
                            match = re.search(r'\{.*\}', clean_text, re.DOTALL)
                            if match:
                                val = json.loads(match.group(0)).get('value')
                                if val is not None:
                                    nymo_val = float(val)
                                    print_ok(f"StockCharts ($NYMO) 视觉提取成功: {nymo_val}")
                                    return nymo_val
                    except Exception as e:
                        print_err(f"AI 视觉识别失败: {e}")
            else:
                print_err(f"Firecrawl 请求失败: {resp.status_code}")

        except Exception as e:
            print_err(f"NYMO 抓取流程异常: {e}")
            
        return nymo_val

    # --- 8.6 [升级] MCO 双重抓取 ---
    def fetch_dual_mco(self):
        print_step("[MCO] 启动官方源 + NYMO 双重抓取...")
        mco_official = None
        nymo_ratio = None
        try:
            url_off = "https://www.mcoscillator.com/"
            resp = self.app.scrape(url_off, formats=['markdown'])
            md = getattr(resp, 'markdown', '')
            if md:
                match = re.search(r'McC\s*OSC\s*\|?\s*([-\d\.]+)', md, re.I)
                if match:
                    mco_official = float(match.group(1))
                    print_ok(f"[MCO] 官方源抓取成功: {mco_official}")
        except Exception as e:
            print_err(f"MCO 官方源异常: {e}")
        
        # 使用新的 StockCharts 抓取函数
        nymo_ratio = self.fetch_nymo_vision()
        return mco_official, nymo_ratio

    # --- [重构] TradingView 市场宽度 (直接复用 WSJ 数据) ---
    def fetch_tv_breadth_vision(self):
        print_h("[TradingView 替代方案] 复用 WSJ NASDAQ 数据 (更稳更准)...")
        if hasattr(self, 'cached_nasdaq') and self.cached_nasdaq:
            def clean(v):
                if isinstance(v, str): 
                    v = v.replace(',', '')
                    if 'K' in v: v = float(v.replace('K','')) * 1000
                    return int(float(v))
                return v
            adv = clean(self.cached_nasdaq.get('adv'))
            dec = clean(self.cached_nasdaq.get('dec'))
            if adv and dec:
                print_ok(f"WSJ NASDAQ 数据复用成功: Adv={adv}, Dec={dec}")
                return adv, dec
        print_warn("WSJ NASDAQ 数据缺失，跳过广度显示。")
        return None, None

    # --- 15. CBOE Put/Call Ratio [保持 PCR 模块不变] ---
    def fetch_pcr_robust(self):
        print_h("[PCR] 启动直连 API 抓取 (MacroMicro)...")
        target_url = "https://en.macromicro.me/charts/449/us-cboe-options-put-call-ratio"
        headers = {"Authorization": f"Bearer {self.firecrawl_key}", "Content-Type": "application/json"}
        payload = {"url": target_url, "formats": ["markdown", "screenshot"], "waitFor": 15000, "mobile": True}
        try:
            print_step("发送 API 请求 (Text + Vision)...")
            response = requests.post("https://api.firecrawl.dev/v1/scrape", headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                md = data.get('data', {}).get('markdown', '')
                if md:
                    pattern = r'(20\d{2}-\d{2}-\d{2})(?:[^0-9]{0,200})\s*(\d{1,2}\.\d{2})'
                    matches = re.findall(pattern, md, re.DOTALL)
                    if matches:
                        matches.sort(key=lambda x: x[0], reverse=True)
                        val = float(matches[0][1])
                        print_ok(f"PCR 抓取成功: {val}")
                        return val, val
        except Exception as e:
            print_err(f"PCR 抓取异常: {e}")
        return None, None

    # --- 16. NFCI ---
    def fetch_nfci(self):
        print_step("[NFCI] 启动 FRED API 获取 (替代旧版)...")
        try:
            fred = Fred(api_key=self.fred_key)
            s = fred.get_series('NFCI', observation_start=datetime.now() - timedelta(weeks=4))
            if s.empty: return None
            val = s.iloc[-1]
            print_ok(f"[NFCI] FRED数据获取成功: {val:.4f}")
            return float(val)
        except Exception as e:
            print_err(f"NFCI 获取失败: {e}")
            return None

# ==========================================
# 【核心程序】
# ==========================================
class CrashWarningSystem:
    def __init__(self):
        self.scraper = WebScraper()
        self.setup_fonts()
        self.colors = {
            'bg': '#4B535C', 'table_header': '#3E4953', 
            'row_safe': '#2E8B57', 'text_safe': '#FFFFFF', 
            'row_warn': '#8B0000', 'text_warn': '#FFFFFF', 
            'row_risk': '#B8860B', 'text_risk': '#FFFFFF', 
            'title': '#FFEE88', 'edge': '#606972'
        }
        self.shared_wsj_data = None
        self.shared_breadth_200 = None

    def setup_fonts(self):
        # [Fix] 针对 Linux 环境 (如 Streamlit Cloud) 增加 WenQuanYi Zen Hei 支持
        if platform.system() == "Windows":
            sys_font = ['Microsoft YaHei', 'SimHei']
        else:
            # 优先使用 WenQuanYi Zen Hei (用户已在 packages.txt 安装)
            sys_font = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'PingFang SC', 'Arial Unicode MS']
            
        plt.rcParams['font.sans-serif'] = sys_font + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

    def get_tickers(self):
        # [Fix] 使用缓存
        return get_cached_tickers()

    def download_5y_data(self):
        # [Fix] 使用缓存的全局函数
        tickers = self.get_tickers()
        if not tickers: return pd.DataFrame()
        return get_cached_sp500_data(tickers)

    def calculate_spx_breadth_deep(self):
        try:
            full_data = self.download_5y_data()
            if full_data.empty: return None, None
            print_step("正在本地计算 SMA50 和 SMA20 (及 SMA200)...")
            
            last_close = full_data.iloc[-1]
            sma50 = full_data.rolling(50).mean().iloc[-1]
            pct50 = (last_close > sma50).mean() * 100
            sma20 = full_data.rolling(20).mean().iloc[-1]
            pct20 = (last_close > sma20).mean() * 100
            sma200 = full_data.rolling(200).mean().iloc[-1]
            valid200 = last_close.notna() & sma200.notna()
            pct200 = (last_close[valid200] > sma200[valid200]).mean() * 100
            self.shared_breadth_200 = pct200
            
            print_ok(f"市场广度计算完成: >50MA={pct50:.1f}%, >20MA={pct20:.1f}%, >200MA={pct200:.1f}%")
            return pct50, pct20
        except Exception as e:
            print_err(f"市场广度计算错误: {e}")
            return None, None

    def analyze_market_trends_console(self):
        print("\n" + "="*75)
        print(f" \U0001f3e6 启动深度宏观预警模块 (Deep Macro) - {datetime.now().strftime('%Y-%m-%d')}") 
        print("="*75)
        try:
            fred = Fred(api_key=USER_FRED_KEY)
            start_d = datetime.now() - timedelta(weeks=5)
            s_walcl = fred.get_series('WALCL', observation_start=start_d)   
            s_tga = fred.get_series('WTREGEN', observation_start=start_d)   
            s_rrp = fred.get_series('RRPONTSYD', observation_start=start_d) 
            
            v_walcl = s_walcl.iloc[-1]; v_tga = s_tga.iloc[-1]; v_rrp = s_rrp.iloc[-1]
            p_walcl = s_walcl.iloc[0]; p_tga = s_tga.iloc[0]; p_rrp = s_rrp.iloc[0]

            liq_now = (v_walcl / 1000000.0) - (v_tga / 1000.0) - (v_rrp / 1000.0)
            liq_prev = (p_walcl / 1000000.0) - (p_tga / 1000.0) - (p_rrp / 1000.0)
            liq_chg = liq_now - liq_prev
            liq_signal = "\U0001f7e2 扩张 (利好)" if liq_chg > 0 else "\U0001f534 收缩 (利空)" 
            
            print(f"1. 美联储净流动性: ${liq_now:.3f}T (Trillion)")
            print(f"   -> 4周变化: {liq_chg:+.3f}T ({liq_signal})")
            print(f"   -> 规则: 流动性增加 = 股市燃料增加")
        except: pass
        
        try:
            print_step("计算股权风险溢价 (Equity Risk Premium)...")
            dgs10 = fred.get_series('DGS10', sort_order='desc', limit=1).iloc[-1]
            shiller_pe = self.scraper.fetch_shiller_pe()
            if not shiller_pe: shiller_pe = 35.0 
            earnings_yield = (1.0 / shiller_pe) * 100
            erp = earnings_yield - dgs10
            erp_signal = "\U0001f7e2 正常" 
            if erp < 1.0: erp_signal = "\U0001f534 极度危险 (股不如债)" 
            elif erp < 2.5: erp_signal = "\U0001f7e0 偏低 (吸引力差)" 
            print(f"2. 股权风险溢价 (ERP): {erp:.2f}%  [{erp_signal}]")
        except: pass

        try:
            print_step("分析市场广度 (RSP vs SPY 20日趋势)...")
            df = yf.download(['SPY', 'RSP'], period="3mo", progress=False)['Close']
            if not df.empty:
                ratio = df['RSP'] / df['SPY']
                curr_ratio = ratio.iloc[-1]
                ago_20_ratio = ratio.iloc[-20]
                change_20d = ((curr_ratio - ago_20_ratio) / ago_20_ratio) * 100
                spy_trend = "上涨" if df['SPY'].iloc[-1] > df['SPY'].iloc[-20] else "下跌"
                trend_signal = "\U0001f7e2 结构健康" 
                if spy_trend == "上涨" and change_20d < -1.0:
                    trend_signal = "\U0001f534 严重背离 (大票涨,小票跌)" 
                elif change_20d < 0:
                    trend_signal = "\U0001f7e0 跑输 (小票弱势)" 
                print(f"3. RSP/SPY 相对强度 (20日): {change_20d:+.2f}%  [{trend_signal}]")
        except: pass

        print_step("检查市场内部结构 (WSJ & Local Calc)...")
        nh_val = "N/A"
        nh_signal = "\u26aa 未知" 
        if self.shared_wsj_data and 'high' in self.shared_wsj_data:
            def c(v): return int(str(v).replace(',','')) if v else 0
            val = c(self.shared_wsj_data['high']) - c(self.shared_wsj_data['low'])
            nh_val = f"{val:.0f}"
            nh_signal = "\U0001f7e2 多头主导" if val > 0 else "\U0001f534 空头主导" 
        print(f"4. WSJ 净新高 (Net Highs): {nh_val}  [{nh_signal}]")
        print("="*75 + "\n")

    def fetch_and_calculate(self):
        print_h("开始执行数据获取与计算")
        ma50_pct, ma20_pct = self.calculate_spx_breadth_deep()
        print_step("获取核心指数与宏观数据 (全动态抓取模式)...")
        indicators = []
        trend_desc = "趋势判断: 数据不足"; pos_str = "位置: N/A"
        try:
            tickers = yf.Tickers("^GSPC ^VIX ^TNX ^IRX RSP SPY ^NYA")
            hist = tickers.history(period="3y", group_by='ticker')
            spx = hist['^GSPC']['Close'].dropna()
            vix = hist['^VIX']['Close'].dropna()
            tnx = hist['^TNX']['Close'].dropna()
            irx = hist['^IRX']['Close'].dropna()
            rsp = hist['RSP']['Close'].dropna()
            spy = hist['SPY']['Close'].dropna()
            nya = hist['^NYA']['Close'].dropna()
            spx_weekly = spx.resample('W').last().dropna()
            sma50 = spx.rolling(50).mean()
            spx_trend_up = spx.iloc[-1] > sma50.iloc[-1]
            spx_trend_txt = "向上" if spx_trend_up else "向下"
            if len(spx) > 250:
                curr_px = spx.iloc[-1]
                year_high = spx.iloc[-250:].max()
                ma20 = spx.rolling(20).mean().iloc[-1]
                ma60 = spx.rolling(60).mean().iloc[-1]
                ma120 = spx.rolling(120).mean().iloc[-1]
                ma250 = spx.rolling(250).mean().iloc[-1]
                ma_list = [ma20, ma60, ma120, ma250]
                up_count = sum(1 for m in ma_list if curr_px > m)
                trend_desc = "震荡"
                if up_count == 4: trend_desc = "强多头 (站上所有均线)"
                elif up_count == 0: trend_desc = "强空头 (跌破所有均线)"
                elif curr_px > ma250: trend_desc = "偏多 (年线之上)"
                dist_high = (curr_px / year_high - 1) * 100
                pos_desc = "逼近52周新高" if dist_high > -2 else "区间震荡"
                pos_str = f"距52周高: {dist_high:.1f}% | {pos_desc}"
                print_h("【简单结论】标普500趋势")
                print(f"  当前价格: {curr_px:.2f}"); print(f"  趋势定性: {trend_desc}"); print("-" * 30)
        except: return [], []

        print_h("启动宏观指标动态抓取 (Firecrawl)")
        real_shiller = self.scraper.fetch_shiller_pe()
        real_sahm = self.scraper.fetch_sahm_rule()
        real_fg, fg_source = self.scraper.fetch_fear_greed()
        val_buffett = self.scraper.fetch_buffett_indicator()
        val_margin_yoy, margin_amt, margin_ratio = self.scraper.fetch_margin_debt()
        lei_depth, lei_diff = self.scraper.fetch_lei()
        pcr_avg, pcr_curr = self.scraper.fetch_pcr_robust()
        print_h("芝加哥金融状况指数 (NFCI)")
        val_nfci = self.scraper.fetch_nfci() 
        print("\n")

        print_h("Hindenburg Omen (HO) & McClellan Oscillator (MCO) & Volume")
        real_mco, real_nymo = self.scraper.fetch_dual_mco() 
        ho_res = self.scraper.fetch_nyse_internals_robust()
        if ho_res: self.shared_wsj_data = ho_res
        
        adv_vol, dec_vol = 0, 0
        if ho_res and isinstance(ho_res, dict) and ho_res.get('high') is not None:
            def c(v):
                if isinstance(v, str): 
                    v = v.replace(',', '')
                    if 'B' in v: v = float(v.replace('B',''))*1000000000
                    elif 'M' in v: v = float(v.replace('M',''))*1000000
                    return float(v)
                return float(v) if v else 0
            h_raw = c(ho_res['high']); l_raw = c(ho_res['low'])
            adv = c(ho_res['adv']); dec = c(ho_res['dec'])
            adv_vol = c(ho_res['adv_vol']); dec_vol = c(ho_res['dec_vol'])
            total_issues = adv + dec + c(ho_res.get('unch', 0))
            h_pct = (h_raw / total_issues) * 100 if total_issues > 0 else 0
            l_pct = (l_raw / total_issues) * 100 if total_issues > 0 else 0
            net_issues = adv - dec 
            print_h("抛压指标计算过程 (Daily)")
            print(f"1. Net Issues = Adv({adv}) - Dec({dec}) = {net_issues}")
            
            # --- [TRIN Upgrade Start] ---
            # 升级：TRIN 深度分析 (PDF 策略植入)
            trin_val = None
            trin_stat = 0
            trin_txt = "数据不足"
            trin_logic_short = "数据不足"
            
            if dec > 0 and dec_vol > 0 and adv_vol > 0:
                trin_val = (adv / dec) / (adv_vol / dec_vol)
                print(f"2. TRIN = {trin_val:.2f}")
                
                # --- 控制台深度输出 ---
                print("\n" + "-"*40)
                print(f"【TRIN 指标深度分析】(基于 PDF 实战标准)")
                print(f"   当前读数: {C.BOLD}{trin_val:.2f}{C.ENDC}")
                
                # 1. 区间判断
                status_desc = ""
                if trin_val < 0.5:
                    status_desc = f"{C.RED}极度强势/严重超买 (<0.5){C.ENDC} -> 警惕顶部"
                    trin_stat = 2 # 红色警告: 见顶风险
                    trin_txt = f"TRIN: {trin_val:.2f}\n极度超买 (<0.5)"
                    trin_logic_short = "极度贪婪 (<0.5)\n见顶风险极高"
                elif 0.5 <= trin_val <= 0.8:
                    status_desc = f"{C.GREEN}强势/买方主导 (0.5-0.8){C.ENDC} -> 健康上涨"
                    trin_stat = 0 # 安全
                    trin_txt = f"TRIN: {trin_val:.2f}\n强势买方 (0.5-0.8)"
                    trin_logic_short = "多头占优\n趋势健康"
                elif 0.8 < trin_val <= 1.2:
                    status_desc = f"{C.GREEN}中性/平衡 (0.8-1.2){C.ENDC} -> 观望/跟随"
                    trin_stat = 0 # 安全
                    trin_txt = f"TRIN: {trin_val:.2f}\n多空平衡 (0.8-1.2)"
                    trin_logic_short = "无明显方向\n跟随趋势"
                elif 1.2 < trin_val <= 2.0:
                    status_desc = f"{C.YELLOW}弱势/卖压显现 (1.2-2.0){C.ENDC} -> 谨慎减仓"
                    trin_stat = 1 # 预警
                    trin_txt = f"TRIN: {trin_val:.2f}\n卖压显现 (1.2-2.0)"
                    trin_logic_short = "空头稍强\n注意下行风险"
                elif trin_val > 2.0:
                    status_desc = f"{C.RED}极度恐慌/超卖 (>2.0){C.ENDC} -> 抄底机会"
                    # 这里虽然是买入机会，但对于“崩盘预警”来说，也是极度波动状态。
                    # 设为 1 (黄色) 或 2 (红色) 视作波动预警，但在文字中明确“抄底”。
                    # 按照 PDF >2.0 是 "Bold Buy"，为了区分“见顶风险”，这里给 Status 1 (Yellow) 提示关注。
                    # 或者给 Status 2 表示极端状态。用户希望“结论”，这里给黄色作为“强关注”。
                    trin_stat = 1 
                    trin_txt = f"TRIN: {trin_val:.2f}\n极度恐慌 (>2.0)"
                    trin_logic_short = "恐慌抛售 (>2.0)\n寻找抄底机会"

                if trin_val > 3.0:
                    status_desc = f"{C.HEADER}极端崩溃 (>3.0){C.ENDC} -> 神迹/强力抄底"
                    trin_stat = 1
                    trin_txt = f"TRIN: {trin_val:.2f}\n极端崩溃 (>3.0)"
                    trin_logic_short = "极端洗盘\n神级买点"

                print(f"   状态判定: {status_desc}")

                # 2. 趋势配合/背离分析 (Console Only)
                print(f"   趋势配合:")
                if spx_trend_up: # 大盘处于上升趋势 (50MA之上)
                    if trin_val < 1.0:
                        print(f"   \U0001f7e2 [健康] SPX上涨 + TRIN<1.0 -> 买气充足，升势稳健")
                    elif trin_val > 1.2:
                        print(f"   \U0001f534 [背离] SPX上涨 + TRIN>1.2 -> 价格涨但内部虚弱 (小心诱多)")
                    else:
                        print(f"   \u26aa [中性] SPX上涨 + TRIN正常")
                else: # 大盘处于下降趋势
                    if trin_val > 1.0:
                         print(f"   \U0001f7e2 [正常] SPX下跌 + TRIN>1.0 -> 正常的获利回吐/下跌")
                    elif trin_val < 0.8:
                         print(f"   \U0001f534 [背离] SPX下跌 + TRIN<0.8 -> 价格跌但内部惜售 (小心诱空)")

                # 3. 极值提示
                if trin_val < 0.5:
                    print(f"   \U0001f6a8 [警报] TRIN < 0.5: 无论大盘涨跌，均为短期【见顶】信号！")
                elif trin_val > 2.0:
                    print(f"   \U0001f4b0 [机会] TRIN > 2.0: 无论大盘多恐慌，均为短期【见底】信号！")
                
                print(f"   口诀: 低于0.5要当心(见顶)，高于2.0要激动(抄底)！")
                print("-" * 40 + "\n")

            else: 
                print("2. TRIN: 数据不足 (Adv/Dec/Vol 缺失)")
            # --- [TRIN Upgrade End] ---

            if adv_vol > 0:
                vr_calc = dec_vol / adv_vol
                print(f"3. Vol Ratio = {vr_calc:.2f}")
            else: print("3. Vol Ratio: 数据不足")
            print("\n")
            i_split = (h_pct > 2.2 and l_pct > 2.2)
            mco_condition = False
            if real_mco is not None: mco_condition = (real_mco < 0)
            else: mco_condition = (net_issues < 0)
            nymo_str = f"NYMO:{real_nymo:.2f}" if real_nymo is not None else "NYMO:N/A"
            mco_str = f"MCO_Off:{real_mco:.2f}" if real_mco is not None else "MCO:缺失"
            h_ctx = f"SPX状态: {trend_desc}\n{pos_str}\n新高:{h_raw}({h_pct:.2f}%) | 新低:{l_raw}({l_pct:.2f}%)\n{mco_str}"
            h_log = "趋势标准: 20/60/120/250均线综合\n& (新高/低同时>2.2%)\n& 新高 < 2×新低\n& MCO < 0"
            h_stat = 2 if (spx_trend_up and i_split and mco_condition) else (1 if i_split else 0)
            indicators.append(["Hindenburg Omen (凶兆)", h_stat, h_ctx, h_log])
            net_stat = 0
            if net_issues < -2000: net_stat = 2
            elif net_issues < -1000: net_stat = 1
            indicators.append(["抛压监测 I: 广度 (Net Issues)", net_stat, f"Net Issues: {net_issues}", "标准: <-1000 显著\n<-2000 恐慌"])
            
            # --- TRIN Append (Updated) ---
            indicators.append(["抛压监测 II: 力度 (TRIN Index)", trin_stat, trin_txt, trin_logic_short])
            # -----------------------------

            vol_stat = 0; vol_txt = "数据不足"
            if adv_vol > 0 and dec_vol > 0:
                dn_up_ratio = dec_vol / adv_vol
                if dn_up_ratio > 9.0: vol_stat = 2 
                elif dn_up_ratio > 4.0: vol_stat = 1
                def human_format(num):
                    num = float('{:.3g}'.format(num))
                    magnitude = 0
                    while abs(num) >= 1000:
                        magnitude += 1
                        num /= 1000.0
                    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
                vol_txt = f"Ratio (Dn/Up): {dn_up_ratio:.1f}\nUp: {human_format(adv_vol)} | Dn: {human_format(dec_vol)}"
            indicators.append(["抛压监测 III: 资金 (Vol Flow)", vol_stat, vol_txt, "标准: Dn/Up > 4.0 (资金出逃)\nDn/Up > 9.0 (极致洗盘)"])
        else:
            indicators.append(["Hindenburg Omen (凶兆)", 0, "抓取失败", "Firecrawl 无响应或数据无效"])
            indicators.append(["抛压监测 I: 广度", 0, "无数据", ""])
            indicators.append(["抛压监测 II: 力度", 0, "无数据", ""])
            indicators.append(["抛压监测 III: 资金", 0, "无数据", ""])

        tv_adv, tv_decl = self.scraper.fetch_tv_breadth_vision()
        if tv_adv is not None and tv_decl is not None:
            print_h("【重点数据】NASDAQ 广度 (源自 WSJ Text)")
            print(f"  \U0001f4c8 上涨家数 (ADV) : {tv_adv}") 
            print(f"  \U0001f4c9 下跌家数 (DECL): {tv_decl}") 
            print("") 
        if tv_adv and tv_decl:
            tv_ratio = round(tv_adv / tv_decl, 2)
            st = 2 if tv_ratio < 0.5 else (1 if tv_ratio < 1.0 else 0)
            indicators.append(["NASDAQ 广度 (A/D Ratio)", st, f"Adv: {tv_adv} | Dec: {tv_decl}\nRatio: {tv_ratio}", "标准: Ratio < 1.0 (跌多涨少)\nRatio < 0.5 (空头主导)"])
        else:
            indicators.append(["NASDAQ 广度 (A/D Ratio)", 0, "抓取失败", "无数据"])

        try:
            r = rsp/spy
            curr, ma = r.iloc[-1], r.rolling(50).mean().iloc[-1]
            chg = (curr/r.iloc[-20]-1)*100
            st = 2 if (curr<ma and chg<-2.0) else (1 if curr<ma else 0)
            indicators.append(["市场广度 (RSP vs SPY)", st, f"比率:{curr:.3f} (MA50:{ma:.3f})\n20日变化:{chg:.1f}%", "逻辑: 比率跌破50MA (广度变差)\n& 20日急跌(严重背离)<-2.0%"])
        except: indicators.append(["市场广度 (RSP vs SPY)", 0, "计算失败", "数据不足"])

        try:
            n_ok = nya.iloc[-1] > nya.rolling(50).mean().iloc[-1]
            st = 2 if (spx_trend_up and not n_ok) else (1 if not n_ok else 0)
            indicators.append(["全市场参与度 (^NYA)", st, f"SPX:{spx_trend_txt}\nNYA:{'强' if n_ok else '弱'}", "逻辑: SPX 强 (>50MA) 但 NYA 弱 (<50MA) = 风险触发"])
        except: indicators.append(["全市场参与度 (^NYA)", 0, "计算失败", "数据不足"])

        try:
            spr = tnx.iloc[-1] - irx.iloc[-1]
            indicators.append(["收益率倒挂 (10Y-3M)", 2 if spr<0 else 0, f"利差:{spr:.2f}%", "标准: 短端利率(3M) > 长端利率(10Y)\n(Fed黄金标准)"])
        except: indicators.append(["收益率倒挂 (10Y-3M)", 0, "计算失败", "数据不足"])

        if real_shiller:
            indicators.append(["Shiller PE (周期调整)", 2 if real_shiller>30 else 0, f"{real_shiller:.1f}", "标准: PE > 30 (高风险区)"])
        else:
            indicators.append(["Shiller PE (周期调整)", 0, "数据缺失", "Multpl源无响应"])

        if val_buffett:
            indicators.append(["巴菲特指标 (市值/GDP)", 2 if val_buffett>140 else 0, f"{val_buffett:.1f}%", "标准: 总市值/GDP > 140% (高估)"])
        else:
            indicators.append(["巴菲特指标 (市值/GDP)", 0, "数据缺失", "源无响应"])

        if margin_amt:
            is_high_risk = False
            if margin_ratio and margin_ratio >= 3.5: is_high_risk = True
            if val_margin_yoy is not None and val_margin_yoy > 50: is_high_risk = True
            ratio_str = f"{margin_ratio:.1f}%" if margin_ratio else "N/A"
            line1 = f"{margin_amt:.3f}万亿, GDP%:{ratio_str}"
            line2 = f"YoY:{val_margin_yoy:+.1f}%" if val_margin_yoy is not None else "YoY: N/A"
            st = 1 if is_high_risk else 0
            indicators.append(["美股保证金债务 Margin Debt", st, f"{line1}\n{line2}", "标准: GDP比≥3.5% (预警)\n或 YoY > 50%"])
        else:
            indicators.append(["美股保证金债务 Margin Debt", 0, "数据抓取无效", "FINRA源无响应"])

        try:
            v = vix.iloc[-1]
            chg = (v/vix.iloc[-15]-1)*100
            st = 2 if (v>25 or chg>40) else 0
            indicators.append(["VIX 恐慌指数 (异动)", st, f"现值:{v:.1f}\n14天涨幅:{chg:.0f}%", "标准: 14天涨幅>40% (提早预警)\n或 绝对值>25 (高压区)"])
        except: indicators.append(["VIX 恐慌指数 (异动)", 0, "数据不足", ""])

        if ma50_pct is not None:
            st = 2 if ma50_pct<40 else (1 if ma50_pct<60 else 0)
            indicators.append(["市场广度 (>50MA & >20MA)", st, f">50MA: {ma50_pct:.1f}%\n>20MA: {ma20_pct:.1f}%", "50MA: <60%警 <40%险\n20MA: <50%警 <30%险"])
        else:
            indicators.append(["市场广度 (>50MA & >20MA)", 0, "计算失败", "成分股获取失败"])

        # --- [RSI 顶背离核心升级模块 (V10.045 Tuning)] ---
        try:
            # 1. 算法升级: Wilder's Smoothing (更平滑) + 数学防呆
            delta = spx_weekly.diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            # [Fix] 防止除零错误 (虽罕见但稳健)
            loss = loss.replace(0, 1e-9) 
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # 2. 峰值检测引擎 (Tuned: 灵敏度提升)
            # [Tuning] 改为 +/- 1 周 (更灵敏，无需等待2周确认)
            is_peak = (spx_weekly > spx_weekly.shift(1)) & \
                      (spx_weekly > spx_weekly.shift(-1))
            
            # 获取所有峰值的日期
            peak_dates = spx_weekly[is_peak].index
            
            # 3. 背离判定逻辑
            divergence_detected = False
            div_msg = f"现值:{rsi.iloc[-1]:.1f} (无背离)"
            
            # 至少需要两个峰值才能比对
            if len(peak_dates) >= 2:
                # 获取最近的两个峰值时间 (Last 和 Previous)
                p2_date = peak_dates[-1] # 最近的一个峰值
                p1_date = peak_dates[-2] # 再前一个峰值
                
                # [Tuning] 时间窗口校验: 两个峰值之间不能太久 (比如60天)，否则失效
                # 且最近一个峰值要在近期
                days_between = (p2_date - p1_date).days
                days_since_last = (spx_weekly.index[-1] - p2_date).days
                
                if days_between < 60 and days_since_last < 45:
                    price_h2 = spx_weekly[p2_date]; price_h1 = spx_weekly[p1_date]
                    rsi_h2 = rsi[p2_date];       rsi_h1 = rsi[p1_date]
                    
                    # 判定条件:
                    # A. 价格创新高 (P2 > P1)
                    # B. RSI 未创新高 (R2 < R1)
                    # C. RSI 处于高位区间 (R1 > 60, 过滤弱势波动)
                    if price_h2 > price_h1 and rsi_h2 < rsi_h1 and rsi_h1 > 60:
                        divergence_detected = True
                        div_msg = f"顶背离确认!\n价格:{price_h1:.0f}->{price_h2:.0f}(新高)\nRSI:{rsi_h1:.1f}->{rsi_h2:.1f}(走低)"
            
            indicators.append(["RSI 周线顶背离", 2 if divergence_detected else 0, div_msg, "标准: 价格HH + RSI LH\n(灵敏度: +/-1周 | Wilder算法)"])
            
        except Exception as e:
            # print_err(f"RSI 计算出错: {e}") 
            indicators.append(["RSI 周线顶背离", 0, "计算失败", "数据不足"])

        try:
            # 【逻辑修正】牛市支撑带：由 20周SMA 和 21周EMA 共同构成的区间
            sma20 = spx_weekly.rolling(20).mean().iloc[-1]
            ema21 = spx_weekly.ewm(span=21, adjust=False).mean().iloc[-1]
            
            # 定义带状区域：取两者的最大值和最小值作为上下轨
            band_upper = max(sma20, ema21)
            band_lower = min(sma20, ema21)
            
            now = spx.iloc[-1] # 当前最新价格
            
            # 判断逻辑：只有价格跌破“下轨”才算真正跌破支撑带
            status = 2 if now < band_lower else 0
            
            # 优化显示：展示支撑带的范围
            msg = f"现价:{now:.0f}\n区间:{band_lower:.0f}~{band_upper:.0f}"
            indicators.append(["牛市支撑带 (20SMA/21EMA)", status, msg, "标准: 跌穿 20周SMA 与 21周EMA 构成的双线区间"])
            
        except Exception as e:
            # print(e) # 调试用
            indicators.append(["牛市支撑带 (20SMA/21EMA)", 0, "计算失败", "数据不足"])

        if real_fg is not None:
            indicators.append(["Fear & Greed", 2 if real_fg<45 else 0, f"指数:{real_fg} ({fg_source})", "标准: 指数进入恐惧区间 (< 45)\n/ 抓取失败时使用手动值"])
        else:
            indicators.append(["Fear & Greed", 0, "获取失败", "CNN源无响应"])

        try:
            if len(spx_weekly) > 30:
                e12 = spx_weekly.ewm(span=12, adjust=False).mean()
                e26 = spx_weekly.ewm(span=26, adjust=False).mean()
                macd = e12 - e26
                sig = macd.ewm(span=9, adjust=False).mean()
                m, s = macd.iloc[-1], sig.iloc[-1]
                mp, sp = macd.iloc[-2], sig.iloc[-2]
                dead = (mp>sp) and (m<s) and (m>0)
                state_str = "死叉 (触发)" if dead else ("金叉 (多头)" if m>s else "空头排列")
                indicators.append(["MACD 周线死叉", 2 if dead else 0, f"状态: {state_str}\nMACD:{m:.1f} Sig:{s:.1f}", "标准: 零轴上方 MACD 线向下穿越信号线"])
            else:
                indicators.append(["MACD 周线死叉", 0, "数据不足", ""])
        except: indicators.append(["MACD 周线死叉", 0, "计算错误", ""])

        if real_sahm is not None:
            indicators.append(["Sahm Rule (衰退规则)", 2 if real_sahm>=0.5 else 0, f"失业率升幅:{real_sahm:.2f}%", "标准: 早期预警(>0.2%)\n/ 确认衰退(>=0.5%)"])
        else:
            indicators.append(["Sahm Rule (衰退规则)", 0, "获取失败", "FRED源无响应"])

        if lei_depth is not None:
            st = 2 if lei_depth < -4.1 else 0
            indicators.append(["LEI 领先指标 (3Ds)", st, f"Depth:{lei_depth:.1f}%\nDiffusion:{lei_diff}", "标准: Depth < -4.1% & Diffusion ≤50 (衰退触发)\n/ Depth <0 或 Diffusion <100 (预警)"])
        else:
            indicators.append(["LEI 领先指标 (3Ds)", 0, "抓取失败", "Firecrawl/AI 无结果"])

        if pcr_avg is not None:
            st = 2 if pcr_avg < 0.8 else 0
            indicators.append(["CBOE Put/Call Ratio", st, f"读数: {pcr_curr:.2f}\n(源:10日均值版)", "标准: < 0.8 (贪婪/短线高点)\n> 1.1 (恐慌/短线低点)"])
        else:
            indicators.append(["CBOE Put/Call Ratio", 0, "抓取失败", "MacroMicro源无响应"])

        if val_nfci is not None:
            if val_nfci > -0.2: st = 2
            elif val_nfci > -0.35: st = 1
            else: st = 0
            indicators.append(["芝加哥金融状况指数 (NFCI)", st, f"读数:{val_nfci:.2f}", "标准: > -0.35 (预警)\n> -0.2 (触发)"])
        else:
            indicators.append(["芝加哥金融状况指数 (NFCI)", 0, "抓取失败", "源无响应"])

        ho_data = indicators[0] 
        new_net = indicators[1]
        new_trin = indicators[2]
        new_vol = indicators[3]
        new_tv = indicators[4] 
        rest = indicators[5:] 
        
        nymo_stat = 0
        nymo_txt = "数据获取失败"
        nymo_desc = "数据不足"
        if real_nymo is not None:
            if real_nymo < -60: 
                nymo_stat = 2; nymo_desc = "历史低谷区 (极度超卖)"
            elif real_nymo > 60:
                nymo_stat = 2; nymo_desc = "历史高峰区 (极度超买)"
            elif real_nymo < 0: 
                nymo_stat = 1; nymo_desc = "弱势区 (零轴下方)"
            else:
                nymo_desc = "中性区 (正常波动)"
            nymo_txt = f"读数: {real_nymo:.2f}\n【定性】{nymo_desc}"
            print_h("【简单结论】NYMO 广度")
            print(f"  当前读数: {real_nymo}")
            print(f"  区域判断: {nymo_desc}")
            print("-" * 30)
        
        nymo_data = ["StockCharts 广度 ($NYMO)", nymo_stat, nymo_txt, "极值: <-60恐慌底 / >+60过热顶\n趋势: 0轴上方看多 / 下方看空\n预警: 股价创新高但NYMO未跟(背离)"]
        
        final_list = [ho_data, nymo_data] + rest + [new_net, new_trin, new_vol, new_tv]
        
        return ho_data, final_list[1:]

    def generate_chart(self):
        ho_data, other_data = self.fetch_and_calculate()
        data = [ho_data] + other_data
        if not ho_data and not other_data: return

        risk_score = sum(1 for d in data if d and d[1] == 2) + sum(0.5 for d in data if d and d[1] == 1)
        
        fig = plt.figure(figsize=(33.06, 46.0), facecolor=self.colors['bg'])
        ax = fig.add_subplot(111)
        ax.axis('off')

        ax.text(0.5, 0.96, "美股崩盘预警系统 - 21因子 V10.049 (Bull Support Band Fix)", ha='center', va='center', fontsize=38, fontweight='bold', color=self.colors['title'])
        ax.text(0.5, 0.935, f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')} ", ha='center', va='center', fontsize=18, color='#CCCCCC')

        table_data = []
        if ho_data:
            _, h_st, h_ctx, h_log = ho_data
            st_txt = "【√】安全" if h_st == 0 else ("【!】触发" if h_st == 2 else "【!】预警")
            if "失败" in str(h_ctx) or "无效" in str(h_ctx): st_txt = "【?】缺失"
            
            c3 = h_ctx.split('\n')
            c4 = h_log.split('\n')
            
            val_row1 = '\n'.join(c3[:2]) if len(c3)>=2 else h_ctx
            log_row1 = '\n'.join(c4[:2])
            table_data.append([ho_data[0], st_txt, val_row1, log_row1])
            
            val_row2 = '\n'.join(c3[2:]) if len(c3)>2 else ""
            log_row2 = '\n'.join(c4[2:])
            table_data.append(['', st_txt, val_row2, log_row2])
        
        for d in other_data:
            st_txt = "【√】安全"
            if d[1] == 2: st_txt = "【!】触发"
            elif d[1] == 1: st_txt = "【!】预警"
            if "失败" in str(d[2]) or "缺失" in str(d[2]) or "不足" in str(d[2]): st_txt = "【?】缺失"
            table_data.append([d[0], st_txt, d[2], d[3]])
        
        table = ax.table(cellText=table_data, colLabels=['监测指标 (21因子)', '状态评级', '当前读数 (提供上下文)', '判断逻辑 (清晰标准)'], cellLoc='center', loc='center', colWidths=[0.25, 0.12, 0.25, 0.38]) 
        
        table.scale(1, 6.75) 
        table.auto_set_font_size(False); table.set_fontsize(23)

        # --- [UI 优化: 动态行距调整] ---
        # 寻找 RSI 所在的行号
        target_row_idx = -1
        for i, row_cont in enumerate(table_data):
            if row_cont and "RSI" in str(row_cont[0]):
                target_row_idx = i + 1 # +1 是因为 header 占了第 0 行
                break
        
        # 默认兜底 (如果没有找到 RSI，还是针对原定行)
        if target_row_idx == -1: target_row_idx = 12 

        # 扩大约 35% 以容纳 3 行文字
        target_height_factor = 1.35 
        
        std_height = table[0, 0].get_height()
        extra_h = std_height * (target_height_factor - 1.0)

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(self.colors['edge']); cell.set_linewidth(1.5)
            
            if row == target_row_idx:
                cell.set_height(std_height * target_height_factor)
            elif row > target_row_idx:
                current_y = cell.get_y()
                # 向下顺延 (y坐标减小)
                cell.set_y(current_y - extra_h)

            if row in [1, 2]: cell.set_edgecolor(self.colors['edge']) 

            if row == 0:
                cell.set_facecolor(self.colors['table_header']); cell.set_text_props(weight='bold', color='#FFFFFF')
            else:
                if row <= 2: idx = 0 
                elif row == 3: idx = 1 
                else: idx = row - 2 
                
                if idx >= len(data): continue
                lvl = data[idx][1]
                
                bg, c_txt = self.colors['row_safe'], self.colors['text_safe']
                val_txt = str(data[idx][2])
                if "失败" in val_txt or "缺失" in val_txt: bg = '#555555' 
                elif lvl == 2: bg, c_txt = self.colors['row_warn'], self.colors['text_warn']
                elif lvl == 1: bg, c_txt = self.colors['row_risk'], self.colors['text_risk']
                
                cell.set_facecolor(bg)
                cell.set_text_props(color=c_txt, weight='bold')
                if row == 2 and (col == 0 or col == 1): cell.set_text_props(color=bg)

        if risk_score <= 5: msg, clr = f"风险评分 {risk_score:.1f}/21.0 - 市场结构大致健康，可保持观察", self.colors['text_safe']
        elif risk_score <= 10: msg, clr = f"风险评分 {risk_score:.1f}/21.0 - 内部背离，中期风险累积，建议谨慎", self.colors['text_risk']
        else: msg, clr = f"严重警告：风险评分 {risk_score:.1f}/21.0 - 崩盘信号共振，建议立即减仓", self.colors['text_warn']
        
        ax.text(0.5, 0.05, msg, ha='center', va='center', fontsize=34, fontweight='bold', color=clr)

        try:
            save_dir = Path(os.path.dirname(os.path.abspath(__file__))) 
            fname = save_dir / f"Warning_21Factors_Pro_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
            plt.savefig(fname, bbox_inches='tight', facecolor=self.colors['bg'], dpi=400) 
            plt.close()
            print_ok(f"报表已生成: {fname}")
            if platform.system() == 'Windows': os.startfile(str(fname))
            else: webbrowser.open(str(fname))
        except Exception as e:
            print_err(f"图片保存失败: {e}")

# ==============================================================================
# 模块：板块轮动引擎 (Fix: 白底 + 汉字乱码修复 + 大白话坐标 + 10日爆发)
# ==============================================================================
class SectorRotationEngine:
    def __init__(self):
        self.sectors = {
            'XLK': '科技', 'XLF': '金融', 'XLV': '医疗',
            'XLE': '能源', 'XLY': '可选消费', 'XLP': '必选消费',
            'XLI': '工业', 'XLC': '通讯', 'XLB': '材料',
            'XLRE': '房地产', 'XLU': '公用事业', 'SPY': '基准'
        }
        self.rs_window = 60 
        self.mom_window = 10 

    def run_analysis(self):
        print("\n" + "="*75)
        print(f" \U0001f504 启动板块轮动分析模块 (Sector Rotation RRG) - {datetime.now().strftime('%Y-%m-%d')}") 
        print("="*75)
        
        try:
            tickers = list(self.sectors.keys())
            start_date = (datetime.now() - timedelta(days=300)).strftime('%Y-%m-%d')
            # [Fix] 使用缓存下载
            data = self._get_data_with_cache(tickers, start_date)
            
            if data is None or data.empty:
                print_err("数据下载失败，跳过板块轮动分析。")
                return None

            results = self._calculate_rrg(data)
            short_term_movers = self._calculate_10d_movers(data)
            self._print_console_summary(results, short_term_movers)
            
            return {
                "results": results,
                "summary": self._generate_summary_text(results, short_term_movers)
            }
        except Exception as e:
            print_err(f"板块轮动分析异常: {e}")
            return None

    def _get_data_with_cache(self, tickers, start_date):
        # [Fix] 调用全局缓存函数
        raw_data = get_cached_sector_data(tickers, start_date)
        
        if raw_data.empty: return None

        data = None
        if isinstance(raw_data.columns, pd.MultiIndex):
            try:
                data = raw_data['Adj Close']
            except KeyError:
                try:
                    data = raw_data['Close']
                    print_info("提示: 使用 'Close' 列代替 'Adj Close'")
                except KeyError:
                    print_err("未能在下载数据中找到价格列")
                    return None
        else:
            if 'Adj Close' in raw_data:
                data = raw_data['Adj Close']
            elif 'Close' in raw_data:
                data = raw_data['Close']
            else:
                data = raw_data
        return data

    def _calculate_10d_movers(self, data):
        if 'SPY' not in data.columns: return []
        
        movers = []
        spy = data['SPY']
        spy_10d = (spy.iloc[-1] - spy.iloc[-11]) / spy.iloc[-11] if len(spy) > 11 else 0

        for ticker in data.columns:
            if ticker == 'SPY': continue
            series = data[ticker]
            if len(series) > 11:
                pct_10d = (series.iloc[-1] - series.iloc[-11]) / series.iloc[-11]
                alpha_10d = pct_10d - spy_10d
                movers.append({
                    'Ticker': ticker,
                    'Name': self.sectors.get(ticker, ticker),
                    'Alpha_10d': alpha_10d * 100 
                })
        
        movers.sort(key=lambda x: x['Alpha_10d'], reverse=True)
        return movers[:3] 

    def _calculate_rrg(self, data):
        rs_data = pd.DataFrame()
        if 'SPY' not in data.columns:
            raise ValueError("基准数据 SPY 缺失")

        benchmark = data['SPY']
        
        for ticker in data.columns:
            if ticker != 'SPY':
                rs_data[ticker] = data[ticker] / benchmark
        
        x_values = []; y_values = []; names = []
        quadrants = []
        
        for ticker in rs_data.columns:
            rs_series = rs_data[ticker]
            ma_rs = rs_series.rolling(window=self.rs_window).mean()
            rs_ratio = 100 * (rs_series / ma_rs)
            rs_mom = 100 + ((rs_series - rs_series.shift(self.mom_window)) / rs_series.shift(self.mom_window) * 100)
            
            if rs_ratio.dropna().empty or rs_mom.dropna().empty:
                continue

            latest_x = rs_ratio.dropna().iloc[-1]
            latest_y = rs_mom.dropna().iloc[-1]
            
            q = "Unknown"
            if latest_x > 100 and latest_y > 100: q = "Leading (领涨)"
            elif latest_x < 100 and latest_y > 100: q = "Improving (改善)"
            elif latest_x < 100 and latest_y < 100: q = "Lagging (落后)"
            elif latest_x > 100 and latest_y < 100: q = "Weakening (转弱)"
            
            names.append(ticker)
            x_values.append(latest_x)
            y_values.append(latest_y)
            quadrants.append(q)
            
        return pd.DataFrame({
            'Ticker': names,
            'Name': [self.sectors.get(t, t) for t in names], 
            'RS_Ratio': x_values,
            'RS_Momentum': y_values,
            'Quadrant': quadrants
        })

    def _print_console_summary(self, df, movers):
        print("\n\U0001f4ca [RRG 象限分布] - 研报版") 
        for q in ["Leading (领涨)", "Improving (改善)", "Weakening (转弱)", "Lagging (落后)"]:
            items = df[df['Quadrant'] == q]
            if not items.empty:
                ticks = ", ".join([f"{r['Name']}" for _, r in items.iterrows()])
                icon = "\U0001f7e2" if "Leading" in q else ("\U0001f535" if "Improving" in q else ("\U0001f7e1" if "Weakening" in q else "\U0001f534")) 
                print(f"   {icon} {q}: {ticks}")
        
        print("\n\U0001f680 [10日 资金抢筹榜] (短期爆发力)") 
        if movers:
            for m in movers:
                print(f"   \U0001f525 {m['Name']}: 跑赢大盘 {m['Alpha_10d']:.2f}%") 
        else:
            print("   (近期无明显异动板块)")
        print("="*75 + "\n")

    def _generate_summary_text(self, df, movers):
        leaders = df[df['Quadrant'] == "Leading (领涨)"]['Name'].tolist()
        improvers = df[df['Quadrant'] == "Improving (改善)"]['Name'].tolist()
        movers_str = "无"
        if movers:
            movers_str = ", ".join([f"{m['Name']}(+{m['Alpha_10d']:.1f}%)" for m in movers])
        period_info = f"(基于日线: {self.rs_window}日趋势 vs {self.mom_window}日动量)"
        return f"领涨板块: {', '.join(leaders) if leaders else '无'}\n改善板块: {', '.join(improvers) if improvers else '无'}\n🚀 10日抢筹: {movers_str}\n{period_info}"

# ==========================================
# 【附加功能：FRED 收益率曲线/失业率红绿灯】
# ==========================================
def run_fred_traffic_light(fred_key):
    print("\n" + "="*50)
    print("\U0001f6a6 收益率曲线 + 失业率红绿灯系统 (FRED直连 - 智能修复版)") 
    print("="*50)
    
    def get_valid_fred_data(series_id, count=1):
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={fred_key}&file_type=json&sort_order=desc&limit=10"
        try:
            r = requests.get(url, timeout=10).json()
            obs_list = r.get('observations', [])
            clean_data = []
            for obs in obs_list:
                val = obs['value']
                if val != '.': 
                    try:
                        clean_data.append({'value': float(val), 'date': obs['date']})
                    except: pass
                if len(clean_data) >= count: break
            return clean_data
        except Exception as e:
            print_err(f"获取 {series_id} 失败: {e}")
            return []

    try:
        data_curve = get_valid_fred_data('T10Y2Y', 1)
        if not data_curve:
            print_err("T10Y2Y 数据获取失败")
            return
        val_curve = data_curve[0]['value']
        date_curve = data_curve[0]['date']
        data_unrate = get_valid_fred_data('UNRATE', 2)
        if len(data_unrate) < 2:
            print_err("UNRATE 数据不足")
            return

        val_unrate = data_unrate[0]['value']
        date_unrate = data_unrate[0]['date']
        prev_unrate = data_unrate[1]['value']
        
        print(f"数据源: St. Louis Fed (API Key已验证)")
        print(f"1. 10Y-2Y 利差 (T10Y2Y): {val_curve:+.2f}%  (日期: {date_curve})")
        print(f"2. 失业率 (UNRATE)     : {val_unrate}%  (日期: {date_unrate}) [前值: {prev_unrate}%]")
        print("-" * 50)

        signal = ""
        advice = ""
        if val_curve < 0 and (val_unrate > 5.0 or val_unrate > prev_unrate):
            signal = "\U0001f534\U0001f534 红灯 (大衰退警报)" 
            advice = "赶紧减仓或卖出！衰退风险极高，股市大概率大跌，转防御股或现金。"
        elif val_curve < 0 and val_unrate < 5.0:
            signal = "\U0001f7e1 黄灯 (左侧预警)" 
            advice = "先观望，别急着全仓卖，但不要猛加仓，准备防守。"
        elif val_curve > 0 and val_unrate >= 5.0:
             signal = "\U0001f7e1 黄灯 (经济放缓)" 
             advice = "小心点，关注后续数据，可能经济在放缓，适当减仓。"
        elif val_curve > 0 and val_unrate < prev_unrate:
            signal = "\U0001f7e2\U0001f7e2 超级绿灯 (最佳买点)" 
            advice = "最佳买入时机！往往是大牛市起点，大胆加仓周期股和成长股。"
        elif val_curve > 0 and val_unrate < 4.5:
             signal = "\U0001f7e2 绿灯 (安全期)" 
             advice = "放心大胆买！经济扩张期，股市最好。"
        else:
             signal = "\U0001f7e2 绿灯 (当前稳健)" 
             advice = "继续持有或加仓！经济还稳，股市有支撑。"

        print(f"\U0001f6a6 信号灯状态: {signal}") 
        print(f"\U0001f4a1 操作建议  : {advice}") 
        print("="*50 + "\n")

    except Exception as e:
        print_err(f"FRED API 调用失败: {e}")

# ==========================================
# 【新增集成：FRED V10.003 精简版】
# ==========================================
def run_fred_v10_dashboard(api_key):
    C_PURPLE = '\033[95m'; C_CYAN = '\033[96m'; C_BLUE = '\033[94m'; C_END = '\033[0m'
    masked_key = api_key[:6] + "..." if len(api_key) > 6 else "xxxx..."
    print(f"{C_PURPLE}▬ ₪  FRED 集成版 (V10.003) - 补充宏观快照  ▬{C_END}")
    print(f"{C_CYAN}\U0001f539 正在连接 St. Louis Fed (Key: {masked_key})...{C_END}") 
    
    try:
        from fredapi import Fred
        if 'fred' not in locals(): fred = Fred(api_key=api_key)
        curve_series = fred.get_series('T10Y2Y', sort_order='desc', limit=1)
        curve_val = curve_series.iloc[0]
    except: curve_val = 0.0

    try:
        vix_dat = yf.Ticker("^VIX").history(period="1d")
        if not vix_dat.empty: vix_val = vix_dat['Close'].iloc[-1]
        else: vix_val = 0.0
    except: vix_val = 0.0

    current_date_str = datetime.now().strftime('%Y-%m-%d')
    if vix_val < 20: vix_status = "\U0001f7e2 正常" 
    elif vix_val > 30: vix_status = "\U0001f534 恐慌" 
    else: vix_status = "\U0001f7e1 警告" 

    if curve_val > 0: yield_status = "\U0001f7e2 正向" 
    else: yield_status = "\U0001f534 倒挂" 

    print("") 
    print("-" * 40)
    print(f"\U0001f4ca 宏观与市场快照 ({current_date_str})") 
    print("-" * 40)
    print(f"1. 市场恐慌指数 VIX: {vix_val:.2f} ({vix_status})")
    print(f"2. 10Y-2Y 收益率差 : {curve_val:.2f}% ({yield_status})")
    print("-" * 40)
    print("")

# ==========================================
# 【NEW MODULE】SMT 背离分析引擎 (V3 Pro - 经典回归+深度解读)
# ==========================================
class SMTDivergenceAnalyzer:
    def __init__(self):
        # 1. 经典四指数 (ETF/Index) - 保持原汁原味
        self.tickers_classic = ['^IXIC', '^GSPC', 'QQQ', 'SPY']
        
        # 2. Pro 级标的：期货与等权
        self.tickers_pro = ['NQ=F', 'ES=F', 'RSP']
        
        self.all_tickers = self.tickers_classic + self.tickers_pro
        
        self.names = {
            '^IXIC': '纳指(IXIC)', '^GSPC': '标普(SPX)', 
            'QQQ': '纳指ETF(QQQ)', 'SPY': '标普ETF(SPY)',
            'NQ=F': '纳指期货(NQ)', 'ES=F': '标普期货(ES)',
            'RSP': '标普等权(RSP)'
        }
        # 【修改】增加 3日 极速窗口，响应你的灵敏度需求
        self.periods = [3, 5, 10, 20, 60] 
        self.signals = [] # 收集所有信号用于总结

    def run(self):
        print("\n" + "="*75)
        print(f" \U0001f9ed 启动 SMT 背离分析模块 (Pro V3) - {datetime.now().strftime('%Y-%m-%d')}")
        print("="*75)

        # [Fix] 使用缓存下载
        df_close = self._get_data_with_cache()
        if df_close is None: return

        print_ok("数据获取成功，开始计算...")
        print("-" * 75)

        # 2. 经典 SMT (恢复老版样式)
        print_h("1. 经典 SMT 分析 (纳指/标普/QQQ/SPY)")
        for period in self.periods:
            self._analyze_classic_style(df_close, period)
        
        print("-" * 75)
        
        # 3. Pro SMT (增强信息量)
        print_h("2. 进阶 SMT 分析 (期货 & 市场广度)")
        print_info("💡 期货(NQ/ES)包含夜盘，反应更真实；SPY/RSP揭示只有巨头在涨还是普涨。")
        self._analyze_pro_futures(df_close, 10) # 10日是期货背离黄金窗口
        self._analyze_pro_breadth(df_close, 20) # 20日看广度最准
        
        print("-" * 75)

        # 4. 关键位与入场
        self._analyze_entry_signals(df_close)

        # 5. 市场总评
        self._summarize_market()

        # 6. 图例
        self._print_legend()
    
    def _get_data_with_cache(self):
         # [Fix] 调用全局缓存函数
        data = get_cached_smt_data(self.all_tickers, "6mo")
        
        if isinstance(data.columns, pd.MultiIndex):
            try: df_close = data['Close']
            except KeyError: df_close = data 
        else:
            df_close = data
        
        df_close = df_close.ffill().dropna() 
        
        if df_close.empty:
            print_err("SMT 数据下载为空，跳过分析。")
            return None
        return df_close

    # --- 风格1：经典老版样式 (你喜欢的) ---
    def _analyze_classic_style(self, df, period):
        if len(df) < period + 2: return
        
        target_tickers = self.tickers_classic
        window_df = df.iloc[-(period+1):]
        current_prices = window_df.iloc[-1]
        period_highs = window_df.max()
        period_lows = window_df.min()
        
        made_new_high = []
        made_new_low = []
        
        for t in target_tickers:
            if t not in df.columns: continue
            if current_prices[t] >= period_highs[t] * 0.9995: made_new_high.append(t)
            if current_prices[t] <= period_lows[t] * 1.0005: made_new_low.append(t)
            
        # 只打印有信号的窗口，避免刷屏
        if not made_new_high and not made_new_low:
            return 

        print(f"[{period}日窗口]")
        
        if len(made_new_high) > 0 and len(made_new_high) < len(target_tickers):
            failed = [self.names[t] for t in target_tickers if t not in made_new_high]
            success = [self.names[t] for t in made_new_high]
            msg = f"**看跌背离 (Bearish)** - 预示顶部"
            print(f"   \U0001f534 状态: {msg}") 
            print(f"   -> 创新高: {', '.join(success)}")
            print(f"   -> 未确认: {', '.join(failed)} (虚弱)")
            self.signals.append(-1)
        
        elif len(made_new_low) > 0 and len(made_new_low) < len(target_tickers):
            failed = [self.names[t] for t in target_tickers if t not in made_new_low]
            success = [self.names[t] for t in made_new_low]
            msg = f"**看涨背离 (Bullish)** - 预示底部"
            print(f"   \U0001f7e2 状态: {msg}") 
            print(f"   -> 创新低: {', '.join(success)}")
            print(f"   -> 未确认: {', '.join(failed)} (抗跌)")
            self.signals.append(1)
            
        elif len(made_new_high) == len(target_tickers):
            print(f"   \U0001f525 状态: 强多头共振 (全部创新高)") 
            self.signals.append(0.5)
        elif len(made_new_low) == len(target_tickers):
            print(f"   \U0001f9ca 状态: 强空头共振 (全部创新低)") 
            self.signals.append(-0.5)

    # --- 风格2：Pro 期货分析 (信息更充分) ---
    def _analyze_pro_futures(self, df, period):
        t1, t2 = 'NQ=F', 'ES=F'
        if t1 not in df.columns or t2 not in df.columns: return
        
        w = df.iloc[-(period+1):]
        curr = w.iloc[-1]
        highs = w.max()
        lows = w.min()
        
        # 判定
        nq_high = curr[t1] >= highs[t1] * 0.9995
        es_high = curr[t2] >= highs[t2] * 0.9995
        nq_low = curr[t1] <= lows[t1] * 1.0005
        es_low = curr[t2] <= lows[t2] * 1.0005
        
        res = ""
        detail = ""
        if nq_high and not es_high:
            res = "\U0001f534 [看跌] 科技拉升，标普不跟"
            detail = "解读: 资金只敢做多高流动性的纳指，不敢全面做多，是诱多信号。"
            self.signals.append(-2)
        elif not nq_high and es_high:
            res = "\U0001f534 [看跌] 标普补涨，科技滞涨"
            detail = "解读: 领头羊纳指动能衰竭，补涨通常是行情尾声。"
            self.signals.append(-1)
        elif nq_low and not es_low:
            res = "\U0001f7e2 [看涨] 纳指杀跌，标普拒绝"
            detail = "解读: 科技股恐慌抛售，但大盘蓝筹拒绝创新低，有护盘资金。"
            self.signals.append(2)
        elif not nq_low and es_low:
            res = "\U0001f7e2 [看涨] 标普新低，纳指抗跌"
            detail = "解读: 领头羊纳指率先止跌，通常是反转先兆。"
            self.signals.append(1)
        else:
            res = "\u26aa [中性] 期货步调一致"
            
        print(f"📊 [{period}日 期货SMT]: {res}")
        if detail: print(f"   {detail}")

    # --- 风格3：Pro 广度分析 (RSP) ---
    def _analyze_pro_breadth(self, df, period):
        t1, t2 = 'SPY', 'RSP'
        if t1 not in df.columns or t2 not in df.columns: return
        
        w = df.iloc[-(period+1):]
        curr = w.iloc[-1]
        start = w.iloc[0]
        highs = w.max()
        
        # 1. 经典新高检测
        spy_high = curr[t1] >= highs[t1] * 0.9995
        rsp_high = curr[t2] >= highs[t2] * 0.9995
        
        # 2. 相对涨幅 (Performance Check)
        spy_perf = (curr[t1] - start[t1]) / start[t1] * 100
        rsp_perf = (curr[t2] - start[t2]) / start[t2] * 100
        
        # 3. 判定逻辑
        # 情况A: SPY创新高，RSP没创新高，且RSP涨幅落后SPY -> 危险 (只有巨头在涨)
        if spy_high and not rsp_high and spy_perf > rsp_perf:
            print(f"📊 [{period}日 内部背离]: \U0001f534 极度危险 (虚假繁荣)")
            print(f"   数据: SPY(+{spy_perf:.1f}%) 创新高 | RSP(+{rsp_perf:.1f}%) 滞涨")
            print(f"   解读: 只有几只巨头(SPY权重)在涨，490只成分股(RSP)没跟。")
            self.signals.append(-2)
        
        # 情况B: 虽然RSP没创新高，但是跑赢了SPY (或涨幅差不多) -> 良性轮动
        elif spy_high and not rsp_high and rsp_perf >= spy_perf:
            print(f"📊 [{period}日 广度修复]: \U0001f7e2 良性轮动 (RSP跑赢)")
            print(f"   数据: RSP(+{rsp_perf:.1f}%) > SPY(+{spy_perf:.1f}%)")
            print(f"   解读: 虽然RSP未创新高(前期跌多了)，但近期反弹强于大盘，市场广度在变好。")
            self.signals.append(1)

        # 情况C: 普涨
        elif spy_high and rsp_high:
            print(f"📊 [{period}日 内部健康]: \U0001f7e2 市场普涨 (健康牛市)")
            self.signals.append(1)
            
        else:
             print(f"📊 [{period}日 市场广度]: \u26aa 正常波动 (RSP: {rsp_perf:.1f}%)")


    # --- Vincent 策略: 入场信号 (更清晰的标准) ---
    def _analyze_entry_signals(self, df):
        print_h("3. 关键位与入场信号 (Vincent 策略)")
        
        for ticker in ['SPY', 'QQQ']:
            if ticker not in df.columns: continue
            close = df[ticker]
            curr = close.iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1]
            high20 = close.iloc[-20:].max()
            
            # 计算距离
            dist_ma20 = (curr - ma20) / ma20 * 100
            
            print(f"📌 {self.names[ticker]} 价格行为:")
            print(f"   现价: {curr:.2f} (MA20: {ma20:.2f})")
            
            # 判断逻辑
            if abs(dist_ma20) < 0.6 and curr > ma20:
                print(f"   🔥 [信号]: 完美回踩 MA20")
                print(f"   👉 操作: 若 SMT 同时出现看涨背离，则是绝佳【做多】点位。")
            elif abs(dist_ma20) < 0.6 and curr < ma20:
                print(f"   ❄️ [信号]: 反抽 MA20 受阻")
                print(f"   👉 操作: 若 SMT 同时出现看跌背离，则是绝佳【做空】点位。")
            elif (high20 - curr)/curr < 0.005:
                print(f"   🚧 [信号]: 逼近前高阻力")
                print(f"   👉 操作: 观察是否假突破(SFP)。若创新高后迅速跌回，做空。")
            else:
                print(f"   🌊 [状态]: 趋势运行中，等待关键位测试...")
            print("")

    # --- 新增: 市场趋势总结 ---
    def _summarize_market(self):
        print_h("4. \U0001f31f 市场趋势总汇 (Executive Summary)")
        
        bull_score = sum([s for s in self.signals if s > 0])
        bear_score = sum([abs(s) for s in self.signals if s < 0])
        
        trend = ""
        if bear_score > bull_score and bear_score >= 2:
            trend = "\U0001f534 趋势转弱 (空头占优)"
            action = "防守/减仓，关注做空机会"
        elif bull_score > bear_score and bull_score >= 2:
            trend = "\U0001f7e2 趋势增强 (多头占优)"
            action = "持股待涨，寻找回踩做多机会"
        else:
            trend = "\u26aa 趋势震荡 (多空纠缠)"
            action = "多看少动，等待SMT共振信号"
            
        print(f"   总评: {trend}")
        print(f"   建议: {action}")
        print(f"   信号强度: 多头({bull_score}) vs 空头({bear_score})")

    def _print_legend(self):
        print("\n" + "-"*75)
        print("【SMT Pro 策略说明书】")
        print("1. \U0001f525 期货先行: NQ/ES 期货包含夜盘，比ETF早 1-4 小时反应。")
        print("2. \u2696\ufe0f 内部广度: 若 SPY 涨但 RSP 跌 = 虚假繁荣 (看跌)。")
        print("3. \U0001f3af Vincent战法: SMT只是过滤器，必须配合“关键位”。")
        print("   - 买入公式: SMT看涨背离 + 价格回踩MA20不破。")
        print("   - 卖出公式: SMT看跌背离 + 价格假突破前高 (或跌破MA20)。")
        print("="*75 + "\n")


if __name__ == "__main__":
    try:
        app = CrashWarningSystem()
        
        # 1. 核心图片与报告 
        app.generate_chart()
        
        # 2. 附加功能模块
        # [安全修复] 下面这行也改为从 secrets 读取
        try:
            run_fred_traffic_light(USER_FRED_KEY)
            run_fred_v10_dashboard(USER_FRED_KEY)
        except NameError:
            print("FRED Key 未配置，跳过附加模块。")
        
        # 3. 趋势分析 (深度宏观)
        app.analyze_market_trends_console()
        
        # 4. 板块轮动模块
        try:
            sr_engine = SectorRotationEngine()
            sr_engine.run_analysis()
        except Exception as e:
            print(f"\n{C.RED}\u274c 板块轮动模块运行中断: {e}{C.ENDC}")

        # 5. SMT 背离分析模块 (放在最后)
        try:
            # 确保此类在此之前已定义
            smt_analyzer = SMTDivergenceAnalyzer()
            smt_analyzer.run()
        except Exception as e:
            print(f"\n{C.RED}\u274c SMT分析模块运行中断: {e}{C.ENDC}")
            traceback.print_exc()
            
    except Exception as e:
        print_err(f"程序运行出错: {e}")
        traceback.print_exc() 
    print("\n")
    # input(">>> 计算完成。按 Enter 键退出程序...")

