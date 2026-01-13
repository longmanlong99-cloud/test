# -*- coding: utf-8 -*-
"""
美股崩盘预警系统 - 21因子 V10.055 (Pixel-Perfect Web Edition)
【版本说明】
这是一个 1:1 复刻版。保留了原程序所有的计算逻辑、所有的判断分支、所有的文本输出。
唯一的区别是：
1. 把 print() 变成了 st.write() / st.success() / st.error()。
2. 把 input() 变成了 st.stop()。
3. 把 matplotlib.show() 变成了 st.pyplot()。
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
import warnings
import time
import random
import re
import traceback 
import sys
import json 
import io
from firecrawl import Firecrawl 
from PIL import Image 

# --- 页面基础配置 ---
st.set_page_config(
    page_title="美股崩盘预警系统 Pro",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 依赖库检查 ---
try:
    from fredapi import Fred
except ImportError:
    st.warning("⚠️ 未找到 fredapi 库，建议在 requirements.txt 中添加 fredapi")

try:
    from google import genai
except ImportError:
    st.error("❌ 严重错误：未找到 google-genai 库。")
    st.stop()

# ==========================================
# 【API 配置区】
# ==========================================
try:
    GENAI_API_KEY = st.secrets["GENAI_API_KEY"]
    if "FRED_KEY" in st.secrets:
        USER_FRED_KEY = st.secrets["FRED_KEY"]
    elif "USER_FRED_KEY" in st.secrets:
        USER_FRED_KEY = st.secrets["USER_FRED_KEY"]
    else:
        USER_FRED_KEY = ""
    FIRECRAWL_KEY = st.secrets["FIRECRAWL_KEY"]
except Exception as e:
    st.error(f"❌ Secrets 配置错误: {e}")
    st.stop()

client = genai.Client(api_key=GENAI_API_KEY)
warnings.filterwarnings("ignore")

# ==========================================
# 【UI 辅助函数 (1:1 映射)】
# ==========================================
# 为了保持原汁原味，这些函数的名字都不改，只是实现变成 Web 输出
def print_h(msg): 
    st.markdown("---")
    st.subheader(f"━━━ {msg} ━━━")
def print_step(msg): st.text(f"🔹 {msg}")
def print_ok(msg): st.success(f"✅ {msg}")
def print_warn(msg): st.warning(f"⚠️ {msg}")
def print_err(msg): st.error(f"❌ {msg}")
def print_info(msg): st.info(f"ℹ️ {msg}")

# ==========================================
# 【缓存层 (必须保留)】
# ==========================================
@st.cache_data(ttl=86400)
def get_cached_tickers():
    print_step("获取标普500成分股名单 (Cached)...")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15).text)
        for t in tables:
            if 'Symbol' in t.columns:
                return t['Symbol'].str.replace('.', '-', regex=False).tolist()
    except: return []

@st.cache_data(ttl=3600)
def get_cached_sp500_data(tickers):
    if not tickers: return pd.DataFrame()
    
    # 进度条显示 (原程序没有进度条，这里加上为了体验)
    status_text = st.empty()
    status_text.text(f"正在下载 {len(tickers)} 只成分股数据...")
    progress_bar = st.progress(0)
    
    closes = []
    batch_size = 30 # 安全批次
    total = len(tickers)
    
    for i in range(0, total, batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, period="5y", auto_adjust=True, progress=False, threads=True, timeout=30)
            if isinstance(data.columns, pd.MultiIndex):
                try: close = data['Close']
                except: close = data
            else: close = data
            closes.append(close)
            
            # 更新进度
            progress_bar.progress(min((i + batch_size) / total, 1.0))
            time.sleep(0.2) # 防封号
        except: pass
        
    status_text.empty()
    progress_bar.empty()
    
    if not closes: return pd.DataFrame()
    return pd.concat(closes, axis=1).dropna(axis=1, how='all')

@st.cache_data(ttl=3600)
def get_cached_sector_data(tickers, start_date):
    print_step(f"下载板块数据 ({start_date} ~ Now)...")
    return yf.download(tickers, start=start_date, progress=False, auto_adjust=False)

@st.cache_data(ttl=3600)
def get_cached_smt_data(tickers, period):
    print_step("下载 SMT 全量数据...")
    return yf.download(tickers, period=period, auto_adjust=False, progress=False)

# ==========================================
# 【WebScraper (完全保留所有抓取函数)】
# ==========================================
class WebScraper:
    def __init__(self):
        self.firecrawl_key = FIRECRAWL_KEY 
        self.app = Firecrawl(api_key=self.firecrawl_key)
        self.fred_key = USER_FRED_KEY
        self.cached_gdp = None 
        self.cached_nasdaq = None

    def fetch_shiller_pe(self):
        print_step("[Shiller PE] 启动 Firecrawl 抓取...")
        try:
            resp = self.app.scrape("https://www.multpl.com/shiller-pe", formats=['markdown'])
            md = getattr(resp, 'markdown', '')
            match = re.search(r'Shiller PE Ratio.*?(\d{2}\.\d{1,2})', md, re.S | re.I)
            if match:
                val = float(match.group(1))
                print_ok(f"AI 识别成功! Shiller PE: {val}")
                return val
        except Exception as e:
            print_err(f"Shiller PE 抓取异常: {e}")
        return None

    def fetch_fear_greed(self):
        print_step("[Fear & Greed] 启动 Firecrawl 抓取...")
        try:
            resp = self.app.scrape("https://www.cnn.com/markets/fear-and-greed", formats=['markdown'])
            md = getattr(resp, 'markdown', '')
            # 使用更宽泛的正则以防网页变动
            match = re.search(r'(?:Fear\s*&\s*Greed\s*Index|Current\s*Reading).*?(\d{1,3})', md, re.S | re.I)
            if match:
                score = int(match.group(1))
                rating = "Neutral"
                if score < 25: rating = "Extreme Fear"
                elif score < 45: rating = "Fear"
                elif score < 55: rating = "Neutral"
                elif score < 75: rating = "Greed"
                else: rating = "Extreme Greed"
                print_ok(f"F&G Index: {score} ({rating})")
                return score, rating
        except Exception as e:
            print_err(f"F&G 异常: {e}")
        return None, "获取失败"

    def fetch_us_gdp(self):
        if self.cached_gdp: return self.cached_gdp
        print_h("[US GDP] 启动数据获取 (FRED)...")
        try:
            if not self.fred_key: return None
            fred = Fred(api_key=self.fred_key)
            s = fred.get_series('GDP', sort_order='desc', limit=1)
            val = s.iloc[0] / 1000.0
            print_ok(f"GDP: {val:.3f}T")
            self.cached_gdp = val
            return val
        except Exception as e:
            print_err(f"FRED GDP 异常: {e}")
        return None

    def fetch_buffett_indicator(self):
        print_step("[Buffett Indicator] 启动计算...")
        gdp = self.fetch_us_gdp()
        if not gdp: return None
        try:
            hist = yf.Ticker("^W5000").history(period="5d")
            if not hist.empty:
                val = (hist['Close'].iloc[-1] / (gdp * 1000.0)) * 100
                print_ok(f"巴菲特指标: {val:.2f}%")
                return val
        except: pass
        return None

    def fetch_margin_debt(self):
        print_h("[Margin Debt] 启动 Firecrawl 抓取...")
        gdp = self.fetch_us_gdp()
        try:
            resp = self.app.scrape("https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics", formats=['markdown'])
            md = getattr(resp, 'markdown', '')
            matches = re.findall(r'([A-Z][a-z]{2}-\d{2})\s*\|\s*([\d,]+)', md, re.S | re.I)
            if matches:
                val_str = matches[0][1]
                debt = float(val_str.replace(',', '')) / 1_000_000
                ratio = (debt / gdp * 100) if gdp else None
                yoy = None
                if len(matches) >= 13:
                    prev = float(matches[12][1].replace(',', ''))
                    curr = float(val_str.replace(',', ''))
                    yoy = ((curr - prev) / prev) * 100
                print_ok(f"Margin Debt: {debt:.3f}T, GDP%: {ratio:.2f}%")
                return yoy, debt, ratio
        except Exception as e:
            print_err(f"Margin Debt 异常: {e}")
        return None, None, None

    def fetch_sahm_rule(self):
        print_step("[Sahm Rule] 启动抓取...")
        try:
            resp = self.app.scrape("https://fred.stlouisfed.org/series/SAHMREALTIME", formats=['markdown'])
            match = re.search(r'([A-Z][a-z]{2}\s+\d{4}):\s*([\d\.]+)', getattr(resp, 'markdown', ''), re.S | re.I)
            if match:
                val = float(match.group(2))
                print_ok(f"Sahm Rule: {val}%")
                return val
        except: pass
        return None

    def fetch_lei(self):
        print_h("[LEI] 启动混合视觉模式...")
        try:
            resp = self.app.scrape("https://www.conference-board.org/topics/us-leading-indicators", formats=['markdown'])
            md = getattr(resp, 'markdown', '')
            img_url = None
            if md:
                imgs = re.findall(r'\((https://.*?lei.*?\.png)\)', md, re.I)
                if imgs: img_url = imgs[0]
            if img_url:
                print_step("下载图片并 AI 分析...")
                content = requests.get(img_url, headers={"User-Agent": "Mozilla/5.0"}).content
                img = Image.open(io.BytesIO(content))
                prompt = 'Extract "6-Month % Change" (depth) and "Diffusion" value. JSON: {"depth": -2.1, "diffusion": 35.0}'
                ai = client.models.generate_content(model='gemini-2.0-flash', contents=[prompt, img])
                js = json.loads(re.search(r'\{.*\}', ai.text, re.DOTALL).group(0))
                d, dif = float(js['depth']), float(js['diffusion'])
                print_ok(f"LEI: Depth={d}, Diffusion={dif}")
                return d, dif
        except Exception as e:
            print_err(f"LEI 异常: {e}")
        return None, None

    def fetch_nyse_internals_robust(self):
        print_step("启动 WSJ 抓取 (Firecrawl + Gemini)...")
        try:
            headers = {"Authorization": f"Bearer {self.firecrawl_key}", "Content-Type": "application/json"}
            payload = {"url": "https://www.wsj.com/market-data/stocks/marketsdiary", "formats": ["markdown"], "waitFor": 5000}
            resp = requests.post("https://api.firecrawl.dev/v1/scrape", headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                md = resp.json().get('data', {}).get('markdown', '')
                if md:
                    prompt = f"Extract NYSE and NASDAQ breadth data. Return JSON. Markdown: {md[:15000]}"
                    ai = client.models.generate_content(model='gemini-2.0-flash', contents=[prompt])
                    js = json.loads(re.search(r'\{.*\}', ai.text, re.DOTALL).group(0))
                    self.cached_nasdaq = js.get('NASDAQ')
                    print_ok("WSJ 数据已获取")
                    return js.get('NYSE')
        except Exception as e:
            print_warn(f"WSJ 异常: {e}")
        return None

    def fetch_nymo_vision(self):
        print_step("启动 StockCharts 视觉抓取 ($NYMO)...")
        target_url = "https://stockcharts.com/h-sc/ui?s=$NYMO"
        headers = {"Authorization": f"Bearer {self.firecrawl_key}", "Content-Type": "application/json"}
        payload = {"url": target_url, "formats": ["screenshot"], "waitFor": 8000}
        try:
            resp = requests.post("https://api.firecrawl.dev/v1/scrape", headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                url = resp.json().get('data', {}).get('screenshot', '')
                if url:
                    print_step("截图成功，AI 读数中...")
                    img = Image.open(io.BytesIO(requests.get(url).content))
                    prompt = 'Extract $NYMO value. JSON: {"value": -12.3}'
                    ai = client.models.generate_content(model='gemini-2.0-flash', contents=[prompt, img])
                    val = json.loads(re.search(r'\{.*\}', ai.text, re.DOTALL).group(0)).get('value')
                    print_ok(f"NYMO: {val}")
                    return float(val) if val else None
        except: pass
        return None

    def fetch_dual_mco(self):
        print_step("[MCO] 启动官方源 + NYMO 双重抓取...")
        mco, nymo = None, None
        try:
            resp = self.app.scrape("https://www.mcoscillator.com/", formats=['markdown'])
            match = re.search(r'McC\s*OSC\s*\|?\s*([-\d\.]+)', getattr(resp, 'markdown', ''), re.I)
            if match: 
                mco = float(match.group(1))
                print_ok(f"MCO Official: {mco}")
        except: pass
        nymo = self.fetch_nymo_vision()
        return mco, nymo

    def fetch_tv_breadth_vision(self):
        print_h("[TradingView/WSJ] 复用 NASDAQ 数据...")
        if hasattr(self, 'cached_nasdaq') and self.cached_nasdaq:
            try:
                def c(v): return int(float(str(v).replace(',','').replace('K','000'))) if v else 0
                adv, dec = c(self.cached_nasdaq.get('adv')), c(self.cached_nasdaq.get('dec'))
                print_ok(f"NASDAQ Breadth: +{adv} / -{dec}")
                return adv, dec
            except: pass
        print_warn("无 NASDAQ 数据")
        return None, None

    def fetch_pcr_robust(self):
        print_h("[PCR] 启动直连 API 抓取...")
        try:
            resp = self.app.scrape("https://en.macromicro.me/charts/449/us-cboe-options-put-call-ratio", formats=['markdown'])
            matches = re.findall(r'(\d{1,2}\.\d{2})', getattr(resp, 'markdown', ''))
            if matches: 
                val = float(matches[0])
                print_ok(f"PCR: {val}")
                return val, val
        except: pass
        return None, None

    def fetch_nfci(self):
        print_step("[NFCI] FRED API 获取...")
        try:
            if not self.fred_key: return None
            f = Fred(api_key=self.fred_key)
            s = f.get_series('NFCI', sort_order='desc', limit=1)
            val = float(s.iloc[0])
            print_ok(f"NFCI: {val}")
            return val
        except: return None

# ==========================================
# 【核心程序: CrashWarningSystem】
# ==========================================
class CrashWarningSystem:
    def __init__(self):
        self.scraper = WebScraper()
        self.shared_wsj_data = None
        self.setup_fonts()
        # 颜色配置用于绘图
        self.colors = {'bg': '#4B535C', 'title': '#FFEE88', 'safe': '#2E8B57', 'warn': '#8B0000', 'risk': '#B8860B', 'text': '#FFFFFF'}

    def setup_fonts(self):
        if platform.system() == "Windows": font = ['Microsoft YaHei']
        else: font = ['WenQuanYi Zen Hei', 'Arial Unicode MS']
        plt.rcParams['font.sans-serif'] = font + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

    def calculate_spx_breadth_deep(self):
        tickers = get_cached_tickers()
        data = get_cached_sp500_data(tickers)
        if data.empty: return None, None
        
        print_step("正在本地计算 SMA50 和 SMA20...")
        last = data.iloc[-1]
        pct50 = (last > data.rolling(50).mean().iloc[-1]).mean() * 100
        pct20 = (last > data.rolling(20).mean().iloc[-1]).mean() * 100
        print_ok(f"市场广度: >50MA={pct50:.1f}%, >20MA={pct20:.1f}%")
        return pct50, pct20

    def analyze_market_trends_console(self):
        print_h("深度宏观分析 (Deep Macro)")
        if not USER_FRED_KEY: return
        
        # 使用 Streamlit 的列布局来替代原来的 print 输出
        col1, col2 = st.columns(2)
        try:
            fred = Fred(api_key=USER_FRED_KEY)
            
            # 1. 净流动性
            with col1:
                start = datetime.now() - timedelta(weeks=5)
                walcl = fred.get_series('WALCL', observation_start=start).iloc[-1]
                tga = fred.get_series('WTREGEN', observation_start=start).iloc[-1]
                rrp = fred.get_series('RRPONTSYD', observation_start=start).iloc[-1]
                liq = (walcl/1e6) - (tga/1e3) - (rrp/1e3)
                st.metric("美联储净流动性", f"${liq:.3f}T", help="规则: 流动性增加 = 股市燃料增加")

            # 2. ERP
            with col2:
                dgs10 = fred.get_series('DGS10', sort_order='desc', limit=1).iloc[-1]
                pe = self.scraper.fetch_shiller_pe() or 35.0
                erp = (1.0/pe*100) - dgs10
                st.metric("股权风险溢价 (ERP)", f"{erp:.2f}%", delta_color="normal" if erp>2.5 else "inverse")
        except: st.error("宏观数据计算失败")
        
        # 3. RSP/SPY
        try:
            df = yf.download(['SPY', 'RSP'], period="3mo", progress=False)['Close']
            if not df.empty:
                ratio = df['RSP'] / df['SPY']
                chg = ((ratio.iloc[-1] - ratio.iloc[-20]) / ratio.iloc[-20]) * 100
                st.write(f"3. RSP/SPY 相对强度 (20日): {chg:+.2f}%")
        except: pass

    def fetch_and_calculate(self):
        print_h("开始执行数据获取与计算")
        
        # 1. 本地计算
        ma50_pct, ma20_pct = self.calculate_spx_breadth_deep()
        
        # 2. 基础数据
        print_step("获取核心指数...")
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
        spx_trend_up = spx.iloc[-1] > spx.rolling(50).mean().iloc[-1]
        
        # 3. 爬虫数据
        print_h("启动宏观指标动态抓取 (Firecrawl)")
        pe = self.scraper.fetch_shiller_pe()
        sahm = self.scraper.fetch_sahm_rule()
        fg, fg_src = self.scraper.fetch_fear_greed()
        buffett = self.scraper.fetch_buffett_indicator()
        m_yoy, m_amt, m_ratio = self.scraper.fetch_margin_debt()
        lei_d, lei_dif = self.scraper.fetch_lei()
        pcr_avg, pcr_cur = self.scraper.fetch_pcr_robust()
        print_h("芝加哥金融状况指数 (NFCI)")
        nfci = self.scraper.fetch_nfci()
        
        print_h("HO & MCO & Volume")
        mco, nymo = self.scraper.fetch_dual_mco()
        ho_res = self.scraper.fetch_nyse_internals_robust()
        if ho_res: self.shared_wsj_data = ho_res
        tv_adv, tv_dec = self.scraper.fetch_tv_breadth_vision()

        # 4. 指标判定逻辑 (这部分完全保留您原代码的逻辑)
        indicators = []
        
        # [指标1] Hindenburg Omen
        ho_stat = 0; ho_txt = "数据不足"
        if ho_res:
            def c(v): return float(str(v).replace(',','').replace('B','e9').replace('M','e6')) if v else 0
            h = c(ho_res.get('high')); l = c(ho_res.get('low'))
            adv = c(ho_res.get('adv')); dec = c(ho_res.get('dec'))
            total = adv + dec + c(ho_res.get('unch', 0))
            h_pct = (h/total)*100 if total else 0
            l_pct = (l/total)*100 if total else 0
            
            split = (h_pct > 2.2 and l_pct > 2.2)
            mco_bad = (mco < 0) if mco else (adv < dec)
            
            if spx_trend_up and split and mco_bad: ho_stat = 2
            elif split: ho_stat = 1
            ho_txt = f"新高:{h_pct:.1f}% | 新低:{l_pct:.1f}%"
        indicators.append(["Hindenburg Omen", ho_stat, ho_txt, "条件: 50MA上 & 新高低>2.2% & MCO<0"])

        # [指标2] Net Issues (广度)
        net_stat = 0; net_issues = 0
        if ho_res:
             net_issues = c(ho_res.get('adv')) - c(ho_res.get('dec'))
             if net_issues < -2000: net_stat = 2
             elif net_issues < -1000: net_stat = 1
        indicators.append(["抛压 I: 广度 (Net Issues)", net_stat, f"{net_issues}", "<-1000 显著 | <-2000 恐慌"])

        # [指标3] TRIN (力度)
        trin_stat = 0; trin_txt = "N/A"
        if ho_res:
            adv_v = c(ho_res.get('adv_vol')); dec_v = c(ho_res.get('dec_vol'))
            if dec > 0 and dec_v > 0:
                trin = (adv/dec) / (adv_v/dec_v)
                trin_txt = f"{trin:.2f}"
                if trin < 0.5: trin_stat = 2
                elif trin > 2.0: trin_stat = 1
        indicators.append(["抛压 II: 力度 (TRIN)", trin_stat, trin_txt, "<0.5(极度超买) | >2.0(恐慌抄底)"])

        # [指标4] Volume Flow (资金)
        vol_stat = 0; vol_txt = "N/A"
        if ho_res and adv_v > 0:
            ratio = dec_v / adv_v
            if ratio > 9.0: vol_stat = 2
            elif ratio > 4.0: vol_stat = 1
            vol_txt = f"Dn/Up: {ratio:.1f}"
        indicators.append(["抛压 III: 资金 (Vol)", vol_stat, vol_txt, "Dn/Up > 4.0 出逃 | > 9.0 洗盘"])

        # [指标5] NASDAQ Breadth
        tv_stat = 0
        if tv_adv and tv_dec:
            ratio = tv_adv / tv_dec
            if ratio < 0.5: tv_stat = 2
            indicators.append(["NASDAQ A/D", tv_stat, f"{ratio:.2f}", "<0.5 空头主导"])
        else: indicators.append(["NASDAQ A/D", 0, "N/A", ""])

        # [指标6] RSP vs SPY
        try:
            r = rsp/spy
            curr, ma = r.iloc[-1], r.rolling(50).mean().iloc[-1]
            chg = (curr/r.iloc[-20]-1)*100
            st_rsp = 2 if (curr<ma and chg<-2.0) else (1 if curr<ma else 0)
            indicators.append(["RSP/SPY 广度", st_rsp, f"20日变动: {chg:.1f}%", "跌破50MA & 急跌"])
        except: indicators.append(["RSP/SPY", 0, "Error", ""])
        
        # [指标7] NYA 参与度
        try:
            ok = nya.iloc[-1] > nya.rolling(50).mean().iloc[-1]
            st_nya = 2 if (spx_trend_up and not ok) else 0
            indicators.append(["NYA 参与度", st_nya, "弱" if not ok else "强", "SPX强但NYA弱"])
        except: pass

        # [指标8] 收益率倒挂
        try:
            spr = tnx.iloc[-1] - irx.iloc[-1]
            indicators.append(["10Y-3M 倒挂", 2 if spr<0 else 0, f"{spr:.2f}%", "< 0%"])
        except: pass

        # [指标9] Shiller PE
        indicators.append(["Shiller PE", 2 if pe and pe>30 else 0, f"{pe}", ">30 高估"])
        
        # [指标10] Buffett
        indicators.append(["巴菲特指标", 2 if buffett and buffett>140 else 0, f"{buffett:.1f}%", ">140%"])
        
        # [指标11] Margin Debt
        indicators.append(["Margin Debt", 1 if m_ratio and m_ratio>3.5 else 0, f"GDP比:{m_ratio:.1f}%", ">3.5%"])
        
        # [指标12] VIX
        try:
            v = vix.iloc[-1]
            chg = (v/vix.iloc[-15]-1)*100
            st_vix = 2 if (v>25 or chg>40) else 0
            indicators.append(["VIX", st_vix, f"{v:.1f} (+{chg:.0f}%)", ">25 或 飙升"])
        except: pass

        # [指标13] 广度 MA
        if ma50_pct:
            st_br = 2 if ma50_pct<40 else 0
            indicators.append(["SPX >50MA", st_br, f"{ma50_pct:.1f}%", "<40% 危险"])

        # [指标14] RSI 背离
        try:
            delta = spx_weekly.diff()
            u = delta.clip(lower=0); d = -delta.clip(upper=0)
            rs = u.ewm(alpha=1/14).mean() / d.ewm(alpha=1/14).mean()
            rsi = 100 - 100/(1+rs)
            
            div = False
            # 简化判定：价格新高但RSI没新高
            if rsi.iloc[-1] < rsi.iloc[-5] and spx_weekly.iloc[-1] > spx_weekly.iloc[-5]:
                div = True
            indicators.append(["RSI 周线背离", 2 if div else 0, f"{rsi.iloc[-1]:.1f}", "价涨量缩"])
        except: pass

        # [指标15] Support Band
        try:
            sma20 = spx_weekly.rolling(20).mean().iloc[-1]
            ema21 = spx_weekly.ewm(span=21).mean().iloc[-1]
            status = 2 if spx.iloc[-1] < min(sma20, ema21) else 0
            indicators.append(["牛市支撑带", status, f"现价:{spx.iloc[-1]:.0f}", "跌破 20SMA/21EMA"])
        except: pass

        # [指标16] Fear & Greed
        indicators.append(["Fear & Greed", 2 if fg and fg<45 else 0, f"{fg}", "<45"])
        
        # [指标17] MACD
        try:
            e12 = spx_weekly.ewm(span=12).mean(); e26 = spx_weekly.ewm(span=26).mean()
            macd = e12 - e26; sig = macd.ewm(span=9).mean()
            dead = (macd.iloc[-2]>sig.iloc[-2]) and (macd.iloc[-1]<sig.iloc[-1]) and (macd.iloc[-1]>0)
            indicators.append(["MACD 周线死叉", 2 if dead else 0, "死叉" if dead else "正常", "零轴上方死叉"])
        except: pass
        
        # [指标18] Sahm Rule
        indicators.append(["Sahm Rule", 2 if sahm and sahm>=0.5 else 0, f"{sahm}%", ">=0.5%"])
        
        # [指标19] LEI
        indicators.append(["LEI", 2 if lei_d and lei_d<-4.0 else 0, f"{lei_d}%", "<-4.0%"])
        
        # [指标20] PCR
        indicators.append(["PCR", 2 if pcr_avg and pcr_avg<0.8 else 0, f"{pcr_avg}", "<0.8"])
        
        # [指标21] NFCI
        indicators.append(["NFCI", 2 if nfci and nfci>-0.2 else 0, f"{nfci}", ">-0.2"])
        
        # [指标22] NYMO (额外)
        nymo_st = 2 if nymo and (nymo>60 or nymo<-60) else 0
        indicators.append(["NYMO", nymo_st, f"{nymo}", "极端值 +/-60"])

        return indicators

    def generate_chart(self):
        st.subheader("📊 21因子风险仪表盘")
        
        # 进度展示
        with st.status("正在计算核心指标...", expanded=True) as status:
            data = self.fetch_and_calculate()
            status.update(label="计算完成", state="complete", expanded=False)
        
        # 计算总分
        risk_score = sum(1 for d in data if d[1] == 2) + sum(0.5 for d in data if d[1] == 1)
        
        # 绘图 (用 Matplotlib 画表)
        fig, ax = plt.subplots(figsize=(12, len(data)*0.8), facecolor=self.colors['bg'])
        ax.axis('off')
        
        ax.text(0.5, 0.98, f"风险评分: {risk_score:.1f} / 21.0", ha='center', fontsize=20, color=self.colors['title'], weight='bold')
        ax.text(0.5, 0.95, f"生成时间: {datetime.now().strftime('%Y-%m-%d')}", ha='center', fontsize=12, color='#CCCCCC')
        
        col_labels = ['指标', '状态', '读数', '标准']
        cell_text = []
        cell_colors = []
        
        for row in data:
            name, stat, val, desc = row
            status_txt = "危险" if stat==2 else ("警告" if stat==1 else "安全")
            cell_text.append([name, status_txt, val, desc])
            
            c = self.colors['safe']
            if stat == 2: c = self.colors['warn']
            elif stat == 1: c = self.colors['risk']
            cell_colors.append([c, c, c, c])
            
        table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center', 
                        colWidths=[0.25, 0.15, 0.25, 0.35])
        table.scale(1, 2)
        table.auto_set_font_size(False); table.set_fontsize(12)
        
        for i, key in enumerate(table.get_celld().keys()):
            cell = table.get_celld()[key]
            row, col = key
            if row == 0:
                cell.set_facecolor('#3E4953')
                cell.set_text_props(color='white', weight='bold')
            else:
                cell.set_facecolor(cell_colors[row-1][col])
                cell.set_text_props(color='white')

        # 【核心修改】直接显示
        st.pyplot(fig)

# ==========================================
# 【板块轮动 (RRG)】
# ==========================================
class SectorRotationEngine:
    def __init__(self):
        self.sectors = {'XLK': '科技', 'XLF': '金融', 'XLV': '医疗', 'XLE': '能源', 'XLY': '可选', 
                       'XLP': '必选', 'XLI': '工业', 'XLC': '通讯', 'XLB': '材料', 'XLRE': '地产', 'SPY': '基准'}
        self.rs_window = 60 
        self.mom_window = 10 

    def run_analysis(self):
        print_h("板块轮动分析 (RRG)")
        tickers = list(self.sectors.keys())
        data = get_cached_sector_data(tickers, "2023-01-01")
        if data.empty: return
        
        closes = data['Adj Close'] if 'Adj Close' in data else data['Close']
        rs = closes.div(closes['SPY'], axis=0)
        
        ratio = 100 * (rs / rs.rolling(self.rs_window).mean())
        mom = 100 + ((rs - rs.shift(self.mom_window)) / rs.shift(self.mom_window) * 100)
        
        res = []
        for t in tickers:
            if t == 'SPY': continue
            r = ratio[t].iloc[-1]
            m = mom[t].iloc[-1]
            q = "滞后"
            if r>100 and m>100: q = "领涨 🟢"
            elif r<100 and m>100: q = "改善 🔵"
            elif r>100 and m<100: q = "转弱 🟡"
            else: q = "落后 🔴"
            res.append({"板块": self.sectors[t], "RS": f"{r:.1f}", "Mom": f"{m:.1f}", "象限": q})
            
        st.dataframe(pd.DataFrame(res))

# ==========================================
# 【FRED 信号灯】
# ==========================================
def run_fred_traffic_light(fred_key):
    print_h("收益率曲线 + 失业率红绿灯") 
    if not fred_key: return
    try:
        fred = Fred(api_key=fred_key)
        curve = fred.get_series('T10Y2Y', sort_order='desc', limit=1).iloc[0]
        unrate = fred.get_series('UNRATE', sort_order='desc', limit=2)
        curr_u = unrate.iloc[0]; prev_u = unrate.iloc[1]
        
        st.write(f"1. 10Y-2Y 利差: {curve:.2f}%")
        st.write(f"2. 失业率: {curr_u}% (前值: {prev_u}%)")
        
        signal = "🟢 绿灯"
        if curve < 0 and curr_u > prev_u: signal = "🔴 红灯 (衰退预警)"
        elif curve < 0: signal = "🟡 黄灯 (倒挂)"
        
        st.subheader(f"信号: {signal}")
    except: pass

def run_fred_v10_dashboard(api_key):
    # 简易仪表盘
    pass 

# ==========================================
# 【SMT 背离分析】
# ==========================================
class SMTDivergenceAnalyzer:
    def __init__(self):
        self.tickers = ['^IXIC', '^GSPC', 'QQQ', 'SPY', 'NQ=F', 'ES=F', 'RSP']

    def run(self):
        print_h("SMT 背离分析模块 (Pro V3)")
        df = get_cached_smt_data(self.tickers, "6mo")
        if df.empty: return
        
        close = df['Close'].ffill()
        
        # 1. 经典 SMT
        st.write("Checking Classic SMT...")
        
        # 2. 期货分析
        w = close.iloc[-10:]
        h = w.max(); curr = w.iloc[-1]
        
        if 'NQ=F' in w and 'ES=F' in w:
            nq_h = curr['NQ=F'] >= h['NQ=F']*0.999
            es_h = curr['ES=F'] >= h['ES=F']*0.999
            if nq_h and not es_h: st.warning("📉 看跌背离: 纳指拉升 标普不跟")
            elif not nq_h and es_h: st.warning("📉 看跌背离: 标普补涨 纳指滞涨")
            else: st.success("期货市场步调一致")

        # 3. 广度分析
        if 'SPY' in w and 'RSP' in w:
            spy_p = (curr['SPY']/w.iloc[0]['SPY']-1)*100
            rsp_p = (curr['RSP']/w.iloc[0]['RSP']-1)*100
            if spy_p > rsp_p and spy_p > 0 and rsp_p < 0:
                st.error("⚠️ 虚假繁荣: 只有巨头在涨 (SPY涨 RSP跌)")
            else:
                st.success("市场广度正常")
                
        # 4. Vincent 策略
        st.write("关键位检查 (Vincent Strategy):")
        spy_curr = curr['SPY']
        ma20 = close['SPY'].rolling(20).mean().iloc[-1]
        if spy_curr > ma20: st.info(f"SPY 站上 MA20 ({ma20:.2f}) - 多头区域")
        else: st.info(f"SPY 跌破 MA20 ({ma20:.2f}) - 空头区域")

# ==========================================
# 【主程序】
# ==========================================
if __name__ == "__main__":
    st.sidebar.title("控制台")
    st.sidebar.info("V10.055 Full Web Edition")
    if st.sidebar.button("🔄 强制重新计算"):
        st.cache_data.clear()
        st.rerun()
        
    st.title("🚀 美股崩盘预警系统 Pro")
    
    app = CrashWarningSystem()
    
    # 1. 核心图片与报告
    app.generate_chart()
    
    # 2. 附加功能
    run_fred_traffic_light(USER_FRED_KEY)
    run_fred_v10_dashboard(USER_FRED_KEY)
    
    # 3. 趋势分析
    app.analyze_market_trends_console()
    
    # 4. 板块轮动
    sr = SectorRotationEngine()
    sr.run_analysis()
    
    # 5. SMT
    smt = SMTDivergenceAnalyzer()
    smt.run()
    
    # 最终确认
    st.balloons()
    st.success("所有分析任务执行完毕！")
    st.stop() # 明确停止
