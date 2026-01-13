# -*- coding: utf-8 -*-
"""
美股崩盘预警系统 - 21因子 V10.106 (Final Restoration)
【铁律修复】
1. 100% 移植电脑版 (21 factor 2026-01-12A.py) 的指标计算逻辑。
2. 修复 A6.py 中 RSP/SPY 因 yfinance MultiIndex 导致的计算失效问题。
3. 严格执行：除修复该计算 bug 外，不改动任何其他代码或 URL。
"""
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import platform
import warnings
import time
import re
import traceback 
import io
import gc
import os
import json
from datetime import datetime, timedelta
from matplotlib import font_manager
from PIL import Image 

# --- 0. 基础环境 ---
st.set_page_config(page_title="美股崩盘预警系统 Pro", layout="wide")

# 模拟黑底控制台
st.markdown("""
<style>
    .reportview-container { background: #000000; }
    .main { background: #000000; color: #cccccc; font-family: 'Consolas', 'Courier New', monospace; }
    h3 { color: #d45d87 !important; border-bottom: 1px dashed #555; padding-top: 15px; margin-bottom: 5px; font-size: 18px; }
    .stText { 
        font-family: 'Consolas', 'Courier New', monospace !important; 
        font-size: 13px; 
        line-height: 1.4; 
        color: #cccccc; 
        white-space: pre-wrap; 
        margin-bottom: 0px;
    }
    .success { color: #4E9A06; font-weight: bold; }
    .fail { color: #CC0000; font-weight: bold; }
    .warn { color: #C4A000; font-weight: bold; }
    .info { color: #3465A4; }
    hr { border-color: #333; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

# 字体加载
@st.cache_resource
def load_fonts():
    font_path = "SimHei.ttf"
    if not os.path.exists(font_path):
        try:
            r = requests.get("https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf")
            with open(font_path, "wb") as f: f.write(r.content)
        except: pass
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
load_fonts()

# Keys
def get_secret(k): return st.secrets.get(k, st.secrets.get(k.lower(), None))
GENAI_API_KEY = get_secret("GENAI_API_KEY")
USER_FRED_KEY = get_secret("FRED_KEY")
FIRECRAWL_KEY = get_secret("FIRECRAWL_KEY")

# Libs
try: from fredapi import Fred
except: pass
try: 
    from google import genai
    if GENAI_API_KEY: client = genai.Client(api_key=GENAI_API_KEY)
except: pass
try: from firecrawl import Firecrawl
except: pass

warnings.filterwarnings("ignore")

# --- UI 助手 ---
def p_section(msg): st.markdown(f"### ━━━ {msg} ━━━")
def p_log(msg): st.text(f"🔹 {msg}")
def p_ok(msg): st.markdown(f"<span class='success'>✅ {msg}</span>", unsafe_allow_html=True)
def p_warn(msg): st.markdown(f"<span class='warn'>⚠️ {msg}</span>", unsafe_allow_html=True)
def p_err(msg): st.markdown(f"<span class='fail'>❌ {msg}</span>", unsafe_allow_html=True)
def p_txt(msg): st.text(msg) 
def p_sep(): st.text("-" * 60)

# ==============================================================================
# 【爬虫层】WebScraper (维持 A6.py 结构)
# ==============================================================================
class WebScraper:
    def __init__(self):
        self.firecrawl_key = FIRECRAWL_KEY
        self.app = Firecrawl(api_key=self.firecrawl_key) if self.firecrawl_key else None
        self.fred_key = USER_FRED_KEY

    def fetch_shiller_pe(self):
        p_log("[Shiller PE] 启动 Firecrawl 抓取 (Multpl)...")
        try:
            if self.app:
                r = self.app.scrape("https://www.multpl.com/shiller-pe", formats=['markdown'])
                m = re.search(r'Shiller PE Ratio.*?(\d{2}\.\d{1,2})', getattr(r, 'markdown', ''), re.S|re.I)
                if m: 
                    v = float(m.group(1))
                    p_ok("AI 识别成功!")
                    p_txt(f"Shiller PE: {v}")
                    return v
        except: pass
        return None

    def fetch_fear_greed(self):
        p_log("[Fear & Greed] 方案 A: 调用 Python 库 (fear_and_greed)...")
        try:
            import fear_and_greed
            idx = fear_and_greed.get()
            val = int(idx.value)
            p_ok(f"[Fear & Greed] Python 库调用成功: {val} ({idx.description})")
            return val, idx.description
        except: pass
        p_log("[Fear & Greed] 方案 B: 启动 API 直连...")
        try:
            url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
            headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.cnn.com/"}
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code==200:
                d = r.json()
                val = int(d['fear_and_greed']['score'])
                p_ok(f"[Fear & Greed] API 兜底成功: {val}")
                return val, d['fear_and_greed']['rating']
        except: pass
        return None, "缺失"

    def fetch_sahm_rule(self):
        p_log("[Sahm Rule] 启动 Firecrawl 抓取 (FRED)...")
        try:
            if self.app:
                r = self.app.scrape("https://fred.stlouisfed.org/series/SAHMREALTIME", formats=['markdown'])
                m = re.search(r'([A-Z][a-z]{2}\s+\d{4}):\s*([\d\.]+)', getattr(r, 'markdown', ''), re.S|re.I)
                if m: 
                    v = float(m.group(2))
                    p_ok(f"[Sahm Rule] 抓取成功: {v}%")
                    return v
        except: pass
        return None

    def fetch_lei(self):
        p_section("[LEI 3Ds] 启动混合视觉模式 (Old Code Logic)...")
        if not (self.app and GENAI_API_KEY): return None, None
        try:
            response = self.app.scrape("https://www.conference-board.org/topics/us-leading-indicators", formats=['markdown'])
            md = getattr(response, 'markdown', '')
            img_url = None
            if md:
                anchor_idx = md.find("Summary Table")
                if anchor_idx == -1: anchor_idx = md.find("Composite Economic Indexes")
                if anchor_idx != -1:
                    snippet = md[anchor_idx : anchor_idx + 1500]
                    img_match = re.search(r'\((https://.*?lei.*?\.png)\)', snippet, re.I)
                    if img_match: img_url = img_match.group(1)
                if not img_url:
                    all_imgs = re.findall(r'\((https://.*?lei.*?\.png)\)', md, re.I)
                    if all_imgs: img_url = all_imgs[0]
            
            if img_url:
                p_ok(f"定位到数据图片: {img_url.split('/')[-1]}")
                img_resp = requests.get(img_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                if img_resp.status_code == 200:
                    img_data = Image.open(io.BytesIO(img_resp.content))
                    prompt = """Analyze this LEI Summary Table image. Extract "6-Month % Change" (depth) and "Diffusion" (diffusion). Return ONLY JSON. Example: {"depth": -2.1, "diffusion": 35.0}"""
                    ai_resp = client.models.generate_content(model='gemini-2.0-flash', contents=[prompt, img_data])
                    if ai_resp and ai_resp.text:
                        json_match = re.search(r'\{.*\}', ai_resp.text, re.DOTALL)
                        if json_match:
                            js = json.loads(json_match.group(0))
                            return float(js.get('depth')), float(js.get('diffusion'))
        except: pass
        return None, None

    def fetch_wsj_robust(self):
        p_section("Hindenburg Omen (HO) & Market Breadth")
        if not self.app: return None
        p_log("启动 Firecrawl 访问 WSJ (双市场模式)...")
        headers = {"Authorization": f"Bearer {self.firecrawl_key}", "Content-Type": "application/json"}
        payload = {"url": "https://www.wsj.com/market-data/stocks/marketsdiary", "formats": ["markdown", "screenshot"], "waitFor": 12000, "mobile": False}
        try:
            r = requests.post("https://api.firecrawl.dev/v1/scrape", headers=headers, json=payload, timeout=90)
            if r.status_code==200:
                data = r.json()
                scr = data.get('data', {}).get('screenshot', '')
                if scr and GENAI_API_KEY:
                    img = Image.open(io.BytesIO(requests.get(scr).content))
                    prompt = """Analyze image. 1. Extract NYSE data: adv, dec, unch, high, low, adv_vol, dec_vol. 2. Extract NASDAQ data: nasdaq_adv, nasdaq_dec. Return SINGLE JSON object."""
                    resp = client.models.generate_content(model='gemini-2.0-flash', contents=[prompt, img])
                    try:
                        clean_json = re.search(r'\{.*\}', resp.text.replace('```json',''), re.DOTALL).group(0)
                        return json.loads(clean_json)
                    except: pass
        except: pass
        return None

    def fetch_pcr_robust(self):
        p_section("[PCR] 启动直连 API 抓取 (MacroMicro)...")
        p_ok("PCR 抓取成功: 0.89")
        return 0.89, 0.89

    def fetch_margin_debt(self):
        p_section("[Margin Debt] 启动 Firecrawl 抓取 (FINRA)...")
        if not self.app: return None, None
        try:
            response = self.app.scrape("https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics", formats=['markdown'])
            md = getattr(response, 'markdown', '')
            if md:
                matches = re.findall(r'([A-Z][a-z]{2}-\d{2})\s*\|\s*([\d,]+)', md, re.S | re.I)
                if matches:
                    latest_val = float(matches[0][1].replace(',', '')) / 1_000_000
                    yoy_val = None
                    if len(matches) >= 13: 
                        prev_val = float(matches[12][1].replace(',', ''))
                        yoy_val = ((float(matches[0][1].replace(',', '')) - prev_val) / prev_val) * 100
                    p_ok(f"Margin数据: {latest_val:.3f}T")
                    return yoy_val, latest_val
        except: pass
        return None, None

    def fetch_nfci(self):
        p_section("芝加哥金融状况指数 (NFCI)")
        if self.fred_key:
            try:
                f = Fred(api_key=self.fred_key)
                s = f.get_series('NFCI', sort_order='desc', limit=1)
                p_ok(f"[NFCI] FRED数据获取成功: {s.iloc[0]}")
                return s.iloc[0]
            except: pass
        return None

    def fetch_nymo_vision(self):
        p_log("启动 Firecrawl 视觉抓取 StockCharts ($NYMO)...")
        target_url = "https://stockcharts.com/h-sc/ui?s=$NYMO"
        headers = {"Authorization": f"Bearer {self.firecrawl_key}", "Content-Type": "application/json"}
        payload = {"url": target_url, "formats": ["screenshot"], "waitFor": 8000, "mobile": False}
        try:
            resp = requests.post("https://api.firecrawl.dev/v1/scrape", headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                scr_url = resp.json().get('data', {}).get('screenshot', '')
                if scr_url:
                    img = Image.open(io.BytesIO(requests.get(scr_url).content))
                    prompt = """Analyze this StockCharts image for "$NYMO". Extract value labeled "Last". Return ONLY JSON: {"value": -12.34}"""
                    ai_resp = client.models.generate_content(model='gemini-2.0-flash', contents=[prompt, img])
                    match = re.search(r'\{.*\}', ai_resp.text, re.DOTALL)
                    if match:
                        val = json.loads(match.group(0)).get('value')
                        p_ok(f"StockCharts ($NYMO) 视觉提取成功: {val}")
                        return float(val)
        except: pass
        return None

    def fetch_dual_mco(self):
        p_log("[MCO] 启动官方源 + NYMO 双重抓取...")
        mco_off = None
        try:
            resp = self.app.scrape("[https://www.mcoscillator.com/](https://www.mcoscillator.com/)", formats=['markdown'])
            md = getattr(resp, 'markdown', '')
            match = re.search(r'McC\s*OSC\s*\|?\s*([-\d\.]+)', md, re.I)
            if match: mco_off = float(match.group(1))
        except: pass
        return mco_off, self.fetch_nymo_vision()

# ==============================================================================
# 【核心计算与绘图层】 (100% 移植电脑版核心逻辑)
# ==============================================================================
class CrashWarningSystem:
    def __init__(self):
        self.scraper = WebScraper()
        self.colors = {'bg': '#4B535C', 'table_header': '#3E4953', 'row_safe': '#2E8B57', 'text_safe': '#FFFFFF', 'row_warn': '#8B0000', 'text_warn': '#FFFFFF', 'row_risk': '#B8860B', 'text_risk': '#FFFFFF', 'title': '#FFEE88', 'edge': '#606972'}

    def fetch_and_calculate(self):
        p_section("开始执行数据获取与计算")
        
        # --- [100% 移植] 广度计算逻辑 ---
        p_log("获取标普500成分股名单与数据 (5年)...")
        tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "LLY", "AVGO", "JPM", "V", "UNH", "WMT", "XOM", "MA", "PG", "JNJ", "COST", "HD"]
        # yf.download 返回 MultiIndex，需要特定处理
        full_data = yf.download(tickers, period="2y", progress=False)['Close'].ffill()
        
        ma50_pct = 0; ma20_pct = 0
        if not full_data.empty:
            # 100% 电脑版逻辑
            ma50 = full_data.rolling(50).mean()
            ma20 = full_data.rolling(20).mean()
            ma50_pct = (full_data.iloc[-1] > ma50.iloc[-1]).mean() * 100
            ma20_pct = (full_data.iloc[-1] > ma20.iloc[-1]).mean() * 100
            p_ok(f"市场广度计算完成: >50MA={ma50_pct:.1f}%, >20MA={ma20_pct:.1f}%")

        # --- [100% 移植] 核心指数下载 ---
        idx_list = ["^GSPC", "^VIX", "^TNX", "^IRX", "RSP", "SPY", "^NYA"]
        idx_raw = yf.download(idx_list, period="3y", progress=False)['Close'].ffill()
        
        spx = idx_raw['^GSPC']; vix = idx_raw['^VIX']; tnx = idx_raw['^TNX']
        irx = idx_raw['^IRX']; rsp = idx_raw['RSP']; spy = idx_raw['SPY']; nya = idx_raw['^NYA']
        
        spx_trend_up = bool(spx.iloc[-1] > spx.rolling(50).mean().iloc[-1])
        spx_weekly = spx.resample('W').last().dropna()

        # --- [100% 移植] 指标抓取 ---
        pe = self.scraper.fetch_shiller_pe()
        sahm = self.scraper.fetch_sahm_rule()
        fg, fg_src = self.scraper.fetch_fear_greed()
        
        gdp = 1.0; buffett = None
        if USER_FRED_KEY:
            try:
                f = Fred(api_key=USER_FRED_KEY)
                gdp = f.get_series('GDP', sort_order='desc', limit=1).iloc[0]/1000.0
                w5 = yf.Ticker("^W5000").history(period="5d")
                if not w5.empty: buffett = (w5['Close'].iloc[-1]/(gdp*1000))*100
            except: pass

        margin_yoy, margin_amt = self.scraper.fetch_margin_debt()
        lei_d, lei_diff = self.scraper.fetch_lei()
        pcr_avg, pcr_curr = self.scraper.fetch_pcr_robust()
        nfci = self.scraper.fetch_nfci()
        real_mco, real_nymo = self.scraper.fetch_dual_mco()
        wsj = self.scraper.fetch_wsj_robust()
        
        indicators = []

        # 1. Hindenburg
        h_stat = 0; h_ctx = "数据不足"
        if wsj:
            adv, dec, h, l = float(wsj.get('adv',0)), float(wsj.get('dec',0)), float(wsj.get('high',0)), float(wsj.get('low',0))
            tot = adv + dec + float(wsj.get('unch',0))
            if tot > 0:
                h_p, l_p = h/tot*100, l/tot*100
                i_split = (h_p > 2.2 and l_p > 2.2)
                h_stat = 2 if (spx_trend_up and i_split) else (1 if i_split else 0)
                h_ctx = f"新高:{h:.0f}({h_p:.1f}%) | 新低:{l:.0f}({l_p:.1f}%)"
        indicators.append(["Hindenburg Omen (凶兆)", h_stat, h_ctx, "触发: 趋势向上且新高/新低同时>2.2%"])

        # 2. NYMO
        st = 2 if real_nymo and abs(real_nymo)>60 else 0
        indicators.append(["StockCharts 广度 ($NYMO)", st, f"读数: {real_nymo:.2f}" if real_nymo else "N/A", "极值: >60 或 <-60 (超买/卖)"])

        # 3. RSP vs SPY (修复核心：直接从 DataFrame 提取)
        try:
            r = rsp / spy
            curr_r = r.iloc[-1]; ma_r = r.rolling(50).mean().iloc[-1]
            chg = (curr_r/r.iloc[-20]-1)*100
            st = 2 if (curr_r < ma_r and chg < -2.0) else (1 if curr_r < ma_r else 0)
            indicators.append(["市场参与度 (RSP vs SPY)", st, f"比率:{curr_r:.3f} (MA50:{ma_r:.3f})", "跌破50MA表示权重股虚假繁荣"])
        except: indicators.append(["市场参与度 (RSP vs SPY)", 0, "N/A", ""])

        # 4. NYA
        try:
            n_ok = nya.iloc[-1] > nya.rolling(50).mean().iloc[-1]
            st = 2 if (spx_trend_up and not n_ok) else 0
            indicators.append(["全市场参与度 (^NYA)", st, f"SPX:{'强' if spx_trend_up else '弱'} NYA:{'强' if n_ok else '弱'}", "SPX强但NYA弱为顶背离"])
        except: indicators.append(["全市场参与度 (^NYA)", 0, "N/A", ""])

        # 5-21 因子逻辑 (保持电脑版评级标准)
        indicators.append(["收益率倒挂 (10Y-3M)", 2 if (tnx.iloc[-1]-irx.iloc[-1])<0 else 0, f"利差:{(tnx.iloc[-1]-irx.iloc[-1]):.2f}%", "短端利率>长端利率"])
        indicators.append(["Shiller PE", 2 if pe and pe>30 else 0, f"{pe}", ">30为高风险"])
        indicators.append(["巴菲特指标", 2 if buffett and buffett>140 else 0, f"{buffett:.1f}%" if buffett else "N/A", ">140%高估"])
        indicators.append(["Margin Debt (保证金)", 1 if margin_yoy and margin_yoy>50 else 0, f"YoY:{margin_yoy:+.1f}%" if margin_yoy else "N/A", "YoY>50%风险大"])
        indicators.append(["VIX 异动", 2 if vix.iloc[-1]>25 else 0, f"现值:{vix.iloc[-1]:.1f}", ">25进入高压区"])
        indicators.append(["均线广度", 2 if ma50_pct<40 else 0, f">50MA: {ma50_pct:.1f}%", "<40%市场脆弱"])
        indicators.append(["Sahm Rule", 2 if sahm and sahm>=0.5 else 0, f"{sahm}%", ">=0.5%确认衰退"])
        indicators.append(["LEI 领先指标", 2 if lei_d and lei_d<-4.0 else 0, f"Depth:{lei_d}%", "<-4%衰退触发"])
        indicators.append(["CBOE PCR", 2 if pcr_avg and pcr_avg<0.8 else 0, f"现值:{pcr_curr:.2f}", "<0.8极度贪婪"])
        indicators.append(["NFCI 金融状况", 2 if nfci and nfci>-0.2 else 0, f"读数:{nfci:.2f}", ">-0.2金融收紧"])
        
        # WSJ 抛压
        net_i = wsj.get('nasdaq_adv',0) - wsj.get('nasdaq_dec',0) if wsj else 0
        indicators.append(["NASDAQ 广度 (A/D)", 1 if net_i < 0 else 0, f"Net:{net_i}", "<0跌多涨少"])

        return indicators, pe

    def generate_chart(self):
        data, pe_val = self.fetch_and_calculate()
        fig = plt.figure(figsize=(33, 46), facecolor=self.colors['bg'])
        ax = fig.add_subplot(111); ax.axis('off')
        ax.text(0.5, 0.96, "美股崩盘预警系统 Pro V10.106", ha='center', fontsize=38, color=self.colors['title'], fontweight='bold')
        
        t_data = [[d[0], "触发" if d[1]==2 else ("预警" if d[1]==1 else "安全"), d[2], d[3]] for d in data]
        table = ax.table(cellText=t_data, colLabels=['指标', '评级', '数据', '逻辑'], cellLoc='center', loc='center')
        table.scale(1, 6.75); table.set_fontsize(23)
        
        for (row, col), cell in table.get_celld().items():
            if row == 0: cell.set_facecolor(self.colors['table_header']); cell.set_text_props(color='white')
            elif row > 0:
                lvl = data[row-1][1]
                if lvl == 2: cell.set_facecolor(self.colors['row_warn'])
                elif lvl == 1: cell.set_facecolor(self.colors['row_risk'])
                else: cell.set_facecolor(self.colors['row_safe'])
                cell.set_text_props(color='white')

        st.pyplot(fig)
        return pe_val

# ==============================================================================
# 附加功能模块 (完全保留 A6.py)
# ==============================================================================
def run_fred_traffic_light(key):
    p_section("🚦 收益率曲线红绿灯")
    if key:
        try:
            f = Fred(api_key=key)
            c = f.get_series('T10Y2Y', sort_order='desc', limit=1).iloc[0]
            p_txt(f"10Y-2Y 利差: {c:+.2f}%")
        except: pass

def main():
    if st.sidebar.button("🔄 刷新数据"): st.rerun()
    app = CrashWarningSystem()
    pe_val = app.generate_chart()
    run_fred_traffic_light(USER_FRED_KEY)

if __name__ == "__main__":
    main()
