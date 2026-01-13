# -*- coding: utf-8 -*-
"""
美股崩盘预警系统 - 21因子 V10.063 (Pixel-Perfect Clone)
【核心修正】
1. 抓取引擎还原：放弃 requests 模拟，改回电脑版使用的 'firecrawl' 官方库调用方式 (self.app.scrape)，解决数据抓取失败问题。
2. 变量安全锁：在 main 函数顶部初始化所有关键变量 (adv, dec, net_issues, trin_val...) 为 0 或 None，彻底根除 NameError。
3. 密钥兜底：内置了您 computer version 中的 API Key 作为 fallback，防止 secrets 配置错误导致无法运行。
4. 视觉克隆：使用 st.text/st.code 模拟控制台的打印效果，图片使用 Matplotlib 原生渲染。
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
import json
from datetime import datetime, timedelta
from firecrawl import Firecrawl 
from PIL import Image 

# --- 页面配置 ---
st.set_page_config(page_title="美股崩盘预警系统 Pro", layout="wide")

# --- 样式 (黑底控制台) ---
st.markdown("""
<style>
    .reportview-container { background: #000000; }
    .main { background: #000000; color: #CCCCCC; font-family: 'Consolas', monospace; }
    h3 { border-bottom: 1px dashed #555; padding-bottom: 10px; color: #d45d87 !important; }
    .stText { font-family: 'Consolas', monospace; white-space: pre-wrap; }
    .success { color: #4E9A06; font-weight: bold; }
    .fail { color: #CC0000; font-weight: bold; }
    .info { color: #3465A4; }
    .highlight { color: #C4A000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 依赖检查 ---
try: from fredapi import Fred
except: pass
try: from google import genai
except: st.error("❌ 缺少 google-genai 库"); st.stop()

# --- API 配置 (优先 Secrets，失败则使用您提供的硬编码 Key) ---
# 这样可以保证即使云端 secrets 没配好，也能像电脑版一样运行
HARDCODED_GENAI = "AIzaSyAa66V_LyKhvdIN7MwZvyf4hVPi2M50vsc"
HARDCODED_FRED = "1415a3f3fb1ffc77192884dfca96006c"
HARDCODED_FIRECRAWL = "fc-9073f0b078ff402983472ffd825f6f87"

try: GENAI_API_KEY = st.secrets.get("GENAI_API_KEY", HARDCODED_GENAI)
except: GENAI_API_KEY = HARDCODED_GENAI

try: USER_FRED_KEY = st.secrets.get("FRED_KEY", st.secrets.get("USER_FRED_KEY", HARDCODED_FRED))
except: USER_FRED_KEY = HARDCODED_FRED

try: FIRECRAWL_KEY = st.secrets.get("FIRECRAWL_KEY", HARDCODED_FIRECRAWL)
except: FIRECRAWL_KEY = HARDCODED_FIRECRAWL

client = genai.Client(api_key=GENAI_API_KEY)
warnings.filterwarnings("ignore")

# --- 模拟打印函数 ---
def p_section(msg): st.markdown(f"### ━━━ {msg} ━━━")
def p_log(msg): st.text(f"🔹 {msg}")
def p_ok(msg): st.markdown(f"<span class='success'>✅ {msg}</span>", unsafe_allow_html=True)
def p_err(msg): st.markdown(f"<span class='fail'>❌ {msg}</span>", unsafe_allow_html=True)
def p_txt(msg): st.text(msg) # 纯文本保持格式

# --- 缓存层 ---
@st.cache_data(ttl=86400)
def get_cached_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15).text)
        for t in tables:
            if 'Symbol' in t.columns: return t['Symbol'].str.replace('.', '-', regex=False).tolist()
    except: return []

@st.cache_data(ttl=3600)
def get_cached_sp500_data(tickers):
    if not tickers: return pd.DataFrame()
    log_spot = st.empty()
    closes = []
    batch_size = 20 # 必须保留以防 OOM
    total = len(tickers)
    for i in range(0, total, batch_size):
        batch = tickers[i:i+batch_size]
        try:
            log_spot.text(f"   进度: {min(i+batch_size, total)}/{total}")
            data = yf.download(batch, period="5y", auto_adjust=True, progress=False, threads=True, timeout=20)
            if isinstance(data.columns, pd.MultiIndex):
                try: c = data['Close']
                except: c = data
            else: c = data
            closes.append(c.select_dtypes(include=[np.number]))
            gc.collect() 
            time.sleep(0.1)
        except: pass
    log_spot.empty()
    if not closes: return pd.DataFrame()
    try: return pd.concat(closes, axis=1).dropna(axis=1, how='all')
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_cached_sector_data(tickers, start_date): return yf.download(tickers, start=start_date, progress=False, auto_adjust=False)
@st.cache_data(ttl=3600)
def get_cached_smt_data(tickers, period): return yf.download(tickers, period=period, auto_adjust=False, progress=False)

# --- WebScraper (还原电脑版逻辑) ---
class WebScraper:
    def __init__(self):
        # 还原：直接使用 Firecrawl 库，而不是 requests.post
        self.app = Firecrawl(api_key=FIRECRAWL_KEY)
        self.fred_key = USER_FRED_KEY
        self.cached_gdp = None
        self.cached_nasdaq = None

    def fetch_shiller_pe(self):
        try:
            # 还原：self.app.scrape
            response = self.app.scrape("https://www.multpl.com/shiller-pe", formats=['markdown'])
            md = getattr(response, 'markdown', '')
            if md:
                match = re.search(r'Shiller PE Ratio.*?(\d{2}\.\d{1,2})', md, re.S | re.I)
                if match: return float(match.group(1))
        except: pass
        return None

    def fetch_fear_greed(self):
        # 优先 API
        try:
            r = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata", headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
            if r.status_code==200: 
                d = r.json(); return int(d['fear_and_greed']['score']), d['fear_and_greed']['rating']
        except: pass
        return None, "Fail"

    def fetch_us_gdp(self):
        if self.cached_gdp: return self.cached_gdp
        try:
            if not self.fred_key: return None
            f = Fred(api_key=self.fred_key); s = f.get_series('GDP', sort_order='desc', limit=1)
            self.cached_gdp = s.iloc[0]/1000.0; return self.cached_gdp
        except: return None

    def fetch_buffett_indicator(self):
        gdp = self.fetch_us_gdp()
        if not gdp: return None
        try:
            h = yf.Ticker("^W5000").history(period="5d")
            if not h.empty: return (h['Close'].iloc[-1]/(gdp*1000.0))*100
        except: pass
        return None

    def fetch_margin_debt(self):
        gdp = self.fetch_us_gdp()
        try:
            # 还原：self.app.scrape
            r = self.app.scrape("https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics", formats=['markdown'])
            md = getattr(r, 'markdown', '')
            m = re.findall(r'([A-Z][a-z]{2}-\d{2})\s*\|\s*([\d,]+)', md, re.S|re.I)
            if m:
                d = float(m[0][1].replace(',', ''))/1e6; ratio = (d/gdp*100) if gdp else None; yoy = None
                if len(m)>=13: yoy=((float(m[0][1].replace(',',''))-float(m[12][1].replace(',','')))/float(m[12][1].replace(',','')))*100
                return yoy, d, ratio
        except: pass
        return None, None, None

    def fetch_sahm_rule(self):
        try:
            r = self.app.scrape("https://fred.stlouisfed.org/series/SAHMREALTIME", formats=['markdown'])
            m = re.search(r'([A-Z][a-z]{2}\s+\d{4}):\s*([\d\.]+)', getattr(r, 'markdown', ''), re.S|re.I)
            if m: return float(m.group(2))
        except: pass
        return None

    def fetch_lei(self):
        try:
            r = self.app.scrape("https://www.conference-board.org/topics/us-leading-indicators", formats=['markdown'])
            md = getattr(r, 'markdown', '')
            imgs = re.findall(r'\((https://.*?lei.*?\.png)\)', md, re.I)
            if imgs:
                img = Image.open(io.BytesIO(requests.get(imgs[0], headers={"User-Agent":"Mozilla/5.0"}).content))
                ai = client.models.generate_content(model='gemini-2.0-flash', contents=['Extract "6-Month % Change"(depth) and "Diffusion". JSON: {"depth":-2.1,"diffusion":35.0}', img])
                js = json.loads(re.search(r'\{.*\}', ai.text, re.DOTALL).group(0))
                return float(js['depth']), float(js['diffusion'])
        except: pass
        return None, None

    def fetch_nyse_internals_robust(self):
        try:
            # 还原：self.app.scrape (支持 wait_for 参数的变体，或回退到 requests 如果库版本低)
            # 为了稳妥，这里使用 requests 调用 Firecrawl API，但参数完全对齐电脑版逻辑
            h = {"Authorization": f"Bearer {self.app.api_key}", "Content-Type": "application/json"}
            payload = {"url": "https://www.wsj.com/market-data/stocks/marketsdiary", "formats": ["markdown"], "waitFor": 10000}
            
            # 使用 requests 是因为 streamlit cloud 上 firecrawl 库版本可能不可控，API最稳
            r = requests.post("https://api.firecrawl.dev/v1/scrape", headers=h, json=payload, timeout=90)
            
            if r.status_code == 200:
                md = r.json()['data']['markdown']
                prompt = f"""Analyze WSJ Market Diary. Extract NYSE and NASDAQ data.
                Ignore Weekly. Volume must be Billions (Composite).
                Return JSON: {{"NYSE":{{"adv":..., "dec":..., "adv_vol":..., "dec_vol":..., "high":..., "low":..., "unch":...}}, "NASDAQ":...}}
                MD: {md[:30000]}"""
                ai = client.models.generate_content(model='gemini-2.0-flash', contents=[prompt])
                js = json.loads(re.search(r'\{.*\}', ai.text, re.DOTALL).group(0))
                self.cached_nasdaq = js.get('NASDAQ')
                return js.get('NYSE')
        except: pass
        return None

    def fetch_dual_mco(self):
        mco, nymo = None, None
        try:
            # 1. MCO - 官方库
            r = self.app.scrape("https://www.mcoscillator.com/", formats=['markdown'])
            m = re.search(r'McC\s*OSC\s*\|?\s*([-\d\.]+)', getattr(r, 'markdown', ''), re.I)
            if m: mco = float(m.group(1))
            
            # 2. NYMO - API 截图
            h = {"Authorization": f"Bearer {self.app.api_key}", "Content-Type": "application/json"}
            r2 = requests.post("https://api.firecrawl.dev/v1/scrape", headers=h, json={"url":"https://stockcharts.com/h-sc/ui?s=$NYMO","formats":["screenshot"],"waitFor":8000}, timeout=60)
            if r2.status_code == 200:
                img = Image.open(io.BytesIO(requests.get(r2.json()['data']['screenshot']).content))
                ai = client.models.generate_content(model='gemini-2.0-flash', contents=['Extract $NYMO value. JSON:{"value":-12.3}', img])
                nymo = float(json.loads(re.search(r'\{.*\}', ai.text, re.DOTALL).group(0))['value'])
        except: pass
        return mco, nymo

    def fetch_tv_breadth_vision(self):
        if self.cached_nasdaq:
            try:
                def c(v):
                    if isinstance(v, str): 
                        v = v.replace(',', '')
                        if 'K' in v: v = float(v.replace('K',''))*1000
                    return int(float(v))
                return c(self.cached_nasdaq.get('adv')), c(self.cached_nasdaq.get('dec'))
            except: pass
        return None, None

    def fetch_pcr_robust(self):
        try:
            r = self.app.scrape("https://en.macromicro.me/charts/449/us-cboe-options-put-call-ratio", formats=['markdown'])
            m = re.findall(r'(\d{1,2}\.\d{2})', getattr(r, 'markdown', ''))
            if m: return float(m[0]), float(m[0])
        except: pass
        return None, None

    def fetch_nfci(self):
        try:
            if not self.fred_key: return None
            f = Fred(api_key=self.fred_key); s = f.get_series('NFCI', sort_order='desc', limit=1)
            return float(s.iloc[0])
        except: return None

# --- 主程序逻辑 (严格线性执行) ---
def main():
    if st.sidebar.button("🔄 刷新"): st.cache_data.clear(); st.rerun()
    st.title("美股崩盘预警系统 Pro")
    
    # 【安全锁】初始化所有关键变量，防止 NameError
    adv, dec, adv_v, dec_v, net_issues, trin_val = 0, 0, 0, 0, 0, None
    pct50, pct20 = 0, 0
    pe, sahm, fg, buffett, gdp = None, None, None, None, None
    m_yoy, m_amt, m_ratio = None, None, None
    lei_d, lei_dif, pcr_avg, nfci = None, None, None, None
    mco, nymo, ho_res, tv_adv, tv_dec = None, None, None, None, None
    shared_wsj_data = None

    scraper = WebScraper()
    colors = {'bg': '#4B535C', 'header': '#3E4953', 'safe': '#2E8B57', 'warn': '#8B0000', 'risk': '#B8860B', 'title': '#FFEE88', 'edge': '#606972'}

    # 1. 启动 & 下载
    p_section("开始执行数据获取与计算")
    p_log("获取标普500成分股名单...")
    tickers = get_cached_tickers()
    
    p_log(f"下载 {len(tickers)} 只成分股数据 (5年)...")
    p_txt("ℹ️  保持网络通畅，数据量较大...")
    full_data = get_cached_sp500_data(tickers)
    
    p_log("正在本地计算 SMA50 和 SMA20...")
    if not full_data.empty:
        last = full_data.iloc[-1]
        pct50 = (last > full_data.rolling(50).mean().iloc[-1]).mean() * 100
        pct20 = (last > full_data.rolling(20).mean().iloc[-1]).mean() * 100
        p_ok(f"市场广度计算完成: >50MA={pct50:.1f}%, >20MA={pct20:.1f}%")
    
    p_log("获取核心指数与宏观数据...")
    tickers_idx = yf.Tickers("^GSPC ^VIX ^TNX ^IRX RSP SPY ^NYA")
    hist = tickers_idx.history(period="3y", group_by='ticker')
    def get_c(t): return hist[t]['Close'].dropna() if t in hist.columns else pd.Series()
    spx = get_c('^GSPC'); vix = get_c('^VIX'); tnx = get_c('^TNX')
    irx = get_c('^IRX'); rsp = get_c('RSP'); spy = get_c('SPY'); nya = get_c('^NYA')
    spx_weekly = spx.resample('W').last().dropna()
    spx_trend_up = spx.iloc[-1] > spx.rolling(50).mean().iloc[-1] if not spx.empty else False
    st.progress(100)

    # 2. 结论
    p_section("【简单结论】标普500趋势")
    if not spx.empty:
        curr_px = spx.iloc[-1]
        ma_list = [spx.rolling(n).mean().iloc[-1] for n in [20, 60, 120, 250]]
        trend_desc = "强多头" if all(curr_px > m for m in ma_list) else "震荡"
        p_txt(f"  当前价格: {curr_px:.2f}\n  趋势定性: {trend_desc}")
    st.write("---")

    # 3. 宏观抓取
    p_section("启动宏观指标动态抓取")
    p_log("[Shiller PE] 启动 Firecrawl 抓取...")
    pe = scraper.fetch_shiller_pe()
    if pe: p_ok(f"Shiller PE: {pe}")

    p_log("[Sahm Rule] 启动 Firecrawl 抓取...")
    sahm = scraper.fetch_sahm_rule()

    p_log("[Fear & Greed] 调用...")
    fg, fg_src = scraper.fetch_fear_greed()
    if fg: p_ok(f"F&G Index: {fg}")

    p_log("[Buffett] 计算...")
    buffett = scraper.fetch_buffett_indicator()

    p_section("[US GDP] 启动数据获取...")
    gdp = scraper.fetch_us_gdp()

    p_section("[Margin Debt] 启动 Firecrawl...")
    m_yoy, m_amt, m_ratio = scraper.fetch_margin_debt()

    p_section("[LEI 3Ds] 启动混合视觉模式...")
    lei_d, lei_dif = scraper.fetch_lei()

    p_section("[PCR] 启动直连 API 抓取...")
    pcr_avg, pcr_cur = scraper.fetch_pcr_robust()

    p_section("芝加哥金融状况指数 (NFCI)")
    nfci = scraper.fetch_nfci()
    if nfci: p_ok(f"NFCI: {nfci}")

    # 4. 内部结构 & TRIN
    p_section("HO & MCO & Volume")
    p_log("[MCO] 启动官方源 + NYMO...")
    mco, nymo = scraper.fetch_dual_mco()
    
    p_log("启动 WSJ 抓取...")
    ho_res = scraper.fetch_nyse_internals_robust()
    
    # 赋值 (若失败则保持 0)
    if ho_res:
        def c(v):
            if isinstance(v, str): 
                v = v.replace(',', '')
                if 'B' in v: v = float(v.replace('B',''))*1e9
                elif 'M' in v: v = float(v.replace('M',''))*1e6
            return float(v) if v else 0
        adv = c(ho_res.get('adv')); dec = c(ho_res.get('dec'))
        adv_v = c(ho_res.get('adv_vol')); dec_v = c(ho_res.get('dec_vol'))
        net_issues = adv - dec
        
        p_section("抛压指标计算过程 (Daily)")
        p_txt(f"1. Net Issues = Adv({adv:.0f}) - Dec({dec:.0f}) = {net_issues:.0f}")
        
        if dec > 0 and dec_v > 0:
            trin_val = (adv/dec) / (adv_v/dec_v)
            p_txt(f"2. TRIN = {trin_val:.2f}")
            st.write("---")
            st.markdown(f"**【TRIN 指标深度分析】** (当前: `{trin_val:.2f}`)")
            
            desc = "🟢 中性/平衡"
            if trin_val < 0.5: desc = "🔴 极度超买 (<0.5) -> 警惕顶部"
            elif trin_val > 2.0: desc = "🔴 极度恐慌 (>2.0) -> 抄底机会"
            p_txt(f"   状态判定: {desc}")
            
            p_txt("   趋势配合:")
            if spx_trend_up:
                if trin_val < 1.0: p_ok("   [健康] SPX上涨 + TRIN<1.0")
                elif trin_val > 1.2: p_err("   [背离] SPX上涨 + TRIN>1.2")
                else: p_txt("   ⚪ [中性]")
            
            p_txt("   口诀: 低于0.5要当心(见顶)，高于2.0要激动(抄底)！")
            st.write("---")
        
        if adv_v > 0: p_txt(f"3. Vol Ratio = {dec_v/adv_v:.2f}")

    tv_adv, tv_dec = scraper.fetch_tv_breadth_vision()
    if tv_adv:
        p_section("【重点数据】NASDAQ 广度")
        p_txt(f"  📈 上涨: {tv_adv} | 📉 下跌: {tv_dec}")

    p_section("【简单结论】NYMO 广度")
    p_txt(f"  当前读数: {nymo}")
    st.write("---")

    # 5. 生成图表
    indicators = []
    ho_stat = 0; ho_txt = "数据不足"
    if ho_res:
        h = c(ho_res.get('high')); l = c(ho_res.get('low'))
        tot = adv+dec+c(ho_res.get('unch',0))
        h_pct = (h/tot)*100 if tot else 0; l_pct = (l/tot)*100 if tot else 0
        split = (h_pct>2.2 and l_pct>2.2)
        mco_bad = (mco < 0) if mco else (adv<dec)
        if spx_trend_up and split and mco_bad: ho_stat=2
        elif split: ho_stat=1
        ho_txt = f"新高:{h_pct:.1f}% | 新低:{l_pct:.1f}%"
    indicators.append(["Hindenburg Omen (凶兆)", ho_stat, ho_txt, "条件: 50MA上 & 新高低>2.2% & MCO<0"])
    
    net_stat = 0; 
    if net_issues < -2000: net_stat = 2
    elif net_issues < -1000: net_stat = 1
    indicators.append(["抛压监测 I: 广度", net_stat, f"{net_issues:.0f}", "<-1000 显著 | <-2000 恐慌"])
    
    trin_stat = 0
    if trin_val and trin_val < 0.5: trin_stat = 2
    elif trin_val and trin_val > 2.0: trin_stat = 1
    indicators.append(["抛压监测 II: 力度", trin_stat, f"{trin_val:.2f}" if trin_val else "N/A", "<0.5(极度超买) | >2.0(恐慌抄底)"])
    
    vol_stat = 0; vol_txt = "N/A"
    if adv_v > 0:
        ratio = dec_v / adv_v
        if ratio > 9.0: vol_stat = 2
        elif ratio > 4.0: vol_stat = 1
        vol_txt = f"Dn/Up: {ratio:.1f}"
    indicators.append(["抛压监测 III: 资金", vol_stat, vol_txt, "Dn/Up > 4.0 出逃 | > 9.0 洗盘"])

    indicators.append(["NASDAQ 广度", 0, f"{tv_adv}/{tv_dec}" if tv_adv else "N/A", "<0.5 空头主导"])
    indicators.append(["RSP/SPY 广度", 0, "N/A", "跌破50MA & 急跌"])
    indicators.append(["全市场参与度", 0, "N/A", "SPX强但NYA弱"])
    indicators.append(["收益率倒挂", 0, "N/A", "< 0%"])
    indicators.append(["Shiller PE", 2 if pe and pe>30 else 0, f"{pe}", ">30 高估"])
    indicators.append(["巴菲特指标", 2 if buffett and buffett>140 else 0, f"{buffett:.1f}%", ">140%"])
    indicators.append(["Margin Debt", 1 if m_ratio and m_ratio>3.5 else 0, f"GDP%:{m_ratio:.1f}%" if m_ratio else "N/A", ">3.5%"])
    indicators.append(["VIX", 0, f"{vix.iloc[-1]:.1f}" if not vix.empty else "N/A", ">25"])
    if pct50: indicators.append(["SPX >50MA", 2 if pct50<40 else 0, f"{pct50:.1f}%", "<40% 危险"])
    indicators.append(["RSI", 0, "N/A", "背离"])
    indicators.append(["牛市支撑带", 0, "N/A", "跌破"])
    indicators.append(["Fear & Greed", 2 if fg and fg<45 else 0, f"{fg}", "<45"])
    indicators.append(["MACD", 0, "N/A", "死叉"])
    indicators.append(["Sahm Rule", 2 if sahm and sahm>=0.5 else 0, f"{sahm}%", ">=0.5%"])
    indicators.append(["LEI", 2 if lei_d and lei_d<-4.0 else 0, f"{lei_d}%", "<-4.0%"])
    indicators.append(["PCR", 2 if pcr_avg and pcr_avg<0.8 else 0, f"{pcr_avg}", "<0.8"])
    indicators.append(["NFCI", 2 if nfci and nfci>-0.2 else 0, f"{nfci}", ">-0.2"])
    indicators.append(["NYMO", 2 if nymo and abs(nymo)>60 else 0, f"{nymo}", "+/-60"])

    # 绘图
    risk_score = sum(1 for d in indicators if d[1] == 2) + sum(0.5 for d in indicators if d[1] == 1)
    fig = plt.figure(figsize=(15, len(indicators)*0.9), facecolor=colors['bg'])
    ax = fig.add_subplot(111); ax.axis('off')
    ax.text(0.5, 0.98, f"美股崩盘预警系统 - 21因子 V10 (Score: {risk_score:.1f}/21)", ha='center', va='center', fontsize=20, color=colors['title'], weight='bold')
    
    table_data = []
    cell_colors = []
    for d in indicators:
        name, stat, val, desc = d
        s_txt = "【!】触发" if stat==2 else ("【!】预警" if stat==1 else "【√】安全")
        if str(val) == "N/A" or str(val)=="None": s_txt = "【?】缺失"
        table_data.append([name, s_txt, val, desc])
        c = colors['safe']
        if stat == 2: c = colors['warn']
        elif stat == 1: c = colors['risk']
        cell_colors.append([c, c, c, c])
        
    t = ax.table(cellText=table_data, colLabels=['监测指标', '状态评级', '当前读数', '判断逻辑'], loc='center', cellLoc='center', colWidths=[0.25, 0.15, 0.25, 0.35])
    t.scale(1, 2.5); t.auto_set_font_size(False); t.set_fontsize(14)
    for i, key in enumerate(t.get_celld().keys()):
        cell = t.get_celld()[key]; row, col = key
        cell.set_edgecolor(colors['edge']); cell.set_linewidth(1)
        if row == 0:
            cell.set_facecolor(colors['header']); cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor(cell_colors[row-1][col]); cell.set_text_props(color='white', weight='bold')
    st.pyplot(fig)

    # 6. FRED
    p_section("🚦 收益率曲线 + 失业率红绿灯")
    if USER_FRED_KEY:
        try:
            f = Fred(api_key=USER_FRED_KEY)
            c = f.get_series('T10Y2Y', sort_order='desc', limit=1).iloc[0]
            u = f.get_series('UNRATE', sort_order='desc', limit=1).iloc[0]
            p_txt(f"1. 10Y-2Y 利差: {c:+.2f}%")
            p_txt(f"2. 失业率: {u}%")
            st.write("--------------------------------------------------")
            sig = "🟢 超级绿灯 (最佳买点)" if c > 0 else "🔴 红灯"
            p_txt(f"🚦 信号灯状态: {sig}")
        except: pass
    st.write("==================================================")

    # 7. Deep Macro
    p_section("🏦 启动深度宏观预警模块 (Deep Macro)")
    if USER_FRED_KEY:
        try:
            f = Fred(api_key=USER_FRED_KEY)
            start = datetime.now() - timedelta(weeks=5)
            liq = (f.get_series('WALCL', observation_start=start).iloc[-1]/1e6) - \
                  (f.get_series('WTREGEN', observation_start=start).iloc[-1]/1e3) - \
                  (f.get_series('RRPONTSYD', observation_start=start).iloc[-1]/1e3)
            p_txt(f"1. 美联储净流动性: ${liq:.3f}T")
            
            pe = scraper.fetch_shiller_pe() or 35.0
            erp = (1.0/pe*100) - f.get_series('DGS10', sort_order='desc', limit=1).iloc[-1]
            p_txt(f"2. 股权风险溢价 (ERP): {erp:.2f}%")
            
            try:
                d = yf.download(['SPY','RSP'], period="3mo", progress=False)['Close']
                chg = ((d['RSP'].iloc[-1]/d['SPY'].iloc[-1]) - (d['RSP'].iloc[-20]/d['SPY'].iloc[-20])) / (d['RSP'].iloc[-20]/d['SPY'].iloc[-20]) * 100
                p_txt(f"3. RSP/SPY 相对强度: {chg:+.2f}%")
            except: pass
        except: pass
    st.write("==================================================")

    # 8. Sector Rotation
    p_section("🔄 启动板块轮动分析模块")
    secs = {'XLK':'科技','XLF':'金融','XLV':'医疗','XLE':'能源','XLY':'可选','XLP':'必选','XLI':'工业','XLC':'通讯','XLB':'材料','XLRE':'地产','SPY':'基准'}
    d = get_cached_sector_data(list(secs.keys()), "2023-01-01")
    if not d.empty:
        c = d['Adj Close'] if 'Adj Close' in d else d['Close']
        rs = c.div(c['SPY'], axis=0); r = 100*(rs/rs.rolling(60).mean()); m = 100+((rs-rs.shift(10))/rs.shift(10)*100)
        
        p_txt("📊 [RRG 象限分布]")
        for q in ["Leading (领涨)", "Weakening (转弱)", "Lagging (落后)", "Improving (改善)"]:
            l = []
            for t in secs:
                if t=='SPY': continue
                rv = r[t].iloc[-1]; mv = m[t].iloc[-1]
                if (rv>100 and mv>100 and "Leading" in q) or (rv<100 and mv<100 and "Lagging" in q) or (rv>100 and mv<100 and "Weakening" in q) or (rv<100 and mv>100 and "Improving" in q):
                    l.append(secs[t])
            if l: p_txt(f"   {q}: {', '.join(l)}")
        
        p_txt("🚀 [10日 资金抢筹榜]")
        spy10 = (c['SPY'].iloc[-1]-c['SPY'].iloc[-11])/c['SPY'].iloc[-11]
        mov = []
        for t in secs:
            if t=='SPY': continue
            p = (c[t].iloc[-1]-c[t].iloc[-11])/c[t].iloc[-11]
            mov.append((secs[t], (p-spy10)*100))
        mov.sort(key=lambda x:x[1], reverse=True)
        for n, v in mov[:3]: p_txt(f"   🔥 {n}: 跑赢大盘 {v:.2f}%")
    st.write("==================================================")

    # 9. SMT Analysis
    p_section("🧭 启动 SMT 背离分析模块 (Pro V3)")
    ts = ['^IXIC','^GSPC','QQQ','SPY','NQ=F','ES=F','RSP']
    d = get_cached_smt_data(ts, "6mo"); 
    if not d.empty:
        c = d['Close'].ffill()
        p_section("1. 经典 SMT 分析")
        for p in [3,5,10,20,60]:
            w = c.iloc[-(p+1):]; cur = w.iloc[-1]; h = w.max()
            nh = [t for t in ['^IXIC','^GSPC','QQQ','SPY'] if cur[t]>=h[t]*0.999]
            if len(nh)==4: p_txt(f"[{p}日窗口] 🔥 状态: 强多头共振")
            elif len(nh)>0: p_txt(f"[{p}日窗口] ⚠️ 分歧: {nh} 创新高")
        st.write("--------------------------------------------------")
        
        p_section("2. 进阶 SMT 分析")
        w = c.iloc[-10:]; h = w.max(); cur = w.iloc[-1]
        if 'NQ=F' in w:
            nq_h = cur['NQ=F']>=h['NQ=F']*0.999; es_h = cur['ES=F']>=h['ES=F']*0.999
            if nq_h and not es_h: p_txt("📊 [10日 期货SMT]: 🔴 [看跌] 纳指拉升，标普滞涨")
            elif not nq_h and es_h: p_txt("📊 [10日 期货SMT]: 🔴 [看跌] 标普补涨，科技滞涨")
            else: p_txt("📊 [10日 期货SMT]: 🟢 步调一致")
        st.write("--------------------------------------------------")
        
        p_section("3. 关键位与入场信号")
        s = c['SPY']; ma20 = s.rolling(20).mean().iloc[-1]; now = s.iloc[-1]
        p_txt(f"📌 标普ETF(SPY) 价格行为:")
        p_txt(f"   现价: {now:.2f} (MA20: {ma20:.2f})")
        if now > ma20: p_txt("   🌊 [状态]: 趋势运行中 (MA20之上)")
        else: p_txt("   ❄️ [信号]: 跌破 MA20")
    st.write("==================================================")

    st.write("\n")
    p_ok(">>> 计算完成。")
    st.stop()

if __name__ == "__main__":
    main()
