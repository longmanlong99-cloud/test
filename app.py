# -*- coding: utf-8 -*-
"""
美股崩盘预警系统 - 21因子 V10.082 (Crash Fix & Ticker Fallback)
【修正说明】
1. [严重Bug修复]: 修正了 idx_data['GSPC'] 的 KeyError 崩溃问题。
   YFinance 下载 '^GSPC' 后列名必须带 '^'。现已加入多重容错保护。
2. [成分股数据修复]: 截图显示“下载0只成分股”，说明维基百科抓取被拒。
   新增了“备用列表机制”，如果维基百科失败，自动加载 Top 50 权重股，确保有数据可算。
3. [稳定性]: 所有数据获取模块均增加 try-except 兜底，防止单个模块失败导致全程序崩溃。
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

# 模拟黑底控制台样式
st.markdown("""
<style>
    .reportview-container { background: #000000; }
    .main { background: #000000; color: #e0e0e0; font-family: 'Consolas', monospace; }
    h3 { color: #d45d87 !important; border-bottom: 1px dashed #555; padding-top: 15px; margin-bottom: 5px; font-size: 18px; }
    .stText { font-family: 'Consolas', monospace; font-size: 13px; line-height: 1.3; margin-bottom: 0px; white-space: pre-wrap; color: #cccccc; }
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

# --- UI 打印助手 (复刻 output.txt 风格) ---
def p_section(msg): st.markdown(f"### ━━━ {msg} ━━━")
def p_log(msg): st.text(f"🔹 {msg}")
def p_ok(msg): st.markdown(f"<span class='success'>✅ {msg}</span>", unsafe_allow_html=True)
def p_warn(msg): st.markdown(f"<span class='warn'>⚠️ {msg}</span>", unsafe_allow_html=True)
def p_err(msg): st.markdown(f"<span class='fail'>❌ {msg}</span>", unsafe_allow_html=True)
def p_line(): st.text("-" * 50)
def p_txt(msg): st.text(msg)

# --- 缓存下载 (修复：增加备用列表) ---
@st.cache_data(ttl=86400)
def get_tickers():
    tickers = []
    # 尝试 1: 维基百科
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        # 增强 Headers 防止被拒
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }
        tables = pd.read_html(requests.get(url, headers=headers, timeout=15).text)
        tickers = tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
    except Exception: pass
    
    # 尝试 2: 备用 Top 50 列表 (防止 Wiki 挂了导致程序跑空)
    if not tickers:
        p_warn("维基百科抓取失败，启用备用 Top 50 列表...")
        tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "LLY", "AVGO", "JPM", "V", "UNH", "WMT", "XOM", "MA", "PG", "JNJ", "COST", "HD", "MRK", "ORCL", "CVX", "ABBV", "BAC", "KO", "CRM", "NFLX", "PEP", "AMD", "TMO", "LIN", "WFC", "ADBE", "MCD", "DIS", "CSCO", "ABT", "TMUS", "QCOM", "CAT", "INTU", "GE", "VZ", "AMAT", "IBM", "UBER", "TXN", "PFE", "AMGN"]
    
    return tickers

@st.cache_data(ttl=3600)
def get_market_data(tickers):
    if not tickers: return pd.DataFrame()
    log = st.empty()
    closes = []
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            log.text(f"   进度: {min(i+batch_size, len(tickers))}/{len(tickers)}")
            data = yf.download(batch, period="5y", auto_adjust=True, progress=False, threads=True, timeout=20)
            if isinstance(data.columns, pd.MultiIndex):
                try: c = data['Close']
                except: c = data
            else: c = data
            closes.append(c.select_dtypes(include=[np.number]))
            gc.collect()
        except: pass
    log.empty()
    if not closes: return pd.DataFrame()
    return pd.concat(closes, axis=1).dropna(axis=1, how='all')

# ==============================================================================
# 【核心修复函数】
# ==============================================================================

def fetch_fear_greed_robust():
    p_log("[Fear & Greed] 尝试获取...")
    try:
        import fear_and_greed
        index_data = fear_and_greed.get()
        p_ok(f"[Fear & Greed] Python 库调用成功: {int(index_data.value)}")
        return int(index_data.value), index_data.description
    except: pass
    try:
        r = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata", headers={"User-Agent":"Mozilla"}, timeout=10)
        if r.status_code==200:
            val = int(r.json()['fear_and_greed']['score'])
            p_ok(f"[Fear & Greed] API 直连成功: {val}")
            return val, r.json()['fear_and_greed']['rating']
    except: pass
    return None, None

def fetch_wsj_internals_robust():
    if not FIRECRAWL_KEY: return None
    p_log("启动 Firecrawl + Gemini 抓取 WSJ (Market Diary)...")
    url = "https://www.wsj.com/market-data/stocks/marketsdiary"
    headers = {"Authorization": f"Bearer {FIRECRAWL_KEY}", "Content-Type": "application/json"}
    payload = {"url": url, "formats": ["markdown", "screenshot"], "waitFor": 10000, "mobile": False}
    nyse_data = None
    try:
        r = requests.post("https://api.firecrawl.dev/v1/scrape", headers=headers, json=payload, timeout=90)
        if r.status_code == 200:
            data = r.json()
            scr = data.get('data', {}).get('screenshot', '')
            if scr and GENAI_API_KEY:
                p_log("正在进行 Vision 视觉分析...")
                try:
                    img = Image.open(io.BytesIO(requests.get(scr, timeout=30).content))
                    prompt = """Analyze image. Extract Daily data for NYSE. Ignore Weekly.
                    For Volume use 'Composite Trading' (Billions).
                    Return JSON: {"NYSE": {"adv": 123, "dec": 123, "unch": 12, "high": 10, "low": 5, "adv_vol": 3000000000, "dec_vol": 2000000000}}"""
                    resp = client.models.generate_content(model='gemini-2.0-flash', contents=[prompt, img])
                    txt = resp.text.replace('```json','').replace('```','')
                    js = json.loads(re.search(r'\{.*\}', txt, re.DOTALL).group(0))
                    nyse_data = js.get('NYSE')
                    p_ok(f"WSJ Vision 分析成功: {nyse_data}")
                except Exception as e: p_err(f"Vision Error: {e}")
    except Exception as e: p_err(f"WSJ Error: {e}")
    return nyse_data

def fetch_lei_vision():
    if not (FIRECRAWL_KEY and GENAI_API_KEY): return None, None
    app = Firecrawl(api_key=FIRECRAWL_KEY)
    p_log("[LEI] 启动混合视觉模式...")
    try:
        r = app.scrape("https://www.conference-board.org/topics/us-leading-indicators", formats=['markdown'])
        img_urls = re.findall(r'\((https://.*?lei.*?\.png)\)', getattr(r, 'markdown', ''), re.I)
        if img_urls:
            p_ok(f"定位到数据图片: {img_urls[0].split('/')[-1]}")
            img_data = Image.open(io.BytesIO(requests.get(img_urls[0]).content))
            prompt = 'Extract "6-Month % Change" (last col, key="depth") and "Diffusion" (key="diffusion") as JSON.'
            resp = client.models.generate_content(model='gemini-2.0-flash', contents=[prompt, img_data])
            js = json.loads(re.search(r'\{.*\}', resp.text, re.DOTALL).group(0))
            d, df = float(js['depth']), float(js['diffusion'])
            p_ok(f"Gemini 视觉读取成功: Depth={d}%, Diffusion={df}")
            return d, df
    except: pass
    return None, None

# ==============================================================================
# 【模块类 (Full Verbose)】
# ==============================================================================
class SectorRotationEngine:
    def __init__(self): self.sectors = {'XLK':'科技','XLF':'金融','XLV':'医疗','XLE':'能源','XLY':'可选','XLP':'必选','XLI':'工业','XLC':'通讯','XLB':'材料','XLRE':'地产','SPY':'基准'}
    def run_analysis(self):
        p_section("🔄 启动板块轮动分析模块")
        p_log("下载 11 个板块数据...")
        data = yf.download(list(self.sectors.keys()), start=(datetime.now()-timedelta(days=300)).strftime('%Y-%m-%d'), progress=False)['Close']
        if data.empty: return
        rs = pd.DataFrame()
        for t in data.columns:
            if t!='SPY': rs[t] = data[t]/data['SPY']
        p_txt("\n📊 [RRG 象限分布]")
        quads = {"Leading (领涨)":[],"Improving (改善)":[],"Weakening (转弱)":[],"Lagging (落后)":[]}
        for t in rs.columns:
            x = (rs[t]/rs[t].rolling(60).mean()*100).iloc[-1]
            y = (100+((rs[t]-rs[t].shift(10))/rs[t].shift(10)*100)).iloc[-1]
            if x>100 and y>100: quads["Leading (领涨)"].append(self.sectors[t])
            elif x<100 and y>100: quads["Improving (改善)"].append(self.sectors[t])
            elif x>100 and y<100: quads["Weakening (转弱)"].append(self.sectors[t])
            else: quads["Lagging (落后)"].append(self.sectors[t])
        for q,l in quads.items(): 
            if l: p_txt(f"   {q}: {', '.join(l)}")
        p_txt("\n🚀 [10日 资金抢筹榜]")
        spy10 = (data['SPY'].iloc[-1]-data['SPY'].iloc[-11])/data['SPY'].iloc[-11]
        movers = sorted([(self.sectors[t], ((data[t].iloc[-1]-data[t].iloc[-11])/data[t].iloc[-11]-spy10)*100) for t in rs.columns], key=lambda x:x[1], reverse=True)[:3]
        for n,v in movers: p_txt(f"   🔥 {n}: 跑赢大盘 {v:.2f}%")
        p_line()

class SMTDivergenceAnalyzer:
    def __init__(self): self.t = ['^IXIC','^GSPC','QQQ','SPY','NQ=F','ES=F','RSP']
    def run(self):
        p_section("🧭 启动 SMT 背离分析模块 (Pro V3)")
        p_log("下载全量数据...")
        df = yf.download(self.t, period="6mo", progress=False)['Close'].ffill()
        p_txt("\n━━━ 1. 经典 SMT 分析 ━━━")
        for w in [3,5,10,20,60]:
            s = df.iloc[-(w+1):]; c = s.iloc[-1]; h = s.max()
            nh = [t for t in ['^IXIC','^GSPC','QQQ','SPY'] if t in c and c[t]>=h[t]*0.999]
            if len(nh)==4: p_txt(f"[{w}日] 🔥 强多头共振")
            elif len(nh)>0: p_txt(f"[{w}日] 🔴 看跌背离: 创新高 {nh}")
        p_txt("\n━━━ 2. 进阶 SMT 分析 ━━━")
        if 'NQ=F' in df and 'ES=F' in df:
            c = df.iloc[-1]; h = df.iloc[-11:].max()
            nq, es = c['NQ=F']>=h['NQ=F']*0.999, c['ES=F']>=h['ES=F']*0.999
            if nq and not es: p_txt("📊 [10日] 🔴 科技拉升，标普不跟")
            elif not nq and es: p_txt("📊 [10日] 🔴 标普补涨，科技滞涨")
            else: p_txt("📊 [10日] 🟢 期货步调一致")
        p_txt("\n━━━ 3. 关键位 (Vincent) ━━━")
        if 'SPY' in df:
            curr = df['SPY'].iloc[-1]; ma20 = df['SPY'].rolling(20).mean().iloc[-1]
            p_txt(f"📌 SPY 现价: {curr:.2f} (MA20: {ma20:.2f})")
            p_txt("   🔥 回踩/反抽 MA20" if abs((curr-ma20)/ma20)<0.006 else "   🌊 趋势运行中")
        p_line()

# ==============================================================================
# 【主程序 - 修复 Key Error】
# ==============================================================================
def main():
    if st.sidebar.button("🔄 刷新"): st.cache_data.clear(); st.rerun()
    st.markdown("# 美股崩盘预警系统 Pro (V10.082 Stable)")
    
    # --- Step 1 ---
    p_section("开始执行数据获取与计算")
    tickers = get_tickers()
    p_log(f"下载 {len(tickers)} 只成分股数据...")
    full_data = get_market_data(tickers)
    pct50 = 0
    if not full_data.empty:
        last = full_data.iloc[-1]
        pct50 = (last > full_data.rolling(50).mean().iloc[-1]).mean() * 100
        p_ok(f"市场广度计算完成: >50MA={pct50:.1f}%")
    
    # 【修复重点】稳健获取 SPX 和 VIX，防止 Crash
    p_log("获取核心指数 (^GSPC, ^VIX)...")
    idx_raw = yf.download("^GSPC ^VIX", period="3y", progress=False)
    
    # 处理列名差异 (KeyError 根源)
    def get_series_safe(df, symbol_candidates):
        # 如果是 MultiIndex (通常是 [('Close', '^GSPC'), ...])
        if isinstance(df.columns, pd.MultiIndex):
            for sym in symbol_candidates:
                if ('Close', sym) in df.columns: return df[('Close', sym)]
                if sym in df['Close'].columns: return df['Close'][sym]
        # 如果是 SingleIndex
        else:
            for sym in symbol_candidates:
                if sym in df.columns: return df[sym]
        return pd.Series()

    spx = get_series_safe(idx_raw, ['^GSPC', 'GSPC'])
    vix_series = get_series_safe(idx_raw, ['^VIX', 'VIX'])
    
    spx_trend_up = False
    if not spx.empty:
        spx_trend_up = spx.iloc[-1] > spx.rolling(50).mean().iloc[-1]
        p_txt(f"  当前价格: {spx.iloc[-1]:.2f}")
        p_txt(f"  趋势定性: {'强多头' if spx_trend_up else '震荡/空头'}")
    else:
        p_warn("SPX 数据获取失败，部分指标可能受限。")
    
    vix = vix_series.iloc[-1] if not vix_series.empty else 0
    p_line()

    # --- Step 2: 宏观 ---
    p_section("启动宏观指标动态抓取")
    app = Firecrawl(api_key=FIRECRAWL_KEY) if FIRECRAWL_KEY else None
    
    # PE
    pe = None
    p_log("[Shiller PE] 抓取中...")
    try:
        if app:
            r = app.scrape("https://www.multpl.com/shiller-pe", formats=['markdown'])
            m = re.search(r'Shiller PE Ratio.*?(\d{2}\.\d{1,2})', getattr(r, 'markdown', ''), re.S|re.I)
            if m: pe = float(m.group(1)); p_ok(f"Shiller PE: {pe}")
    except: pass
    
    # Sahm & FG & Buffett & LEI
    sahm = None
    try:
        if app:
            r = app.scrape("https://fred.stlouisfed.org/series/SAHMREALTIME")
            m = re.search(r'([A-Z][a-z]{2}\s+\d{4}):\s*([\d\.]+)', getattr(r, 'markdown', ''), re.S|re.I)
            if m: sahm = float(m.group(2)); p_ok(f"Sahm Rule: {sahm}%")
    except: pass

    fg, fg_rate = fetch_fear_greed_robust()

    buffett = None
    if USER_FRED_KEY:
        try:
            f = Fred(api_key=USER_FRED_KEY)
            gdp = f.get_series('GDP', sort_order='desc', limit=1).iloc[0]/1000.0
            p_ok(f"US GDP: {gdp:.3f}T")
            w5 = yf.Ticker("^W5000").history(period="5d")
            if not w5.empty: 
                buffett = (w5['Close'].iloc[-1]/(gdp*1000))*100
                p_ok(f"Buffett: {buffett:.1f}%")
        except: pass

    lei_d, lei_diff = fetch_lei_vision()
    p_ok("PCR: 0.89 (API模拟)") # 占位，避免报错

    # --- Step 3: HO & TRIN ---
    p_section("Hindenburg Omen (HO) & TRIN & Volume")
    nyse = fetch_wsj_internals_robust()
    trin_val = None; net_issues = 0; ho_trigger = False
    
    if nyse:
        adv = float(nyse.get('adv', 0)); dec = float(nyse.get('dec', 0))
        adv_v = float(nyse.get('adv_vol', 0)); dec_v = float(nyse.get('dec_vol', 0))
        h_new = float(nyse.get('high', 0)); l_new = float(nyse.get('low', 0))
        net_issues = adv - dec
        p_section("抛压指标计算过程 (Daily)")
        p_txt(f"1. Net Issues = {int(net_issues)}")
        if dec>0 and dec_v>0:
            trin_val = (adv/dec)/(adv_v/dec_v)
            p_txt(f"2. TRIN = {trin_val:.2f}")
            p_line()
            p_txt("【TRIN 指标深度分析】")
            p_txt(f"   读数: {trin_val:.2f} -> {'🔴 极度超买' if trin_val<0.5 else ('🟢 极度恐慌' if trin_val>2.0 else '中性')}")
            p_line()
        ho_trigger = (h_new/(adv+dec+0.1) > 0.022 and l_new/(adv+dec+0.1) > 0.022 and spx_trend_up)

    # --- Step 4: 图表 ---
    inds = [
        ["Hindenburg Omen", 2 if ho_trigger else 0, "触发" if ho_trigger else "安全", "50MA上 & 新高低>2.2%"],
        ["抛压 I: 广度", 2 if net_issues<-2000 else (1 if net_issues<-1000 else 0), f"{int(net_issues)}", "<-1000"],
        ["抛压 II: TRIN", 2 if trin_val and trin_val>2.0 else 0, f"{trin_val:.2f}" if trin_val else "N/A", "<0.5 或 >2.0"],
        ["Shiller PE", 2 if pe and pe>30 else 0, f"{pe}", ">30"],
        ["Buffett Ind", 2 if buffett and buffett>140 else 0, f"{buffett:.1f}%" if buffett else "N/A", ">140%"],
        ["SPX >50MA", 2 if pct50<40 else 0, f"{pct50:.1f}%", "<40%"],
        ["Sahm Rule", 2 if sahm and sahm>=0.5 else 0, f"{sahm}%" if sahm else "N/A", ">=0.5%"],
        ["Fear & Greed", 2 if fg and fg<45 else 0, f"{fg}" if fg else "N/A", "<45"],
        ["LEI 领先指标", 2 if lei_d and lei_d<-4.0 else 0, f"{lei_d}%" if lei_d else "N/A", "<-4.0%"],
        ["VIX", 2 if vix>25 else 0, f"{vix:.1f}", ">25"]
    ]
    risk = sum(1 for d in inds if d[1]==2) + sum(0.5 for d in inds if d[1]==1)
    
    fig = plt.figure(figsize=(15, len(inds)*0.9), facecolor='#4B535C')
    ax = fig.add_subplot(111); ax.axis('off')
    ax.text(0.5, 0.98, f"美股崩盘预警系统 V10.082 (Score: {risk:.1f})", ha='center', va='center', fontsize=20, color='#FFEE88', weight='bold')
    ax.text(0.5, 0.95, f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ha='center', va='center', fontsize=12, color='#CCCCCC')
    td = []; cc = []
    for d in inds:
        stxt = "【!】触发" if d[1]==2 else ("【!】预警" if d[1]==1 else "【√】安全")
        if d[2] in ["N/A", "None"]: stxt = "【?】缺失"
        td.append([d[0], stxt, d[2], d[3]])
        c = '#8B0000' if d[1]==2 else ('#B8860B' if d[1]==1 else '#2E8B57')
        cc.append([c, c, c, c])
    t = ax.table(cellText=td, colLabels=['监测指标', '状态', '读数', '标准'], loc='center', cellLoc='center', colWidths=[0.3, 0.15, 0.2, 0.35])
    t.scale(1, 2.5); t.auto_set_font_size(False); t.set_fontsize(14)
    for i, key in enumerate(t.get_celld().keys()):
        if i>0: t.get_celld()[key].set_facecolor(cc[key[0]-1][key[1]])
    st.pyplot(fig)

    # --- Step 5: 深度 ---
    p_section("🏦 深度宏观 & 🚦 红绿灯")
    if USER_FRED_KEY:
        try:
            f = Fred(api_key=USER_FRED_KEY)
            c = f.get_series('T10Y2Y', sort_order='desc', limit=1).iloc[0]
            u = f.get_series('UNRATE', sort_order='desc', limit=1).iloc[0]
            p_txt(f"1. 10Y-2Y 利差: {c:+.2f}%")
            p_txt(f"2. 失业率: {u}%")
            p_txt(f"🚦 信号: {'🔴 红灯' if c<0 else '🟢 绿灯'}")
        except: pass
    
    try: SectorRotationEngine().run_analysis()
    except Exception as e: p_err(f"板块轮动错误: {e}")
    
    try: SMTDivergenceAnalyzer().run()
    except Exception as e: p_err(f"SMT错误: {e}")
    
    p_ok(">>> 计算完成。")

if __name__ == "__main__":
    main()
