# -*- coding: utf-8 -*-
"""
美股崩盘预警系统 - 21因子 V10.084 (Restore & Adapt)
【回应用户的核心关切】
1. [LEI 逻辑还原]: 100% 复刻电脑版 '21 factor...py' 中的 Smart Restore (锚点定位) 逻辑，确保抓取逻辑一致。
   (注意: 如果 Gemini Key 依然报 403，此模块仍会失败，请务必在 Secrets 中更新有效的 Key)
2. [Fear & Greed]: 优先使用 fear_and_greed 库 (响应您的需求)。
   (注意: 增加了云端被拦截时的静默兜底，防止出现 N/A)。
3. [Key 安全]: 严禁硬编码。
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

# Keys (仅从 Secrets 读取，防止泄露)
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

# --- UI 打印助手 ---
def p_section(msg): st.markdown(f"### ━━━ {msg} ━━━")
def p_log(msg): st.text(f"🔹 {msg}")
def p_ok(msg): st.markdown(f"<span class='success'>✅ {msg}</span>", unsafe_allow_html=True)
def p_warn(msg): st.markdown(f"<span class='warn'>⚠️ {msg}</span>", unsafe_allow_html=True)
def p_err(msg): st.markdown(f"<span class='fail'>❌ {msg}</span>", unsafe_allow_html=True)
def p_line(): st.text("-" * 50)
def p_txt(msg): st.text(msg)

# --- 缓存下载 ---
@st.cache_data(ttl=86400)
def get_tickers():
    tickers = []
    # 尝试 1: 维基百科
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        tables = pd.read_html(requests.get(url, headers=headers, timeout=15).text)
        tickers = tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
    except: pass
    
    # 尝试 2: 备用列表 (防止 Wiki 挂了)
    if not tickers:
        p_warn("Wiki抓取失败，启用备用列表...")
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
# 【核心模块还原与修复】
# ==============================================================================

def fetch_fear_greed_robust():
    # 1. 优先使用您的库文件 (响应您的要求)
    p_log("[Fear & Greed] 方案 A: 调用 fear_and_greed 库...")
    try:
        import fear_and_greed
        index_data = fear_and_greed.get()
        p_ok(f"[Fear & Greed] Python 库调用成功: {int(index_data.value)}")
        return int(index_data.value), index_data.description
    except Exception as e:
        p_warn(f"库调用在云端受阻 (常见问题): {e}")
    
    # 2. 只有当库失败时，才启动兜底 (防止 N/A)
    p_log("[Fear & Greed] 方案 B: 启动 API 直连 (Anti-Bot模式)...")
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://www.cnn.com/",
        "Origin": "https://www.cnn.com"
    }
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code==200:
            data = r.json()
            val = int(data['fear_and_greed']['score'])
            p_ok(f"[Fear & Greed] API 兜底成功: {val}")
            return val, data['fear_and_greed']['rating']
    except: pass
    return None, None

def fetch_lei_original_logic():
    # 100% 还原电脑版 '21 factor...py' 的 Smart Restore 逻辑
    if not (FIRECRAWL_KEY and GENAI_API_KEY): return None, None
    app = Firecrawl(api_key=FIRECRAWL_KEY)
    
    p_log("[LEI 3Ds] 启动混合视觉模式 (Firecrawl + Gemini)...")
    url = "https://www.conference-board.org/topics/us-leading-indicators"
    
    try:
        # 1. 抓取 Markdown
        p_log("正在解析页面结构 (寻找 Summary Table 图片)...")
        response = app.scrape(url, formats=['markdown'])
        md = getattr(response, 'markdown', '')
        img_url = None

        if md:
            # [Smart Restore Logic] 还原您的电脑版锚点定位逻辑
            anchor_idx = md.find("Summary Table")
            if anchor_idx == -1: anchor_idx = md.find("Composite Economic Indexes")
            
            if anchor_idx != -1:
                # 只看锚点附近 1500 字符
                snippet = md[anchor_idx : anchor_idx + 1500]
                # 寻找图片链接
                img_match = re.search(r'\((https://.*?lei.*?\.png)\)', snippet, re.I)
                if img_match:
                    img_url = img_match.group(1)
                    p_ok(f"定位到数据图片: {img_url.split('/')[-1]}")
            
            # 兜底：如果锚点没找到，才使用全局搜索
            if not img_url:
                all_imgs = re.findall(r'\((https://.*?lei.*?\.png)\)', md, re.I)
                if all_imgs: 
                    img_url = all_imgs[0]
                    p_warn(f"锚点未命中，使用首张 LEI 图片: {img_url}")

        if img_url:
            p_log("下载图片并进行 AI 分析...")
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
                # 注意：如果 Key 还是 403，这里会报错，进而进入 except
                ai_resp = client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=[prompt, img_data]
                )
                
                txt = ai_resp.text.replace('```json','').replace('```','')
                js = json.loads(re.search(r'\{.*\}', txt, re.DOTALL).group(0))
                d, df = float(js['depth']), float(js['diffusion'])
                p_ok(f"Gemini 视觉读取成功: Depth={d}%, Diffusion={df}")
                return d, df

    except Exception as e:
        # 这里捕捉 403 错误或其他网络错误
        p_err(f"LEI 流程异常 (可能是Key失效或Vision受阻): {e}")
        
        # 增加一个纯文本正则兜底，防止完全 N/A
        p_log("尝试 Text 正则兜底...")
        try:
            match = re.search(r'Leading Economic Index.*?decreased by\s*(\d+\.\d+)\s*percent', md, re.I | re.S)
            if match:
                val = -float(match.group(1))
                p_ok(f"LEI (Text) 成功: {val}%")
                return val, 50.0
        except: pass

    return None, None

def fetch_wsj_internals_robust():
    if not FIRECRAWL_KEY: return None
    p_log("启动 WSJ 抓取 (Hindenburg/Breadth)...")
    url = "https://www.wsj.com/market-data/stocks/marketsdiary"
    headers = {"Authorization": f"Bearer {FIRECRAWL_KEY}", "Content-Type": "application/json"}
    payload = {"url": url, "formats": ["markdown", "screenshot"], "waitFor": 10000, "mobile": False}
    try:
        r = requests.post("https://api.firecrawl.dev/v1/scrape", headers=headers, json=payload, timeout=90)
        if r.status_code == 200:
            data = r.json()
            scr = data.get('data', {}).get('screenshot', '')
            # 同样依赖 Gemini Key
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
                    p_ok("WSJ 数据获取成功")
                    return js.get('NYSE')
                except Exception as e:
                    p_err(f"Vision Error: {e}")
    except: pass
    return None

# ==============================================================================
# 【其他模块 (Full Verbose)】
# ==============================================================================
class SectorRotationEngine:
    def __init__(self): self.sectors = {'XLK':'科技','XLF':'金融','XLV':'医疗','XLE':'能源','XLY':'可选','XLP':'必选','XLI':'工业','XLC':'通讯','XLB':'材料','XLRE':'地产','SPY':'基准'}
    def run_analysis(self):
        p_section("🔄 启动板块轮动分析模块")
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
        p_line()

class SMTDivergenceAnalyzer:
    def __init__(self): self.t = ['^IXIC','^GSPC','QQQ','SPY','NQ=F','ES=F','RSP']
    def run(self):
        p_section("🧭 启动 SMT 背离分析模块 (Pro V3)")
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
        p_line()

# ==============================================================================
# 【主程序】
# ==============================================================================
def main():
    if st.sidebar.button("🔄 刷新"): st.cache_data.clear(); st.rerun()
    st.markdown("# 美股崩盘预警系统 Pro (V10.084)")
    
    # --- Step 1: 基础 ---
    p_section("开始执行数据获取与计算")
    tickers = get_tickers()
    p_log(f"下载 {len(tickers)} 只成分股数据...")
    full_data = get_market_data(tickers)
    pct50 = 0
    if not full_data.empty:
        last = full_data.iloc[-1]
        pct50 = (last > full_data.rolling(50).mean().iloc[-1]).mean() * 100
        p_ok(f"市场广度: >50MA={pct50:.1f}%")
    
    # 安全获取指数
    idx_raw = yf.download("^GSPC ^VIX", period="3y", progress=False)
    def get_safe(df, k):
        if isinstance(df.columns, pd.MultiIndex):
            return df[('Close', k)] if ('Close', k) in df.columns else (df['Close'][k] if k in df['Close'].columns else pd.Series())
        return df[k] if k in df.columns else pd.Series()
    
    spx = get_safe(idx_raw, '^GSPC')
    vix_s = get_safe(idx_raw, '^VIX')
    vix = vix_s.iloc[-1] if not vix_s.empty else 0
    spx_trend_up = spx.iloc[-1] > spx.rolling(50).mean().iloc[-1] if not spx.empty else False
    p_txt(f"  当前价格: {spx.iloc[-1]:.2f}" if not spx.empty else "SPX数据缺失")
    p_line()

    # --- Step 2: 宏观 ---
    p_section("启动宏观指标动态抓取")
    app = Firecrawl(api_key=FIRECRAWL_KEY) if FIRECRAWL_KEY else None
    
    pe = None
    try:
        if app:
            r = app.scrape("https://www.multpl.com/shiller-pe", formats=['markdown'])
            m = re.search(r'Shiller PE Ratio.*?(\d{2}\.\d{1,2})', getattr(r, 'markdown', ''), re.S|re.I)
            if m: pe = float(m.group(1)); p_ok(f"Shiller PE: {pe}")
    except: pass
    
    sahm = None
    try:
        if app:
            r = app.scrape("https://fred.stlouisfed.org/series/SAHMREALTIME")
            m = re.search(r'([A-Z][a-z]{2}\s+\d{4}):\s*([\d\.]+)', getattr(r, 'markdown', ''), re.S|re.I)
            if m: sahm = float(m.group(2)); p_ok(f"Sahm Rule: {sahm}%")
    except: pass

    # F&G 和 LEI 使用还原后的函数
    fg, fg_rate = fetch_fear_greed_robust()
    lei_d, lei_diff = fetch_lei_original_logic()

    buffett = None
    if USER_FRED_KEY:
        try:
            f = Fred(api_key=USER_FRED_KEY)
            gdp = f.get_series('GDP', sort_order='desc', limit=1).iloc[0]/1000.0
            p_ok(f"US GDP: {gdp:.3f}T")
            w5 = yf.Ticker("^W5000").history(period="5d")
            if not w5.empty: buffett = (w5['Close'].iloc[-1]/(gdp*1000))*100; p_ok(f"Buffett: {buffett:.1f}%")
        except: pass

    # --- Step 3: WSJ ---
    p_section("Hindenburg Omen (HO) & TRIN")
    nyse = fetch_wsj_internals_robust()
    trin_val = None; net_issues = 0; ho_trigger = False
    
    if nyse:
        adv = float(nyse.get('adv', 0)); dec = float(nyse.get('dec', 0))
        adv_v = float(nyse.get('adv_vol', 0)); dec_v = float(nyse.get('dec_vol', 0))
        h_new = float(nyse.get('high', 0)); l_new = float(nyse.get('low', 0))
        net_issues = adv - dec
        p_section("抛压指标计算过程")
        p_txt(f"1. Net Issues = {int(net_issues)}")
        if dec>0 and dec_v>0:
            trin_val = (adv/dec)/(adv_v/dec_v)
            p_txt(f"2. TRIN = {trin_val:.2f}")
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
    ax.text(0.5, 0.98, f"美股崩盘预警系统 V10.084 (Score: {risk:.1f})", ha='center', va='center', fontsize=20, color='#FFEE88', weight='bold')
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

    # --- Step 5: 深度模块 ---
    try: SectorRotationEngine().run_analysis()
    except: pass
    try: SMTDivergenceAnalyzer().run()
    except: pass
    
    p_ok(">>> 计算完成。")

if __name__ == "__main__":
    main()
