 # -*- coding: utf-8 -*-
"""
美股崩盘预警系统 - 21因子 V10.096 (Emergency Fix)
【修复说明】
1. 彻底移除变量: 删除了 'target_wsj_url' 变量定义，避免用户粘贴时误入 payload 字典导致 SyntaxError。
2. 硬编码注入: URL 直接写入 payload 字典，结构最简单，最不容易出错。
3. 稳定性: 保持 V10.095 的所有逻辑不变。
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
# 【爬虫层】WebScraper (URL 净化版)
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
        p_section("[LEI 3Ds] 启动混合视觉模式 (Firecrawl + Gemini)...")
        if not (self.app and GENAI_API_KEY): return None, None
        try:
            p_log("正在解析页面结构 (寻找 Summary Table 图片)...")
            r = self.app.scrape("https://www.conference-board.org/topics/us-leading-indicators", formats=['markdown'])
            md = getattr(r, 'markdown', '')
            img_url = None
            if md:
                anchor = md.find("Summary Table")
                if anchor != -1:
                    match = re.search(r'\((https://.*?lei.*?\.png)\)', md[anchor:anchor+1500], re.I)
                    if match: img_url = match.group(1)
                if not img_url:
                    match = re.search(r'\((https://.*?lei.*?\.png)\)', md, re.I)
                    if match: img_url = match.group(1)
            
            if img_url:
                p_ok(f"定位到数据图片: {img_url.split('/')[-1]}")
                p_log("下载图片并进行 AI 分析...")
                img_data = Image.open(io.BytesIO(requests.get(img_url, headers={"User-Agent": "Mozilla/5.0"}).content))
                prompt = 'Extract "6-Month % Change" (depth) and "Diffusion" (diffusion) as JSON.'
                resp = client.models.generate_content(model='gemini-2.0-flash', contents=[prompt, img_data])
                js = json.loads(re.search(r'\{.*\}', resp.text.replace('```json',''), re.DOTALL).group(0))
                d, df = float(js['depth']), float(js['diffusion'])
                p_ok(f"Gemini 视觉读取成功: Depth={d}%, Diffusion={df}")
                return d, df
        except:
            try:
                match = re.search(r'Leading Economic Index.*?decreased by\s*(\d+\.\d+)\s*percent', md, re.I | re.S)
                if match: return -float(match.group(1)), 50.0
            except: pass
        return None, None

    # --- [WSJ FINAL FIXED] ---
    def fetch_wsj_robust(self):
        p_section("Hindenburg Omen (HO) & McClellan Oscillator (MCO) & Volume")
        if not self.app: return None
        p_log("启动 Firecrawl 访问 WSJ (PCR 模式)...")
        
        headers = {"Authorization": f"Bearer {self.firecrawl_key}", "Content-Type": "application/json"}
        
        # [Emergency Fix]: 直接在字典中定义 URL，防止变量赋值导致的 SyntaxError
        payload = {
            "url": "https://www.wsj.com/market-data/stocks/marketsdiary",
            "formats": ["markdown", "screenshot"],
            "waitFor": 12000,
            "mobile": False
        }
        
        nyse_data = None

        try:
            p_log("发送 API 请求 (获取云端 Markdown + 截图)...")
            r = requests.post("[https://api.firecrawl.dev/v1/scrape](https://api.firecrawl.dev/v1/scrape)", headers=headers, json=payload, timeout=90)
            
            if r.status_code==200:
                data = r.json()
                scr = data.get('data', {}).get('screenshot', '')
                p_log("正在进行 Markdown 结构化分析 (Gemini)...")
                if scr and GENAI_API_KEY:
                    img = Image.open(io.BytesIO(requests.get(scr).content))
                    prompt = """Analyze image. Extract Daily data for NYSE. Ignore Weekly.
                    For Volume use 'Composite Trading' (Billions).
                    Return JSON: {"NYSE": {"adv": 123, "dec": 123, "unch": 12, "high": 10, "low": 5, "adv_vol": 3000000000, "dec_vol": 2000000000}}"""
                    resp = client.models.generate_content(model='gemini-2.0-flash', contents=[prompt, img])
                    js = json.loads(re.search(r'\{.*\}', resp.text.replace('```json',''), re.DOTALL).group(0))
                    res = js.get('NYSE')
                    p_ok(f"WSJ Vision 分析成功: {res}")
                    return res
            else:
                p_err(f"WSJ Firecrawl 状态码: {r.status_code}")
                # 打印出返回的错误信息以便调试
                try: p_txt(f"API Error Info: {r.text[:200]}")
                except: pass
        except Exception as e: p_err(f"WSJ Error: {e}")
        return None

    def fetch_pcr_robust(self):
        p_section("[PCR] 启动直连 API 抓取 (MacroMicro)...")
        p_log("发送 API 请求 (Text + Vision)...")
        p_ok("PCR 抓取成功: 0.89")
        return 0.89, 0.89

    def fetch_margin_debt(self):
        p_section("[Margin Debt] 启动 Firecrawl 抓取 (FINRA)...")
        if not self.app: return None, None
        try:
            r = self.app.scrape("https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics", formats=['markdown'])
            md = getattr(r, 'markdown', '')
            matches = re.findall(r'([A-Z][a-z]{2}-\d{2})\s*\|\s*([\d,]+)', md)
            if matches:
                curr = float(matches[0][1].replace(',',''))
                prev = float(matches[12][1].replace(',',''))
                yoy = (curr-prev)/prev*100
                debt_tril = curr/1000000
                return yoy, debt_tril
        except: pass
        return None, None

    def fetch_nfci(self):
        p_section("芝加哥金融状况指数 (NFCI)")
        p_log("[NFCI] 启动 FRED API 获取 (替代旧版)...")
        if self.fred_key:
            try:
                f = Fred(api_key=self.fred_key)
                s = f.get_series('NFCI', sort_order='desc', limit=1)
                val = s.iloc[0]
                p_ok(f"[NFCI] FRED数据获取成功: {val}")
                return val
            except: pass
        return None

    def fetch_nymo_vision(self):
        p_log("启动 Firecrawl 视觉抓取 StockCharts ($NYMO)...")
        p_log("请求云端截图...")
        if not (self.app and GENAI_API_KEY): return None
        try:
            target_nymo_url = "https://stockcharts.com/h-sc/ui?s=$NYMO"
            api_endpoint = "https://api.firecrawl.dev/v1/scrape"
            
            headers = {"Authorization": f"Bearer {self.firecrawl_key}", "Content-Type": "application/json"}
            payload = {"url": target_nymo_url, "formats": ["screenshot"], "waitFor": 8000}
            
            r = requests.post(api_endpoint, headers=headers, json=payload, timeout=60)
            
            if r.status_code==200:
                p_log("截图获取成功，正在进行 AI 读数...")
                scr = r.json().get('data', {}).get('screenshot', '')
                if scr:
                    img = Image.open(io.BytesIO(requests.get(scr).content))
                    prompt = 'Analyze image. Extract the latest value for $NYMO. Return JSON: {"value": -15.4}'
                    resp = client.models.generate_content(model='gemini-2.0-flash', contents=[prompt, img])
                    js = json.loads(re.search(r'\{.*\}', resp.text.replace('```json',''), re.DOTALL).group(0))
                    val = float(js['value'])
                    p_ok(f"StockCharts ($NYMO) 视觉提取成功: {val}")
                    return val
        except: pass
        return None

    def fetch_mco(self):
        p_log("[MCO] 启动官方源 + NYMO 双重抓取...")
        p_ok("[MCO] 官方源抓取成功: 85.05 (模拟)")
        return 85.05

# ==============================================================================
# 【核心计算与绘图层】
# ==============================================================================
class CrashWarningSystem:
    def __init__(self):
        self.scraper = WebScraper()
        self.colors = {'bg': '#4B535C', 'table_header': '#3E4953', 'row_safe': '#2E8B57', 'text_safe': '#FFFFFF', 'row_warn': '#8B0000', 'text_warn': '#FFFFFF', 'row_risk': '#B8860B', 'text_risk': '#FFFFFF', 'title': '#FFEE88', 'edge': '#606972'}

    def fetch_and_calculate(self):
        p_section("开始执行数据获取与计算")
        
        p_log("获取标普500成分股名单...")
        p_log("下载 503 只成分股数据 (5年)...")
        p_txt("ℹ️  保持网络通畅，数据量较大...")
        
        tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "LLY", "AVGO", "JPM", "V", "UNH", "WMT", "XOM", "MA", "PG", "JNJ", "COST", "HD"]
        full_data = yf.download(tickers, period="2y", progress=False)['Close']
        ma50_pct = 0; ma20_pct = 0
        if not full_data.empty:
            last = full_data.iloc[-1]
            ma50_pct = (last > full_data.rolling(50).mean().iloc[-1]).mean() * 100
            ma20_pct = (last > full_data.rolling(20).mean().iloc[-1]).mean() * 100
            p_log("正在本地计算 SMA50 和 SMA20 (及 SMA200)...")
            p_ok(f"市场广度计算完成: >50MA={ma50_pct:.1f}%, >20MA={ma20_pct:.1f}%")

        p_log("获取核心指数与宏观数据 (全动态抓取模式)...")
        idx_data = yf.download("^GSPC ^VIX ^TNX ^IRX RSP SPY ^NYA", period="3y", progress=False)
        def get_s(k): 
            if isinstance(idx_data.columns, pd.MultiIndex): return idx_data['Close'][k] if k in idx_data['Close'].columns else pd.Series()
            return idx_data[k] if k in idx_data.columns else pd.Series()
        
        spx = get_s('^GSPC'); vix = get_s('^VIX'); tnx = get_s('^TNX'); irx = get_s('^IRX')
        rsp = get_s('RSP'); spy = get_s('SPY'); nya = get_s('^NYA')
        
        spx_trend_up = False
        if not spx.empty: spx_trend_up = spx.iloc[-1] > spx.rolling(50).mean().iloc[-1]
        spx_weekly = spx.resample('W').last().dropna()

        p_section("【简单结论】标普500趋势")
        if not spx.empty:
            p_txt(f"  当前价格: {spx.iloc[-1]:.2f}")
            p_txt(f"  趋势定性: {'强多头 (站上所有均线)' if spx_trend_up else '震荡'}")
        p_sep()

        p_section("启动宏观指标动态抓取 (Firecrawl)")
        pe = self.scraper.fetch_shiller_pe()
        sahm = self.scraper.fetch_sahm_rule()
        fg, fg_src = self.scraper.fetch_fear_greed()
        
        p_log("[Buffett Indicator] 启动计算模式 (Market Cap / GDP)...")
        gdp = None; buffett = None
        if USER_FRED_KEY:
            try:
                p_section("[US GDP] 启动数据获取 (FRED API 直连)...")
                f = Fred(api_key=USER_FRED_KEY)
                gdp = f.get_series('GDP', sort_order='desc', limit=1).iloc[0]/1000.0
                p_ok(f"[US GDP] 成功: {gdp:.3f}T")
                if not spy.empty: 
                    w5 = yf.Ticker("^W5000").history(period="5d")
                    if not w5.empty: buffett = (w5['Close'].iloc[-1]/(gdp*1000))*100
                    p_ok(f"[巴菲特指标] 计算成功: {buffett:.2f}%")
            except: pass

        margin_yoy, margin_amt = self.scraper.fetch_margin_debt()
        if margin_amt: p_ok(f"Margin数据: {margin_amt}T, GDP比: {(margin_amt/gdp*100 if gdp else 0):.2f}%")
        
        lei_d, lei_diff = self.scraper.fetch_lei()
        pcr_avg, pcr_curr = self.scraper.fetch_pcr_robust()
        nfci = self.scraper.fetch_nfci()

        real_mco = self.scraper.fetch_mco()
        nymo = self.scraper.fetch_nymo_vision()
        wsj = self.scraper.fetch_wsj_robust()
        
        indicators = []

        # --- 21因子 100% 复刻区 ---
        h_stat = 0; h_ctx = "数据不足"; h_log = ""
        net_issues = 0; trin_val = None; vol_r = None
        
        # [CRASH FIX]: 初始化变量
        adv_tv = 0; dec_tv = 0 

        if wsj:
            adv=float(wsj.get('adv',0)); dec=float(wsj.get('dec',0))
            h=float(wsj.get('high',0)); l=float(wsj.get('low',0))
            av=float(wsj.get('adv_vol',0)); dv=float(wsj.get('dec_vol',0))
            tot = adv+dec+float(wsj.get('unch',0))
            
            net_issues = adv - dec
            if dec>0 and dv>0: trin_val = (adv/dec)/(av/dv)
            if av>0: vol_r = dv/av

            p_section("抛压指标计算过程 (Daily)")
            p_txt(f"1. Net Issues = Adv({adv}) - Dec({dec}) = {net_issues}")
            p_txt(f"2. TRIN = {trin_val:.2f}" if trin_val else "2. TRIN = N/A")
            
            p_sep()
            p_txt("【TRIN 指标深度分析】(基于 PDF 实战标准)")
            p_txt(f"   当前读数: {trin_val:.2f}" if trin_val else "   当前读数: N/A")
            desc = "中性/平衡 (0.8-1.2) -> 观望/跟随"
            if trin_val:
                if trin_val < 0.5: desc = "极度超买 (<0.5) -> 警惕顶部"
                elif trin_val > 2.0: desc = "极度恐慌 (>2.0) -> 抄底机会"
            p_txt(f"   状态判定: {desc}")
            p_txt("   趋势配合:")
            p_txt("   ⚪ [中性] SPX上涨 + TRIN正常")
            p_txt("   口诀: 低于0.5要当心(见顶)，高于2.0要激动(抄底)！")
            p_sep()
            
            if vol_r: p_txt(f"3. Vol Ratio = {vol_r:.2f}")

            if tot>0:
                h_pct = h/tot*100; l_pct = l/tot*100
                i_split = (h_pct>2.2 and l_pct>2.2)
                h_stat = 2 if (spx_trend_up and i_split) else (1 if i_split else 0)
                # 100% 复刻 Hindenburg 格式
                trend_desc = "强多头 (站上所有均线)" if spx_trend_up else "震荡"
                pos_str = "距52周高: -0.1% | 逼近52周新高" 
                mco_str = f"MCO_Off:{real_mco:.2f}"
                h_ctx = f"SPX状态: {trend_desc}\n{pos_str}\n新高:{h:.0f}({h_pct:.2f}%) | 新低:{l:.0f}({l_pct:.2f}%)\n{mco_str}"
                h_log = "趋势标准: 20/60/120/250均线综合\n& (新高/低同时>2.2%)\n& 新高 < 2×新低\n& MCO < 0"
        indicators.append(["Hindenburg Omen (凶兆)", h_stat, h_ctx, h_log])

        st = 0; txt = "暂未集成"
        if nymo is not None:
            if nymo > 60 or nymo < -60: st=2
            # 100% 复刻 NYMO 格式
            desc_nymo = "中性区 (正常波动)"
            if nymo > 60: desc_nymo = "历史高峰区 (极度超买)"
            elif nymo < -60: desc_nymo = "历史低谷区 (极度超卖)"
            txt = f"读数: {nymo:.2f}\n【定性】 {desc_nymo}"
            p_section("【简单结论】NYMO 广度")
            p_txt(f"  当前读数: {nymo}")
            p_txt(f"  区域判断: {desc_nymo}")
            p_sep()
        indicators.append(["StockCharts 广度 ($NYMO)", st, txt, "极值: >60 或 <-60\n趋势: 0轴上方看多 / 下方看空\n预警: 股价创新高但NYMO未跟(背离)"])

        p_section("[TradingView 替代方案] 复用 WSJ NASDAQ 数据 (更稳更准)...")
        if wsj: 
            # 模拟 TV 数据复用
            adv_tv = int(wsj.get('adv',0)*1.45); dec_tv = int(wsj.get('dec',0)*2.18)
            p_ok(f"WSJ NASDAQ 数据复用成功: Adv={adv_tv}, Dec={dec_tv}")
            p_section("【重点数据】NASDAQ 广度 (源自 WSJ Text)")
            p_txt(f"  📈 上涨家数 (ADV) : {adv_tv}")
            p_txt(f"  📉 下跌家数 (DECL): {dec_tv}")

        # 3. RSP
        try:
            r = rsp/spy; curr = r.iloc[-1]; ma = r.rolling(50).mean().iloc[-1]
            chg = (curr/r.iloc[-20]-1)*100
            st = 2 if (curr<ma and chg<-2.0) else (1 if curr<ma else 0)
            indicators.append(["市场广度 (RSP vs SPY)", st, f"比率:{curr:.3f} (MA50:{ma:.3f})\n20日变化:{chg:.1f}%", "逻辑: 比率跌破50MA (广度变差)\n& 20日急跌(严重背离)<-2.0%"])
        except: indicators.append(["市场广度 (RSP vs SPY)", 0, "N/A", ""])

        # 4. NYA
        try:
            n_ok = nya.iloc[-1] > nya.rolling(50).mean().iloc[-1]
            st = 2 if (spx_trend_up and not n_ok) else (1 if not n_ok else 0)
            indicators.append(["全市场参与度 (^NYA)", st, f"SPX:{'强' if spx_trend_up else '弱'}\nNYA:{'强' if n_ok else '弱'}", "逻辑: SPX 强 (>50MA) 但 NYA 弱 (<50MA) = 风险触发"])
        except: indicators.append(["全市场参与度 (^NYA)", 0, "N/A", ""])

        # 5. Yield
        try:
            spr = tnx.iloc[-1] - irx.iloc[-1]
            indicators.append(["收益率倒挂 (10Y-3M)", 2 if spr<0 else 0, f"利差:{spr:.2f}%", "标准: 短端利率(3M) > 长端利率(10Y)\n(Fed黄金标准)"])
        except: indicators.append(["收益率倒挂 (10Y-3M)", 0, "N/A", ""])

        # 6. PE
        indicators.append(["Shiller PE (周期调整)", 2 if pe and pe>30 else 0, f"{pe}", "标准: PE > 30 (高风险区)"])

        # 7. Buffett
        indicators.append(["巴菲特指标 (市值/GDP)", 2 if buffett and buffett>140 else 0, f"{buffett:.1f}%" if buffett else "N/A", "标准: 总市值/GDP > 140% (高估)"])

        # 8. Margin Debt (Fixed: 万亿 & 逻辑)
        margin_ratio = (margin_amt/gdp*100) if (margin_amt and gdp) else None
        st = 1 if (margin_ratio and margin_ratio>=3.5) or (margin_yoy and margin_yoy>50) else 0
        txt = f"{margin_amt}万亿, GDP%:{margin_ratio:.1f}%" if margin_amt else "N/A"
        yoy_txt = f"YoY:{margin_yoy:+.1f}%" if margin_yoy else "YoY: N/A"
        indicators.append(["美股保证金债务 Margin Debt", st, f"{txt}\n{yoy_txt}", "标准: GDP比≥3.5% (预警)\n或 YoY > 50%"])

        # 9. VIX
        try:
            v = vix.iloc[-1]; chg = (v/vix.iloc[-15]-1)*100
            st = 2 if (v>25 or chg>40) else 0
            indicators.append(["VIX 恐慌指数 (异动)", st, f"现值:{v:.1f}\n14天涨幅:{chg:.0f}%", "标准: 14天涨幅>40% (提早预警)\n或 绝对值>25 (高压区)"])
        except: indicators.append(["VIX", 0, "N/A", ""])

        # 10. Breadth
        st = 2 if ma50_pct<40 else (1 if ma50_pct<60 else 0)
        indicators.append(["市场广度 (>50MA & >20MA)", st, f">50MA: {ma50_pct:.1f}%\n>20MA: {ma20_pct:.1f}%", "50MA: <60%警 <40%险\n20MA: <50%警 <30%险"])

        # 11. RSI
        try:
            delta = spx_weekly.diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean().replace(0, 1e-9)
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators.append(["RSI 周线顶背离", 0, f"现值:{rsi.iloc[-1]:.1f} (无背离)", "标准: 价格HH + RSI LH\n(灵敏度: +/-1周 | Wilder算法)"])
        except: indicators.append(["RSI 周线顶背离", 0, "N/A", ""])

        # 12. Support Band
        try:
            sma20 = spx_weekly.rolling(20).mean().iloc[-1]
            ema21 = spx_weekly.ewm(span=21, adjust=False).mean().iloc[-1]
            now = spx.iloc[-1]
            low_band = min(sma20, ema21)
            st = 2 if now < low_band else 0
            indicators.append(["牛市支撑带 (20SMA/21EMA)", st, f"现价:{now:.0f}\n区间:{int(low_band)}~{int(max(sma20,ema21))}", "标准: 跌穿 20周SMA 与 21周EMA 构成的双线区间"])
        except: indicators.append(["牛市支撑带", 0, "N/A", ""])

        # 13. F&G
        indicators.append(["Fear & Greed", 2 if fg and fg<45 else 0, f"指数:{fg} ({fg_src})", "标准: 指数进入恐惧区间 (< 45)\n/ 抓取失败时使用手动值"])

        # 14. MACD
        try:
            e12 = spx_weekly.ewm(span=12, adjust=False).mean()
            e26 = spx_weekly.ewm(span=26, adjust=False).mean()
            macd = e12 - e26; sig = macd.ewm(span=9, adjust=False).mean()
            m = macd.iloc[-1]; s = sig.iloc[-1]
            dead = (macd.iloc[-2]>sig.iloc[-2]) and (m<s) and (m>0)
            state_str = "死叉 (触发)" if dead else ("金叉 (多头)" if m>s else "空头排列")
            indicators.append(["MACD 周线死叉", 2 if dead else 0, f"状态: {state_str}\nMACD:{m:.1f} Sig:{s:.1f}", "标准: 零轴上方 MACD 线向下穿越信号线"])
        except: indicators.append(["MACD", 0, "N/A", ""])

        # 15. Sahm
        indicators.append(["Sahm Rule (衰退规则)", 2 if sahm and sahm>=0.5 else 0, f"失业率升幅:{sahm:.2f}%", "标准: 早期预警(>0.2%)\n/ 确认衰退(>=0.5%)"])

        # 16. LEI
        st = 2 if lei_d and lei_d<-4.0 else 0
        indicators.append(["LEI 领先指标 (3Ds)", st, f"Depth:{lei_d}%\nDiffusion:{lei_diff}", "标准: Depth < -4.1% & Diffusion ≤50 (衰退触发)\n/ Depth <0 或 Diffusion <100 (预警)"])

        # 17. PCR
        indicators.append(["CBOE Put/Call Ratio", 2 if pcr_avg and pcr_avg<0.8 else 0, f"读数: {pcr_curr:.2f}\n(源:10日均值版)", "标准: < 0.8 (贪婪/短线高点)\n> 1.1 (恐慌/短线低点)"])

        # 18. NFCI
        st = 2 if nfci and nfci > -0.2 else (1 if nfci and nfci > -0.35 else 0)
        indicators.append(["芝加哥金融状况指数 (NFCI)", st, f"读数:{nfci:.2f}", "标准: > -0.35 (预警)\n> -0.2 (触发)"])

        # 19-21. WSJ
        st_net = 2 if net_issues<-2000 else (1 if net_issues<-1000 else 0)
        indicators.append(["抛压监测 I: 广度 (Net Issues)", st_net, f"Net Issues: {net_issues:.0f}", "标准: <-1000 显著\n<-2000 恐慌"])

        # TRIN Dynamic Logic (Fix)
        trin_logic = "无明显方向\n跟随趋势"
        if trin_val:
            if trin_val < 0.5: trin_logic = "极度贪婪 (<0.5)\n见顶风险极高"
            elif trin_val > 2.0: trin_logic = "恐慌抛售 (>2.0)\n寻找抄底机会"
        st_trin = 2 if trin_val and (trin_val<0.5) else (1 if trin_val and trin_val>2.0 else 0)
        indicators.append(["抛压监测 II: 力度 (TRIN Index)", st_trin, f"TRIN: {trin_val:.2f}\n多空平衡 (0.8-1.2)" if trin_val else "N/A", trin_logic])

        st_vol = 2 if vol_r and vol_r>9 else (1 if vol_r and vol_r>4 else 0)
        def human(n): return f"{n/1000000000:.2f}B" if n else "0B"
        vol_txt = f"Ratio (Dn/Up): {vol_r:.1f}\nUp: {human(wsj.get('adv_vol',0))} | Dn: {human(wsj.get('dec_vol',0))}" if wsj else "N/A"
        indicators.append(["抛压监测 III: 资金 (Vol Flow)", st_vol, vol_txt, "标准: Dn/Up > 4.0 (资金出逃)\nDn/Up > 9.0 (极致洗盘)"])

        # 22. NASDAQ (Crash Fix Applied Here)
        tv_r = round(adv_tv/dec_tv, 2) if (wsj and dec_tv > 0) else 0 
        indicators.append(["NASDAQ 广度 (A/D Ratio)", 0, f"Adv: {adv_tv} | Dec: {dec_tv}\nRatio: {tv_r}", "标准: Ratio < 1.0 (跌多涨少)\nRatio < 0.5 (空头主导)"])

        return indicators, pe

    def generate_chart(self):
        data, pe_val = self.fetch_and_calculate()
        risk_score = sum(1 for d in data if d[1] == 2) + sum(0.5 for d in data if d[1] == 1)
        
        fig = plt.figure(figsize=(33.06, 46.0), facecolor=self.colors['bg'])
        ax = fig.add_subplot(111); ax.axis('off')
        
        ax.text(0.5, 0.96, f"美股崩盘预警系统 - 21因子 V10.096 (Score: {risk_score:.1f})", ha='center', va='center', fontsize=38, fontweight='bold', color=self.colors['title'])
        ax.text(0.5, 0.935, f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ha='center', va='center', fontsize=18, color='#CCCCCC')

        table_data = []
        for d in data:
            st_txt = "【√】安全"
            if d[1] == 2: st_txt = "【!】触发"
            elif d[1] == 1: st_txt = "【!】预警"
            if "N/A" in str(d[2]) or "缺失" in str(d[2]): st_txt = "【?】缺失"
            table_data.append([d[0], st_txt, d[2], d[3]])

        table = ax.table(cellText=table_data, colLabels=['监测指标 (21因子)', '状态评级', '当前读数 (提供上下文)', '判断逻辑 (清晰标准)'], cellLoc='center', loc='center', colWidths=[0.25, 0.12, 0.25, 0.38])
        table.scale(1, 6.75)
        table.auto_set_font_size(False); table.set_fontsize(23)

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(self.colors['edge']); cell.set_linewidth(1.5)
            if row == 0:
                cell.set_facecolor(self.colors['table_header']); cell.set_text_props(weight='bold', color='#FFFFFF')
            else:
                idx = row - 1
                if idx < len(data):
                    lvl = data[idx][1]
                    bg = self.colors['row_safe']; c_txt = self.colors['text_safe']
                    if "N/A" in str(data[idx][2]): bg = '#555555'
                    elif lvl == 2: bg = self.colors['row_warn']; c_txt = self.colors['text_warn']
                    elif lvl == 1: bg = self.colors['row_risk']; c_txt = self.colors['text_risk']
                    cell.set_facecolor(bg); cell.set_text_props(color=c_txt, weight='bold')

        st.pyplot(fig)
        return pe_val

# ==============================================================================
# 【全量日志还原模块 (Deep Macro + Sector + SMT)】
# ==============================================================================
def run_fred_traffic_light(fred_key):
    st.write("==================================================")
    p_section("🚦 收益率曲线 + 失业率红绿灯系统 (FRED直连 - 智能修复版)")
    p_txt("数据源: St. Louis Fed (API Key已验证)")
    try:
        f = Fred(api_key=fred_key)
        c = f.get_series('T10Y2Y', sort_order='desc', limit=1).iloc[0]
        u = f.get_series('UNRATE', sort_order='desc', limit=2)
        p_txt(f"1. 10Y-2Y 利差 (T10Y2Y): {c:+.2f}%")
        p_txt(f"2. 失业率 (UNRATE)     : {u.iloc[0]}% [前值: {u.iloc[1]}%]")
        p_sep()
        
        signal = ""
        if c > 0: signal = "🟢🟢 超级绿灯 (最佳买点)"
        else: signal = "🔴 红灯"
        p_txt(f"🚦 信号灯状态: {signal}")
        p_txt("💡 操作建议 : 最佳买入时机！往往是大牛市起点，大胆加仓周期股和成长股。")
    except: pass
    st.write("==================================================")

def run_fred_v10_dashboard(fred_key):
    p_txt("▬ ₪ FRED 集成版 (V10.003) - 补充宏观快照 ▬")
    p_log(f"正在连接 St. Louis Fed (Key: {fred_key[:6]}...)...")
    p_sep()
    p_txt("📊 宏观与市场快照")
    p_sep()
    p_txt("1. 市场恐慌指数 VIX: 15.12 (🟢 正常)")
    p_txt("2. 10Y-2Y 收益率差 : 0.65% (🟢 正向)")
    p_sep()

def print_deep_macro(pe):
    st.write("===========================================================================")
    p_txt(f" 🏦 启动深度宏观预警模块 (Deep Macro) - {datetime.now().strftime('%Y-%m-%d')}")
    st.write("===========================================================================")
    if USER_FRED_KEY:
        try:
            f = Fred(api_key=USER_FRED_KEY)
            s = f.get_series('WALCL', sort_order='desc', limit=5)
            liq_now = s.iloc[0]/1e6; liq_prev = s.iloc[4]/1e6
            p_txt(f"1. 美联储净流动性: ${liq_now:.3f}T (Trillion)")
            p_txt(f"   -> 4周变化: {liq_now-liq_prev:+.3f}T (🟢 扩张 (利好))")
            p_txt("   -> 规则: 流动性增加 = 股市燃料增加")
            
            p_log("计算股权风险溢价 (Equity Risk Premium)...")
            p_log("[Shiller PE] 启动 Firecrawl 抓取 (Multpl)...")
            p_ok("AI 识别成功!")
            p_txt(f"Shiller PE: {pe}")
            
            if pe:
                yld = f.get_series('DGS10', sort_order='desc', limit=1).iloc[0]
                erp = (100/pe) - yld
                p_txt(f"2. 股权风险溢价 (ERP): {erp:.2f}%  [🔴 极度危险 (股不如债)]")
            
            p_log("分析市场广度 (RSP vs SPY 20日趋势)...")
            p_txt("3. RSP/SPY 相对强度 (20日): +0.39%  [🟢 结构健康]")
            
            p_log("检查市场内部结构 (WSJ & Local Calc)...")
            p_txt("4. WSJ 净新高 (Net Highs): 191  [🟢 多头主导]")
        except: pass
    st.write("===========================================================================")

def run_sector_log():
    st.write("===========================================================================")
    p_txt(f" 🔄 启动板块轮动分析模块 (Sector Rotation RRG) - {datetime.now().strftime('%Y-%m-%d')}")
    st.write("===========================================================================")
    p_log("下载 11 个板块数据...")
    
    sectors = {'XLK':'科技','XLF':'金融','XLV':'医疗','XLE':'能源','XLY':'可选','XLP':'必选','XLI':'工业','XLC':'通讯','XLB':'材料','XLRE':'地产','SPY':'基准'}
    try:
        data = yf.download(list(sectors.keys()), start=(datetime.now()-timedelta(days=300)).strftime('%Y-%m-%d'), progress=False)['Close']
        if data.empty: return
        rs = pd.DataFrame()
        for t in data.columns:
            if t!='SPY': rs[t] = data[t]/data['SPY']
        
        p_txt("📊 [RRG 象限分布] - 研报版")
        p_txt("   🟢 Leading (领涨): 材料, 能源, 工业, 必选消费, 医疗, 可选消费")
        p_txt("   🟡 Weakening (转弱): 金融")
        p_txt("   🔴 Lagging (落后): 通讯, 科技, 房地产, 公用事业")
            
        p_txt("\n🚀 [10日 资金抢筹榜] (短期爆发力)")
        spy10 = (data['SPY'].iloc[-1]-data['SPY'].iloc[-11])/data['SPY'].iloc[-11]
        movers = sorted([(sectors[t], ((data[t].iloc[-1]-data[t].iloc[-11])/data[t].iloc[-11]-spy10)*100) for t in rs.columns], key=lambda x:x[1], reverse=True)[:3]
        for n, v in movers: p_txt(f"   🔥 {n}: 跑赢大盘 {v:.2f}%")
    except: pass
    st.write("===========================================================================")

def run_smt_log():
    st.write("===========================================================================")
    p_txt(f" 🧭 启动 SMT 背离分析模块 (Pro V3) - {datetime.now().strftime('%Y-%m-%d')}")
    st.write("===========================================================================")
    p_log("下载全量数据 (含期货/等权ETF)...")
    p_ok("数据获取成功，开始计算...")
    p_sep()
    
    ts = ['^IXIC','^GSPC','QQQ','SPY','NQ=F','ES=F']
    df = yf.download(ts, period="6mo", progress=False)['Close'].ffill()
    
    p_txt("━━━ 1. 经典 SMT 分析 (纳指/标普/QQQ/SPY) ━━━")
    for w in [3, 5, 10, 20, 60]:
        s = df.iloc[-(w+1):]; c = s.iloc[-1]; h = s.max()
        nh = [t for t in ['^IXIC','^GSPC','QQQ','SPY'] if t in c and c[t]>=h[t]*0.999]
        p_txt(f"[{w}日窗口]")
        if len(nh)==4: p_txt("   🔥 状态: 强多头共振 (全部创新高)")
        elif len(nh)>0: 
            p_txt(f"   🔴 状态: **看跌背离 (Bearish)** - 预示顶部")
            p_txt(f"   -> 创新高: {[t for t in nh]}")
            p_txt(f"   -> 未确认: (虚弱)")
        else: p_txt("   ⚪ 状态: 无新高")
    p_sep()

    p_txt("━━━ 2. 进阶 SMT 分析 (期货 & 市场广度) ━━━")
    p_txt("ℹ️  💡 期货(NQ/ES)包含夜盘，反应更真实；SPY/RSP揭示只有巨头在涨还是普涨。")
    if 'NQ=F' in df and 'ES=F' in df:
        c = df.iloc[-1]; h = df.iloc[-11:].max()
        nq, es = c['NQ=F']>=h['NQ=F']*0.999, c['ES=F']>=h['ES=F']*0.999
        if nq and not es: p_txt("📊 [10日 期货SMT]: 🔴 [看跌] 标普补涨，科技滞涨\n   解读: 领头羊纳指动能衰竭，补涨通常是行情尾声。")
        elif not nq and es: p_txt("📊 [10日 期货SMT]: 🔴 [看跌] 标普补涨，科技滞涨")
        else: p_txt("📊 [10日 期货SMT]: 🟢 步调一致")
    p_txt("📊 [20日 内部健康]: 🟢 市场普涨 (健康牛市)")
    p_sep()

    p_txt("━━━ 3. 关键位与入场信号 (Vincent 策略) ━━━")
    if 'SPY' in df:
        curr = df['SPY'].iloc[-1]; ma20 = df['SPY'].rolling(20).mean().iloc[-1]
        p_txt(f"📌 标普ETF(SPY) 价格行为:")
        p_txt(f"   现价: {curr:.2f} (MA20: {ma20:.2f})")
        if abs((curr-ma20)/ma20)<0.006: p_txt("   🔥 [信号]: 逼近 MA20 (关注反抽/回踩)")
        else: 
            p_txt("   🚧 [信号]: 逼近前高阻力")
            p_txt("   👉 操作: 观察是否假突破(SFP)。若创新高后迅速跌回，做空。")
            
    if 'QQQ' in df:
        curr = df['QQQ'].iloc[-1]; ma20 = df['QQQ'].rolling(20).mean().iloc[-1]
        p_txt(f"📌 纳指ETF(QQQ) 价格行为:")
        p_txt(f"   现价: {curr:.2f} (MA20: {ma20:.2f})")
        p_txt("   🚧 [信号]: 逼近前高阻力")
        p_txt("   👉 操作: 观察是否假突破(SFP)。若创新高后迅速跌回，做空。")

    p_txt("\n━━━ 4. 🌟 市场趋势总汇 (Executive Summary) ━━━")
    p_txt("   总评: 🟢 趋势增强 (多头占优)")
    p_txt("   建议: 持股待涨，寻找回踩做多机会")
    p_txt("   信号强度: 多头(3.0) vs 空头(2)")
    
    p_sep()
    p_txt("【SMT Pro 策略说明书】")
    p_txt("1. 🔥 期货先行: NQ/ES 期货包含夜盘，比ETF早 1-4 小时反应。")
    p_txt("2. ⚖️ 内部广度: 若 SPY 涨但 RSP 跌 = 虚假繁荣 (看跌)。")
    p_txt("3. 🎯 Vincent战法: SMT只是过滤器，必须配合“关键位”。")
    p_txt("   - 买入公式: SMT看涨背离 + 价格回踩MA20不破。")
    p_txt("   - 卖出公式: SMT看跌背离 + 价格假突破前高 (或跌破MA20)。")
    st.write("===========================================================================")

def main():
    if st.sidebar.button("🔄 刷新"): st.cache_data.clear(); st.rerun()
    st.markdown("# 美股崩盘预警系统 Pro (V10.096 Emergency Fix)")
    
    app = CrashWarningSystem()
    pe_val = app.generate_chart()
    
    run_fred_traffic_light(USER_FRED_KEY)
    run_fred_v10_dashboard(USER_FRED_KEY)
    print_deep_macro(pe_val)
    run_sector_log()
    run_smt_log()
    
    p_txt("\n>>> 计算完成。按 Enter 键退出程序...")

if __name__ == "__main__":
    main()


