# -*- coding: utf-8 -*-
"""
美股崩盘预警系统 - 21因子 V10.056 (Report Edition)
【修正说明】
1. 移除 st.status 折叠：所有分析过程（TRIN、宏观、SMT）直接展示在页面，不再隐藏。
2. 表格原生化：放弃 Matplotlib 图片，改用 st.table 展示 21 因子表，清晰度最高。
3. 增强可读性：使用 Markdown 增强排版，还原“研报”阅读体验。
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import platform
import warnings
import time
import re
import traceback 
import io
from firecrawl import Firecrawl 
from PIL import Image 

# --- 页面配置 ---
st.set_page_config(
    page_title="美股崩盘预警研报",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 样式增强 ---
st.markdown("""
<style>
    .report-title { font-size: 32px; font-weight: bold; color: #FFEE88; text-align: center; background-color: #4B535C; padding: 10px; border-radius: 5px; }
    .section-header { font-size: 24px; font-weight: bold; color: #4DA6FF; border-bottom: 2px solid #4DA6FF; margin-top: 20px; margin-bottom: 10px; }
    .metric-box { border: 1px solid #444; padding: 10px; border-radius: 5px; background-color: #262730; }
    .safe { color: #2E8B57; font-weight: bold; }
    .warn { color: #FFA500; font-weight: bold; }
    .danger { color: #FF4B4B; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 依赖检查 ---
try:
    from fredapi import Fred
except ImportError:
    st.warning("⚠️ 建议安装 fredapi")

try:
    from google import genai
except ImportError:
    st.error("❌ 缺少 google-genai 库")
    st.stop()

# ==========================================
# 【API 配置】
# ==========================================
try:
    GENAI_API_KEY = st.secrets["GENAI_API_KEY"]
    USER_FRED_KEY = st.secrets.get("FRED_KEY", st.secrets.get("USER_FRED_KEY", ""))
    FIRECRAWL_KEY = st.secrets["FIRECRAWL_KEY"]
except Exception as e:
    st.error(f"❌ Secrets 配置错误: {e}")
    st.stop()

client = genai.Client(api_key=GENAI_API_KEY)
warnings.filterwarnings("ignore")

# ==========================================
# 【UI 输出函数 (直出不折叠)】
# ==========================================
def print_h(msg): 
    st.markdown(f"<div class='section-header'>{msg}</div>", unsafe_allow_html=True)

def print_step(msg): 
    st.markdown(f"🔹 *{msg}*")

def print_ok(msg): 
    st.success(f"✅ {msg}")

def print_warn(msg): 
    st.warning(f"⚠️ {msg}")

def print_err(msg): 
    st.error(f"❌ {msg}")

def print_info(msg): 
    st.info(f"ℹ️ {msg}")

# ==========================================
# 【缓存层】
# ==========================================
@st.cache_data(ttl=86400)
def get_cached_tickers():
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
    status = st.empty()
    status.write(f"⏳ 正在下载 {len(tickers)} 只成分股数据 (批次下载中)...")
    progress = st.progress(0)
    
    closes = []
    batch_size = 50
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
            progress.progress(min((i + batch_size) / total, 1.0))
            time.sleep(0.1)
        except: pass
        
    status.empty(); progress.empty()
    if not closes: return pd.DataFrame()
    return pd.concat(closes, axis=1).dropna(axis=1, how='all')

@st.cache_data(ttl=3600)
def get_cached_sector_data(tickers, start_date):
    return yf.download(tickers, start=start_date, progress=False, auto_adjust=False)

@st.cache_data(ttl=3600)
def get_cached_smt_data(tickers, period):
    return yf.download(tickers, period=period, auto_adjust=False, progress=False)

# ==========================================
# 【WebScraper (Firecrawl)】
# ==========================================
class WebScraper:
    def __init__(self):
        self.app = Firecrawl(api_key=FIRECRAWL_KEY)
        self.fred_key = USER_FRED_KEY
        self.cached_gdp = None 
        self.cached_nasdaq = None

    def fetch_shiller_pe(self):
        print_step("抓取 Shiller PE...")
        try:
            resp = self.app.scrape("https://www.multpl.com/shiller-pe", formats=['markdown'])
            match = re.search(r'Shiller PE Ratio.*?(\d{2}\.\d{1,2})', getattr(resp, 'markdown', ''), re.S | re.I)
            if match:
                val = float(match.group(1))
                print_ok(f"Shiller PE: {val}")
                return val
        except: pass
        return None

    def fetch_fear_greed(self):
        print_step("抓取 Fear & Greed...")
        try:
            resp = self.app.scrape("https://www.cnn.com/markets/fear-and-greed", formats=['markdown'])
            match = re.search(r'(?:Fear\s*&\s*Greed\s*Index|Current\s*Reading).*?(\d{1,3})', getattr(resp, 'markdown', ''), re.S | re.I)
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
        except: pass
        return None, "获取失败"

    def fetch_us_gdp(self):
        if self.cached_gdp: return self.cached_gdp
        try:
            if not self.fred_key: return None
            fred = Fred(api_key=self.fred_key)
            s = fred.get_series('GDP', sort_order='desc', limit=1)
            val = s.iloc[0] / 1000.0
            self.cached_gdp = val
            return val
        except: return None

    def fetch_buffett_indicator(self):
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
        gdp = self.fetch_us_gdp()
        try:
            resp = self.app.scrape("https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics", formats=['markdown'])
            matches = re.findall(r'([A-Z][a-z]{2}-\d{2})\s*\|\s*([\d,]+)', getattr(resp, 'markdown', ''), re.S | re.I)
            if matches:
                val_str = matches[0][1]
                debt = float(val_str.replace(',', '')) / 1_000_000
                ratio = (debt / gdp * 100) if gdp else None
                yoy = None
                if len(matches) >= 13:
                    prev = float(matches[12][1].replace(',', ''))
                    curr = float(val_str.replace(',', ''))
                    yoy = ((curr - prev) / prev) * 100
                print_ok(f"Margin Debt: {debt:.3f}T (YoY: {yoy:.1f}%)")
                return yoy, debt, ratio
        except: pass
        return None, None, None

    def fetch_sahm_rule(self):
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
        print_step("分析 LEI (AI Vision)...")
        try:
            resp = self.app.scrape("https://www.conference-board.org/topics/us-leading-indicators", formats=['markdown'])
            md = getattr(resp, 'markdown', '')
            img_url = None
            if md:
                imgs = re.findall(r'\((https://.*?lei.*?\.png)\)', md, re.I)
                if imgs: img_url = imgs[0]
            if img_url:
                content = requests.get(img_url, headers={"User-Agent": "Mozilla/5.0"}).content
                img = Image.open(io.BytesIO(content))
                prompt = 'Extract "6-Month % Change" (depth) and "Diffusion" value. JSON: {"depth": -2.1, "diffusion": 35.0}'
                ai = client.models.generate_content(model='gemini-2.0-flash', contents=[prompt, img])
                js = json.loads(re.search(r'\{.*\}', ai.text, re.DOTALL).group(0))
                d, dif = float(js['depth']), float(js['diffusion'])
                print_ok(f"LEI: Depth={d}, Diffusion={dif}")
                return d, dif
        except: pass
        return None, None

    def fetch_nyse_internals_robust(self):
        print_step("抓取 WSJ 内部数据...")
        try:
            headers = {"Authorization": f"Bearer {FIRECRAWL_KEY}", "Content-Type": "application/json"}
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
        except: pass
        return None

    def fetch_dual_mco(self):
        mco, nymo = None, None
        try:
            # MCO
            resp = self.app.scrape("https://www.mcoscillator.com/", formats=['markdown'])
            match = re.search(r'McC\s*OSC\s*\|?\s*([-\d\.]+)', getattr(resp, 'markdown', ''), re.I)
            if match: 
                mco = float(match.group(1))
                print_ok(f"MCO Official: {mco}")
            
            # NYMO Vision
            headers = {"Authorization": f"Bearer {FIRECRAWL_KEY}", "Content-Type": "application/json"}
            payload = {"url": "https://stockcharts.com/h-sc/ui?s=$NYMO", "formats": ["screenshot"], "waitFor": 6000}
            r = requests.post("https://api.firecrawl.dev/v1/scrape", headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                url = r.json().get('data', {}).get('screenshot', '')
                if url:
                    img = Image.open(io.BytesIO(requests.get(url).content))
                    prompt = 'Extract $NYMO value. JSON: {"value": -12.3}'
                    ai = client.models.generate_content(model='gemini-2.0-flash', contents=[prompt, img])
                    nymo = float(json.loads(re.search(r'\{.*\}', ai.text, re.DOTALL).group(0)).get('value'))
                    print_ok(f"NYMO: {nymo}")
        except: pass
        return mco, nymo

    def fetch_tv_breadth_vision(self):
        if hasattr(self, 'cached_nasdaq') and self.cached_nasdaq:
            try:
                def c(v): return int(float(str(v).replace(',','').replace('K','000'))) if v else 0
                adv, dec = c(self.cached_nasdaq.get('adv')), c(self.cached_nasdaq.get('dec'))
                print_ok(f"NASDAQ Breadth: +{adv} / -{dec}")
                return adv, dec
            except: pass
        return None, None

    def fetch_pcr_robust(self):
        print_step("抓取 PCR...")
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
        try:
            if not self.fred_key: return None
            f = Fred(api_key=self.fred_key)
            s = f.get_series('NFCI', sort_order='desc', limit=1)
            val = float(s.iloc[0])
            print_ok(f"NFCI: {val}")
            return val
        except: return None

# ==========================================
# 【核心程序】
# ==========================================
class CrashWarningSystem:
    def __init__(self):
        self.scraper = WebScraper()
        self.shared_wsj_data = None

    def calculate_spx_breadth_deep(self):
        tickers = get_cached_tickers()
        data = get_cached_sp500_data(tickers)
        if data.empty: return None, None
        
        last = data.iloc[-1]
        pct50 = (last > data.rolling(50).mean().iloc[-1]).mean() * 100
        pct20 = (last > data.rolling(20).mean().iloc[-1]).mean() * 100
        print_ok(f"市场广度: >50MA={pct50:.1f}%, >20MA={pct20:.1f}%")
        return pct50, pct20

    def analyze_market_trends_console(self):
        print_h("深度宏观分析 (Deep Macro)")
        if not USER_FRED_KEY: return
        
        col1, col2 = st.columns(2)
        try:
            fred = Fred(api_key=USER_FRED_KEY)
            with col1:
                start = datetime.now() - timedelta(weeks=5)
                walcl = fred.get_series('WALCL', observation_start=start).iloc[-1]
                tga = fred.get_series('WTREGEN', observation_start=start).iloc[-1]
                rrp = fred.get_series('RRPONTSYD', observation_start=start).iloc[-1]
                liq = (walcl/1e6) - (tga/1e3) - (rrp/1e3)
                st.metric("美联储净流动性", f"${liq:.3f}T", help="规则: 流动性增加 = 股市燃料增加")

            with col2:
                dgs10 = fred.get_series('DGS10', sort_order='desc', limit=1).iloc[-1]
                pe = self.scraper.fetch_shiller_pe() or 35.0
                erp = (1.0/pe*100) - dgs10
                st.metric("股权风险溢价 (ERP)", f"{erp:.2f}%")
        except: st.error("宏观数据计算失败")
        
        # 3. RSP/SPY
        try:
            df = yf.download(['SPY', 'RSP'], period="3mo", progress=False)['Close']
            if not df.empty:
                ratio = df['RSP'] / df['SPY']
                chg = ((ratio.iloc[-1] - ratio.iloc[-20]) / ratio.iloc[-20]) * 100
                
                trend_signal = "🟢 结构健康" 
                if df['SPY'].iloc[-1] > df['SPY'].iloc[-20] and chg < -1.0:
                    trend_signal = "🔴 严重背离 (大票涨,小票跌)" 
                
                st.write(f"📈 **RSP/SPY 相对强度 (20日):** {chg:+.2f}%  [{trend_signal}]")
        except: pass

    def fetch_and_calculate(self):
        print_h("核心数据获取与计算")
        
        ma50_pct, ma20_pct = self.calculate_spx_breadth_deep()
        
        print_step("获取指数数据...")
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
        
        # 爬虫数据
        pe = self.scraper.fetch_shiller_pe()
        sahm = self.scraper.fetch_sahm_rule()
        fg, fg_src = self.scraper.fetch_fear_greed()
        buffett = self.scraper.fetch_buffett_indicator()
        m_yoy, m_amt, m_ratio = self.scraper.fetch_margin_debt()
        lei_d, lei_dif = self.scraper.fetch_lei()
        pcr_avg, pcr_cur = self.scraper.fetch_pcr_robust()
        nfci = self.scraper.fetch_nfci()
        
        print_h("市场内部结构 (HO & MCO & Volume)")
        mco, nymo = self.scraper.fetch_dual_mco()
        ho_res = self.scraper.fetch_nyse_internals_robust()
        if ho_res: self.shared_wsj_data = ho_res
        tv_adv, tv_dec = self.scraper.fetch_tv_breadth_vision()

        # --- 深度 TRIN 分析 (用户想看的) ---
        if ho_res:
            def c(v): return float(str(v).replace(',','').replace('B','e9').replace('M','e6')) if v else 0
            adv = c(ho_res.get('adv')); dec = c(ho_res.get('dec'))
            adv_v = c(ho_res.get('adv_vol')); dec_v = c(ho_res.get('dec_vol'))
            
            if dec > 0 and dec_v > 0:
                trin_val = (adv / dec) / (adv_v / dec_v)
                st.markdown("---")
                st.markdown(f"**【TRIN 指标深度分析】** 读数: `{trin_val:.2f}`")
                
                status_desc = ""
                if trin_val < 0.5: status_desc = "🔴 极度强势/严重超买 (<0.5) -> 警惕顶部"
                elif 0.5 <= trin_val <= 0.8: status_desc = "🟢 强势/买方主导 (0.5-0.8) -> 健康上涨"
                elif 0.8 < trin_val <= 1.2: status_desc = "🟢 中性/平衡 (0.8-1.2) -> 观望/跟随"
                elif 1.2 < trin_val <= 2.0: status_desc = "🟡 弱势/卖压显现 (1.2-2.0) -> 谨慎减仓"
                elif trin_val > 2.0: status_desc = "🔴 极度恐慌/超卖 (>2.0) -> 抄底机会"
                st.write(f"状态判定: {status_desc}")
                
                if spx_trend_up:
                    if trin_val < 1.0: st.success("健康: SPX上涨 + TRIN<1.0 -> 买气充足")
                    elif trin_val > 1.2: st.warning("背离: SPX上涨 + TRIN>1.2 -> 价格涨但内部虚弱")
                st.markdown("---")

        # --- 指标判定逻辑 ---
        indicators = []
        
        # 1. Hindenburg Omen
        ho_stat = 0; ho_txt = "数据不足"
        if ho_res:
            h = c(ho_res.get('high')); l = c(ho_res.get('low'))
            total = adv + dec + c(ho_res.get('unch', 0))
            h_pct = (h/total)*100 if total else 0
            l_pct = (l/total)*100 if total else 0
            split = (h_pct > 2.2 and l_pct > 2.2)
            mco_bad = (mco < 0) if mco else (adv < dec)
            if spx_trend_up and split and mco_bad: ho_stat = 2
            elif split: ho_stat = 1
            ho_txt = f"新高:{h_pct:.1f}% | 新低:{l_pct:.1f}%"
        indicators.append(["Hindenburg Omen", ho_stat, ho_txt, "条件: 50MA上 & 新高低>2.2% & MCO<0"])

        # 2. Net Issues
        net_stat = 0; net_issues = adv - dec
        if net_issues < -2000: net_stat = 2
        elif net_issues < -1000: net_stat = 1
        indicators.append(["抛压 I: 广度", net_stat, f"{net_issues:.0f}", "<-1000 显著 | <-2000 恐慌"])

        # 3. TRIN (Status)
        trin_stat = 0
        if dec > 0 and dec_v > 0:
            if trin_val < 0.5: trin_stat = 2
            elif trin_val > 2.0: trin_stat = 1
        indicators.append(["抛压 II: 力度 (TRIN)", trin_stat, f"{trin_val:.2f}" if 'trin_val' in locals() else "N/A", "<0.5(极度超买) | >2.0(恐慌抄底)"])

        # 4. Volume Flow
        vol_stat = 0; vol_txt = "N/A"
        if adv_v > 0:
            ratio = dec_v / adv_v
            if ratio > 9.0: vol_stat = 2
            elif ratio > 4.0: vol_stat = 1
            vol_txt = f"Dn/Up: {ratio:.1f}"
        indicators.append(["抛压 III: 资金 (Vol)", vol_stat, vol_txt, "Dn/Up > 4.0 出逃 | > 9.0 洗盘"])

        # 5. NASDAQ Breadth
        tv_stat = 0
        if tv_adv and tv_dec:
            ratio = tv_adv / tv_dec
            if ratio < 0.5: tv_stat = 2
            indicators.append(["NASDAQ A/D", tv_stat, f"{ratio:.2f}", "<0.5 空头主导"])
        else: indicators.append(["NASDAQ A/D", 0, "N/A", ""])

        # 6. RSP vs SPY
        try:
            r = rsp/spy
            curr, ma = r.iloc[-1], r.rolling(50).mean().iloc[-1]
            chg = (curr/r.iloc[-20]-1)*100
            st_rsp = 2 if (curr<ma and chg<-2.0) else (1 if curr<ma else 0)
            indicators.append(["RSP/SPY 广度", st_rsp, f"20日变动: {chg:.1f}%", "跌破50MA & 急跌"])
        except: indicators.append(["RSP/SPY", 0, "Error", ""])
        
        # 7. NYA
        try:
            ok = nya.iloc[-1] > nya.rolling(50).mean().iloc[-1]
            st_nya = 2 if (spx_trend_up and not ok) else 0
            indicators.append(["NYA 参与度", st_nya, "弱" if not ok else "强", "SPX强但NYA弱"])
        except: pass

        # 8. 倒挂
        try:
            spr = tnx.iloc[-1] - irx.iloc[-1]
            indicators.append(["10Y-3M 倒挂", 2 if spr<0 else 0, f"{spr:.2f}%", "< 0%"])
        except: pass

        # 9-12 宏观
        indicators.append(["Shiller PE", 2 if pe and pe>30 else 0, f"{pe}", ">30 高估"])
        indicators.append(["巴菲特指标", 2 if buffett and buffett>140 else 0, f"{buffett:.1f}%", ">140%"])
        indicators.append(["Margin Debt", 1 if m_ratio and m_ratio>3.5 else 0, f"GDP比:{m_ratio:.1f}%", ">3.5%"])
        
        # 13 VIX
        try:
            v = vix.iloc[-1]
            chg = (v/vix.iloc[-15]-1)*100
            st_vix = 2 if (v>25 or chg>40) else 0
            indicators.append(["VIX", st_vix, f"{v:.1f} (+{chg:.0f}%)", ">25 或 飙升"])
        except: pass

        # 14 广度
        if ma50_pct:
            st_br = 2 if ma50_pct<40 else 0
            indicators.append(["SPX >50MA", st_br, f"{ma50_pct:.1f}%", "<40% 危险"])

        # 15 RSI
        try:
            delta = spx_weekly.diff()
            u = delta.clip(lower=0); d = -delta.clip(upper=0)
            rs = u.ewm(alpha=1/14).mean() / d.ewm(alpha=1/14).mean()
            rsi = 100 - 100/(1+rs)
            div = False
            if rsi.iloc[-1] < rsi.iloc[-5] and spx_weekly.iloc[-1] > spx_weekly.iloc[-5]: div = True
            indicators.append(["RSI 周线背离", 2 if div else 0, f"{rsi.iloc[-1]:.1f}", "价涨量缩"])
        except: pass

        # 16 Support
        try:
            sma20 = spx_weekly.rolling(20).mean().iloc[-1]
            ema21 = spx_weekly.ewm(span=21).mean().iloc[-1]
            status = 2 if spx.iloc[-1] < min(sma20, ema21) else 0
            indicators.append(["牛市支撑带", status, f"现价:{spx.iloc[-1]:.0f}", "跌破 20SMA/21EMA"])
        except: pass

        # 17-21 其他
        indicators.append(["Fear & Greed", 2 if fg and fg<45 else 0, f"{fg}", "<45"])
        try:
            e12 = spx_weekly.ewm(span=12).mean(); e26 = spx_weekly.ewm(span=26).mean()
            macd = e12 - e26; sig = macd.ewm(span=9).mean()
            dead = (macd.iloc[-2]>sig.iloc[-2]) and (macd.iloc[-1]<sig.iloc[-1]) and (macd.iloc[-1]>0)
            indicators.append(["MACD 周线死叉", 2 if dead else 0, "死叉" if dead else "正常", "零轴上方死叉"])
        except: pass
        indicators.append(["Sahm Rule", 2 if sahm and sahm>=0.5 else 0, f"{sahm}%", ">=0.5%"])
        indicators.append(["LEI", 2 if lei_d and lei_d<-4.0 else 0, f"{lei_d}%", "<-4.0%"])
        indicators.append(["PCR", 2 if pcr_avg and pcr_avg<0.8 else 0, f"{pcr_avg}", "<0.8"])
        indicators.append(["NFCI", 2 if nfci and nfci>-0.2 else 0, f"{nfci}", ">-0.2"])
        nymo_st = 2 if nymo and (nymo>60 or nymo<-60) else 0
        indicators.append(["NYMO", nymo_st, f"{nymo}", "极端值 +/-60"])

        return indicators

    def generate_table(self):
        print_h("21因子风险仪表盘")
        data = self.fetch_and_calculate()
        
        # 计算总分
        risk_score = sum(1 for d in data if d[1] == 2) + sum(0.5 for d in data if d[1] == 1)
        
        # 风险评级
        st.markdown(f"<div class='report-title'>风险评分: {risk_score:.1f} / 21.0</div>", unsafe_allow_html=True)
        if risk_score <= 5: st.success("🟢 市场结构健康，可保持观察")
        elif risk_score <= 10: st.warning("🟡 中期风险累积，建议谨慎")
        else: st.error("🔴 崩盘信号共振，建议立即减仓")
        
        # 构建 DataFrame
        df_display = []
        for row in data:
            name, stat, val, desc = row
            status_txt = "🔴 危险" if stat==2 else ("🟡 警告" if stat==1 else "🟢 安全")
            df_display.append({"监测指标": name, "状态": status_txt, "当前读数": val, "判断标准": desc})
        
        df = pd.DataFrame(df_display)
        
        # 使用 Streamlit 原生表格 (比图片清晰 100 倍)
        st.table(df)

# ==========================================
# 【板块轮动】
# ==========================================
class SectorRotationEngine:
    def __init__(self):
        self.sectors = {'XLK': '科技', 'XLF': '金融', 'XLV': '医疗', 'XLE': '能源', 'XLY': '可选', 
                       'XLP': '必选', 'XLI': '工业', 'XLC': '通讯', 'XLB': '材料', 'XLRE': '地产', 'SPY': '基准'}
        self.rs_window = 60 
        self.mom_window = 10 

    def run_analysis(self):
        print_h("板块轮动 RRG")
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
            if r>100 and m>100: q = "🟢 领涨"
            elif r<100 and m>100: q = "🔵 改善"
            elif r>100 and m<100: q = "🟡 转弱"
            else: q = "🔴 落后"
            res.append({"板块": self.sectors[t], "RS (趋势)": f"{r:.1f}", "Mom (动量)": f"{m:.1f}", "象限": q})
            
        st.dataframe(pd.DataFrame(res), use_container_width=True)

# ==========================================
# 【SMT 背离】
# ==========================================
class SMTDivergenceAnalyzer:
    def __init__(self):
        self.tickers = ['^IXIC', '^GSPC', 'QQQ', 'SPY', 'NQ=F', 'ES=F', 'RSP']

    def run(self):
        print_h("SMT 背离分析 (Pro V3)")
        df = get_cached_smt_data(self.tickers, "6mo")
        if df.empty: return
        close = df['Close'].ffill()
        
        st.markdown("**1. 期货先行指标 (NQ vs ES)**")
        w = close.iloc[-10:]
        h = w.max(); curr = w.iloc[-1]
        if 'NQ=F' in w and 'ES=F' in w:
            nq_h = curr['NQ=F'] >= h['NQ=F']*0.999
            es_h = curr['ES=F'] >= h['ES=F']*0.999
            if nq_h and not es_h: st.error("🔴 看跌背离: 纳指拉升，标普滞涨 (诱多)")
            elif not nq_h and es_h: st.error("🔴 看跌背离: 标普补涨，纳指动能衰竭")
            elif not nq_h and not es_h: st.info("⚪ 正常调整")
            else: st.success("🟢 步调一致")

        st.markdown("**2. 内部广度 (SPY vs RSP)**")
        if 'SPY' in w and 'RSP' in w:
            spy_p = (curr['SPY']/w.iloc[0]['SPY']-1)*100
            rsp_p = (curr['RSP']/w.iloc[0]['RSP']-1)*100
            if spy_p > rsp_p and spy_p > 0 and rsp_p < 0:
                st.error(f"🔴 虚假繁荣: SPY涨({spy_p:.1f}%) RSP跌({rsp_p:.1f}%)")
            else:
                st.success(f"🟢 广度健康: RSP({rsp_p:.1f}%) vs SPY({spy_p:.1f}%)")

# ==========================================
# 【主程序】
# ==========================================
if __name__ == "__main__":
    st.sidebar.title("操作台")
    if st.sidebar.button("🔄 刷新报告"):
        st.cache_data.clear()
        st.rerun()
        
    st.title("🚀 美股崩盘预警研报")
    st.markdown(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    app = CrashWarningSystem()
    
    # 1. 深度宏观
    app.analyze_market_trends_console()
    
    # 2. 21因子大表 (核心)
    app.generate_table()
    
    # 3. 补充模型
    sr = SectorRotationEngine()
    sr.run_analysis()
    
    smt = SMTDivergenceAnalyzer()
    smt.run()
    
    st.balloons()
    st.success("✅ 所有分析任务执行完毕")
    st.stop()
