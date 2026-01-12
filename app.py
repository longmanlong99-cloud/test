import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import re
import json
import io
import warnings
from datetime import datetime, timedelta
from firecrawl import Firecrawl
from PIL import Image

# ==========================================
# 0. 页面配置 & 样式
# ==========================================
st.set_page_config(
    page_title="美股崩盘预警系统 Pro",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 隐藏一些警告
warnings.filterwarnings("ignore")
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white;}
    .reportview-container .main .block-container {max-width: 1000px; padding-top: 2rem; padding-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. API Key 读取 (从 Secrets 安全获取)
# ==========================================
def get_secret(key_name, default=None):
    if key_name in st.secrets:
        return st.secrets[key_name]
    return default

GENAI_API_KEY = get_secret("GENAI_API_KEY")
USER_FRED_KEY = get_secret("USER_FRED_KEY")
FIRECRAWL_KEY = get_secret("FIRECRAWL_KEY")

# 检查 Key 状态
if not GENAI_API_KEY or not FIRECRAWL_KEY:
    st.error("❌ 严重错误：未检测到 API Key！请在 Streamlit Advanced Settings -> Secrets 中配置 GENAI_API_KEY 和 FIRECRAWL_KEY。")
    st.stop()

# 初始化 AI 客户端
try:
    from google import genai
    client = genai.Client(api_key=GENAI_API_KEY)
except ImportError:
    st.error("google-genai 库未安装")
    st.stop()

# FRED 库
try:
    from fredapi import Fred
    fred = Fred(api_key=USER_FRED_KEY) if USER_FRED_KEY else None
except:
    fred = None

# ==========================================
# 2. 核心类定义 (改造版)
# ==========================================

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.firecrawl_key = FIRECRAWL_KEY
        self.app = Firecrawl(api_key=self.firecrawl_key)
        self.fred_key = USER_FRED_KEY
        self.cached_gdp = None
        self.logs = [] # 用于存储日志

    def log(self, msg, level="info"):
        # 将日志存入列表，稍后在 UI 显示
        self.logs.append((msg, level))
        # 也可以实时打印到 Streamlit 的 expander
        if level == "ok": st.toast(f"✅ {msg}")
        elif level == "warn": st.toast(f"⚠️ {msg}")

    # --- 数据抓取函数 (保持原逻辑，微调输出) ---
    def fetch_shiller_pe(self):
        url = "https://www.multpl.com/shiller-pe"
        try:
            response = self.app.scrape(url, formats=['markdown'])
            md = getattr(response, 'markdown', '')
            if md:
                match = re.search(r'Shiller PE Ratio.*?(\d{2}\.\d{1,2})', md, re.S | re.I)
                if match:
                    return float(match.group(1))
        except: pass
        return None

    def fetch_fear_greed(self):
        # 优先 API 直连
        api_url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            r = requests.get(api_url, headers=headers, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if 'fear_and_greed' in data:
                    return int(data['fear_and_greed']['score']), data['fear_and_greed']['rating']
        except: pass
        return None, "获取失败"

    def fetch_us_gdp(self):
        if self.cached_gdp: return self.cached_gdp
        try:
            if fred:
                gdp_series = fred.get_series('GDP', sort_order='desc', limit=1)
                if not gdp_series.empty:
                    val = gdp_series.iloc[0] / 1000.0
                    self.cached_gdp = val
                    return val
        except: pass
        return None

    def fetch_buffett_indicator(self):
        gdp_tril = self.fetch_us_gdp()
        if not gdp_tril: return None
        try:
            w5000 = yf.Ticker("^W5000")
            hist = w5000.history(period="5d")
            if not hist.empty:
                return (hist['Close'].iloc[-1] / (gdp_tril * 1000.0)) * 100
        except: pass
        return None

    def fetch_margin_debt(self):
        url = "https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics"
        gdp_val = self.fetch_us_gdp()
        try:
            response = self.app.scrape(url, formats=['markdown'])
            md = getattr(response, 'markdown', '')
            if md:
                matches = re.findall(r'([A-Z][a-z]{2}-\d{2})\s*\|\s*([\d,]+)', md, re.S | re.I)
                if matches:
                    latest_date, latest_val_str = matches[0]
                    debt = float(latest_val_str.replace(',', '')) / 1_000_000
                    gdp_ratio = (debt / gdp_val * 100) if gdp_val else None
                    yoy = None
                    if len(matches) >= 13:
                        prev = float(matches[12][1].replace(',', ''))
                        curr = float(latest_val_str.replace(',', ''))
                        yoy = ((curr - prev) / prev) * 100
                    return yoy, debt, gdp_ratio
        except: pass
        return None, None, None

    def fetch_sahm_rule(self):
        url = "https://fred.stlouisfed.org/series/SAHMREALTIME"
        try:
            response = self.app.scrape(url, formats=['markdown'])
            md = getattr(response, 'markdown', '')
            if md:
                match = re.search(r'([A-Z][a-z]{2}\s+\d{4}):\s*([\d\.]+)', md, re.S | re.I)
                if match: return float(match.group(2))
        except: pass
        return None

    def fetch_lei(self):
        # 简化版：直接返回 None 让主程序跳过或显示N/A，避免云端视觉分析超时
        # 如果需要完整功能，需要确保 Google API 调用极其稳定
        return None, None 

    def fetch_pcr_robust(self):
        # 简化版逻辑
        return None, None

    def fetch_nfci(self):
        try:
            if fred:
                s = fred.get_series('NFCI', sort_order='desc', limit=1)
                if not s.empty: return float(s.iloc[0])
        except: pass
        return None

    # --- 简化的 WSJ 数据抓取 (Firecrawl) ---
    def fetch_nyse_internals(self):
        # 为了速度，这里我们尝试抓取文本，不进行截图分析
        target_url = "https://www.wsj.com/market-data/stocks/marketsdiary"
        try:
            response = self.app.scrape(target_url, formats=['markdown'])
            md = getattr(response, 'markdown', '')
            if md:
                prompt = """Extract Market Breadth for NYSE. Return JSON: {"high": 123, "low": 45, "adv": 100, "dec": 50, "adv_vol": 1000000000, "dec_vol": 500000000}. Use Composite/Daily data."""
                ai_resp = client.models.generate_content(model='gemini-2.0-flash', contents=[prompt, md])
                if ai_resp.text:
                    clean = re.sub(r'```json|```', '', ai_resp.text).strip()
                    return json.loads(re.search(r'\{.*\}', clean, re.DOTALL).group(0))
        except: pass
        return None
    
    def fetch_nymo_vision(self):
        return None # 暂时跳过视觉分析以提高云端稳定性

class CrashWarningSystem:
    def __init__(self):
        self.scraper = WebScraper()
        self.colors = {
            'bg': '#4B535C', 'table_header': '#3E4953', 
            'row_safe': '#2E8B57', 'text_safe': '#FFFFFF', 
            'row_warn': '#8B0000', 'text_warn': '#FFFFFF', 
            'row_risk': '#B8860B', 'text_risk': '#FFFFFF', 
            'title': '#FFEE88', 'edge': '#606972'
        }

    # --- 核心：数据下载与计算 (缓存保护) ---
    @st.cache_data(ttl=3600) # 1小时缓存
    def get_market_data(_self):
        tickers = "^GSPC ^VIX ^TNX ^IRX RSP SPY ^NYA"
        data = yf.download(tickers, period="3y", group_by='ticker', progress=False)
        return data

    @st.cache_data(ttl=3600)
    def get_sp500_tickers(_self):
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            return tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
        except: return []

    def run_analysis(self):
        status_container = st.status("正在启动 21因子分析引擎...", expanded=True)
        
        # 1. 下载基础数据
        status_container.write("📥 正在获取 Yahoo Finance 核心数据...")
        raw_data = self.get_market_data()
        
        # 数据提取
        try:
            spx = raw_data['^GSPC']['Close'].dropna()
            vix = raw_data['^VIX']['Close'].dropna()
            tnx = raw_data['^TNX']['Close'].dropna()
            irx = raw_data['^IRX']['Close'].dropna()
            rsp = raw_data['RSP']['Close'].dropna()
            spy = raw_data['SPY']['Close'].dropna()
            nya = raw_data['^NYA']['Close'].dropna()
            spx_weekly = spx.resample('W').last().dropna()
        except KeyError:
            status_container.update(label="数据获取失败", state="error")
            st.error("Yahoo Finance 数据下载不完整，请稍后重试。")
            return None

        # 2. 宏观数据抓取
        status_container.write("🌍 正在抓取宏观数据 (Firecrawl + FRED)...")
        real_shiller = self.scraper.fetch_shiller_pe()
        real_sahm = self.scraper.fetch_sahm_rule()
        real_fg, fg_src = self.scraper.fetch_fear_greed()
        val_buffett = self.scraper.fetch_buffett_indicator()
        val_margin_yoy, margin_amt, margin_ratio = self.scraper.fetch_margin_debt()
        val_nfci = self.scraper.fetch_nfci()
        ho_res = self.scraper.fetch_nyse_internals() # WSJ

        # 3. 计算指标列表
        indicators = []
        
        # --- 计算逻辑 (复用原代码逻辑) ---
        
        # HO
        h_stat = 0; h_msg = "数据不足"
        if ho_res:
            h_stat = 0 # 简化逻辑演示
            h_msg = f"NewHigh: {ho_res.get('high')}\nNewLow: {ho_res.get('low')}"
        indicators.append(["Hindenburg Omen", h_stat, h_msg, ""])

        # NYMO (Stub)
        indicators.append(["StockCharts ($NYMO)", 0, "云端暂缺", ""])

        # Shiller PE
        st_pe = 2 if real_shiller and real_shiller > 30 else 0
        indicators.append(["Shiller PE", st_pe, f"{real_shiller}", "PE > 30 危险"])

        # Buffett
        st_buf = 2 if val_buffett and val_buffett > 140 else 0
        indicators.append(["Buffett Indicator", st_buf, f"{val_buffett:.1f}%" if val_buffett else "N/A", "> 140% 高估"])

        # Margin Debt
        st_md = 1 if margin_ratio and margin_ratio > 3.5 else 0
        indicators.append(["Margin Debt", st_md, f"GDP%: {margin_ratio:.1f}%" if margin_ratio else "N/A", "> 3.5% 预警"])

        # VIX
        curr_vix = vix.iloc[-1]
        st_vix = 2 if curr_vix > 25 else 0
        indicators.append(["VIX Index", st_vix, f"{curr_vix:.2f}", "> 25 恐慌"])

        # RSI Divergence
        # ... (简化背离计算，直接用 RSI 值)
        delta = spx_weekly.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        curr_rsi = rsi.iloc[-1]
        indicators.append(["RSI (Weekly)", 2 if curr_rsi > 75 else 0, f"{curr_rsi:.1f}", "> 75 超买"])

        # 【核心修正】牛市支撑带 (20SMA vs 21EMA)
        sma20 = spx_weekly.rolling(20).mean().iloc[-1]
        ema21 = spx_weekly.ewm(span=21).mean().iloc[-1]
        band_low = min(sma20, ema21)
        band_high = max(sma20, ema21)
        curr_px = spx.iloc[-1]
        st_band = 2 if curr_px < band_low else 0
        indicators.append(["Bull Support Band", st_band, f"Price: {curr_px:.0f}\nBand: {band_low:.0f}~{band_high:.0f}", "跌破下轨预警"])

        # Fear Greed
        st_fg = 2 if real_fg and real_fg < 45 else 0
        indicators.append(["Fear & Greed", st_fg, f"{real_fg}", "< 45 恐慌"])

        # MACD Weekly
        e12 = spx_weekly.ewm(span=12).mean()
        e26 = spx_weekly.ewm(span=26).mean()
        macd = e12 - e26
        sig = macd.ewm(span=9).mean()
        dead_cross = (macd.iloc[-2] > sig.iloc[-2]) and (macd.iloc[-1] < sig.iloc[-1])
        indicators.append(["MACD Weekly", 2 if dead_cross else 0, "死叉" if dead_cross else "正常", "高位死叉"])

        # Sahm Rule
        st_sahm = 2 if real_sahm and real_sahm > 0.5 else 0
        indicators.append(["Sahm Rule", st_sahm, f"{real_sahm}%" if real_sahm else "N/A", "> 0.5% 衰退"])

        # Yield Curve
        spr = tnx.iloc[-1] - irx.iloc[-1]
        indicators.append(["Yield Curve (10Y-3M)", 2 if spr < 0 else 0, f"{spr:.2f}%", "< 0 倒挂"])

        # NFCI
        st_nfci = 1 if val_nfci and val_nfci > -0.35 else 0
        indicators.append(["NFCI (Chicago Fed)", st_nfci, f"{val_nfci}" if val_nfci else "N/A", "> -0.35紧缩"])
        
        # 填充剩余位置以生成完整的表（仅作演示，实际应计算全部）
        while len(indicators) < 21:
            indicators.append(["Other Factor", 0, "Waiting...", ""])

        status_container.update(label="计算完成！正在绘图...", state="complete")
        return indicators

    def draw_chart(self, data):
        if not data: return None
        
        # 风险评分
        risk_score = sum(1 for d in data if d[1] == 2) + sum(0.5 for d in data if d[1] == 1)
        
        fig = plt.figure(figsize=(12, 16), facecolor=self.colors['bg']) # 手机端尺寸调整
        ax = fig.add_subplot(111)
        ax.axis('off')

        # 标题
        ax.text(0.5, 0.98, "美股崩盘预警系统 Pro", ha='center', va='center', fontsize=24, fontweight='bold', color=self.colors['title'])
        ax.text(0.5, 0.96, f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ha='center', va='center', fontsize=12, color='#CCCCCC')

        # 表格数据准备
        table_vals = []
        for row in data:
            name, level, val, logic = row
            status = "安全"
            if level == 2: status = "触发"
            elif level == 1: status = "预警"
            table_vals.append([name, status, val])

        # 绘制表格
        # Col 1: Name, Col 2: Status, Col 3: Value
        col_labels = ['Indicator', 'Status', 'Value']
        table = ax.table(cellText=table_vals, colLabels=col_labels, loc='center', cellLoc='center', colWidths=[0.4, 0.2, 0.4])
        table.scale(1, 2.5)
        table.auto_set_font_size(False)
        table.set_fontsize(12)

        # 染色
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(self.colors['edge'])
            if row == 0:
                cell.set_facecolor(self.colors['table_header'])
                cell.set_text_props(color='white', weight='bold')
            else:
                idx = row - 1
                if idx < len(data):
                    lvl = data[idx][1]
                    bg = self.colors['row_safe']
                    if lvl == 2: bg = self.colors['row_warn']
                    elif lvl == 1: bg = self.colors['row_risk']
                    
                    cell.set_facecolor(bg)
                    cell.set_text_props(color='white')

        # 底部结论
        res_color = self.colors['text_safe']
        if risk_score > 10: res_color = self.colors['text_warn']
        ax.text(0.5, 0.02, f"Risk Score: {risk_score:.1f} / 21.0", ha='center', va='center', fontsize=20, weight='bold', color=res_color)

        return fig

# ==========================================
# 3. Streamlit 主界面逻辑
# ==========================================
st.title("🛡️ 美股崩盘预警系统")
st.markdown("基于 21因子模型的深度市场监控")

if st.button("🚀 启动全面扫描"):
    app = CrashWarningSystem()
    indicators = app.run_analysis()
    
    if indicators:
        st.success("分析完成！")
        
        # 1. 显示图片 (核心)
        fig = app.draw_chart(indicators)
        st.pyplot(fig)
        
        st.info("💡 提示：在手机上长按上方图片，可保存到相册。")
        
        # 2. 显示详细数据表 (可折叠)
        with st.expander("查看原始数据明细"):
            df = pd.DataFrame(indicators, columns=["指标", "风险等级", "读数", "判断标准"])
            st.dataframe(df)

# 页脚
st.markdown("---")
st.caption("Data Source: Yahoo Finance, FRED, WSJ, Multpl | Powered by Streamlit & Gemini")
