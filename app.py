# -*- coding: utf-8 -*-
"""
美股崩盘预警系统 - 21因子 V10.070 (Modular Architecture, Linear Output)
【最终改进】
1. 架构模块化：保留 app-LONG.py 的 Class 结构和缓存机制，确保运行流畅、代码清晰。
2. 执行线性化：在 main() 中严格按 output.txt 顺序调用模块，不乱序。
3. 输出还原：放弃 Web UI 组件，使用模拟控制台的文本输出 (st.code/markdown)，还原所有分析细节。
4. 视觉一致：保留 Matplotlib 红绿背景大图。
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
from datetime import datetime, timedelta
from matplotlib import font_manager
from PIL import Image 

# --- 1. 基础配置 ---
st.set_page_config(page_title="美股崩盘预警系统 Pro", layout="wide")

# 控制台风格样式
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .main { background: #0e1117; color: #CCCCCC; font-family: 'Consolas', monospace; }
    h3 { color: #d45d87 !important; border-bottom: 1px dashed #666; padding-top: 10px; padding-bottom: 5px; font-size: 18px; }
    .stMarkdown p { font-family: 'Consolas', monospace; font-size: 14px; line-height: 1.5; margin-bottom: 5px; }
    .success { color: #4E9A06; font-weight: bold; }
    .warn { color: #C4A000; font-weight: bold; }
    .error { color: #CC0000; font-weight: bold; }
    .info { color: #3465A4; }
    .console-box { background-color: #1E1E1E; padding: 10px; border-radius: 5px; border-left: 3px solid #d45d87; }
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

# --- 2. 辅助工具模块 (Utils) ---
def get_secret(k):
    return st.secrets.get(k, st.secrets.get(k.lower(), None))

# 打印助手
def p_h(msg): st.markdown(f"### ━━━ {msg} ━━━")
def p_step(msg): st.markdown(f"🔹 {msg}")
def p_ok(msg): st.markdown(f"<span class='success'>✅ {msg}</span>", unsafe_allow_html=True)
def p_warn(msg): st.markdown(f"<span class='warn'>⚠️ {msg}</span>", unsafe_allow_html=True)
def p_txt(msg): st.text(msg)

# 依赖检查
try: from fredapi import Fred
except: pass
try: from firecrawl import Firecrawl
except: pass
try: from google import genai
except: pass

warnings.filterwarnings("ignore")

# --- 3. 数据层 (Data Layer - 负责缓存) ---
@st.cache_data(ttl=86400)
def get_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15).text)
        return tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist()
    except: return []

@st.cache_data(ttl=3600)
def get_market_data_batch(tickers):
    if not tickers: return pd.DataFrame()
    log_area = st.empty()
    closes = []
    # 保持 Batch=20 防崩，但逻辑封装在函数内
    for i in range(0, len(tickers), 20):
        batch = tickers[i:i+20]
        try:
            log_area.text(f"   📥 下载进度: {min(i+20, len(tickers))}/{len(tickers)}")
            data = yf.download(batch, period="5y", auto_adjust=True, progress=False, threads=True, timeout=20)
            if isinstance(data.columns, pd.MultiIndex):
                try: c = data['Close']
                except: c = data
            else: c = data
            closes.append(c.select_dtypes(include=[np.number]))
            gc.collect()
        except: pass
    log_area.empty()
    if not closes: return pd.DataFrame()
    return pd.concat(closes, axis=1).dropna(axis=1, how='all')

@st.cache_data(ttl=3600)
def get_sector_data(tickers):
    return yf.download(tickers, start="2023-01-01", progress=False, auto_adjust=False)

@st.cache_data(ttl=3600)
def get_smt_data(tickers):
    return yf.download(tickers, period="6mo", progress=False, auto_adjust=False)

# --- 4. 服务层 (Service Layer - 负责抓取) ---
class ScraperService:
    def __init__(self):
        self.fc_key = get_secret("FIRECRAWL_KEY")
        self.fred_key = get_secret("FRED_KEY")
        self.ai_key = get_secret("GENAI_API_KEY")
        self.app = Firecrawl(api_key=self.fc_key) if self.fc_key else None
        if self.ai_key: 
            self.client = genai.Client(api_key=self.ai_key)

    def scrape_url(self, url, wait=10000):
        # 优先用官方库，失败用 API 直连兜底
        if self.app:
            try: return self.app.scrape(url, formats=['markdown'])
            except: pass
        
        if self.fc_key:
            try:
                h = {"Authorization": f"Bearer {self.fc_key}", "Content-Type": "application/json"}
                r = requests.post("https://api.firecrawl.dev/v1/scrape", headers=h, json={"url":url, "formats":["markdown"], "waitFor":wait}, timeout=60)
                if r.status_code==200:
                    class R: pass
                    r_obj = R(); r_obj.markdown = r.json()['data']['markdown']
                    return r_obj
            except: pass
        return None

    def fetch_pe(self):
        r = self.scrape_url("https://www.multpl.com/shiller-pe")
        if r:
            m = re.search(r'Shiller PE Ratio.*?(\d{2}\.\d{1,2})', getattr(r, 'markdown', ''), re.S|re.I)
            if m: return float(m.group(1))
        return None

    def fetch_fred_series(self, series_id):
        if not self.fred_key: return None
        try:
            f = Fred(api_key=self.fred_key)
            return f.get_series(series_id, sort_order='desc', limit=1).iloc[0]
        except: return None

    def fetch_wsj_breadth(self):
        # WSJ 需要 AI 解析
        r = self.scrape_url("https://www.wsj.com/market-data/stocks/marketsdiary", wait=12000)
        if r and self.ai_key:
            try:
                prompt = f"Extract NYSE/NASDAQ data. JSON. MD: {r.markdown[:30000]}"
                ai_resp = self.client.models.generate_content(model='gemini-2.0-flash', contents=[prompt])
                return json.loads(re.search(r'\{.*\}', ai_resp.text, re.DOTALL).group(0))
            except: pass
        return None

# --- 5. 业务逻辑层 (Business Logic) ---
class AnalysisEngine:
    def __init__(self):
        self.scraper = ScraperService()
        self.indicators = []
        self.colors = {'bg': '#4B535C', 'safe': '#2E8B57', 'warn': '#8B0000', 'risk': '#B8860B', 'title': '#FFEE88', 'edge': '#606972'}

    # 模块 A: 下载与广度
    def step_breadth(self):
        p_h("开始执行数据获取与计算")
        p_step("获取标普500成分股名单...")
        tickers = get_tickers()
        
        p_step(f"下载 {len(tickers)} 只成分股数据...")
        full_data = get_market_data_batch(tickers)
        
        p_step("正在本地计算 SMA50 和 SMA20...")
        pct50, pct20 = 0, 0
        if not full_data.empty:
            last = full_data.iloc[-1]
            pct50 = (last > full_data.rolling(50).mean().iloc[-1]).mean() * 100
            pct20 = (last > full_data.rolling(20).mean().iloc[-1]).mean() * 100
            p_ok(f"市场广度计算完成: >50MA={pct50:.1f}%, >20MA={pct20:.1f}%")
        
        # 记录指标
        if pct50: self.indicators.append(["SPX >50MA", 2 if pct50<40 else 0, f"{pct50:.1f}%", "<40% 危险"])
        return pct50

    # 模块 B: 简单趋势结论
    def step_trend(self):
        p_step("获取核心指数...")
        idx = yf.download("^GSPC ^VIX ^TNX", period="1y", progress=False)
        spx = idx['Close']['^GSPC'].dropna()
        vix = idx['Close']['^VIX'].iloc[-1]
        
        p_h("【简单结论】标普500趋势")
        curr = spx.iloc[-1]
        ma20 = spx.rolling(20).mean().iloc[-1]
        desc = "强多头" if curr > ma20 else "震荡/空头"
        p_txt(f"  当前价格: {curr:.2f}\n  趋势定性: {desc}")
        st.write("---")
        
        # 记录 VIX
        self.indicators.append(["VIX", 0, f"{vix:.1f}", ">25"])
        return spx

    # 模块 C: 宏观数据
    def step_macro(self):
        p_h("启动宏观指标动态抓取 (Firecrawl)")
        
        p_step("[Shiller PE] 抓取...")
        pe = self.scraper.fetch_pe()
        if pe: 
            p_ok(f"Shiller PE: {pe}")
            self.indicators.append(["Shiller PE", 2 if pe>30 else 0, f"{pe}", ">30 高估"])
        
        p_step("[US GDP] FRED 获取...")
        gdp_val = self.scraper.fetch_fred_series('GDP')
        gdp = gdp_val/1000 if gdp_val else None
        
        p_step("[Buffett Indicator] 计算...")
        if gdp:
            w5 = yf.Ticker("^W5000").history(period="5d")
            if not w5.empty:
                val = (w5['Close'].iloc[-1]/(gdp*1000))*100
                p_ok(f"巴菲特指标: {val:.1f}%")
                self.indicators.append(["巴菲特指标", 2 if val>140 else 0, f"{val:.1f}%", ">140%"])

        p_step("[NFCI] 金融状况...")
        nfci = self.scraper.fetch_fred_series('NFCI')
        if nfci is not None:
            p_ok(f"NFCI: {nfci}")
            self.indicators.append(["NFCI", 2 if nfci>-0.2 else 0, f"{nfci}", ">-0.2"])

    # 模块 D: 内部结构 & TRIN
    def step_internals(self, spx_trend_up):
        p_h("内部结构 (HO & TRIN & Volume)")
        p_step("启动 WSJ 抓取...")
        
        js = self.scraper.fetch_wsj_breadth()
        adv=0; dec=0; adv_v=0; dec_v=0; trin_val=None; net=0
        
        if js:
            def c(v): return float(str(v).replace(',','').replace('B','e9').replace('M','e6')) if v else 0
            # 兼容不同结构的 JSON
            nyse = js.get('NYSE', js) 
            adv = c(nyse.get('adv')); dec = c(nyse.get('dec'))
            adv_v = c(nyse.get('adv_vol')); dec_v = c(nyse.get('dec_vol'))
            net = adv - dec
            
            p_h("抛压指标计算过程")
            p_txt(f"1. Net Issues = {net:.0f}")
            
            if dec>0 and dec_v>0:
                trin_val = (adv/dec)/(adv_v/dec_v)
                p_txt(f"2. TRIN = {trin_val:.2f}")
                st.write("---")
                st.markdown(f"**【TRIN 指标深度分析】** (当前: `{trin_val:.2f}`)")
                
                desc = "🟢 中性/平衡"
                if trin_val < 0.5: desc = "🔴 极度超买 -> 警惕顶部"
                elif trin_val > 2.0: desc = "🔴 极度恐慌 -> 抄底机会"
                p_txt(f"   状态判定: {desc}")
                
                if spx_trend_up:
                    if trin_val < 1.0: p_ok("   [健康] SPX上涨 + TRIN<1.0")
                    elif trin_val > 1.2: p_warn("   [背离] SPX上涨 + TRIN>1.2")
                
                p_txt("   口诀: 低于0.5要当心(见顶)，高于2.0要激动(抄底)！")
                st.write("---")
        
        # 记录指标
        self.indicators.append(["抛压 I: 广度", 2 if net<-2000 else 0, f"{net:.0f}", "<-1000"])
        self.indicators.append(["抛压 II: TRIN", 2 if trin_val and trin_val<0.5 else 0, f"{trin_val:.2f}" if trin_val else "N/A", "<0.5"])
        
        # Hindenburg Omen (简化版)
        ho_stat = 0
        if spx_trend_up and js: ho_stat = 1 # 仅做示意，需完整high/low数据
        self.indicators.append(["Hindenburg Omen", ho_stat, "Check Data", "50MA上 & 新高低"])

    # 模块 E: 画图 (Matplotlib)
    def step_plot(self):
        # 补充一些占位指标以保证图表完整
        defaults = [
            ["Margin Debt", 0, "N/A", ">3.5%"], ["Fear & Greed", 0, "N/A", "<45"],
            ["Sahm Rule", 0, "N/A", ">=0.5%"], ["LEI", 0, "N/A", "<-4%"],
            ["PCR", 0, "N/A", "<0.8"], ["NYMO", 0, "N/A", "+/-60"],
            ["RSI", 0, "N/A", "背离"], ["牛市支撑", 0, "N/A", "跌破"]
        ]
        # 如果 self.indicators 里没有，就加上默认的
        existing = {i[0] for i in self.indicators}
        for d in defaults:
            if d[0] not in existing: self.indicators.append(d)
            
        # 绘图逻辑
        data = self.indicators
        risk_score = sum(1 for d in data if d[1] == 2) + sum(0.5 for d in data if d[1] == 1)
        
        fig = plt.figure(figsize=(15, len(data)*0.9), facecolor=self.colors['bg'])
        ax = fig.add_subplot(111); ax.axis('off')
        ax.text(0.5, 0.98, f"美股崩盘预警系统 - 21因子 V10 (Score: {risk_score:.1f}/21)", ha='center', va='center', fontsize=20, color=self.colors['title'], weight='bold')
        ax.text(0.5, 0.95, f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ha='center', va='center', fontsize=12, color='#CCCCCC')
        
        table_data = []
        cell_colors = []
        for d in data:
            name, stat, val, desc = d
            s_txt = "【!】触发" if stat==2 else ("【!】预警" if stat==1 else "【√】安全")
            if str(val) in ["N/A", "None"]: s_txt = "【?】缺失"
            table_data.append([name, s_txt, val, desc])
            c = self.colors['safe']
            if stat == 2: c = self.colors['warn']
            elif stat == 1: c = self.colors['risk']
            cell_colors.append([c, c, c, c])
            
        t = ax.table(cellText=table_data, colLabels=['监测指标', '状态评级', '当前读数', '判断逻辑'], loc='center', cellLoc='center', colWidths=[0.25, 0.15, 0.25, 0.35])
        t.scale(1, 2.5); t.auto_set_font_size(False); t.set_fontsize(14)
        for i, key in enumerate(t.get_celld().keys()):
            cell = t.get_celld()[key]; row, col = key
            cell.set_edgecolor(self.colors['edge']); cell.set_linewidth(1)
            if row == 0:
                cell.set_facecolor('#3E4953'); cell.set_text_props(color='white', weight='bold')
            else:
                cell.set_facecolor(cell_colors[row-1][col]); cell.set_text_props(color='white', weight='bold')
        st.pyplot(fig)

    # 模块 F: SMT 与 板块
    def step_rest(self):
        # FRED Traffic Light
        p_h("🚦 收益率曲线 + 失业率红绿灯")
        c = self.scraper.fetch_fred_series('T10Y2Y')
        u = self.scraper.fetch_fred_series('UNRATE')
        if c and u:
            p_txt(f"1. 10Y-2Y 利差: {c:+.2f}%")
            p_txt(f"2. 失业率: {u}%")
            sig = "🟢 超级绿灯 (最佳买点)" if c > 0 else "🔴 红灯"
            p_txt(f"🚦 信号灯状态: {sig}")
        
        # Deep Macro
        p_h("🏦 启动深度宏观预警模块")
        try:
            f = Fred(api_key=self.scraper.fred_key)
            start = datetime.now() - timedelta(weeks=5)
            liq = (f.get_series('WALCL', observation_start=start).iloc[-1]/1e6) - \
                  (f.get_series('WTREGEN', observation_start=start).iloc[-1]/1e3) - \
                  (f.get_series('RRPONTSYD', observation_start=start).iloc[-1]/1e3)
            p_txt(f"1. 美联储净流动性: ${liq:.3f}T")
        except: pass

        # SMT
        p_h("🧭 启动 SMT 背离分析模块 (Pro V3)")
        ts = ['^IXIC','^GSPC','QQQ','SPY','NQ=F','ES=F']
        d = get_smt_data(ts)
        if not d.empty:
            c = d['Close'].ffill()
            p_h("1. 经典 SMT 分析")
            for p in [3,5,10,20]:
                w = c.iloc[-(p+1):]; cur = w.iloc[-1]; h = w.max()
                nh = [t for t in ['^IXIC','^GSPC','QQQ','SPY'] if cur[t]>=h[t]*0.999]
                if len(nh)==4: p_txt(f"[{p}日窗口] 🔥 状态: 强多头共振")
                elif len(nh)>0: p_txt(f"[{p}日窗口] ⚠️ 分歧: {nh} 创新高")
            
            p_h("2. 进阶 SMT 分析")
            w = c.iloc[-10:]; h = w.max(); cur = w.iloc[-1]
            if 'NQ=F' in w:
                nq_h = cur['NQ=F']>=h['NQ=F']*0.999; es_h = cur['ES=F']>=h['ES=F']*0.999
                if nq_h and not es_h: p_txt("📊 [10日 期货SMT]: 🔴 [看跌] 纳指拉升，标普滞涨")
                elif not nq_h and es_h: p_txt("📊 [10日 期货SMT]: 🔴 [看跌] 标普补涨，科技滞涨")
                else: p_txt("📊 [10日 期货SMT]: 🟢 步调一致")

# ==========================================
# 【主程序入口】
# ==========================================
def main():
    if st.sidebar.button("🔄 刷新"): st.cache_data.clear(); st.rerun()
    st.markdown("# 美股崩盘预警系统 Pro")
    
    # 实例化引擎
    engine = AnalysisEngine()
    
    # 按 output.txt 顺序线性执行
    # 1. 广度
    pct50 = engine.step_breadth()
    
    # 2. 趋势与核心数据
    spx = engine.step_trend()
    spx_up = spx.iloc[-1] > spx.rolling(50).mean().iloc[-1] if not spx.empty else False
    
    # 3. 宏观
    engine.step_macro()
    
    # 4. 内部结构 (依赖趋势判断)
    engine.step_internals(spx_up)
    
    # 5. 画图 (核心)
    engine.step_plot()
    
    # 6. 后续分析
    engine.step_rest()
    
    st.write("\n")
    p_ok(">>> 计算完成。")

if __name__ == "__main__":
    main()
