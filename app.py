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
import platform
import matplotlib.font_manager as fm
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
    initial_sidebar_state="expanded"
)

warnings.filterwarnings("ignore")

# 字体修复逻辑 (针对 Linux 云服务器)
def setup_fonts():
    system = platform.system()
    if system == "Linux":
        # 尝试使用文泉驿正黑 (通过 packages.txt 安装)
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
    else:
        # 本地 Windows/Mac
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

setup_fonts()

# ==========================================
# 1. 日志捕获系统 (把 Console 搬到网页)
# ==========================================
class StreamlitLogger:
    def __init__(self):
        self.logs = []
        
    def log(self, msg, color="black", header=False):
        # 存入内存，后续渲染
        self.logs.append({"msg": msg, "color": color, "header": header})
        
    def render(self):
        # 在网页上显示日志
        st.markdown("### 📝 深度分析日志 (Console Log)")
        log_container = st.container(height=400) # 可滚动的区域
        with log_container:
            for item in self.logs:
                if item['header']:
                    st.markdown(f"**{item['msg']}**")
                else:
                    # 简单的颜色处理
                    if "Trigger" in item['msg'] or "触发" in item['msg'] or "危险" in item['msg']:
                        st.markdown(f":red[{item['msg']}]")
                    elif "Safe" in item['msg'] or "安全" in item['msg'] or "健康" in item['msg']:
                        st.markdown(f":green[{item['msg']}]")
                    else:
                        st.text(item['msg'])

# 全局日志实例
logger = StreamlitLogger()

# ==========================================
# 2. API Key 读取
# ==========================================
def get_secret(key_name):
    if key_name in st.secrets:
        return st.secrets[key_name]
    return None

GENAI_API_KEY = get_secret("GENAI_API_KEY")
USER_FRED_KEY = get_secret("USER_FRED_KEY")
FIRECRAWL_KEY = get_secret("FIRECRAWL_KEY")

if not GENAI_API_KEY:
    st.error("请在 Streamlit Secrets 配置 API Key")
    st.stop()

# 初始化 AI
try:
    from google import genai
    client = genai.Client(api_key=GENAI_API_KEY)
except: pass

# ==========================================
# 3. 核心逻辑 (WebScraper + Calculation)
# ==========================================
class WebScraper:
    def __init__(self):
        self.firecrawl_key = FIRECRAWL_KEY
        self.app = Firecrawl(api_key=self.firecrawl_key) if FIRECRAWL_KEY else None
        
    def fetch_shiller_pe(self):
        logger.log("正在抓取 Shiller PE...", "gray")
        try:
            # 简化版：云端为了速度，可以用 yfinance 的某种估算，或者直接硬抓
            # 这里演示 Firecrawl
            if self.app:
                res = self.app.scrape("https://www.multpl.com/shiller-pe", formats=['markdown'])
                md = res.get('markdown', '')
                match = re.search(r'Shiller PE Ratio.*?(\d{2}\.\d{1,2})', md, re.S | re.I)
                if match: 
                    val = float(match.group(1))
                    logger.log(f"Shiller PE 获取成功: {val}")
                    return val
        except: pass
        return None

    # ... (其他抓取函数逻辑类似，为节省篇幅，这里复用核心逻辑) ...
    # 实际部署时，你可以把之前代码里那些 fetch 函数都搬进来
    # 这里为了演示修复乱码，我们先用核心数据
    
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

    @st.cache_data(ttl=3600)
    def get_data(_self):
        logger.log("正在从 Yahoo Finance 下载核心数据...", header=True)
        tickers = "^GSPC ^VIX ^TNX ^IRX RSP SPY"
        data = yf.download(tickers, period="2y", group_by='ticker', progress=False)
        return data

    def run(self):
        data = self.get_data()
        if data is None or data.empty:
            st.error("数据下载失败")
            return []

        spx = data['^GSPC']['Close'].dropna()
        spx_weekly = spx.resample('W').last().dropna()
        
        indicators = []
        
        # --- 1. 牛市支撑带 (修复版逻辑) ---
        sma20 = spx_weekly.rolling(20).mean().iloc[-1]
        ema21 = spx_weekly.ewm(span=21).mean().iloc[-1]
        band_low = min(sma20, ema21)
        band_high = max(sma20, ema21)
        curr = spx.iloc[-1]
        
        st_band = 2 if curr < band_low else 0
        msg = f"现价:{curr:.0f} | 区间:{band_low:.0f}~{band_high:.0f}"
        indicators.append(["牛市支撑带 (20SMA/21EMA)", st_band, msg, "跌破双线区间"])
        
        logger.log(f"牛市支撑带分析: {msg}", "black")
        if st_band == 2: logger.log("警告：价格跌破支撑带！", "red")
        else: logger.log("状态：支撑有效", "green")

        # --- 2. Shiller PE ---
        pe = self.scraper.fetch_shiller_pe()
        if pe:
            st_pe = 2 if pe > 30 else 0
            indicators.append(["Shiller PE", st_pe, f"{pe}", ">30 高估"])
        else:
            indicators.append(["Shiller PE", 0, "N/A", ""])
            
        # --- 3. 填充演示数据 (实际可把21因子全写上) ---
        # 为了展示字体修复效果
        indicators.append(["中文测试 (Test)", 0, "字体正常", "无乱码"])
        
        return indicators

    def draw(self, data):
        fig = plt.figure(figsize=(12, 8), facecolor=self.colors['bg'])
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # 标题 (测试中文)
        ax.text(0.5, 0.95, "美股崩盘预警系统 Pro", ha='center', fontsize=20, color='white', fontweight='bold')
        ax.text(0.5, 0.90, f"生成时间: {datetime.now().strftime('%Y-%m-%d')}", ha='center', color='#ddd')

        # 表格
        cell_text = []
        for row in data:
            name, lvl, val, logic = row
            status = "安全" if lvl == 0 else "触发"
            cell_text.append([name, status, val, logic])
            
        col_labels = ['监测指标', '状态', '当前读数', '判断标准']
        table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
        
        table.scale(1, 2)
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        
        # 简单染色
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(self.colors['table_header'])
                cell.set_text_props(color='white')
            else:
                cell.set_text_props(color='black') # 暂时用黑字测试
                
        return fig

# ==========================================
# 4. 主程序入口
# ==========================================
st.title("🛡️ 美股崩盘预警 Pro (Cloud版)")

if st.button("🚀 开始分析"):
    with st.spinner("正在连接全球数据源..."):
        app = CrashWarningSystem()
        results = app.run()
        
        # 1. 先画图
        fig = app.draw(results)
        st.pyplot(fig)
        
        # 2. 再显示日志 (这就是你的控制台内容！)
        logger.render()
