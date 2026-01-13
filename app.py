# -*- coding: utf-8 -*-
"""
美股崩盘预警系统 - 21因子 V10.051 (Secure Secrets Edition)
【安全与修复更新】
1. 安全升级：移除所有硬编码 API Key，改为从 Streamlit Secrets 读取。
   (GENAI_API_KEY, FRED_KEY, FIRECRAWL_KEY 均需在后台配置)
2. 缓存优化：保留 @st.cache_data 机制，防止 Yahoo Finance 限流。
3. Streamlit 适配：替换所有 print 为 st.write/st.header 等，添加 UI 组件和进度条，防止页面空白/伪死循环。
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
import webbrowser
import warnings
import time
import random
import re
import traceback 
import sys
import json 
import io
import subprocess 
from firecrawl import Firecrawl 
from PIL import Image 

# --- FRED API 库 ---
try:
    from fredapi import Fred
except ImportError:
    st.warning(">>> 提示：未找到 fredapi 库，建议运行 pip install fredapi 以启用完整功能。")

# --- Google Gemini AI 库 ---
try:
    from google import generativeai as genai  # 修正 import 为 generativeai
except ImportError:
    st.error(">>> 严重错误：未找到 google-generativeai 库。请运行 pip install google-generativeai")
    sys.exit(1)

# ==========================================
# 【API 配置区 - 安全版】
# ==========================================
# 自动从 Streamlit Secrets 读取 Key，防止上传 GitHub 后被封
# 请确保在 Streamlit Cloud -> App Settings -> Secrets 中配置了以下三个变量
try:
    GENAI_API_KEY = st.secrets["GENAI_API_KEY"]
    USER_FRED_KEY = st.secrets["FRED_KEY"]
    FIRECRAWL_KEY = st.secrets["FIRECRAWL_KEY"]
except FileNotFoundError:
    st.error("❌ 错误：未检测到 Secrets 配置！请在 Streamlit Cloud 后台 Settings -> Secrets 中填入 API Key。")
    st.stop()
except KeyError as k:
    st.error(f"❌ 错误：Secrets 中缺少键值 {k}。请检查配置文件名拼写是否正确。")
    st.stop()

# 初始化 Google AI
genai.configure(api_key=GENAI_API_KEY)  # 修正为 configure

warnings.filterwarnings("ignore")

# ==========================================
# 【UI 美化工具类】 - 改为 Streamlit 版本
# ==========================================
def st_header(msg): 
    st.header(f"━━━ {msg} ━━━")
def st_step(msg): 
    st.info(f"\U0001f539 {msg}")
def st_ok(msg): 
    st.success(f"\u2705 {msg}")
def st_warn(msg): 
    st.warning(f"\u26a0\ufe0f  {msg}")
def st_err(msg): 
    st.error(f"\u274c {msg}")
def st_info(msg): 
    st.info(f"\u2139\ufe0f  {msg}")

# ==========================================
# 【缓存加速层 (解决 Yahoo 限流问题)】
# ==========================================
@st.cache_data(ttl=86400) # 列表缓存 24 小时
def get_cached_tickers():
    """缓存获取标普500成分股名单"""
    st_step("获取标普500成分股名单 (Cached)...")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15).text)
        for t in tables:
            if 'Symbol' in t.columns:
                return t['Symbol'].str.replace('.', '-', regex=False).tolist()
    except: return []

@st.cache_data(ttl=3600) # 数据缓存 1 小时
def get_cached_sp500_data(tickers):
    """缓存下载 500 只股票数据"""
    if not tickers: return pd.DataFrame()
    st_step(f"下载 {len(tickers)} 只成分股数据 (5年) [Cached]...")
    st_info("提示: 首次运行需联网下载，后续 1 小时内将直接读取缓存...")
    
    closes = []
    progress_bar = st.progress(0)  # 添加进度条
    for i in range(0, len(tickers), 20):  # 减小 batch 防超时
        batch = tickers[i:i+20]
        try:
            data = yf.download(batch, period="5y", auto_adjust=True, progress=False, threads=True, timeout=30)
            if isinstance(data.columns, pd.MultiIndex):
                try: close = data['Close']
                except: close = data
            else: close = data
            closes.append(close)
            st.write(f"   进度: {min(i+20, len(tickers))}/{len(tickers)}")  # 显示进度
            time.sleep(2)  # 防限流
        except: pass
        progress_bar.progress(min((i + 20) / len(tickers), 1.0))
        
    if not closes: return pd.DataFrame()
    return pd.concat(closes, axis=1).dropna(axis=1, how='all')

@st.cache_data(ttl=3600) # 数据缓存 1 小时
def get_cached_sector_data(tickers, start_date):
    """缓存下载板块轮动数据"""
    st_step(f"下载 {len(tickers)} 个板块数据 ({start_date} ~ Now) [Cached]...")
    raw_data = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)
    return raw_data

@st.cache_data(ttl=3600) # 数据缓存 1 小时
def get_cached_smt_data(tickers, period):
    """缓存下载 SMT 分析数据"""
    st_step("下载全量数据 (含期货/等权ETF) [Cached]...")
    data = yf.download(tickers, period=period, auto_adjust=False, progress=False)
    return data

# ==========================================
# 【WebScraper: 纯净 Firecrawl 版】
# ==========================================
class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        # [安全修复] 直接使用从 secrets 读取到的变量，不再硬编码
        self.firecrawl_key = FIRECRAWL_KEY 
        self.app = Firecrawl(api_key=self.firecrawl_key)
        self.fred_key = USER_FRED_KEY
        self.cached_gdp = None 

    # --- 1. Shiller PE ---
    @st.cache_data(ttl=3600)
    def fetch_shiller_pe(self):
        st_step("[Shiller PE] 启动 Firecrawl 抓取 (Multpl)...")
        url = "https://www.multpl.com/shiller-pe"
        try:
            response = self.app.scrape(url, formats=['markdown'])
            md = getattr(response, 'markdown', '')
            if md:
                match = re.search(r'Shiller PE Ratio.*?(\d{2}\.\d{1,2})', md, re.S | re.I)
                if match:
                    val = float(match.group(1))
                    st_ok(f"AI 识别成功! Shiller PE: {val}")
                    return val
        except Exception as e:
            st_err(f"Shiller PE 抓取异常: {e}")
        return None

    # --- 2. Fear & Greed ---
    # ... (类似替换所有 print 为 st. 函数，添加缓存如果需要)

    # 其他方法类似替换 print 为 st.write/st.success 等
    # 由于代码长，这里省略，但原则是全局搜索替换 print_xxx 为 st_xxx

# ... (你的其他类定义，如 CrashWarningSystem, SectorRotationEngine, SMTDivergenceAnalyzer 等)
# 在每个类的方法中，替换 print 为 st. 版本，例如：
# def generate_chart(self):
#     st_step("开始执行数据获取与计算...")
#     # ... 原逻辑
#     fig, ax = plt.subplots()  # 如果有图
#     # ... 画图
#     st.pyplot(fig)  # 显示图表
#     st.image("你的生成png路径", caption="预警报表")  # 或显示生成的 PNG

# ==========================================
# 主程序 - Streamlit 版本
# ==========================================
st.title("美股崩盘预警系统 - 21因子 V10.051")
st.write("欢迎使用！系统将自动计算预警指标。请耐心等待数据加载（首次可能需 1-5 分钟）。")

progress_bar = st.progress(0)
status_text = st.empty()

if __name__ == "__main__":
    try:
        app = CrashWarningSystem()
        
        status_text.write("1. 核心图片与报告生成中...")
        app.generate_chart()  # 里面已替换为 st.
        progress_bar.progress(20)
        
        status_text.write("2. 附加功能模块运行中...")
        try:
            run_fred_traffic_light(USER_FRED_KEY)
            run_fred_v10_dashboard(USER_FRED_KEY)
        except NameError:
            st_warn("FRED Key 未配置，跳过附加模块。")
        progress_bar.progress(40)
        
        status_text.write("3. 趋势分析 (深度宏观)...")
        app.analyze_market_trends_console()
        progress_bar.progress(60)
        
        status_text.write("4. 板块轮动模块...")
        try:
            sr_engine = SectorRotationEngine()
            sr_engine.run_analysis()
        except Exception as e:
            st_err(f"板块轮动模块运行中断: {e}")
        progress_bar.progress(80)
        
        status_text.write("5. SMT 背离分析模块...")
        try:
            smt_analyzer = SMTDivergenceAnalyzer()
            smt_analyzer.run()
        except Exception as e:
            st_err(f"SMT分析模块运行中断: {e}")
            st.write(traceback.format_exc())
        progress_bar.progress(100)
        
        st.success(">>> 计算完成！")
        
    except Exception as e:
        st_err(f"程序运行出错: {e}")
        st.write(traceback.format_exc())
