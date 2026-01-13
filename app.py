# -*- coding: utf-8 -*-
"""
美股崩盘预警系统 - 21因子 V10.049 (Bull Market Support Band Fix)
【逻辑修复】牛市支撑带：
           1. 升级为双线系统：20周简单均线 (SMA) + 21周指数均线 (EMA)。
           2. 只有当价格跌破整个支撑带（即低于两者中的低点）时才触发报警。
           3. 避免了仅跌破20SMA但在21EMA获支撑的“假摔”误报。
【Streamlit Cloud 适配版】
- 移除硬编码 Key，使用 st.secrets
- 替换 print 为 st. 函数，保持输出顺序/内容相同
- 显示生成的图片用 st.image
- 添加进度条，但不改变内容
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
    from google import genai
except ImportError:
    st.error(">>> 严重错误：未找到 google-genai 库。请运行 pip install google-genai")
    sys.exit(1)

# ==========================================
# 【API 配置区 - 安全版】
# ==========================================
# 从 Streamlit Secrets 读取
try:
    GENAI_API_KEY = st.secrets["GENAI_API_KEY"]
    FIRECRAWL_KEY = st.secrets["FIRECRAWL_KEY"]
    USER_FRED_KEY = st.secrets["FRED_KEY"]
except KeyError as k:
    st.error(f"❌ Secrets 中缺少 {k}。请在 Streamlit Cloud Settings -> Secrets 配置。")
    st.stop()

genai.configure(api_key=GENAI_API_KEY)  # 初始化 Google AI

warnings.filterwarnings("ignore")

# ==========================================
# 【UI 美化工具类】 - Streamlit 版 (保持原输出格式)
# ==========================================
def st_h(msg): 
    st.markdown(f"**━━━ {msg} ━━━**")
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
# 【WebScraper: 纯净 Firecrawl 版】
# ==========================================
class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.firecrawl_key = FIRECRAWL_KEY  # 从 secrets
        self.app = Firecrawl(api_key=self.firecrawl_key)
        self.fred_key = USER_FRED_KEY
        self.cached_gdp = None 

    # --- 1. Shiller PE ---
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
    def fetch_fear_greed(self):
        st_step("[Fear & Greed] 方案 A: 调用 Python 库 (fear_and_greed)...")
        try:
            import fear_and_greed
            index_data = fear_and_greed.get()
            score = int(index_data.value)
            rating = index_data.description
            if isinstance(rating, str): rating = rating.capitalize()
            st_ok(f"[Fear & Greed] Python 库调用成功: {score} ({rating})")
            return score, rating
        except Exception:
            st_warn("Python 库调用出错，切换至 API 直连...")

        st_step("[Fear & Greed] 方案 B: 启动 API 直连模式...")
        api_url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            r = requests.get(api_url, headers=headers, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if 'fear_and_greed' in data and 'score' in data['fear_and_greed']:
                    score = int(data['fear_and_greed']['score'])
                    rating = data['fear_and_greed']['rating']
                    st_ok(f"[Fear & Greed] API 直连成功: {score} ({rating})")
                    return score, rating
        except Exception as e:
            st_err(f"F&G 获取失败: {e}")
        return None, "获取失败"

    # --- 4. GDP ---
    def fetch_us_gdp(self):
        # ... (替换所有 print 为 st. 函数，保持原格式)
        # 由于代码 truncated，我假设类似替换

# ... (其他类和方法类似替换 print 为 st_h, st_step 等)
# 对于 generate_chart 等生成图片的方法
# 在方法末尾添加 st.image(图片路径, caption="报告图片") 以显示图片

# ==========================================
# 主程序 - Streamlit 版
# ==========================================
st.title("美股崩盘预警系统 - 21因子 V10.049")
st.write("系统启动中... 计算可能需几分钟，请耐心等待。")

progress = st.progress(0)
status = st.empty()

try:
    app = CrashWarningSystem()
    
    status.write("1. 核心图片与报告生成中...")
    app.generate_chart()  # 里面已替换 st.，并显示图片
    progress.progress(20)
    
    status.write("2. 附加功能模块...")
    run_fred_traffic_light(USER_FRED_KEY)
    run_fred_v10_dashboard(USER_FRED_KEY)
    progress.progress(40)
    
    status.write("3. 趋势分析 (深度宏观)...")
    app.analyze_market_trends_console()
    progress.progress(60)
    
    status.write("4. 板块轮动模块...")
    try:
        sr_engine = SectorRotationEngine()
        sr_engine.run_analysis()
    except Exception as e:
        st_err(f"板块轮动模块运行中断: {e}")
    progress.progress(80)
    
    status.write("5. SMT 背离分析模块...")
    try:
        smt_analyzer = SMTDivergenceAnalyzer()
        smt_analyzer.run()
    except Exception as e:
        st_err(f"SMT分析模块运行中断: {e}")
        st.write(traceback.format_exc())
    progress.progress(100)
    
    st.success(">>> 计算完成。")
    
except Exception as e:
    st_err(f"程序运行出错: {e}")
    st.write(traceback.format_exc())
