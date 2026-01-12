import streamlit as st
import yfinance as yf
import pandas as pd

# 1. 网页标题
st.title("📱 我的美股监控站")

# 2. 检查 API Key (这是为了测试后面步骤是否成功)
st.info("正在检查 API Key 连接状态...")
if "GENAI_API_KEY" in st.secrets:
    st.success("API Key 配置成功！安全连接已建立。")
else:
    st.error("未检测到 API Key，请在 Streamlit 后台配置 Secrets。")

# 3. 简单的交互功能
ticker = st.text_input("输入美股代码 (例如 AAPL, NVDA)", "SPY")

if st.button("开始分析"):
    with st.spinner('正在从云端抓取数据...'):
        # 获取数据
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        
        # 显示当前价格
        current_price = hist['Close'].iloc[-1]
        st.metric(label="当前价格", value=f"${current_price:.2f}")
        
        # 画图
        st.subheader(f"{ticker} 近一月走势")
        st.line_chart(hist['Close'])
        
        # 显示数据表
        st.write("详细数据：")
        st.dataframe(hist.tail())