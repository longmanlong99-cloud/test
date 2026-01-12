import streamlit as st
import yfinance as yf
import pandas as pd
import time

# 1. 网页配置
st.set_page_config(page_title="美股监控", page_icon="📈")
st.title("📱 我的美股监控站")

# 2. 检查 API Key
if "GENAI_API_KEY" in st.secrets:
    st.success("API Key 连接正常 ✅")
else:
    st.warning("⚠️ 未检测到 API Key (目前仅测试雅虎数据，暂不需要 Key)")

# --- 核心修改：增加缓存装饰器 ---
# ttl=3600 表示数据在内存里保留 1 小时，这 1 小时内随便点都不会触发雅虎封锁
@st.cache_data(ttl=3600) 
def get_stock_data(symbol):
    try:
        # 伪装成浏览器访问，防止被一眼识破
        stock = yf.Ticker(symbol)
        # 这里的 history 是最容易触发限流的，所以我们把它缓存起来
        df = stock.history(period="1mo")
        return df
    except Exception as e:
        return None

# 3. 交互界面
ticker = st.text_input("输入美股代码", "SPY").upper()

if st.button("开始分析"):
    with st.spinner(f'正在获取 {ticker} 数据...'):
        # 调用我们上面的缓存函数，而不是直接调用 yf
        hist = get_stock_data(ticker)
        
        if hist is None or hist.empty:
            st.error(f"❌ 无法获取 {ticker} 数据。可能是代码输错，或者雅虎接口正在忙碌，请稍后再试。")
        else:
            # 成功获取
            current_price = hist['Close'].iloc[-1]
            last_price = hist['Close'].iloc[-2]
            change = current_price - last_price
            pct_change = (change / last_price) * 100
            
            # 显示漂亮的数据卡片
            st.metric(
                label="当前价格", 
                value=f"${current_price:.2f}", 
                delta=f"{change:+.2f} ({pct_change:+.2f}%)"
            )
            
            # 画图
            st.subheader("近一月走势")
            st.line_chart(hist['Close'])
            
            # 原始数据
            with st.expander("查看详细数据"):
                st.dataframe(hist.tail())
