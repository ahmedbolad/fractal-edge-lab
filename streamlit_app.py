import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(
    page_title="Fractal Edge Lab",
    page_icon="📈",
    layout="wide"
)

st.title("Fractal Edge Lab")
st.caption("منصة شخصية خاصة لتحليل الأسهم واختبار البنية السعرية")

st.warning(
    "هذه أداة شخصية للتحليل والاختبار، وليست نصيحة مالية أو توصية تداول."
)

DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
    "META", "GOOGL", "AMD", "NFLX", "JPM",
    "SPY", "QQQ"
]

with st.sidebar:
    st.header("الإعدادات")
    symbol = st.selectbox("اختر السهم", DEFAULT_SYMBOLS)
    period = st.selectbox("الفترة التاريخية", ["1y", "2y", "5y", "10y"], index=2)
    interval = st.selectbox("الإطار الزمني", ["1d", "1wk"], index=0)

@st.cache_data(show_spinner=False)
def load_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return data

    data = data.reset_index()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    return data

data = load_data(symbol, period, interval)

if data.empty:
    st.error("لم يتم العثور على بيانات لهذا السهم.")
    st.stop()

date_col = "Date" if "Date" in data.columns else "Datetime"

st.subheader(f"تحليل السهم: {symbol}")

latest_close = float(data["Close"].iloc[-1])
first_close = float(data["Close"].iloc[0])
total_return = (latest_close / first_close - 1) * 100

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("آخر سعر", f"{latest_close:.2f}")

with col2:
    st.metric("العائد خلال الفترة", f"{total_return:.2f}%")

with col3:
    st.metric("عدد الشموع", f"{len(data):,}")

fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=data[date_col],
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name=symbol
    )
)

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("المرحلة الحالية من المشروع")

st.info(
    """
    هذه هي نسخة البداية فقط.

    الهدف الحالي:
    1. التأكد من تشغيل المنصة.
    2. تحميل بيانات الأسهم.
    3. عرض الشارت.
    4. تجهيز الأساس لبناء Backtesting Engine.
    5. لاحقًا سنضيف Market Structure وميزات التحليل البنيوي.
    """
)

st.subheader("الخطوة القادمة")

st.write(
    """
    بعد التأكد أن التطبيق يعمل، سنضيف:
    - حساب العوائد اللوغاريتمية
    - ATR
    - القمم والقيعان
    - اتجاه السهم
    - أول Backtest بسيط
    """
)