import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(
    page_title="Fractal Edge Lab",
    page_icon="📈",
    layout="wide"
)

st.title("Fractal Edge Lab")
st.caption("منصة شخصية خاصة لتحليل الأسهم واختبار البنية السعرية")

st.warning("هذه أداة شخصية للتحليل والاختبار، وليست نصيحة مالية أو توصية تداول.")

DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
    "META", "GOOGL", "AMD", "NFLX", "JPM",
    "SPY", "QQQ"
]

with st.sidebar:
    st.header("الإعدادات")
    symbol = st.selectbox("اختر السهم", DEFAULT_SYMBOLS, index=7)
    period = st.selectbox("الفترة التاريخية", ["1y", "2y", "5y", "10y"], index=2)
    interval = st.selectbox("الإطار الزمني", ["1d", "1wk"], index=0)


@st.cache_data(show_spinner=False)
def load_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        return data

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    date_col = "Date" if "Date" in data.columns else "Datetime"
    data = data.rename(columns={date_col: "Date"})

    needed_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    data = data[needed_cols].copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna().reset_index(drop=True)

    return data


def add_market_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()
    df["ma200"] = df["Close"].rolling(200).mean()

    previous_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - previous_close).abs()
    tr3 = (df["Low"] - previous_close).abs()
    df["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = df["true_range"].rolling(14).mean()
    df["atr_pct"] = df["atr14"] / df["Close"] * 100

    df["high_20"] = df["High"].rolling(20).max()
    df["low_20"] = df["Low"].rolling(20).min()

    df["distance_from_high_20_pct"] = (df["Close"] / df["high_20"] - 1) * 100
    df["distance_from_low_20_pct"] = (df["Close"] / df["low_20"] - 1) * 100

    df["volatility_20"] = df["log_return"].rolling(20).std() * np.sqrt(252) * 100
    df["volatility_60"] = df["log_return"].rolling(60).std() * np.sqrt(252) * 100
    df["volatility_compression"] = df["volatility_20"] / df["volatility_60"]

    return df


def classify_market_state(row: pd.Series) -> tuple[str, str]:
    close = row["Close"]
    ma20 = row["ma20"]
    ma50 = row["ma50"]
    ma200 = row["ma200"]
    compression = row["volatility_compression"]

    if pd.isna(ma20) or pd.isna(ma50) or pd.isna(ma200):
        return "غير كافٍ", "No Trade"

    if close > ma20 > ma50 > ma200:
        state = "اتجاه صاعد منظم"
        action = "Watch"
    elif close < ma20 < ma50 < ma200:
        state = "اتجاه هابط منظم"
        action = "Avoid"
    elif compression < 0.75:
        state = "ضغط تذبذب محتمل"
        action = "Watch for Breakout"
    elif close > ma50 and ma20 > ma50:
        state = "ميل صاعد"
        action = "Watch"
    elif close < ma50 and ma20 < ma50:
        state = "ميل هابط"
        action = "Avoid"
    else:
        state = "تذبذب / ضجيج"
        action = "No Trade"

    return state, action


data = load_data(symbol, period, interval)

if data.empty:
    st.error("لم يتم العثور على بيانات لهذا السهم.")
    st.stop()

df = add_market_features(data)
latest = df.dropna().iloc[-1]

market_state, suggested_action = classify_market_state(latest)

st.subheader(f"تحليل السهم: {symbol}")

col1, col2, col3, col4 = st.columns(4)

latest_close = float(latest["Close"])
first_close = float(df["Close"].iloc[0])
total_return = (latest_close / first_close - 1) * 100

with col1:
    st.metric("آخر سعر", f"{latest_close:.2f}")

with col2:
    st.metric("العائد خلال الفترة", f"{total_return:.2f}%")

with col3:
    st.metric("عدد الشموع", f"{len(df):,}")

with col4:
    st.metric("حالة السوق", market_state)

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric("الإجراء الشخصي", suggested_action)

with col6:
    st.metric("ATR %", f"{latest['atr_pct']:.2f}%")

with col7:
    st.metric("البعد عن قمة 20", f"{latest['distance_from_high_20_pct']:.2f}%")

with col8:
    st.metric("البعد عن قاع 20", f"{latest['distance_from_low_20_pct']:.2f}%")

fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name=symbol
    )
)

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["ma20"],
        mode="lines",
        name="MA 20"
    )
)

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["ma50"],
        mode="lines",
        name="MA 50"
    )
)

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["ma200"],
        mode="lines",
        name="MA 200"
    )
)

fig.update_layout(
    height=650,
    xaxis_rangeslider_visible=False,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Market Structure v1")

st.write(
    """
    هذه النسخة تضيف أول طبقة تحليل بنيوي بسيطة:
    - المتوسطات المتحركة 20 / 50 / 200
    - ATR كنسبة من السعر
    - قمة وقاع آخر 20 شمعة
    - ضغط التذبذب
    - تصنيف أولي لحالة السهم
    """
)

st.subheader("قراءة أولية")

if suggested_action == "Watch":
    st.success("السهم يستحق المراقبة، لكن لا يوجد دخول تلقائي بدون شرط واضح.")
elif suggested_action == "Watch for Breakout":
    st.info("يوجد ضغط تذبذب محتمل. الأفضل مراقبة اختراق واضح قبل أي قرار.")
elif suggested_action == "Avoid":
    st.error("الحالة الحالية غير مناسبة للدخول الطويل حسب القواعد البسيطة.")
else:
    st.warning("لا توجد أفضلية واضحة حاليًا. الأفضل الانتظار.")

with st.expander("عرض آخر 10 صفوف من البيانات المحسوبة"):
    st.dataframe(df.tail(10), use_container_width=True)