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

pivot_window = st.slider(
    "حساسية القمم والقيعان",
    min_value=2,
    max_value=10,
    value=3,
    step=1
)

min_swing_atr = st.slider(
    "فلترة قوة الحركة ATR",
    min_value=0.5,
    max_value=5.0,
    value=1.5,
    step=0.5
)

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

def detect_fractal_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """
    Fractal Swing Detection v1:
    - القمة مؤكدة إذا كان High أعلى من عدد شموع قبلها وبعدها.
    - القاع مؤكد إذا كان Low أقل من عدد شموع قبلها وبعدها.
    - ملاحظة: هذه قمم/قيعان مؤكدة، أي تحتاج right شموع للتأكيد.
    """

    out = df.copy()
    out["swing_high"] = False
    out["swing_low"] = False

    for i in range(left, len(out) - right):
        current_high = out.loc[i, "High"]
        current_low = out.loc[i, "Low"]

        left_highs = out.loc[i - left:i - 1, "High"]
        right_highs = out.loc[i + 1:i + right, "High"]

        left_lows = out.loc[i - left:i - 1, "Low"]
        right_lows = out.loc[i + 1:i + right, "Low"]

        if current_high > left_highs.max() and current_high > right_highs.max():
            out.loc[i, "swing_high"] = True

        if current_low < left_lows.min() and current_low < right_lows.min():
            out.loc[i, "swing_low"] = True

    return out


def build_swing_sequence(df: pd.DataFrame) -> list[dict]:
    """
    يحول القمم والقيعان إلى سلسلة بنيوية مرتبة.
    إذا ظهرت قمتان متتاليتان، نحتفظ بالأعلى.
    إذا ظهر قاعان متتاليان، نحتفظ بالأدنى.
    """

    raw_swings = []

    for idx, row in df.iterrows():
        if row.get("swing_high", False):
            raw_swings.append(
                {
                    "index": idx,
                    "date": row["Date"],
                    "type": "high",
                    "price": float(row["High"]),
                }
            )

        if row.get("swing_low", False):
            raw_swings.append(
                {
                    "index": idx,
                    "date": row["Date"],
                    "type": "low",
                    "price": float(row["Low"]),
                }
            )

    raw_swings = sorted(raw_swings, key=lambda x: x["index"])

    if not raw_swings:
        return []

    cleaned = [raw_swings[0]]

    for swing in raw_swings[1:]:
        last = cleaned[-1]

        if swing["type"] != last["type"]:
            cleaned.append(swing)
        else:
            if swing["type"] == "high" and swing["price"] > last["price"]:
                cleaned[-1] = swing
            elif swing["type"] == "low" and swing["price"] < last["price"]:
                cleaned[-1] = swing

    return cleaned

def filter_swings_by_atr(swings: list[dict], df: pd.DataFrame, min_atr_multiple: float = 1.5) -> list[dict]:
    """
    Fractal Structure v2:
    فلترة القمم والقيعان بحيث لا نقبل إلا التحركات التي تتجاوز حدًا أدنى من ATR.
    الهدف: تقليل الضجيج والتركيز على البنية السعرية الأوضح.
    """

    if len(swings) < 2:
        return swings

    filtered = [swings[0]]

    for swing in swings[1:]:
        last = filtered[-1]

        atr_value = df.loc[swing["index"], "atr14"]

        if pd.isna(atr_value) or atr_value == 0:
            continue

        price_move = abs(swing["price"] - last["price"])
        required_move = atr_value * min_atr_multiple

        if price_move >= required_move:
            if swing["type"] != last["type"]:
                filtered.append(swing)
            else:
                if swing["type"] == "high" and swing["price"] > last["price"]:
                    filtered[-1] = swing
                elif swing["type"] == "low" and swing["price"] < last["price"]:
                    filtered[-1] = swing

    return filtered

def analyze_fractal_structure(df: pd.DataFrame, min_swing_atr: float = 1.5) -> tuple[dict, list[dict]]:
    raw_swings = build_swing_sequence(df)
    swings = filter_swings_by_atr(raw_swings, df, min_atr_multiple=min_swing_atr)

    if len(swings) < 4:
        return {
            "structure_trend": "غير كافٍ",
            "cycle_state": "لا توجد دورات مؤكدة كافية",
            "last_swing": "غير متاح",
            "last_swing_price": np.nan,
            "price_position": np.nan,
            "swings_count": len(swings),
        }, swings

    highs = [s for s in swings if s["type"] == "high"]
    lows = [s for s in swings if s["type"] == "low"]

    if len(highs) >= 2 and len(lows) >= 2:
        last_high = highs[-1]
        prev_high = highs[-2]
        last_low = lows[-1]
        prev_low = lows[-2]

        higher_high = last_high["price"] > prev_high["price"]
        higher_low = last_low["price"] > prev_low["price"]
        lower_high = last_high["price"] < prev_high["price"]
        lower_low = last_low["price"] < prev_low["price"]

        if higher_high and higher_low:
            structure_trend = "بنية صاعدة"
        elif lower_high and lower_low:
            structure_trend = "بنية هابطة"
        else:
            structure_trend = "بنية مختلطة"
    else:
        structure_trend = "غير كافٍ"

    last_swing = swings[-1]
    current_close = float(df["Close"].iloc[-1])

    if last_swing["type"] == "low":
        cycle_state = "بعد قاع مؤكد / محاولة بداية دورة صاعدة"
    else:
        cycle_state = "بعد قمة مؤكدة / احتمال تصحيح أو نهاية موجة"

    recent_swings = swings[-4:]
    recent_prices = [s["price"] for s in recent_swings]
    recent_min = min(recent_prices)
    recent_max = max(recent_prices)

    if recent_max > recent_min:
        price_position = (current_close - recent_min) / (recent_max - recent_min) * 100
    else:
        price_position = np.nan

    return {
        "structure_trend": structure_trend,
        "cycle_state": cycle_state,
        "last_swing": "قمة" if last_swing["type"] == "high" else "قاع",
        "last_swing_price": last_swing["price"],
        "price_position": price_position,
        "swings_count": len(swings),
    }, swings

def run_backtest_v1(df: pd.DataFrame, hold_days: int = 20, cost_pct: float = 0.20) -> tuple[pd.DataFrame, dict]:
    """
    Backtest v1:
    - الإشارة تظهر عند إغلاق اليوم إذا كان:
      Close > MA20 > MA50 > MA200
    - الدخول يكون في افتتاح اليوم التالي حتى نتجنب تسريب المستقبل.
    - الخروج بعد عدد أيام محدد.
    - cost_pct تمثل تكلفة إجمالية تقريبية للصفقة: دخول + خروج.
    """

    trades = []
    i = 200

    while i < len(df) - hold_days - 1:
        row = df.iloc[i]

        signal = (
            row["Close"] > row["ma20"]
            and row["ma20"] > row["ma50"]
            and row["ma50"] > row["ma200"]
        )

        if signal:
            entry_index = i + 1
            exit_index = entry_index + hold_days

            entry_date = df.iloc[entry_index]["Date"]
            exit_date = df.iloc[exit_index]["Date"]

            entry_price = float(df.iloc[entry_index]["Open"])
            exit_price = float(df.iloc[exit_index]["Close"])

            gross_return_pct = (exit_price / entry_price - 1) * 100
            net_return_pct = gross_return_pct - cost_pct

            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_return_pct": gross_return_pct,
                    "net_return_pct": net_return_pct,
                    "hold_days": hold_days,
                }
            )

            # منع تداخل الصفقات
            i = exit_index + 1
        else:
            i += 1

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        stats = {
            "trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "profit_factor": 0.0,
            "strategy_total_return": 0.0,
            "buy_hold_return": 0.0,
        }
        return trades_df, stats

    wins = trades_df[trades_df["net_return_pct"] > 0]
    losses = trades_df[trades_df["net_return_pct"] <= 0]

    total_profit = wins["net_return_pct"].sum()
    total_loss = abs(losses["net_return_pct"].sum())

    if total_loss == 0:
        profit_factor = float("inf")
    else:
        profit_factor = total_profit / total_loss

    strategy_total_return = ((1 + trades_df["net_return_pct"] / 100).prod() - 1) * 100
    buy_hold_return = (df["Close"].iloc[-1] / df["Close"].iloc[200] - 1) * 100

    stats = {
        "trades": len(trades_df),
        "win_rate": len(wins) / len(trades_df) * 100,
        "avg_return": trades_df["net_return_pct"].mean(),
        "best_trade": trades_df["net_return_pct"].max(),
        "worst_trade": trades_df["net_return_pct"].min(),
        "profit_factor": profit_factor,
        "strategy_total_return": strategy_total_return,
        "buy_hold_return": buy_hold_return,
    }

    return trades_df, stats

data = load_data(symbol, period, interval)

if data.empty:
    st.error("لم يتم العثور على بيانات لهذا السهم.")
    st.stop()

df = add_market_features(data)
df = detect_fractal_swings(df, left=pivot_window, right=pivot_window)
fractal_summary, swing_sequence = analyze_fractal_structure(df, min_swing_atr=min_swing_atr)

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

    st.divider()

st.subheader("فحص جودة البيانات")

quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)

with quality_col1:
    st.metric("أول تاريخ", str(df["Date"].iloc[0].date()))

with quality_col2:
    st.metric("آخر تاريخ", str(df["Date"].iloc[-1].date()))

with quality_col3:
    st.metric("أدنى سعر", f"{df['Close'].min():.2f}")

with quality_col4:
    st.metric("أعلى سعر", f"{df['Close'].max():.2f}")

missing_values = int(df[["Open", "High", "Low", "Close", "Volume"]].isna().sum().sum())

if missing_values > 0:
    st.error(f"يوجد {missing_values} قيمة ناقصة في البيانات.")
else:
    st.success("البيانات لا تحتوي على قيم ناقصة في الأعمدة الأساسية.")

with st.expander("عرض آخر 5 أسعار إغلاق"):
    st.dataframe(df[["Date", "Open", "High", "Low", "Close", "Volume"]].tail(5), use_container_width=True)

st.divider()

st.divider()

st.subheader("Fractal Structure v1")

st.write(
    """
    هذه الطبقة لا تعطي توصية تداول. وظيفتها قراءة البنية السعرية:
    - قمم وقيعان مؤكدة
    - آخر دورة سعرية
    - اتجاه البنية
    - موقع السعر داخل آخر نطاق بنيوي
    """
)

fs_col1, fs_col2, fs_col3, fs_col4 = st.columns(4)

with fs_col1:
    st.metric("اتجاه البنية", fractal_summary["structure_trend"])

with fs_col2:
    st.metric("حالة الدورة", fractal_summary["cycle_state"])

with fs_col3:
    st.metric("آخر نقطة مؤكدة", fractal_summary["last_swing"])

with fs_col4:
    if pd.isna(fractal_summary["price_position"]):
        st.metric("موقع السعر داخل الدورة", "غير متاح")
    else:
        st.metric("موقع السعر داخل الدورة", f"{fractal_summary['price_position']:.2f}%")

fs_col5, fs_col6 = st.columns(2)

with fs_col5:
    st.metric("عدد القمم والقيعان المؤكدة", fractal_summary["swings_count"])

with fs_col6:
    if pd.isna(fractal_summary["last_swing_price"]):
        st.metric("سعر آخر نقطة مؤكدة", "غير متاح")
    else:
        st.metric("سعر آخر نقطة مؤكدة", f"{fractal_summary['last_swing_price']:.2f}")

if fractal_summary["structure_trend"] == "بنية صاعدة":
    st.success("البنية الحالية صاعدة حسب القمم والقيعان المؤكدة.")
elif fractal_summary["structure_trend"] == "بنية هابطة":
    st.error("البنية الحالية هابطة حسب القمم والقيعان المؤكدة.")
elif fractal_summary["structure_trend"] == "بنية مختلطة":
    st.warning("البنية الحالية مختلطة؛ لا توجد قراءة بنيوية واضحة.")
else:
    st.info("لا توجد بيانات كافية لاستخراج بنية مؤكدة.")

with st.expander("عرض آخر 12 نقطة Fractal Swing"):
    if not swing_sequence:
        st.write("لا توجد قمم أو قيعان مؤكدة.")
    else:
        swings_df = pd.DataFrame(swing_sequence).tail(12)
        swings_df["date"] = swings_df["date"].astype(str)
        st.dataframe(swings_df, use_container_width=True)

st.subheader("Backtest v1")

st.write(
    """
    هذا اختبار تاريخي أولي وبسيط جدًا.

    القاعدة:
    - تظهر الإشارة عندما يكون السعر فوق MA20 و MA50 و MA200.
    - الدخول يتم في افتتاح اليوم التالي.
    - الخروج بعد عدد أيام محدد.
    - يتم منع تداخل الصفقات.
    - يتم خصم تكلفة تقديرية للصفقة.
    """
)

bt_col_settings_1, bt_col_settings_2 = st.columns(2)

with bt_col_settings_1:
    hold_days = st.slider("مدة الاحتفاظ بالصفقة", min_value=5, max_value=60, value=20, step=5)

with bt_col_settings_2:
    cost_pct = st.number_input(
        "تكلفة الصفقة الإجمالية % تقريبية",
        min_value=0.0,
        max_value=5.0,
        value=0.20,
        step=0.05
    )

trades_df, bt_stats = run_backtest_v1(df.dropna().reset_index(drop=True), hold_days=hold_days, cost_pct=cost_pct)

bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)

with bt_col1:
    st.metric("عدد الصفقات", f"{bt_stats['trades']}")

with bt_col2:
    st.metric("نسبة النجاح", f"{bt_stats['win_rate']:.2f}%")

with bt_col3:
    st.metric("متوسط عائد الصفقة", f"{bt_stats['avg_return']:.2f}%")

with bt_col4:
    if bt_stats["profit_factor"] == float("inf"):
        st.metric("Profit Factor", "∞")
    else:
        st.metric("Profit Factor", f"{bt_stats['profit_factor']:.2f}")

bt_col5, bt_col6, bt_col7, bt_col8 = st.columns(4)

with bt_col5:
    st.metric("أفضل صفقة", f"{bt_stats['best_trade']:.2f}%")

with bt_col6:
    st.metric("أسوأ صفقة", f"{bt_stats['worst_trade']:.2f}%")

with bt_col7:
    st.metric("عائد الاستراتيجية المركب", f"{bt_stats['strategy_total_return']:.2f}%")

with bt_col8:
    st.metric("Buy & Hold", f"{bt_stats['buy_hold_return']:.2f}%")

if bt_stats["trades"] == 0:
    st.warning("لا توجد صفقات كافية حسب القاعدة الحالية.")
else:
    if bt_stats["strategy_total_return"] > bt_stats["buy_hold_return"]:
        st.success("الاستراتيجية تفوقت على Buy & Hold في هذا الاختبار الأولي.")
    else:
        st.warning("الاستراتيجية لم تتفوق على Buy & Hold في هذا الاختبار الأولي.")

with st.expander("عرض آخر 20 صفقة"):
    if trades_df.empty:
        st.write("لا توجد صفقات.")
    else:
        display_trades = trades_df.copy()
        display_trades["entry_date"] = display_trades["entry_date"].astype(str)
        display_trades["exit_date"] = display_trades["exit_date"].astype(str)
        st.dataframe(display_trades.tail(20), use_container_width=True)
        
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

swing_highs = df[df["swing_high"]]
swing_lows = df[df["swing_low"]]

fig.add_trace(
    go.Scatter(
        x=swing_highs["Date"],
        y=swing_highs["High"],
        mode="markers",
        name="Fractal High",
        marker=dict(symbol="triangle-down", size=10)
    )
)

fig.add_trace(
    go.Scatter(
        x=swing_lows["Date"],
        y=swing_lows["Low"],
        mode="markers",
        name="Fractal Low",
        marker=dict(symbol="triangle-up", size=10)
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