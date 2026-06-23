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

    st.divider()

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

    train_ratio = st.slider(
        "نسبة التدريب / المراقبة من البيانات",
        min_value=0.50,
        max_value=0.85,
        value=0.70,
        step=0.05
    )

    st.divider()

    if st.button("تحديث البيانات ومسح الكاش"):
        st.cache_data.clear()
        st.rerun()


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
        return "اتجاه صاعد منظم", "Watch"

    if close < ma20 < ma50 < ma200:
        return "اتجاه هابط منظم", "Avoid"

    if compression < 0.75:
        return "ضغط تذبذب محتمل", "Watch for Breakout"

    if close > ma50 and ma20 > ma50:
        return "ميل صاعد", "Watch"

    if close < ma50 and ma20 < ma50:
        return "ميل هابط", "Avoid"

    return "تذبذب / ضجيج", "No Trade"


def detect_fractal_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """
    كشف قمم وقيعان مؤكد بطريقة أسرع من loop.
    القمة/القاع لا يتأكدان إلا بعد مرور right شموع.
    هذا مناسب للعرض البصري، لكنه يحتاج تأخيرًا صريحًا قبل أي Backtest بنيوي.
    """

    out = df.copy()
    window = left + right + 1
    center = left

    rolling_high = out["High"].rolling(window=window, center=True).max()
    rolling_low = out["Low"].rolling(window=window, center=True).min()

    out["swing_high"] = (out["High"] == rolling_high)
    out["swing_low"] = (out["Low"] == rolling_low)

    out.loc[:left - 1, ["swing_high", "swing_low"]] = False
    out.loc[len(out) - right:, ["swing_high", "swing_low"]] = False

    return out


def build_swing_sequence(df: pd.DataFrame, confirmation_delay: int = 0) -> list[dict]:
    raw_swings = []

    for idx, row in df.iterrows():
        confirmed_index = idx + confirmation_delay
        confirmed_index = min(confirmed_index, len(df) - 1)

        if row.get("swing_high", False):
            raw_swings.append(
                {
                    "index": idx,
                    "confirmed_index": confirmed_index,
                    "date": row["Date"],
                    "confirmed_date": df.loc[confirmed_index, "Date"],
                    "type": "high",
                    "price": float(row["High"]),
                }
            )

        if row.get("swing_low", False):
            raw_swings.append(
                {
                    "index": idx,
                    "confirmed_index": confirmed_index,
                    "date": row["Date"],
                    "confirmed_date": df.loc[confirmed_index, "Date"],
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


def filter_swings_by_atr(
    swings: list[dict],
    df: pd.DataFrame,
    min_atr_multiple: float = 1.5
) -> list[dict]:
    """
    فلترة القمم والقيعان مع الحفاظ على تناوب high / low.
    هذه نسخة آمنة بما يكفي للعرض، ولا تزال تحتاج اختبارًا عند بناء Backtest بنيوي.
    """

    if len(swings) < 2:
        return swings

    filtered = [swings[0]]

    for swing in swings[1:]:
        last = filtered[-1]
        atr_value = df.loc[swing["index"], "atr14"]

        if pd.isna(atr_value) or atr_value == 0:
            continue

        if swing["type"] == last["type"]:
            if swing["type"] == "high" and swing["price"] > last["price"]:
                filtered[-1] = swing
            elif swing["type"] == "low" and swing["price"] < last["price"]:
                filtered[-1] = swing
            continue

        price_move = abs(swing["price"] - last["price"])
        required_move = atr_value * min_atr_multiple

        if price_move >= required_move:
            filtered.append(swing)

    return filtered


def analyze_fractal_structure(
    df: pd.DataFrame,
    min_swing_atr: float = 1.5,
    confirmation_delay: int = 0
) -> tuple[dict, list[dict]]:
    raw_swings = build_swing_sequence(df, confirmation_delay=confirmation_delay)
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

    if last_swing["type"] == "low":
        cycle_state = "بعد قاع مؤكد / محاولة بداية دورة صاعدة"
    else:
        cycle_state = "بعد قمة مؤكدة / احتمال تصحيح أو نهاية موجة"

    current_close = float(df["Close"].iloc[-1])
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


def compute_cycle_quality(df: pd.DataFrame, swings: list[dict]) -> dict:
    if len(swings) < 2:
        return {
            "cycle_direction": "غير كافٍ",
            "cycle_move_pct": np.nan,
            "cycle_move_atr": np.nan,
            "cycle_position": np.nan,
            "swing_density": np.nan,
            "cycle_quality": "غير كافٍ",
            "cycle_reading": "لا توجد دورة مؤكدة كافية."
        }

    previous_swing = swings[-2]
    last_swing = swings[-1]

    previous_price = float(previous_swing["price"])
    last_price = float(last_swing["price"])
    current_close = float(df["Close"].iloc[-1])

    move_abs = abs(last_price - previous_price)
    move_pct = (move_abs / previous_price) * 100 if previous_price != 0 else np.nan

    atr_value = df.loc[last_swing["index"], "atr14"]

    if pd.isna(atr_value) or atr_value == 0:
        atr_values = df["atr14"].dropna()
        atr_value = atr_values.iloc[-1] if len(atr_values) else np.nan

    move_atr = move_abs / atr_value if not pd.isna(atr_value) and atr_value > 0 else np.nan

    low_bound = min(previous_price, last_price)
    high_bound = max(previous_price, last_price)

    if high_bound > low_bound:
        cycle_position = (current_close - low_bound) / (high_bound - low_bound) * 100
        cycle_position = max(0, min(100, cycle_position))
    else:
        cycle_position = np.nan

    swing_density = len(swings) / len(df) * 100

    if previous_swing["type"] == "low" and last_swing["type"] == "high":
        cycle_direction = "دورة صاعدة"
    elif previous_swing["type"] == "high" and last_swing["type"] == "low":
        cycle_direction = "دورة هابطة"
    else:
        cycle_direction = "دورة غير واضحة"

    if pd.isna(move_atr):
        cycle_quality = "غير كافٍ"
    elif swing_density > 20:
        cycle_quality = "ضجيج مرتفع"
    elif move_atr >= 5:
        cycle_quality = "قوية"
    elif move_atr >= 3:
        cycle_quality = "متوسطة"
    elif move_atr >= 1.5:
        cycle_quality = "ضعيفة"
    else:
        cycle_quality = "ضجيج"

    if cycle_quality == "قوية":
        cycle_reading = "آخر دورة كبيرة كفاية مقارنة بالتذبذب، ويمكن اعتبارها بنية مهمة."
    elif cycle_quality == "متوسطة":
        cycle_reading = "آخر دورة مقبولة، لكنها تحتاج تأكيد من الفريم الأكبر أو من اختبار تاريخي."
    elif cycle_quality == "ضعيفة":
        cycle_reading = "آخر دورة موجودة لكنها ليست قوية كفاية؛ الحذر أفضل."
    elif cycle_quality == "ضجيج مرتفع":
        cycle_reading = "عدد القمم والقيعان مرتفع؛ القراءة البنيوية قد تكون ملوثة بالضجيج."
    else:
        cycle_reading = "الحركة الأخيرة أقرب إلى ضجيج أو غير كافية كبنية."

    return {
        "cycle_direction": cycle_direction,
        "cycle_move_pct": move_pct,
        "cycle_move_atr": move_atr,
        "cycle_position": cycle_position,
        "swing_density": swing_density,
        "cycle_quality": cycle_quality,
        "cycle_reading": cycle_reading
    }


def make_personal_summary(
    fractal_summary: dict,
    cycle_quality: dict,
    market_state: str
) -> dict:
    structure = fractal_summary["structure_trend"]
    quality = cycle_quality["cycle_quality"]
    position = cycle_quality["cycle_position"]

    if structure == "بنية صاعدة" and quality in ["قوية", "متوسطة"]:
        if not pd.isna(position) and position >= 75:
            decision = "انتظار تصحيح"
            reason = "البنية صاعدة، لكن السعر متقدم داخل الدورة. لا نطارد الحركة."
            waiting_for = "قاع بنيوي جديد، عودة لمنطقة دعم، أو دورة جديدة أوضح."
        elif not pd.isna(position) and position <= 35:
            decision = "مراقبة فرصة"
            reason = "البنية صاعدة والسعر أقرب لبداية الدورة، لكن نحتاج تأكيد."
            waiting_for = "تأكيد سعري بعد قاع بنيوي أو كسر صاعد واضح."
        else:
            decision = "مراقبة"
            reason = "البنية صاعدة والدورة مقبولة، لكن القرار يحتاج شرط واضح."
            waiting_for = "تأكيد من السعر أو من الفريم الأكبر."
    elif structure == "بنية هابطة":
        decision = "تجنب"
        reason = "البنية هابطة حسب القمم والقيعان المؤكدة."
        waiting_for = "تحول بنيوي واضح قبل التفكير بأي فرصة."
    elif quality in ["ضجيج", "ضجيج مرتفع", "ضعيفة"]:
        decision = "لا تداول"
        reason = "الدورة الحالية ضعيفة أو ملوثة بالضجيج."
        waiting_for = "بنية أوضح وعدد أقل من الإشارات العشوائية."
    else:
        decision = "انتظار"
        reason = "لا توجد قراءة بنيوية كافية لاتخاذ قرار شخصي."
        waiting_for = "تكوّن دورة أوضح."

    return {
        "decision": decision,
        "reason": reason,
        "waiting_for": waiting_for,
        "market_state": market_state
    }


def summarize_trades(
    trades_df: pd.DataFrame,
    df: pd.DataFrame,
    hold_days: int,
    start_index: int = 200
) -> dict:
    if len(df) <= start_index + 1:
        buy_hold_return = 0.0
    else:
        buy_hold_return = (df["Close"].iloc[-1] / df["Close"].iloc[start_index] - 1) * 100

    if trades_df.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "profit_factor": 0.0,
            "strategy_total_return": 0.0,
            "buy_hold_return": buy_hold_return,
            "time_in_market": 0.0,
        }

    wins = trades_df[trades_df["net_return_pct"] > 0]
    losses = trades_df[trades_df["net_return_pct"] <= 0]

    total_profit = wins["net_return_pct"].sum()
    total_loss = abs(losses["net_return_pct"].sum())
    profit_factor = float("inf") if total_loss == 0 else total_profit / total_loss

    strategy_total_return = ((1 + trades_df["net_return_pct"] / 100).prod() - 1) * 100
    available_days = max(len(df) - start_index, 1)
    time_in_market = min((len(trades_df) * hold_days) / available_days * 100, 100)

    return {
        "trades": len(trades_df),
        "win_rate": len(wins) / len(trades_df) * 100,
        "avg_return": trades_df["net_return_pct"].mean(),
        "best_trade": trades_df["net_return_pct"].max(),
        "worst_trade": trades_df["net_return_pct"].min(),
        "profit_factor": profit_factor,
        "strategy_total_return": strategy_total_return,
        "buy_hold_return": buy_hold_return,
        "time_in_market": time_in_market,
    }

def analyze_structure_from_swings(
    df: pd.DataFrame,
    swings: list[dict],
    current_index: int
) -> dict:
    """
    يحلل البنية عند لحظة تاريخية محددة باستخدام swings مؤكدة فقط حتى تلك اللحظة.
    هذا مهم لمنع تسريب بيانات المستقبل في Backtest بنيوي.
    """

    if len(swings) < 4:
        return {
            "structure_trend": "غير كافٍ",
            "last_swing": "غير متاح",
            "price_position": np.nan,
            "swings_count": len(swings),
        }

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

    current_close = float(df["Close"].iloc[current_index])
    recent_swings = swings[-4:]
    recent_prices = [s["price"] for s in recent_swings]
    recent_min = min(recent_prices)
    recent_max = max(recent_prices)

    if recent_max > recent_min:
        price_position = (current_close - recent_min) / (recent_max - recent_min) * 100
        price_position = max(0, min(100, price_position))
    else:
        price_position = np.nan

    last_swing = swings[-1]

    return {
        "structure_trend": structure_trend,
        "last_swing": "قمة" if last_swing["type"] == "high" else "قاع",
        "price_position": price_position,
        "swings_count": len(swings),
    }


def compute_cycle_quality_at_index(
    df: pd.DataFrame,
    swings: list[dict],
    current_index: int
) -> dict:
    """
    يحسب جودة الدورة عند لحظة تاريخية محددة بدون استخدام المستقبل.
    """

    if len(swings) < 2:
        return {
            "cycle_direction": "غير كافٍ",
            "cycle_move_pct": np.nan,
            "cycle_move_atr": np.nan,
            "cycle_position": np.nan,
            "swing_density": np.nan,
            "cycle_quality": "غير كافٍ",
        }

    previous_swing = swings[-2]
    last_swing = swings[-1]

    previous_price = float(previous_swing["price"])
    last_price = float(last_swing["price"])
    current_close = float(df["Close"].iloc[current_index])

    move_abs = abs(last_price - previous_price)
    move_pct = (move_abs / previous_price) * 100 if previous_price != 0 else np.nan

    atr_value = df.loc[last_swing["index"], "atr14"]

    if pd.isna(atr_value) or atr_value == 0:
        atr_values = df["atr14"].iloc[:current_index + 1].dropna()
        atr_value = atr_values.iloc[-1] if len(atr_values) else np.nan

    move_atr = move_abs / atr_value if not pd.isna(atr_value) and atr_value > 0 else np.nan

    low_bound = min(previous_price, last_price)
    high_bound = max(previous_price, last_price)

    if high_bound > low_bound:
        cycle_position = (current_close - low_bound) / (high_bound - low_bound) * 100
        cycle_position = max(0, min(100, cycle_position))
    else:
        cycle_position = np.nan

    if previous_swing["type"] == "low" and last_swing["type"] == "high":
        cycle_direction = "دورة صاعدة"
    elif previous_swing["type"] == "high" and last_swing["type"] == "low":
        cycle_direction = "دورة هابطة"
    else:
        cycle_direction = "دورة غير واضحة"

    visible_bars = max(current_index + 1, 1)
    swing_density = len(swings) / visible_bars * 100

    if pd.isna(move_atr):
        cycle_quality = "غير كافٍ"
    elif swing_density > 20:
        cycle_quality = "ضجيج مرتفع"
    elif move_atr >= 5:
        cycle_quality = "قوية"
    elif move_atr >= 3:
        cycle_quality = "متوسطة"
    elif move_atr >= 1.5:
        cycle_quality = "ضعيفة"
    else:
        cycle_quality = "ضجيج"

    return {
        "cycle_direction": cycle_direction,
        "cycle_move_pct": move_pct,
        "cycle_move_atr": move_atr,
        "cycle_position": cycle_position,
        "swing_density": swing_density,
        "cycle_quality": cycle_quality,
    }


def run_fractal_decision_backtest_v1(
    df: pd.DataFrame,
    hold_days: int = 20,
    cost_pct: float = 0.20,
    min_swing_atr: float = 1.5,
    confirmation_delay: int = 3,
    max_cycle_position: float = 35.0,
    start_index: int = 200
) -> tuple[pd.DataFrame, dict]:
    """
    Fractal Decision Backtest v1:
    يختبر قرارًا بنيويًا أوليًا بدون استخدام swings غير مؤكدة.

    القاعدة:
    - لا نستخدم القمة/القاع إلا بعد confirmed_index.
    - البنية يجب أن تكون صاعدة.
    - جودة الدورة قوية أو متوسطة.
    - موقع السعر داخل الدورة <= max_cycle_position.
    - الدخول في افتتاح اليوم التالي.
    - الخروج بعد hold_days.
    """

    raw_swings = build_swing_sequence(df, confirmation_delay=confirmation_delay)

    trades = []
    i = start_index

    while i < len(df) - hold_days - 1:
        confirmed_swings = [
            swing for swing in raw_swings
            if swing["confirmed_index"] <= i
        ]

        filtered_swings = filter_swings_by_atr(
            confirmed_swings,
            df,
            min_atr_multiple=min_swing_atr
        )

        structure_state = analyze_structure_from_swings(
            df,
            filtered_swings,
            current_index=i
        )

        cycle_state = compute_cycle_quality_at_index(
            df,
            filtered_swings,
            current_index=i
        )

        signal = (
            structure_state["structure_trend"] == "بنية صاعدة"
            and cycle_state["cycle_quality"] in ["قوية", "متوسطة"]
            and not pd.isna(cycle_state["cycle_position"])
            and cycle_state["cycle_position"] <= max_cycle_position
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
                    "structure_trend": structure_state["structure_trend"],
                    "cycle_quality": cycle_state["cycle_quality"],
                    "cycle_position": cycle_state["cycle_position"],
                    "cycle_move_atr": cycle_state["cycle_move_atr"],
                    "swings_count": structure_state["swings_count"],
                }
            )

            i = exit_index + 1
        else:
            i += 1

    trades_df = pd.DataFrame(trades)

    stats = summarize_trades(
        trades_df=trades_df,
        df=df,
        hold_days=hold_days,
        start_index=start_index
    )

    return trades_df, stats

def run_backtest_v1(
    df: pd.DataFrame,
    hold_days: int = 20,
    cost_pct: float = 0.20,
    start_index: int = 200
) -> tuple[pd.DataFrame, dict]:
    """
    اختبار تقني لمحرك الـ Backtesting فقط.
    لا يختبر Fractal Structure ولا Cycle Quality.
    """

    trades = []
    i = start_index

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

            i = exit_index + 1
        else:
            i += 1

    trades_df = pd.DataFrame(trades)
    stats = summarize_trades(
        trades_df=trades_df,
        df=df,
        hold_days=hold_days,
        start_index=start_index
    )

    return trades_df, stats


def split_train_test(df: pd.DataFrame, train_ratio: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index].reset_index(drop=True)
    test_df = df.iloc[split_index:].reset_index(drop=True)
    return train_df, test_df


def display_backtest_stats(label: str, stats: dict) -> None:
    st.markdown(f"### {label}")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("عدد الصفقات", f"{stats['trades']}")

    with c2:
        st.metric("نسبة النجاح", f"{stats['win_rate']:.2f}%")

    with c3:
        st.metric("متوسط عائد الصفقة", f"{stats['avg_return']:.2f}%")

    with c4:
        if stats["profit_factor"] == float("inf"):
            st.metric("Profit Factor", "∞")
        else:
            st.metric("Profit Factor", f"{stats['profit_factor']:.2f}")

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        st.metric("أفضل صفقة", f"{stats['best_trade']:.2f}%")

    with c6:
        st.metric("أسوأ صفقة", f"{stats['worst_trade']:.2f}%")

    with c7:
        st.metric("عائد الاستراتيجية", f"{stats['strategy_total_return']:.2f}%")

    with c8:
        st.metric("Buy & Hold", f"{stats['buy_hold_return']:.2f}%")

    c9, c10 = st.columns(2)

    with c9:
        st.metric("نسبة الوقت داخل السوق", f"{stats['time_in_market']:.2f}%")

    with c10:
        if stats["trades"] < 20:
            st.warning("عدد الصفقات أقل من 20؛ الإحصائيات ضعيفة.")
        else:
            st.info("عدد الصفقات مقبول مبدئيًا، لكنه لا يكفي وحده للحكم.")


data = load_data(symbol, period, interval)

if data.empty:
    st.error("لم يتم العثور على بيانات لهذا السهم.")
    st.stop()

df = add_market_features(data)
df = detect_fractal_swings(df, left=pivot_window, right=pivot_window)

fractal_summary, swing_sequence = analyze_fractal_structure(
    df,
    min_swing_atr=min_swing_atr,
    confirmation_delay=pivot_window
)

cycle_quality = compute_cycle_quality(df, swing_sequence)

latest = df.dropna().iloc[-1]
market_state, suggested_action = classify_market_state(latest)

personal_summary = make_personal_summary(
    fractal_summary,
    cycle_quality,
    market_state
)

latest_close = float(latest["Close"])
first_close = float(df["Close"].iloc[0])
total_return = (latest_close / first_close - 1) * 100

tab_summary, tab_structure, tab_chart, tab_tests = st.tabs(
    ["القرار المختصر", "البنية والدورة", "الشارت", "الاختبارات والبيانات"]
)

with tab_summary:
    st.subheader(f"ملخص شخصي: {symbol}")

    main_col1, main_col2, main_col3 = st.columns(3)

    with main_col1:
        st.metric("القرار الشخصي", personal_summary["decision"])

    with main_col2:
        st.metric("البنية", fractal_summary["structure_trend"])

    with main_col3:
        st.metric("جودة الدورة", cycle_quality["cycle_quality"])

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("آخر سعر", f"{latest_close:.2f}")

    with metric_col2:
        if pd.isna(cycle_quality["cycle_position"]):
            st.metric("موقع السعر داخل الدورة", "غير متاح")
        else:
            st.metric("موقع السعر داخل الدورة", f"{cycle_quality['cycle_position']:.2f}%")

    with metric_col3:
        if pd.isna(cycle_quality["cycle_move_atr"]):
            st.metric("حجم الدورة بـ ATR", "غير متاح")
        else:
            st.metric("حجم الدورة بـ ATR", f"{cycle_quality['cycle_move_atr']:.2f}x")

    with metric_col4:
        st.metric("حالة السوق", market_state)

    st.info(f"السبب: {personal_summary['reason']}")
    st.warning(f"ننتظر: {personal_summary['waiting_for']}")

    st.divider()

    st.caption(
        "هذا الملخص قراءة بنيوية شخصية فقط. لا يمثل إشارة دخول، ولا يعتمد على Backtest فراكتلي بعد."
    )

with tab_structure:
    st.subheader("Fractal Structure + Cycle Quality")

    st.warning(
        f"""
        تنبيه مهم: القمم والقيعان هنا مؤكدة بعد مرور {pivot_window} شموع.
        هذا مناسب للقراءة البصرية، لكنه لا يجوز استخدامه مباشرة في Backtest بدون تأخير زمني،
        حتى لا يحدث تسريب بيانات مستقبلية.
        """
    )

    st.info(
        """
        العتبات الحالية مثل جودة الدورة، كثافة القمم والقيعان، وموقع السعر داخل الدورة
        افتراضات أولية غير معايرة. لا يتم تعديلها بناءً على سهم واحد حتى لا نقع في overfitting يدوي.
        """
    )

    fs_col1, fs_col2, fs_col3, fs_col4 = st.columns(4)

    with fs_col1:
        st.metric("اتجاه البنية", fractal_summary["structure_trend"])

    with fs_col2:
        st.metric("آخر نقطة مؤكدة", fractal_summary["last_swing"])

    with fs_col3:
        st.metric("عدد القمم والقيعان", fractal_summary["swings_count"])

    with fs_col4:
        if pd.isna(fractal_summary["price_position"]):
            st.metric("موقع السعر داخل آخر نطاق", "غير متاح")
        else:
            st.metric("موقع السعر داخل آخر نطاق", f"{fractal_summary['price_position']:.2f}%")

    cq_col1, cq_col2, cq_col3, cq_col4 = st.columns(4)

    with cq_col1:
        st.metric("اتجاه آخر دورة", cycle_quality["cycle_direction"])

    with cq_col2:
        if pd.isna(cycle_quality["cycle_move_pct"]):
            st.metric("حجم الدورة %", "غير متاح")
        else:
            st.metric("حجم الدورة %", f"{cycle_quality['cycle_move_pct']:.2f}%")

    with cq_col3:
        if pd.isna(cycle_quality["cycle_move_atr"]):
            st.metric("حجم الدورة بـ ATR", "غير متاح")
        else:
            st.metric("حجم الدورة بـ ATR", f"{cycle_quality['cycle_move_atr']:.2f}x")

    with cq_col4:
        if pd.isna(cycle_quality["swing_density"]):
            st.metric("كثافة القمم والقيعان", "غير متاح")
        else:
            st.metric("كثافة القمم والقيعان", f"{cycle_quality['swing_density']:.2f}%")

    if cycle_quality["cycle_quality"] == "قوية":
        st.success(cycle_quality["cycle_reading"])
    elif cycle_quality["cycle_quality"] == "متوسطة":
        st.info(cycle_quality["cycle_reading"])
    elif cycle_quality["cycle_quality"] == "ضعيفة":
        st.warning(cycle_quality["cycle_reading"])
    else:
        st.error(cycle_quality["cycle_reading"])

    with st.expander("عرض آخر 12 نقطة Fractal Swing"):
        if not swing_sequence:
            st.write("لا توجد قمم أو قيعان مؤكدة.")
        else:
            swings_df = pd.DataFrame(swing_sequence).tail(12)
            swings_df["date"] = swings_df["date"].astype(str)
            swings_df["confirmed_date"] = swings_df["confirmed_date"].astype(str)
            st.dataframe(swings_df, use_container_width=True)

with tab_chart:
    st.subheader("الشارت البنيوي")

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

    fig.add_trace(go.Scatter(x=df["Date"], y=df["ma20"], mode="lines", name="MA 20"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["ma50"], mode="lines", name="MA 50"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["ma200"], mode="lines", name="MA 200"))

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
        height=700,
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)


with tab_tests:
    st.subheader("فحص جودة البيانات")

    q_col1, q_col2, q_col3, q_col4 = st.columns(4)

    with q_col1:
        st.metric("أول تاريخ", str(df["Date"].iloc[0].date()))

    with q_col2:
        st.metric("آخر تاريخ", str(df["Date"].iloc[-1].date()))

    with q_col3:
        st.metric("أدنى سعر", f"{df['Close'].min():.2f}")

    with q_col4:
        st.metric("أعلى سعر", f"{df['Close'].max():.2f}")

    missing_values = int(df[["Open", "High", "Low", "Close", "Volume"]].isna().sum().sum())

    if missing_values > 0:
        st.error(f"يوجد {missing_values} قيمة ناقصة في البيانات.")
    else:
        st.success("البيانات لا تحتوي على قيم ناقصة في الأعمدة الأساسية.")

    with st.expander("عرض آخر 5 أسعار"):
        st.dataframe(
            df[["Date", "Open", "High", "Low", "Close", "Volume"]].tail(5),
            use_container_width=True
        )

    st.divider()

    st.subheader("Backtest Engine Test v1")

    st.error(
        """
        هذا الاختبار لا يقيس الاستراتيجية الفراكتلية ولا القرار البنيوي.
        هو فقط اختبار تقني للتأكد أن محرك الـ Backtesting يعمل.
        لا تستخدم نتائجه كدليل على قوة Fractal Edge Lab.
        """
    )

    st.write(
        """
        القاعدة الحالية كلاسيكية ومؤقتة:
        - تظهر الإشارة عندما يكون السعر فوق MA20 و MA50 و MA200.
        - الدخول يتم في افتتاح اليوم التالي.
        - هذه القاعدة لا تختبر Fractal Structure ولا Cycle Quality.
        - الهدف فقط اختبار الحسابات، الدخول، الخروج، والتكاليف.
        """
    )

    bt_col_settings_1, bt_col_settings_2 = st.columns(2)

    with bt_col_settings_1:
        hold_days = st.slider(
            "مدة الاحتفاظ بالصفقة",
            min_value=5,
            max_value=60,
            value=20,
            step=5
        )

    with bt_col_settings_2:
        cost_pct = st.number_input(
            "تكلفة الصفقة الإجمالية % تقريبية",
            min_value=0.0,
            max_value=5.0,
            value=0.20,
            step=0.05
        )

    st.divider()

    st.subheader("Fractal Decision Backtest v1")

    st.write(
        """
        هذا أول اختبار بنيوي حقيقي مرتبط بفكرة المشروع.

        القاعدة:
        - البنية صاعدة.
        - جودة الدورة قوية أو متوسطة.
        - السعر في أول جزء من الدورة.
        - لا يتم استخدام القمم والقيعان إلا بعد تأكيدها زمنيًا.
        """
    )

    fd_col1, fd_col2 = st.columns(2)

    with fd_col1:
        max_cycle_position = st.slider(
            "أقصى موقع مسموح داخل الدورة للدخول %",
            min_value=10.0,
            max_value=60.0,
            value=35.0,
            step=5.0
        )

    with fd_col2:
        st.info(
            f"سيتم تأخير القمم والقيعان {pivot_window} شموع لمنع تسريب المستقبل."
        )

    clean_df = df.dropna().reset_index(drop=True)
    train_df, test_df = split_train_test(clean_df, train_ratio=train_ratio)

    all_trades, all_stats = run_backtest_v1(
        clean_df,
        hold_days=hold_days,
        cost_pct=cost_pct,
        start_index=200
    )

    train_trades, train_stats = run_backtest_v1(
        train_df,
        hold_days=hold_days,
        cost_pct=cost_pct,
        start_index=200
    )

    test_start_index = min(20, max(len(test_df) // 5, 1))
    test_trades, test_stats = run_backtest_v1(
        test_df,
        hold_days=hold_days,
        cost_pct=cost_pct,
        start_index=test_start_index
    )

    fractal_all_trades, fractal_all_stats = run_fractal_decision_backtest_v1(
        clean_df,
        hold_days=hold_days,
        cost_pct=cost_pct,
        min_swing_atr=min_swing_atr,
        confirmation_delay=pivot_window,
        max_cycle_position=max_cycle_position,
        start_index=200
    )

    fractal_train_trades, fractal_train_stats = run_fractal_decision_backtest_v1(
        train_df,
        hold_days=hold_days,
        cost_pct=cost_pct,
        min_swing_atr=min_swing_atr,
        confirmation_delay=pivot_window,
        max_cycle_position=max_cycle_position,
        start_index=200
    )

    fractal_test_trades, fractal_test_stats = run_fractal_decision_backtest_v1(
        test_df,
        hold_days=hold_days,
        cost_pct=cost_pct,
        min_swing_atr=min_swing_atr,
        confirmation_delay=pivot_window,
        max_cycle_position=max_cycle_position,
        start_index=test_start_index
    )

    st.info(
        """
        يتم عرض النتائج على كامل البيانات، ثم على أول جزء من البيانات، ثم على آخر جزء Out-of-sample.
        لا يتم اعتبار أي نتيجة قوية ما لم تصمد في الجزء الأخير خارج العينة.
        """
    )

    bt_tab_all, bt_tab_train, bt_tab_test = st.tabs(
        ["كامل البيانات", "أول 70% تقريبًا", "آخر 30% Out-of-sample"]
    )

    with bt_tab_all:
        display_backtest_stats("كامل البيانات", all_stats)

    with bt_tab_train:
        display_backtest_stats("أول جزء من البيانات", train_stats)

    with bt_tab_test:
        display_backtest_stats("آخر جزء Out-of-sample", test_stats)

    st.warning("مقارنة Buy & Hold ليست حكمًا نهائيًا، لأن التعرض للسوق والمخاطرة مختلفان.")

    st.divider()

    st.subheader("نتائج Fractal Decision Backtest v1")

    st.success(
        """
        هذا الاختبار أقرب لفكرة المشروع، لأنه يستخدم البنية والدورة مع تأخير تأكيد القمم والقيعان.
        مع ذلك، لا نعتبره دليلاً نهائيًا إلا إذا صمد على آخر 30% Out-of-sample وبعدد صفقات كافٍ.
        """
    )

    fd_tab_all, fd_tab_train, fd_tab_test = st.tabs(
        ["Fractal - كامل البيانات", "Fractal - أول 70%", "Fractal - آخر 30%"]
    )

    with fd_tab_all:
        display_backtest_stats("Fractal Decision - كامل البيانات", fractal_all_stats)

    with fd_tab_train:
        display_backtest_stats("Fractal Decision - أول جزء", fractal_train_stats)

    with fd_tab_test:
        display_backtest_stats("Fractal Decision - Out-of-sample", fractal_test_stats)

    with st.expander("عرض آخر 20 صفقة Fractal Decision"):
        if fractal_all_trades.empty:
            st.write("لا توجد صفقات.")
        else:
            display_fractal_trades = fractal_all_trades.copy()
            display_fractal_trades["entry_date"] = display_fractal_trades["entry_date"].astype(str)
            display_fractal_trades["exit_date"] = display_fractal_trades["exit_date"].astype(str)
            st.dataframe(display_fractal_trades.tail(20), use_container_width=True)

    with st.expander("عرض آخر 20 صفقة من كامل البيانات"):
        if all_trades.empty:
            st.write("لا توجد صفقات.")
        else:
            display_trades = all_trades.copy()
            display_trades["entry_date"] = display_trades["entry_date"].astype(str)
            display_trades["exit_date"] = display_trades["exit_date"].astype(str)
            st.dataframe(display_trades.tail(20), use_container_width=True)
