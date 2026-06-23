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
st.caption("منصة شخصية خاصة لتحليل الأسهم وقراءة البنية السعرية")
st.warning("هذه أداة شخصية للتحليل والاختبار، وليست نصيحة مالية أو توصية تداول.")

with st.sidebar:
    st.header("الإعدادات")

    symbol_input = st.text_input(
        "اكتب رمز السهم",
        value="AMD",
        placeholder="مثال: AAPL أو TSLA أو NVDA"
    )

    symbol = symbol_input.strip().upper()

    if not symbol:
        st.warning("اكتب رمز السهم أولاً.")
        st.stop()

    period = st.selectbox("الفترة التاريخية", ["1y", "2y", "5y", "10y"], index=2)
    interval = st.selectbox("الإطار الزمني", ["1d", "1wk"], index=0)

    with st.expander("إعدادات متقدمة"):
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
            "نسبة أول جزء من البيانات",
            min_value=0.50,
            max_value=0.85,
            value=0.70,
            step=0.05
        )

        max_cycle_position = st.slider(
            "أقصى موقع دخول داخل الدورة %",
            min_value=10.0,
            max_value=60.0,
            value=35.0,
            step=5.0
        )

        hold_days = st.slider(
            "مدة الاختبار بالشموع",
            min_value=5,
            max_value=60,
            value=20,
            step=5
        )

        cost_pct = st.number_input(
            "تكلفة الصفقة التقريبية %",
            min_value=0.0,
            max_value=5.0,
            value=0.20,
            step=0.05
        )

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


def classify_market_state(row: pd.Series) -> str:
    close = row["Close"]
    ma20 = row["ma20"]
    ma50 = row["ma50"]
    ma200 = row["ma200"]
    compression = row["volatility_compression"]

    if pd.isna(ma20) or pd.isna(ma50) or pd.isna(ma200):
        return "غير كافٍ"

    if close > ma20 > ma50 > ma200:
        return "اتجاه صاعد منظم"

    if close < ma20 < ma50 < ma200:
        return "اتجاه هابط منظم"

    if compression < 0.75:
        return "ضغط تذبذب محتمل"

    if close > ma50 and ma20 > ma50:
        return "ميل صاعد"

    if close < ma50 and ma20 < ma50:
        return "ميل هابط"

    return "تذبذب / ضجيج"


def detect_fractal_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    out = df.copy()
    window = left + right + 1

    rolling_high = out["High"].rolling(window=window, center=True).max()
    rolling_low = out["Low"].rolling(window=window, center=True).min()

    out["swing_high"] = out["High"].eq(rolling_high)
    out["swing_low"] = out["Low"].eq(rolling_low)

    out.loc[:left - 1, ["swing_high", "swing_low"]] = False
    out.loc[len(out) - right:, ["swing_high", "swing_low"]] = False

    return out


def build_swing_sequence(df: pd.DataFrame, confirmation_delay: int = 0) -> list[dict]:
    raw_swings = []

    for idx, row in df.iterrows():
        confirmed_index = min(idx + confirmation_delay, len(df) - 1)

        if bool(row.get("swing_high", False)):
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

        if bool(row.get("swing_low", False)):
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


def analyze_structure_from_swings(
    df: pd.DataFrame,
    swings: list[dict],
    current_index: int
) -> dict:
    if len(swings) < 4:
        return {
            "structure_trend": "غير كافٍ",
            "last_swing": "غير متاح",
            "last_swing_price": np.nan,
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
        "last_swing_price": last_swing["price"],
        "price_position": price_position,
        "swings_count": len(swings),
    }


def compute_cycle_quality_at_index(
    df: pd.DataFrame,
    swings: list[dict],
    current_index: int
) -> dict:
    if len(swings) < 2:
        return {
            "cycle_direction": "غير كافٍ",
            "cycle_move_pct": np.nan,
            "cycle_move_atr": np.nan,
            "cycle_position": np.nan,
            "swing_density": np.nan,
            "cycle_quality": "غير كافٍ",
            "cycle_phase": "غير كافٍ",
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

    if pd.isna(cycle_position):
        cycle_phase = "غير واضحة"
    elif cycle_position <= 35:
        cycle_phase = "بداية دورة"
    elif cycle_position >= 75:
        cycle_phase = "نهاية دورة"
    else:
        cycle_phase = "منتصف دورة"

    return {
        "cycle_direction": cycle_direction,
        "cycle_move_pct": move_pct,
        "cycle_move_atr": move_atr,
        "cycle_position": cycle_position,
        "swing_density": swing_density,
        "cycle_quality": cycle_quality,
        "cycle_phase": cycle_phase,
    }


def detect_false_breakout(
    df: pd.DataFrame,
    swings: list[dict],
    lookback: int = 8
) -> dict:
    if len(swings) < 4:
        return {
            "status": "غير كافٍ",
            "risk": "غير كافٍ",
            "note": "لا توجد نقاط مؤكدة كافية."
        }

    highs = [s for s in swings if s["type"] == "high"]
    lows = [s for s in swings if s["type"] == "low"]

    if not highs or not lows:
        return {
            "status": "غير كافٍ",
            "risk": "غير كافٍ",
            "note": "لا توجد قمم أو قيعان كافية."
        }

    last_high = highs[-1]
    last_low = lows[-1]

    recent = df.tail(lookback).copy()
    latest_close = float(df["Close"].iloc[-1])

    broke_high = recent["High"].max() > last_high["price"]
    failed_high = latest_close < last_high["price"]

    broke_low = recent["Low"].min() < last_low["price"]
    failed_low = latest_close > last_low["price"]

    if broke_high and failed_high:
        return {
            "status": "اختراق صاعد كاذب",
            "risk": "مرتفع",
            "note": "السعر كسر آخر قمة مؤكدة ثم عاد دونها."
        }

    if broke_low and failed_low:
        return {
            "status": "اختراق هابط كاذب",
            "risk": "مرتفع",
            "note": "السعر كسر آخر قاع مؤكد ثم عاد فوقه."
        }

    return {
        "status": "لا يوجد اختراق كاذب واضح",
        "risk": "منخفض",
        "note": "لا يوجد فشل واضح بعد كسر آخر قمة أو قاع مؤكد."
    }


def make_personal_decision(
    structure_state: dict,
    cycle_state: dict,
    market_state: str,
    false_breakout: dict
) -> dict:
    structure = structure_state["structure_trend"]
    quality = cycle_state["cycle_quality"]
    position = cycle_state["cycle_position"]
    phase = cycle_state["cycle_phase"]
    fb_status = false_breakout["status"]

    if fb_status == "اختراق صاعد كاذب":
        return {
            "decision": "انتظار",
            "reason": "يوجد فشل بعد كسر قمة مؤكدة؛ هذا يضعف الدخول المباشر.",
            "waiting_for": "إما قاع بنيوي جديد أو اختراق صاعد جديد يثبت فوق القمة.",
            "confidence": "منخفضة"
        }

    if structure == "بنية صاعدة" and quality in ["قوية", "متوسطة"]:
        if not pd.isna(position) and position <= 35:
            return {
                "decision": "مراقبة فرصة",
                "reason": "البنية صاعدة والسعر قريب من بداية الدورة.",
                "waiting_for": "تأكيد سعري واضح بعد قاع بنيوي أو اختراق ثابت.",
                "confidence": "متوسطة"
            }

        if not pd.isna(position) and position >= 75:
            return {
                "decision": "انتظار تصحيح",
                "reason": "البنية صاعدة لكن السعر متقدم داخل الدورة؛ لا نطارد الحركة.",
                "waiting_for": "تراجع واضح أو قاع بنيوي جديد.",
                "confidence": "متوسطة"
            }

        return {
            "decision": "مراقبة",
            "reason": f"البنية صاعدة لكن السعر في {phase}.",
            "waiting_for": "إشارة أوضح من السعر أو عودة لمنطقة أفضل.",
            "confidence": "متوسطة"
        }

    if structure == "بنية هابطة":
        return {
            "decision": "تجنب",
            "reason": "البنية هابطة حسب القمم والقيعان المؤكدة.",
            "waiting_for": "تحول بنيوي واضح قبل أي متابعة.",
            "confidence": "متوسطة"
        }

    if quality in ["ضعيفة", "ضجيج", "ضجيج مرتفع"]:
        return {
            "decision": "لا تداول",
            "reason": "الدورة الحالية ضعيفة أو ملوثة بالضجيج.",
            "waiting_for": "دورة أقوى وبنية أوضح.",
            "confidence": "منخفضة"
        }

    return {
        "decision": "انتظار",
        "reason": "القراءة البنيوية غير كافية.",
        "waiting_for": "تكوّن قمة/قاع مؤكدين وبنية أوضح.",
        "confidence": "منخفضة"
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


def split_train_test(df: pd.DataFrame, train_ratio: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index].reset_index(drop=True)
    test_df = df.iloc[split_index:].reset_index(drop=True)
    return train_df, test_df


def run_fractal_decision_backtest_v1(
    df: pd.DataFrame,
    hold_days: int = 20,
    cost_pct: float = 0.20,
    min_swing_atr: float = 1.5,
    confirmation_delay: int = 3,
    max_cycle_position: float = 35.0,
    start_index: int = 200
) -> tuple[pd.DataFrame, dict]:
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

            entry_price = float(df.iloc[entry_index]["Open"])
            exit_price = float(df.iloc[exit_index]["Close"])

            gross_return_pct = (exit_price / entry_price - 1) * 100
            net_return_pct = gross_return_pct - cost_pct

            trades.append(
                {
                    "entry_date": df.iloc[entry_index]["Date"],
                    "exit_date": df.iloc[exit_index]["Date"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "net_return_pct": net_return_pct,
                    "hold_days": hold_days,
                    "structure_trend": structure_state["structure_trend"],
                    "cycle_quality": cycle_state["cycle_quality"],
                    "cycle_position": cycle_state["cycle_position"],
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


def make_backtest_verdict(stats: dict) -> dict:
    trades = stats["trades"]
    strategy_return = stats["strategy_total_return"]
    pf = stats["profit_factor"]

    if trades == 0:
        return {
            "label": "لا توجد صفقات كافية",
            "tone": "warning",
            "text": "الاختبار لم يعطِ صفقات كافية للحكم."
        }

    if trades < 5:
        return {
            "label": "واعد لكن غير مثبت",
            "tone": "warning",
            "text": "النتيجة جيدة ظاهرياً، لكن عدد الصفقات قليل جداً."
        }

    if trades < 20:
        if strategy_return > 0 and pf >= 1.5:
            return {
                "label": "واعد لكن إحصائياً ضعيف",
                "tone": "info",
                "text": "الاتجاه إيجابي، لكن عدد الصفقات أقل من 20."
            }
        return {
            "label": "غير كافٍ",
            "tone": "warning",
            "text": "عدد الصفقات قليل ولا يدعم حكماً قوياً."
        }

    if strategy_return > 0 and pf >= 1.5:
        return {
            "label": "مقبول مبدئياً",
            "tone": "success",
            "text": "الاختبار مقبول مبدئياً، لكنه لا يغني عن فحص أسهم وفترات أخرى."
        }

    return {
        "label": "ضعيف",
        "tone": "error",
        "text": "الاختبار لا يظهر تفوقاً كافياً."
    }


data = load_data(symbol, period, interval)

if data.empty:
    st.error("لم يتم العثور على بيانات لهذا الرمز. تأكد من كتابة الرمز بشكل صحيح.")
    st.stop()

df = add_market_features(data)
df = detect_fractal_swings(df, left=pivot_window, right=pivot_window)
clean_df = df.dropna().reset_index(drop=True)

if len(clean_df) < 250:
    st.error("البيانات المتاحة قليلة جداً لهذا الإطار. جرّب فترة أطول أو إطار 1d.")
    st.stop()

raw_swings = build_swing_sequence(clean_df, confirmation_delay=pivot_window)
swing_sequence = filter_swings_by_atr(raw_swings, clean_df, min_atr_multiple=min_swing_atr)

current_index = len(clean_df) - 1
structure_state = analyze_structure_from_swings(clean_df, swing_sequence, current_index=current_index)
cycle_state = compute_cycle_quality_at_index(clean_df, swing_sequence, current_index=current_index)
false_breakout = detect_false_breakout(clean_df, swing_sequence)
latest = clean_df.iloc[-1]
market_state = classify_market_state(latest)

decision_state = make_personal_decision(
    structure_state=structure_state,
    cycle_state=cycle_state,
    market_state=market_state,
    false_breakout=false_breakout
)

train_df, test_df = split_train_test(clean_df, train_ratio=train_ratio)
test_start_index = min(20, max(len(test_df) // 5, 1))

fractal_all_trades, fractal_all_stats = run_fractal_decision_backtest_v1(
    clean_df,
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

backtest_verdict = make_backtest_verdict(fractal_test_stats)
latest_close = float(latest["Close"])

tab_decision, tab_chart, tab_test = st.tabs(["القرار", "الشارت", "الاختبار"])

with tab_decision:
    st.subheader(f"{symbol} — الخلاصة")

    decision = decision_state["decision"]

    if decision in ["مراقبة فرصة", "مراقبة"]:
        st.success(f"القرار: {decision}")
    elif decision in ["انتظار", "انتظار تصحيح"]:
        st.warning(f"القرار: {decision}")
    elif decision in ["تجنب", "لا تداول"]:
        st.error(f"القرار: {decision}")
    else:
        st.info(f"القرار: {decision}")

    top_1, top_2, top_3, top_4 = st.columns(4)

    with top_1:
        st.metric("آخر سعر", f"{latest_close:.2f}")

    with top_2:
        st.metric("البنية", structure_state["structure_trend"])

    with top_3:
        st.metric("الدورة", cycle_state["cycle_quality"])

    with top_4:
        if pd.isna(cycle_state["cycle_position"]):
            st.metric("موقع الدورة", "غير متاح")
        else:
            st.metric("موقع الدورة", f"{cycle_state['cycle_position']:.2f}%")

    mid_1, mid_2, mid_3 = st.columns(3)

    with mid_1:
        st.metric("مرحلة الدورة", cycle_state["cycle_phase"])

    with mid_2:
        st.metric("اختراق كاذب", false_breakout["risk"])

    with mid_3:
        st.metric("حكم الاختبار", backtest_verdict["label"])

    st.divider()

    st.info(f"**السبب:** {decision_state['reason']}")
    st.warning(f"**أنتظر:** {decision_state['waiting_for']}")
    st.caption(f"ملاحظة الاختراق الكاذب: {false_breakout['note']}")

    if backtest_verdict["tone"] == "success":
        st.success(f"اختبار آخر 30%: {backtest_verdict['text']}")
    elif backtest_verdict["tone"] == "error":
        st.error(f"اختبار آخر 30%: {backtest_verdict['text']}")
    elif backtest_verdict["tone"] == "info":
        st.info(f"اختبار آخر 30%: {backtest_verdict['text']}")
    else:
        st.warning(f"اختبار آخر 30%: {backtest_verdict['text']}")

with tab_chart:
    st.subheader("الشارت البنيوي")

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=clean_df["Date"],
            open=clean_df["Open"],
            high=clean_df["High"],
            low=clean_df["Low"],
            close=clean_df["Close"],
            name=symbol
        )
    )

    fig.add_trace(go.Scatter(x=clean_df["Date"], y=clean_df["ma20"], mode="lines", name="MA 20"))
    fig.add_trace(go.Scatter(x=clean_df["Date"], y=clean_df["ma50"], mode="lines", name="MA 50"))
    fig.add_trace(go.Scatter(x=clean_df["Date"], y=clean_df["ma200"], mode="lines", name="MA 200"))

    swing_highs = clean_df[clean_df["swing_high"]]
    swing_lows = clean_df[clean_df["swing_low"]]

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

with tab_test:
    st.subheader("Fractal Decision Backtest")

    st.info(
        """
        الحكم الحقيقي هنا هو آخر 30% Out-of-sample.
        عدد الصفقات الأقل من 20 يبقى ضعيفاً إحصائياً حتى لو كانت النتيجة جميلة.
        """
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("صفقات آخر 30%", fractal_test_stats["trades"])

    with c2:
        st.metric("نسبة النجاح", f"{fractal_test_stats['win_rate']:.2f}%")

    with c3:
        st.metric("عائد الاستراتيجية", f"{fractal_test_stats['strategy_total_return']:.2f}%")

    with c4:
        if fractal_test_stats["profit_factor"] == float("inf"):
            st.metric("Profit Factor", "∞")
        else:
            st.metric("Profit Factor", f"{fractal_test_stats['profit_factor']:.2f}")

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        st.metric("متوسط الصفقة", f"{fractal_test_stats['avg_return']:.2f}%")

    with c6:
        st.metric("أسوأ صفقة", f"{fractal_test_stats['worst_trade']:.2f}%")

    with c7:
        st.metric("Buy & Hold", f"{fractal_test_stats['buy_hold_return']:.2f}%")

    with c8:
        st.metric("وقت داخل السوق", f"{fractal_test_stats['time_in_market']:.2f}%")

    if backtest_verdict["tone"] == "success":
        st.success(backtest_verdict["text"])
    elif backtest_verdict["tone"] == "error":
        st.error(backtest_verdict["text"])
    elif backtest_verdict["tone"] == "info":
        st.info(backtest_verdict["text"])
    else:
        st.warning(backtest_verdict["text"])

    with st.expander("تفاصيل إضافية"):
        st.write("كامل البيانات")
        st.json(fractal_all_stats)

        if fractal_all_trades.empty:
            st.write("لا توجد صفقات.")
        else:
            display_trades = fractal_all_trades.copy()
            display_trades["entry_date"] = display_trades["entry_date"].astype(str)
            display_trades["exit_date"] = display_trades["exit_date"].astype(str)
            st.dataframe(display_trades.tail(20), use_container_width=True)
