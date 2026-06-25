import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Fractal Edge Lab",
    page_icon="📈",
    layout="wide"
)

st.title("Fractal Edge Lab")
st.caption("منصة شخصية خاصة لتحليل الأسهم وقراءة البنية السعرية")
st.warning("هذه أداة شخصية للتحليل والاختبار، وليست نصيحة مالية أو توصية تداول.")

# Locked engine settings — ثابتة حسب التقرير المعتمد والملف الأساسي
# لا تظهر للمستخدم حتى تبقى المنصة تعطي إشارة مباشرة بدون ضبط يدوي.
PERIOD = "5y"
INTERVAL = "1d"
PIVOT_WINDOW = 5
MIN_SWING_ATR = 1.5
TRAIN_RATIO = 0.70
MAX_CYCLE_POSITION = 35.0
HOLD_DAYS = 20
COST_PCT = 0.20

SYMBOL_ALIASES = {
    "ذهب": ("GC=F", "Gold Futures"),
    "الذهب": ("GC=F", "Gold Futures"),
    "دهب": ("GC=F", "Gold Futures"),
    "gold": ("GC=F", "Gold Futures"),
    "xau": ("GC=F", "Gold Futures"),
    "xauusd": ("GC=F", "Gold Futures"),
    "xauusd=x": ("XAUUSD=X", "Gold Spot"),
    "gld": ("GLD", "SPDR Gold Shares"),
    "فضة": ("SI=F", "Silver Futures"),
    "الفضة": ("SI=F", "Silver Futures"),
    "silver": ("SI=F", "Silver Futures"),
    "xag": ("SI=F", "Silver Futures"),
    "xagusd": ("SI=F", "Silver Futures"),
    "نفط": ("CL=F", "WTI Crude Oil"),
    "النفط": ("CL=F", "WTI Crude Oil"),
    "oil": ("CL=F", "WTI Crude Oil"),
    "wti": ("CL=F", "WTI Crude Oil"),
    "brent": ("BZ=F", "Brent Crude Oil"),
    "بتكوين": ("BTC-USD", "Bitcoin"),
    "بيتكوين": ("BTC-USD", "Bitcoin"),
    "bitcoin": ("BTC-USD", "Bitcoin"),
    "btc": ("BTC-USD", "Bitcoin"),
    "اثيريوم": ("ETH-USD", "Ethereum"),
    "ethereum": ("ETH-USD", "Ethereum"),
    "eth": ("ETH-USD", "Ethereum"),
    "ناسداك": ("QQQ", "Nasdaq 100 ETF"),
    "nasdaq": ("QQQ", "Nasdaq 100 ETF"),
    "qqq": ("QQQ", "Nasdaq 100 ETF"),
    "sp500": ("SPY", "S&P 500 ETF"),
    "s&p500": ("SPY", "S&P 500 ETF"),
    "spy": ("SPY", "S&P 500 ETF"),
    "داو": ("DIA", "Dow Jones ETF"),
    "dow": ("DIA", "Dow Jones ETF"),
    "apple": ("AAPL", "Apple"),
    "ابل": ("AAPL", "Apple"),
    "tesla": ("TSLA", "Tesla"),
    "تسلا": ("TSLA", "Tesla"),
    "nvidia": ("NVDA", "NVIDIA"),
    "نفيديا": ("NVDA", "NVIDIA"),
    "microsoft": ("MSFT", "Microsoft"),
    "مايكروسوفت": ("MSFT", "Microsoft"),
    "amd": ("AMD", "AMD"),
}


def normalize_symbol_text(value: str) -> str:
    cleaned = str(value).strip().lower()
    for token in [" ", "-", "_", "/", "\\", ":"]:
        cleaned = cleaned.replace(token, "")
    return cleaned


def resolve_symbol(value: str) -> tuple[str, str, str]:
    raw = str(value).strip()
    key = normalize_symbol_text(raw)

    if key in SYMBOL_ALIASES:
        ticker, name = SYMBOL_ALIASES[key]
        return ticker, name, raw

    ticker = raw.upper().replace(" ", "")
    return ticker, ticker, raw


def infer_asset_type(symbol: str) -> str:
    symbol = str(symbol).upper()
    if symbol.endswith("-USD"):
        return "crypto"
    if symbol.endswith("=F") or symbol.endswith("=X"):
        return "non_stock"
    return "stock"

def annualization_factor(asset_type: str) -> int:
    return 365 if asset_type == "crypto" else 252

def next_trading_date(date_value, bars: int, asset_type: str):
    date_value = pd.to_datetime(date_value)
    if asset_type == "crypto":
        return date_value + pd.Timedelta(days=int(bars))
    dates = pd.bdate_range(start=date_value, periods=bars + 1)
    return dates[-1]


input_col, refresh_col = st.columns([4, 1])

with input_col:
    symbol_input = st.text_input(
        "اكتب الرمز أو الاسم",
        value="AMD",
        placeholder="مثال: AMD أو الذهب أو GOLD أو NVDA",
        label_visibility="collapsed"
    )

with refresh_col:
    st.write("")
    refresh_clicked = st.button("تحديث", use_container_width=True)

if refresh_clicked:
    st.cache_data.clear()
    st.rerun()

symbol, instrument_name, raw_symbol_input = resolve_symbol(symbol_input)
asset_type = infer_asset_type(symbol)

if not symbol:
    st.warning("اكتب رمز السهم أو اسم الأصل أولاً.")
    st.stop()

st.caption(f"الأصل: {instrument_name} — الرمز المستخدم للبيانات: {symbol}")


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

    date_col = "Date" if "Date" in data.columns else "Datetime" if "Datetime" in data.columns else None
    if date_col is None:
        return pd.DataFrame()

    data = data.rename(columns={date_col: "Date"})

    required_price_cols = ["Open", "High", "Low", "Close"]
    for col in required_price_cols:
        if col not in data.columns:
            return pd.DataFrame()

    if "Volume" not in data.columns:
        data["Volume"] = 0

    needed_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    data = data[needed_cols].copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["Volume"] = data["Volume"].fillna(0)
    data = data.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    return data


def add_market_features(data: pd.DataFrame, asset_type: str = "stock") -> pd.DataFrame:
    df = data.copy()
    ann_factor = annualization_factor(asset_type)

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

    df["volatility_20"] = df["log_return"].rolling(20).std() * np.sqrt(ann_factor) * 100
    df["volatility_60"] = df["log_return"].rolling(60).std() * np.sqrt(ann_factor) * 100
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


def update_filtered_swings(
    filtered: list[dict],
    swing: dict,
    df: pd.DataFrame,
    min_atr_multiple: float = 1.5
) -> list[dict]:
    if not filtered:
        return [swing]

    last = filtered[-1]
    atr_value = df.loc[swing["index"], "atr14"]

    if pd.isna(atr_value) or atr_value == 0:
        return filtered

    if swing["type"] == last["type"]:
        if swing["type"] == "high" and swing["price"] > last["price"]:
            filtered[-1] = swing
        elif swing["type"] == "low" and swing["price"] < last["price"]:
            filtered[-1] = swing
        return filtered

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

    density_window_start = max(current_index - 100, 0)
    recent_swing_count = len([s for s in swings if s["index"] >= density_window_start])
    visible_bars = max(current_index - density_window_start + 1, 1)
    swing_density = recent_swing_count / visible_bars * 100

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
    lookback: int = 8,
    max_reference_age: int = 80
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

    last_index = len(df) - 1
    last_high = highs[-1]
    last_low = lows[-1]

    recent = df.tail(lookback).copy()
    latest_close = float(df["Close"].iloc[-1])

    high_reference_recent = (last_index - int(last_high["index"])) <= max_reference_age
    low_reference_recent = (last_index - int(last_low["index"])) <= max_reference_age

    broke_high = high_reference_recent and recent["High"].max() > last_high["price"]
    failed_high = high_reference_recent and latest_close < last_high["price"]

    broke_low = low_reference_recent and recent["Low"].min() < last_low["price"]
    failed_low = low_reference_recent and latest_close > last_low["price"]

    if broke_high and failed_high:
        return {
            "status": "اختراق صاعد كاذب",
            "risk": "مرتفع",
            "note": "السعر كسر قمة مؤكدة حديثة ثم عاد دونها."
        }

    if broke_low and failed_low:
        return {
            "status": "اختراق هابط كاذب",
            "risk": "مرتفع",
            "note": "السعر كسر قاعاً مؤكداً حديثاً ثم عاد فوقه."
        }

    return {
        "status": "لا يوجد اختراق كاذب واضح",
        "risk": "منخفض",
        "note": "لا يوجد فشل واضح بعد كسر قمة أو قاع حديث مؤكد."
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
    raw_ptr = 0
    filtered_swings: list[dict] = []

    while i < len(df) - hold_days - 1:
        while raw_ptr < len(raw_swings) and raw_swings[raw_ptr]["confirmed_index"] <= i:
            filtered_swings = update_filtered_swings(
                filtered_swings,
                raw_swings[raw_ptr],
                df,
                min_atr_multiple=min_swing_atr
            )
            raw_ptr += 1

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


def build_forward_scenario(
    df: pd.DataFrame,
    swings: list[dict],
    structure_state: dict,
    cycle_state: dict,
    false_breakout: dict,
    horizon: int = 20
) -> dict:
    latest = df.iloc[-1]
    last_close = float(latest["Close"])
    atr = float(latest["atr14"]) if not pd.isna(latest["atr14"]) and latest["atr14"] > 0 else last_close * 0.03

    highs = [s for s in swings if s["type"] == "high"]
    lows = [s for s in swings if s["type"] == "low"]
    last_high = highs[-1]["price"] if highs else last_close + atr
    last_low = lows[-1]["price"] if lows else last_close - atr

    structure = structure_state["structure_trend"]
    quality = cycle_state["cycle_quality"]
    position = cycle_state["cycle_position"]
    fb_risk = false_breakout["risk"]

    if structure == "بنية صاعدة" and quality in ["قوية", "متوسطة"] and not pd.isna(position) and position <= 35 and fb_risk != "مرتفع":
        label = "امتداد صاعد مشروط"
        bias = "إيجابي"
        low = max(last_close - 1.2 * atr, last_low)
        high = max(last_close + 2.2 * atr, last_high)
        invalidation = min(last_close - 1.5 * atr, last_low)
        note = "السيناريو الصاعد يبقى قائماً ما دام السعر لا يكسر منطقة الإلغاء."
    elif structure == "بنية صاعدة" and (fb_risk == "مرتفع" or (not pd.isna(position) and position >= 75)):
        label = "تصحيح أو تهدئة محتملة"
        bias = "حذر"
        low = max(last_close - 2.0 * atr, last_low)
        high = last_close + 0.8 * atr
        invalidation = max(last_close + 1.2 * atr, last_high)
        note = "السعر متأخر داخل الدورة أو توجد علامة فشل اختراق؛ الأفضل انتظار قاع جديد."
    elif structure == "بنية هابطة":
        label = "ضغط هابط محتمل"
        bias = "سلبي"
        low = min(last_close - 2.0 * atr, last_low)
        high = last_close + 1.0 * atr
        invalidation = max(last_close + 1.5 * atr, last_high)
        note = "البنية الهابطة تلغي فكرة المتابعة حتى يظهر تحول واضح."
    else:
        label = "نطاق انتظار"
        bias = "محايد"
        low = last_close - 1.5 * atr
        high = last_close + 1.5 * atr
        invalidation = np.nan
        note = "لا توجد أفضلية بنيوية كافية؛ الانتظار أفضل من التوقع القسري."

    low, high = sorted([float(low), float(high)])

    return {
        "label": label,
        "bias": bias,
        "range_low": low,
        "range_high": high,
        "mid": (low + high) / 2,
        "invalidation": invalidation,
        "horizon": horizon,
        "note": note,
    }


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


data = load_data(symbol, PERIOD, INTERVAL)

if data.empty:
    st.error("لم يتم العثور على بيانات لهذا الرمز. تأكد من كتابة الرمز بشكل صحيح.")
    st.stop()

df = add_market_features(data, asset_type=asset_type)
df = detect_fractal_swings(df, left=PIVOT_WINDOW, right=PIVOT_WINDOW)
clean_df = df.dropna().reset_index(drop=True)
zero_volume_pct = (data["Volume"].eq(0).mean() * 100) if "Volume" in data.columns and len(data) else 0.0
missing_price_values = int(data[["Open", "High", "Low", "Close"]].isna().sum().sum()) if not data.empty else 0

if len(clean_df) < 250:
    st.error("البيانات المتاحة قليلة جداً لهذا الإطار. جرّب فترة أطول أو إطار 1d.")
    st.stop()

raw_swings = build_swing_sequence(clean_df, confirmation_delay=PIVOT_WINDOW)
swing_sequence = filter_swings_by_atr(raw_swings, clean_df, min_atr_multiple=MIN_SWING_ATR)

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

train_df, test_df = split_train_test(clean_df, train_ratio=TRAIN_RATIO)
test_start_index = min(20, max(len(test_df) // 5, 1))

fractal_all_trades, fractal_all_stats = run_fractal_decision_backtest_v1(
    clean_df,
    hold_days=HOLD_DAYS,
    cost_pct=COST_PCT,
    min_swing_atr=MIN_SWING_ATR,
    confirmation_delay=PIVOT_WINDOW,
    max_cycle_position=MAX_CYCLE_POSITION,
    start_index=200
)

fractal_test_trades, fractal_test_stats = run_fractal_decision_backtest_v1(
    test_df,
    hold_days=HOLD_DAYS,
    cost_pct=COST_PCT,
    min_swing_atr=MIN_SWING_ATR,
    confirmation_delay=PIVOT_WINDOW,
    max_cycle_position=MAX_CYCLE_POSITION,
    start_index=test_start_index
)

backtest_verdict = make_backtest_verdict(fractal_test_stats)
forward_scenario = build_forward_scenario(
    clean_df,
    swing_sequence,
    structure_state,
    cycle_state,
    false_breakout,
    horizon=HOLD_DAYS
)
latest_close = float(latest["Close"])

tab_decision, tab_chart, tab_test = st.tabs(["القرار", "الشارت", "الاختبار"])

with tab_decision:
    st.subheader(f"{instrument_name} — الخلاصة")

    decision = decision_state["decision"]

    if decision in ["مراقبة فرصة", "مراقبة"]:
        st.success(f"القرار: {decision}")
    elif decision in ["انتظار", "انتظار تصحيح"]:
        st.warning(f"القرار: {decision}")
    elif decision in ["تجنب", "لا تداول"]:
        st.error(f"القرار: {decision}")
    else:
        st.info(f"القرار: {decision}")

    top_1, top_2, top_3 = st.columns(3)

    with top_1:
        st.metric("آخر سعر", f"{latest_close:.2f}")
        st.metric("البنية", structure_state["structure_trend"])

    with top_2:
        st.metric("مرحلة الدورة", cycle_state["cycle_phase"])
        if pd.isna(cycle_state["cycle_position"]):
            st.metric("موقع الدورة", "غير متاح")
        else:
            st.metric("موقع الدورة", f"{cycle_state['cycle_position']:.2f}%")

    with top_3:
        st.metric("التوقع القادم", forward_scenario["label"])
        st.metric("حكم الاختبار", backtest_verdict["label"])

    st.divider()

    range_text = f"{forward_scenario['range_low']:.2f} — {forward_scenario['range_high']:.2f}"
    st.info(f"**السبب:** {decision_state['reason']}")
    st.warning(f"**أنتظر:** {decision_state['waiting_for']}")
    st.caption(f"النطاق المتوقع خلال {HOLD_DAYS} شمعة: {range_text}. {forward_scenario['note']}")

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

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.78, 0.22],
    )

    fig.add_trace(
        go.Candlestick(
            x=clean_df["Date"],
            open=clean_df["Open"],
            high=clean_df["High"],
            low=clean_df["Low"],
            close=clean_df["Close"],
            name=symbol,
            increasing_line_color="#26a69a",
            increasing_fillcolor="#26a69a",
            decreasing_line_color="#ef5350",
            decreasing_fillcolor="#ef5350",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(go.Scatter(x=clean_df["Date"], y=clean_df["ma20"], mode="lines", name="MA 20", line=dict(width=1.4, color="#42a5f5")), row=1, col=1)
    fig.add_trace(go.Scatter(x=clean_df["Date"], y=clean_df["ma50"], mode="lines", name="MA 50", line=dict(width=1.4, color="#ffca28")), row=1, col=1)
    fig.add_trace(go.Scatter(x=clean_df["Date"], y=clean_df["ma200"], mode="lines", name="MA 200", line=dict(width=1.6, color="#ab47bc")), row=1, col=1)

    recent_structural_swings = swing_sequence[-32:]
    if len(recent_structural_swings) >= 2:
        fig.add_trace(
            go.Scatter(
                x=[s["date"] for s in recent_structural_swings],
                y=[s["price"] for s in recent_structural_swings],
                mode="lines+markers",
                name="ZigZag بنيوي",
                line=dict(width=1.6, color="#fdd835"),
                marker=dict(size=6, color="#fdd835"),
            ),
            row=1,
            col=1,
        )

    swing_high_points = [s for s in recent_structural_swings if s["type"] == "high"]
    swing_low_points = [s for s in recent_structural_swings if s["type"] == "low"]

    if swing_high_points:
        fig.add_trace(
            go.Scatter(
                x=[s["date"] for s in swing_high_points],
                y=[s["price"] for s in swing_high_points],
                mode="markers",
                name="قمم",
                marker=dict(symbol="triangle-down", size=10, color="#ff7043"),
            ),
            row=1,
            col=1,
        )

    if swing_low_points:
        fig.add_trace(
            go.Scatter(
                x=[s["date"] for s in swing_low_points],
                y=[s["price"] for s in swing_low_points],
                mode="markers",
                name="قيعان",
                marker=dict(symbol="triangle-up", size=10, color="#26c6da"),
            ),
            row=1,
            col=1,
        )

    volume_color = np.where(clean_df["Close"] >= clean_df["Open"], "#26a69a", "#ef5350")
    fig.add_trace(
        go.Bar(
            x=clean_df["Date"],
            y=clean_df["Volume"],
            name="Volume",
            marker_color=volume_color,
            opacity=0.45,
        ),
        row=2,
        col=1,
    )

    highs = [s for s in swing_sequence if s["type"] == "high"]
    lows = [s for s in swing_sequence if s["type"] == "low"]
    last_date = clean_df["Date"].iloc[-1]
    future_date = next_trading_date(last_date, HOLD_DAYS, asset_type)

    if highs:
        last_high = highs[-1]
        fig.add_hline(
            y=last_high["price"],
            line_dash="dot",
            line_color="#ff7043",
            annotation_text="آخر قمة",
            annotation_position="top left",
            row=1,
            col=1,
        )

    if lows:
        last_low = lows[-1]
        fig.add_hline(
            y=last_low["price"],
            line_dash="dot",
            line_color="#26c6da",
            annotation_text="آخر قاع",
            annotation_position="bottom left",
            row=1,
            col=1,
        )

    fig.add_shape(
        type="rect",
        x0=last_date,
        x1=future_date,
        y0=forward_scenario["range_low"],
        y1=forward_scenario["range_high"],
        xref="x",
        yref="y",
        fillcolor="rgba(66, 165, 245, 0.07)",
        line=dict(width=0),
        layer="below",
    )

    fig.add_trace(
        go.Scatter(
            x=[last_date, future_date],
            y=[latest_close, forward_scenario["mid"]],
            mode="lines",
            name="سيناريو",
            line=dict(color="#42a5f5", width=1.5, dash="dash"),
        ),
        row=1,
        col=1,
    )

    if not pd.isna(forward_scenario["invalidation"]):
        fig.add_hline(
            y=forward_scenario["invalidation"],
            line_dash="dash",
            line_color="#fdd835",
            annotation_text="إلغاء السيناريو",
            annotation_position="bottom right",
            row=1,
            col=1,
        )

    fig.update_layout(
        height=780,
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        font=dict(color="#d1d4dc"),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=2, label="2Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ])
            ),
        ),
        xaxis2=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(
            side="right",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            fixedrange=False,
        ),
        yaxis2=dict(
            side="right",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.04)",
            title="Volume",
        ),
        bargap=0,
    )

    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
    st.caption(
        f"السيناريو ليس تنبؤاً معايراً ولا توصية. هو إسقاط بنيوي تقريبي لمدة {HOLD_DAYS} شمعة اعتماداً على الدورة، القمم/القيعان، الاختراق الكاذب، وATR."
    )


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
        st.write("جودة البيانات")
        st.write({
            "start": str(clean_df["Date"].iloc[0].date()),
            "end": str(clean_df["Date"].iloc[-1].date()),
            "rows": int(len(clean_df)),
            "zero_volume_pct": round(float(zero_volume_pct), 2),
            "missing_price_values": int(missing_price_values),
            "asset_type": asset_type,
        })

        st.write("كامل البيانات")
        st.json(fractal_all_stats)

        if fractal_all_trades.empty:
            st.write("لا توجد صفقات.")
        else:
            display_trades = fractal_all_trades.copy()
            display_trades["entry_date"] = display_trades["entry_date"].astype(str)
            display_trades["exit_date"] = display_trades["exit_date"].astype(str)
            st.dataframe(display_trades.tail(20), use_container_width=True)
