import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

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

# قائمة بحث مقفلة لصيد الانفجارات السعرية.
# الهدف ليس وعداً بالصعود، بل فلترة أصول صغيرة/متحركة عندها سيناريو بنيوي لصعود مرتفع يبدأ من 50% ومفتوح للأعلى.
EXPLOSION_MIN_UPSIDE_PCT = 50.0
EXPLOSION_UNIVERSE = [
    # أمثلة أسهم صغيرة/مضاربية عالية المخاطرة؛ بعضها قد لا يعطي بيانات دائماً من Yahoo.
    ("BNAI", "Brand Engagement Network", "سهم صغير"),
    ("MLGO", "MicroAlgo", "سهم صغير"),
    ("HOLO", "MicroCloud Hologram", "سهم صغير"),
    ("LIDR", "AEye", "سهم صغير"),
    ("KULR", "KULR Technology", "سهم صغير"),
    ("BBAI", "BigBear.ai", "AI Small Cap"),
    ("SOUN", "SoundHound AI", "AI Small Cap"),
    ("RGTI", "Rigetti Computing", "Quantum"),
    ("QBTS", "D-Wave Quantum", "Quantum"),
    ("QUBT", "Quantum Computing", "Quantum"),
    ("IONQ", "IonQ", "Quantum"),
    ("LAES", "SEALSQ", "سهم صغير"),
    ("HUMA", "Humacyte", "Biotech"),
    ("ACHR", "Archer Aviation", "Growth"),
    ("JOBY", "Joby Aviation", "Growth"),
    ("LUNR", "Intuitive Machines", "Space"),
    ("RKLB", "Rocket Lab", "Space"),
    ("ASTS", "AST SpaceMobile", "Space"),
    ("OPEN", "Opendoor", "High Beta"),
    ("SOFI", "SoFi", "High Beta"),
    ("RIVN", "Rivian", "High Beta"),
    ("LCID", "Lucid", "High Beta"),
    ("PLTR", "Palantir", "AI"),
    ("MARA", "MARA Holdings", "Bitcoin Miner"),
    ("RIOT", "Riot Platforms", "Bitcoin Miner"),
    ("CIFR", "Cipher Mining", "Bitcoin Miner"),
    ("WULF", "TeraWulf", "Bitcoin Miner"),
    ("BITF", "Bitfarms", "Bitcoin Miner"),
    ("HIVE", "HIVE Digital", "Bitcoin Miner"),
    ("COIN", "Coinbase", "Crypto Equity"),
    ("MSTR", "MicroStrategy", "Bitcoin Proxy"),
    # كريبتو عالي الحركة
    ("BTC-USD", "Bitcoin", "كريبتو"),
    ("ETH-USD", "Ethereum", "كريبتو"),
    ("SOL-USD", "Solana", "كريبتو"),
    ("SUI-USD", "Sui", "كريبتو"),
    ("AVAX-USD", "Avalanche", "كريبتو"),
    ("LINK-USD", "Chainlink", "كريبتو"),
    ("DOGE-USD", "Dogecoin", "كريبتو"),
    ("ADA-USD", "Cardano", "كريبتو"),
    ("XRP-USD", "XRP", "كريبتو"),
]

# إبقاء الاسم القديم كـ alias حتى لا ينكسر أي جزء قديم في الصفحة.
HIGH_UPSIDE_MIN_PCT = EXPLOSION_MIN_UPSIDE_PCT
OPPORTUNITY_UNIVERSE = EXPLOSION_UNIVERSE


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



def clamp_entry_zone_around_midpoint(entry_low, entry_high, max_pct: float = 0.02):
    """Keep the ideal entry zone tight: no more than +/- max_pct around its midpoint."""
    if pd.isna(entry_low) or pd.isna(entry_high):
        return entry_low, entry_high
    low = float(entry_low)
    high = float(entry_high)
    if low <= 0 or high <= 0:
        return low, high
    low, high = sorted([low, high])
    midpoint = (low + high) / 2
    max_low = midpoint * (1 - max_pct)
    max_high = midpoint * (1 + max_pct)
    return max(low, max_low), min(high, max_high)


def format_entry_note(entry_low, entry_high, suffix: str = "") -> str:
    if pd.isna(entry_low) or pd.isna(entry_high):
        return "نقطة الدخول المثالية: لا توجد منطقة دخول صالحة الآن؛ انتظر تحول البنية إلى صاعدة."
    return f"نقطة الدخول المثالية: انتظر قرب {entry_low:.2f} - {entry_high:.2f}، نطاق ضيق لا يتجاوز ±2%{suffix}."

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

    ma20 = float(latest["ma20"]) if "ma20" in latest and not pd.isna(latest["ma20"]) else last_close
    ma50 = float(latest["ma50"]) if "ma50" in latest and not pd.isna(latest["ma50"]) else last_close

    if structure == "بنية صاعدة" and quality in ["قوية", "متوسطة"] and not pd.isna(position) and position <= 35 and fb_risk != "مرتفع":
        label = "امتداد صاعد مشروط"
        bias = "إيجابي"
        low = max(last_close - 1.2 * atr, last_low)
        high = max(last_close + 2.2 * atr, last_high)
        invalidation = min(last_close - 1.5 * atr, last_low)
        entry_low = max(last_low, last_close - 0.8 * atr)
        entry_high = last_close + 0.3 * atr
        entry_low, entry_high = clamp_entry_zone_around_midpoint(entry_low, entry_high, max_pct=0.02)
        entry_note = format_entry_note(entry_low, entry_high)
        note = "السيناريو الصاعد يبقى قائماً ما دام السعر لا يكسر منطقة الإلغاء."
    elif structure == "بنية صاعدة" and (fb_risk == "مرتفع" or (not pd.isna(position) and position >= 75)):
        label = "تصحيح أو تهدئة محتملة"
        bias = "حذر"
        low = max(last_close - 2.0 * atr, last_low)
        high = last_close + 0.8 * atr
        invalidation = max(last_close + 1.2 * atr, last_high)
        pullback_low = max(last_low, min(ma20, ma50, last_close - 2.2 * atr))
        pullback_high = max(pullback_low, min(last_close - 1.0 * atr, max(ma20, ma50)))
        entry_low, entry_high = sorted([float(pullback_low), float(pullback_high)])
        entry_low, entry_high = clamp_entry_zone_around_midpoint(entry_low, entry_high, max_pct=0.02)
        entry_note = format_entry_note(entry_low, entry_high)
        note = "السعر متأخر داخل الدورة أو توجد علامة فشل اختراق؛ الأفضل انتظار قاع جديد."
    elif structure == "بنية هابطة":
        label = "ضغط هابط محتمل"
        bias = "سلبي"
        low = min(last_close - 2.0 * atr, last_low)
        high = last_close + 1.0 * atr
        invalidation = max(last_close + 1.5 * atr, last_high)
        entry_low = np.nan
        entry_high = np.nan
        entry_note = "نقطة الدخول المثالية: لا توجد منطقة دخول صالحة الآن؛ انتظر تحول البنية إلى صاعدة."
        note = "البنية الهابطة تلغي فكرة المتابعة حتى يظهر تحول واضح."
    else:
        label = "نطاق انتظار"
        bias = "محايد"
        low = last_close - 1.5 * atr
        high = last_close + 1.5 * atr
        invalidation = np.nan
        entry_low = max(last_low, last_close - 1.2 * atr)
        entry_high = last_close - 0.4 * atr
        entry_low, entry_high = sorted([float(entry_low), float(entry_high)])
        entry_low, entry_high = clamp_entry_zone_around_midpoint(entry_low, entry_high, max_pct=0.02)
        entry_note = format_entry_note(entry_low, entry_high, suffix=" فقط إذا تحسنت البنية")
        note = "لا توجد أفضلية بنيوية كافية؛ الانتظار أفضل من التوقع القسري."

    low, high = sorted([float(low), float(high)])

    return {
        "label": label,
        "bias": bias,
        "range_low": low,
        "range_high": high,
        "mid": (low + high) / 2,
        "invalidation": invalidation,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "entry_note": entry_note,
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


def safe_ratio(numerator: float, denominator: float, fallback: float = np.nan) -> float:
    try:
        numerator = float(numerator)
        denominator = float(denominator)
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return fallback
        return numerator / denominator
    except Exception:
        return fallback


def analyze_explosion_candidate(ticker: str, name: str, kind: str, min_upside_pct: float = EXPLOSION_MIN_UPSIDE_PCT):
    """
    صيد انفجار سعري محتمل مثل أسهم BNAI قبل القفزة.
    لا يقول "سوف يصعد"، بل يفلتر مرشحاً إذا توفرت: ضغط/تجميع، بداية دورة، حجم يتحسن، مساحة صعود مفتوحة فوق 50%.
    """
    candidate_asset_type = infer_asset_type(ticker)
    data_candidate = load_data(ticker, PERIOD, INTERVAL)

    if data_candidate.empty or len(data_candidate) < 180:
        return None

    df_candidate = add_market_features(data_candidate, asset_type=candidate_asset_type)
    df_candidate = detect_fractal_swings(df_candidate, left=PIVOT_WINDOW, right=PIVOT_WINDOW)
    clean_candidate = df_candidate.dropna().reset_index(drop=True)

    if len(clean_candidate) < 160:
        return None

    raw_candidate_swings = build_swing_sequence(clean_candidate, confirmation_delay=PIVOT_WINDOW)
    candidate_swings = filter_swings_by_atr(raw_candidate_swings, clean_candidate, min_atr_multiple=MIN_SWING_ATR)

    if len(candidate_swings) < 4:
        return None

    candidate_index = len(clean_candidate) - 1
    candidate_structure = analyze_structure_from_swings(clean_candidate, candidate_swings, current_index=candidate_index)
    candidate_cycle = compute_cycle_quality_at_index(clean_candidate, candidate_swings, current_index=candidate_index)
    candidate_false_breakout = detect_false_breakout(clean_candidate, candidate_swings)
    candidate_scenario = build_forward_scenario(
        clean_candidate,
        candidate_swings,
        candidate_structure,
        candidate_cycle,
        candidate_false_breakout,
        horizon=HOLD_DAYS
    )

    latest_candidate = clean_candidate.iloc[-1]
    last_close_candidate = float(latest_candidate["Close"])
    if last_close_candidate <= 0:
        return None

    atr_candidate = float(latest_candidate["atr14"]) if not pd.isna(latest_candidate["atr14"]) and latest_candidate["atr14"] > 0 else last_close_candidate * 0.04

    structure = candidate_structure.get("structure_trend", "")
    cycle_quality = candidate_cycle.get("cycle_quality", "")
    cycle_position = candidate_cycle.get("cycle_position", np.nan)
    cycle_phase = candidate_cycle.get("cycle_phase", "غير متاح")
    breakout_risk = candidate_false_breakout.get("risk", "غير متاح")

    # لا نريد مرشحين مكسورين بنيوياً أو في نهاية دورة واضحة.
    if structure != "بنية صاعدة":
        return None
    if cycle_quality not in ["قوية", "متوسطة"]:
        return None
    if breakout_risk == "مرتفع":
        return None
    if not pd.isna(cycle_position) and float(cycle_position) > 65:
        return None

    entry_low = candidate_scenario.get("entry_low", np.nan)
    entry_high = candidate_scenario.get("entry_high", np.nan)
    if pd.isna(entry_low) or pd.isna(entry_high):
        return None

    entry_low = float(entry_low)
    entry_high = float(entry_high)
    entry_low, entry_high = clamp_entry_zone_around_midpoint(entry_low, entry_high, max_pct=0.02)
    entry_mid = (entry_low + entry_high) / 2
    if entry_mid <= 0:
        return None

    lookback_252 = clean_candidate.tail(min(252, len(clean_candidate)))
    lookback_126 = clean_candidate.tail(min(126, len(clean_candidate)))
    lookback_60 = clean_candidate.tail(min(60, len(clean_candidate)))
    lookback_20 = clean_candidate.tail(min(20, len(clean_candidate)))

    high_52w = float(lookback_252["High"].max())
    high_6m = float(lookback_126["High"].max())
    low_6m = float(lookback_126["Low"].min())
    recent_high_20 = float(lookback_20["High"].max())

    # مساحة الصعود المفتوحة: أولاً إلى قمة سنة/ستة أشهر، ثم هدف بنيوي بدورة ممتدة.
    highs = [s for s in candidate_swings if s["type"] == "high"]
    lows = [s for s in candidate_swings if s["type"] == "low"]
    if not highs or not lows:
        return None

    last_high = float(highs[-1]["price"])
    last_low = float(lows[-1]["price"])
    cycle_amplitude = max(abs(last_high - last_low), atr_candidate * 5)

    historical_target = max(high_52w, high_6m, last_high)
    structural_target = max(historical_target, last_close_candidate + cycle_amplitude * 2.0)

    # للأسهم الصغيرة جداً، إذا كان السعر تحت القمم السابقة بكثير، نسمح بهدف مفتوح أعلى من القمة السابقة.
    if kind != "كريبتو" and last_close_candidate <= 15:
        structural_target = max(structural_target, last_close_candidate + cycle_amplitude * 3.0)

    upside_pct = (structural_target / entry_mid - 1) * 100
    if upside_pct < min_upside_pct:
        return None

    # إشارات ما قبل الانفجار: ضغط/تجميع، اقتراب من اختراق، تحسن حجم، توسع تذبذب.
    range_6m_pct = safe_ratio(high_6m - low_6m, max(low_6m, 0.01), 0.0) * 100
    distance_to_20d_high_pct = (recent_high_20 / last_close_candidate - 1) * 100
    distance_to_52w_high_pct = (high_52w / last_close_candidate - 1) * 100

    vol_now = float(lookback_20["Volume"].tail(5).mean()) if "Volume" in lookback_20.columns else 0.0
    vol_base = float(lookback_60["Volume"].head(max(5, len(lookback_60) - 20)).mean()) if "Volume" in lookback_60.columns and len(lookback_60) > 25 else 0.0
    volume_ratio = safe_ratio(vol_now, vol_base, 0.0)

    atr_pct_now = safe_ratio(atr_candidate, last_close_candidate, 0.0) * 100
    atr_pct_60 = float((lookback_60["atr14"] / lookback_60["Close"]).replace([np.inf, -np.inf], np.nan).dropna().median() * 100) if "atr14" in lookback_60.columns else atr_pct_now
    volatility_expansion = safe_ratio(atr_pct_now, atr_pct_60, 1.0)

    compression_score = 0
    if range_6m_pct < 120:
        compression_score += 1
    if distance_to_20d_high_pct <= 8:
        compression_score += 1
    if distance_to_52w_high_pct >= 50:
        compression_score += 1
    if volume_ratio >= 1.2 or kind == "كريبتو":
        compression_score += 1
    if volatility_expansion >= 1.05:
        compression_score += 1
    if last_close_candidate <= 20 or kind == "كريبتو":
        compression_score += 1

    if compression_score < 3:
        return None

    if len(candidate_swings) >= 2:
        try:
            cycle_bars = int(abs(candidate_swings[-1]["index"] - candidate_swings[-2]["index"]))
        except Exception:
            cycle_bars = HOLD_DAYS * 2
    else:
        cycle_bars = HOLD_DAYS * 2

    horizon_bars = int(np.clip(cycle_bars * 1.6, 20, 160))
    entry_date = next_trading_date(clean_candidate["Date"].iloc[-1], max(2, horizon_bars // 5), candidate_asset_type)
    end_date = next_trading_date(clean_candidate["Date"].iloc[-1], horizon_bars, candidate_asset_type)

    invalidation = candidate_scenario.get("invalidation", np.nan)
    if pd.isna(invalidation):
        invalidation = max(0.01, entry_low - 1.3 * atr_candidate)
    invalidation = float(invalidation)

    risk_pct = max(0.0, (entry_mid / invalidation - 1) * 100) if invalidation > 0 and invalidation < entry_mid else np.nan
    reward_risk_score = upside_pct / max(risk_pct, 3.0) if not pd.isna(risk_pct) else upside_pct / 8

    explosion_score = (
        compression_score * 18
        + min(upside_pct, 400) * 0.35
        + min(max(volume_ratio, 0), 5) * 8
        + min(max(volatility_expansion, 0), 4) * 6
        + min(reward_risk_score, 50)
    )

    reasons = []
    if distance_to_52w_high_pct >= 50:
        reasons.append("مساحة صعود كبيرة مقابل القمم السابقة")
    if distance_to_20d_high_pct <= 8:
        reasons.append("قريب من اختراق قصير")
    if volume_ratio >= 1.2:
        reasons.append("تحسن حجم التداول")
    if volatility_expansion >= 1.05:
        reasons.append("بداية توسع في الحركة")
    if cycle_phase in ["بداية دورة", "منتصف دورة"]:
        reasons.append("الدورة ليست متأخرة")
    if not reasons:
        reasons.append("تجميع بنيوي مع صعود محتمل فوق 50%")

    return {
        "الرمز": ticker,
        "الاسم": name,
        "النوع": kind,
        "نسبة الصعود المفتوحة": f"{upside_pct:.1f}%+",
        "منطقة الدخول": f"{entry_low:.2f} - {entry_high:.2f}",
        "هدف بنيوي مفتوح": f"{structural_target:.2f}",
        "إلغاء السيناريو": f"{invalidation:.2f}",
        "نافذة المراقبة": f"{entry_date.date()} إلى {end_date.date()}",
        "مرحلة الدورة": cycle_phase,
        "خطر فشل الاختراق": breakout_risk,
        "سبب الظهور": " + ".join(reasons[:3]),
        "explosion_score": float(explosion_score),
        "upside_numeric": float(upside_pct),
    }


# إبقاء اسم الدالة القديم كـ wrapper حتى لا ينكسر أي استدعاء قديم.
def analyze_high_upside_candidate(ticker: str, name: str, kind: str, min_upside_pct: float = EXPLOSION_MIN_UPSIDE_PCT):
    return analyze_explosion_candidate(ticker, name, kind, min_upside_pct=min_upside_pct)


@st.cache_data(show_spinner=False, ttl=60 * 30)
def scan_explosion_hunter(min_upside_pct: float = EXPLOSION_MIN_UPSIDE_PCT) -> pd.DataFrame:
    rows = []
    for ticker, name, kind in EXPLOSION_UNIVERSE:
        try:
            result = analyze_explosion_candidate(ticker, name, kind, min_upside_pct=min_upside_pct)
            if result is not None:
                rows.append(result)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(["explosion_score", "upside_numeric"], ascending=[False, False]).reset_index(drop=True)
    return result_df


@st.cache_data(show_spinner=False, ttl=60 * 30)
def scan_high_upside_opportunities(min_upside_pct: float = EXPLOSION_MIN_UPSIDE_PCT) -> pd.DataFrame:
    return scan_explosion_hunter(min_upside_pct=min_upside_pct)


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

tab_decision, tab_chart, tab_test, tab_opportunities = st.tabs(["القرار", "الشارت", "الاختبار", "صيد الانفجارات"])

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
    st.success(f"**{forward_scenario['entry_note']}**")
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

    # الشارت هنا يركز على آخر منطقة فعلية: دورة حالية + دورة توقع + إشارات دخول/خروج مؤكدة.
    display_bars = min(len(clean_df), 260)
    focus_start_date = clean_df["Date"].iloc[-display_bars]
    visible_df = clean_df.tail(display_bars).copy()

    last_date = clean_df["Date"].iloc[-1]
    future_date = next_trading_date(last_date, HOLD_DAYS, asset_type)
    range_end_date = next_trading_date(future_date, 8, asset_type)

    cycle_start = swing_sequence[-2] if len(swing_sequence) >= 2 else None
    cycle_end = swing_sequence[-1] if len(swing_sequence) >= 1 else None
    cycle_start_date = cycle_start["date"] if cycle_start else visible_df["Date"].iloc[0]
    cycle_end_date = cycle_end["date"] if cycle_end else last_date

    visible_low = float(visible_df["Low"].min())
    visible_high = float(visible_df["High"].max())
    price_pad = max((visible_high - visible_low) * 0.08, float(latest["atr14"]) if not pd.isna(latest["atr14"]) else latest_close * 0.02)
    y_min = visible_low - price_pad
    y_max = visible_high + price_pad

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        row_heights=[0.78, 0.22],
    )

    # تظليل الدورة الحالية قبل الشموع، حتى تكون الخلفية واضحة وغير مزعجة.
    fig.add_shape(
        type="rect",
        x0=max(pd.to_datetime(cycle_start_date), pd.to_datetime(focus_start_date)),
        x1=min(pd.to_datetime(last_date), pd.to_datetime(range_end_date)),
        y0=y_min,
        y1=y_max,
        xref="x",
        yref="y",
        fillcolor="rgba(255,255,255,0.035)",
        line=dict(color="rgba(255,255,255,0.11)", width=1),
        layer="below",
    )

    # دورة التوقع القادمة: واضحة على يمين آخر شمعة، لكنها ليست وعداً سعرياً.
    fig.add_shape(
        type="rect",
        x0=last_date,
        x1=future_date,
        y0=forward_scenario["range_low"],
        y1=forward_scenario["range_high"],
        xref="x",
        yref="y",
        fillcolor="rgba(47,128,237,0.11)",
        line=dict(color="rgba(47,128,237,0.45)", width=1, dash="dot"),
        layer="below",
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
            whiskerwidth=0.35,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(go.Scatter(x=clean_df["Date"], y=clean_df["ma20"], mode="lines", name="MA 20", line=dict(width=1.15, color="#2f80ed")), row=1, col=1)
    fig.add_trace(go.Scatter(x=clean_df["Date"], y=clean_df["ma50"], mode="lines", name="MA 50", line=dict(width=1.15, color="#f2c94c")), row=1, col=1)
    fig.add_trace(go.Scatter(x=clean_df["Date"], y=clean_df["ma200"], mode="lines", name="MA 200", line=dict(width=1.25, color="#bb6bd9")), row=1, col=1)

    # ZigZag بنيوي للمنطقة الظاهرة فقط.
    recent_structural_swings = [s for s in swing_sequence if pd.to_datetime(s["date"]) >= pd.to_datetime(focus_start_date)]
    if len(recent_structural_swings) < 2:
        recent_structural_swings = swing_sequence[-18:]

    if len(recent_structural_swings) >= 2:
        fig.add_trace(
            go.Scatter(
                x=[s["date"] for s in recent_structural_swings],
                y=[s["price"] for s in recent_structural_swings],
                mode="lines+markers",
                name="ZigZag",
                line=dict(width=1.8, color="#f2c94c"),
                marker=dict(size=5, color="#f2c94c"),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra>ZigZag</extra>",
            ),
            row=1,
            col=1,
        )

    # حجم التداول.
    volume_color = np.where(clean_df["Close"] >= clean_df["Open"], "rgba(38,166,154,0.45)", "rgba(239,83,80,0.45)")
    fig.add_trace(
        go.Bar(
            x=clean_df["Date"],
            y=clean_df["Volume"],
            name="Volume",
            marker_color=volume_color,
            opacity=0.55,
            hovertemplate="%{x|%Y-%m-%d}<br>Volume: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    highs = [s for s in swing_sequence if s["type"] == "high"]
    lows = [s for s in swing_sequence if s["type"] == "low"]

    if highs:
        last_high = highs[-1]
        fig.add_hline(
            y=last_high["price"],
            line_dash="dot",
            line_color="rgba(255,112,67,0.95)",
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
            line_color="rgba(38,198,218,0.95)",
            annotation_text="آخر قاع",
            annotation_position="bottom left",
            row=1,
            col=1,
        )

    # بداية ونهاية الدورة الحالية.
    fig.add_vline(
        x=cycle_start_date,
        line_dash="dash",
        line_color="rgba(255,255,255,0.55)",
        annotation_text="بداية الدورة",
        annotation_position="top left",
        row=1,
        col=1,
    )

    fig.add_vline(
        x=cycle_end_date,
        line_dash="dash",
        line_color="rgba(255,255,255,0.34)",
        annotation_text="آخر تأكيد",
        annotation_position="top right",
        row=1,
        col=1,
    )

    fig.add_vline(
        x=future_date,
        line_dash="dot",
        line_color="rgba(47,128,237,0.65)",
        annotation_text="نهاية دورة التوقع",
        annotation_position="top right",
        row=1,
        col=1,
    )

    fig.add_annotation(
        x=last_date,
        y=y_max,
        text="الدورة الحالية",
        showarrow=False,
        font=dict(color="#d1d4dc", size=11),
        bgcolor="rgba(11,15,25,0.72)",
        bordercolor="rgba(255,255,255,0.18)",
        xanchor="right",
        yanchor="top",
        row=1,
        col=1,
    )

    fig.add_annotation(
        x=future_date,
        y=forward_scenario["range_high"],
        text="دورة التوقع",
        showarrow=False,
        font=dict(color="#9ec5ff", size=11),
        bgcolor="rgba(11,15,25,0.78)",
        bordercolor="rgba(47,128,237,0.45)",
        xanchor="right",
        yanchor="bottom",
        row=1,
        col=1,
    )

    # منطقة الدخول المثالية على آخر منطقة فقط.
    if not pd.isna(forward_scenario.get("entry_low", np.nan)) and not pd.isna(forward_scenario.get("entry_high", np.nan)):
        entry_low = float(forward_scenario["entry_low"])
        entry_high = float(forward_scenario["entry_high"])
        entry_x0 = clean_df["Date"].iloc[-min(len(clean_df), 70)]
        fig.add_shape(
            type="rect",
            x0=entry_x0,
            x1=future_date,
            y0=entry_low,
            y1=entry_high,
            xref="x",
            yref="y",
            fillcolor="rgba(38,166,154,0.13)",
            line=dict(color="rgba(38,166,154,0.80)", width=1),
            layer="below",
        )
        fig.add_annotation(
            x=future_date,
            y=(entry_low + entry_high) / 2,
            text="منطقة دخول مثالية",
            showarrow=False,
            font=dict(color="#80cbc4", size=11),
            bgcolor="rgba(11,15,25,0.82)",
            bordercolor="rgba(38,166,154,0.65)",
            xanchor="right",
            yanchor="middle",
            row=1,
            col=1,
        )

    # إذا لم توجد منطقة دخول صالحة، نعرض مساراً بنيوياً عاماً فقط بدون مثلث دخول/خروج.
    no_forecast_entry = pd.isna(forward_scenario.get("entry_low", np.nan)) or pd.isna(forward_scenario.get("entry_high", np.nan))
    if no_forecast_entry:
        fig.add_trace(
            go.Scatter(
                x=[last_date, future_date],
                y=[latest_close, forward_scenario["mid"]],
                mode="lines+markers",
                name="سيناريو بنيوي",
                line=dict(color="#2f80ed", width=1.5, dash="dot"),
                marker=dict(size=5, color="#2f80ed"),
                hovertemplate="سيناريو بنيوي<br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if not pd.isna(forward_scenario["invalidation"]):
        fig.add_hline(
            y=forward_scenario["invalidation"],
            line_dash="dash",
            line_color="rgba(253,216,53,0.95)",
            annotation_text="إلغاء السيناريو",
            annotation_position="bottom right",
            row=1,
            col=1,
        )

    # إشارات التوقع القادمة: المثلث الأخضر والأحمر هنا ليسا من الباك تست؛ هما نقاط مستقبلية محسوبة من دورة التوقع.
    forecast_entry_valid = (
        not pd.isna(forward_scenario.get("entry_low", np.nan))
        and not pd.isna(forward_scenario.get("entry_high", np.nan))
    )

    if forecast_entry_valid:
        forecast_entry_price = (float(forward_scenario["entry_low"]) + float(forward_scenario["entry_high"])) / 2
        forecast_entry_date = next_trading_date(last_date, max(2, HOLD_DAYS // 3), asset_type)
        forecast_exit_date = future_date

        # الخروج المتوقع هو الهدف البنيوي داخل دورة التوقع، وليس صفقة مؤكدة.
        if forward_scenario.get("bias") == "إيجابي":
            forecast_exit_price = float(forward_scenario["range_high"])
        elif forward_scenario.get("bias") == "حذر":
            forecast_exit_price = max(float(forward_scenario["range_high"]), forecast_entry_price * 1.02)
        else:
            forecast_exit_price = float(forward_scenario["mid"])

        # خط توقع بثلاث نقاط: الآن -> دخول متوقع -> خروج متوقع.
        fig.add_trace(
            go.Scatter(
                x=[last_date, forecast_entry_date, forecast_exit_date],
                y=[latest_close, forecast_entry_price, forecast_exit_price],
                mode="lines",
                name="مسار التوقع",
                line=dict(color="#2f80ed", width=2.2, dash="dot"),
                hovertemplate="مسار التوقع<br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[forecast_entry_date],
                y=[forecast_entry_price],
                mode="markers+text",
                name="دخول متوقع",
                marker=dict(
                    symbol="triangle-up",
                    size=42,
                    color="#00e676",
                    line=dict(color="#ffffff", width=3),
                ),
                text=["دخول"],
                textposition="bottom center",
                textfont=dict(color="#00e676", size=15),
                hovertemplate=(
                    "دخول متوقع<br>"
                    "%{x|%Y-%m-%d}<br>"
                    "السعر التقريبي: %{y:.2f}<br>"
                    "حسب منطقة الدخول المثالية ±2%<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[forecast_exit_date],
                y=[forecast_exit_price],
                mode="markers+text",
                name="خروج متوقع",
                marker=dict(
                    symbol="triangle-down",
                    size=46,
                    color="#ff1744",
                    line=dict(color="#ffffff", width=3),
                ),
                text=["خروج"],
                textposition="top center",
                textfont=dict(color="#ff5252", size=15),
                hovertemplate=(
                    "خروج متوقع<br>"
                    "%{x|%Y-%m-%d}<br>"
                    "السعر التقريبي: %{y:.2f}<br>"
                    "حسب نهاية دورة التوقع<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        # خطوط عمودية خفيفة لتوضيح يوم الدخول ويوم الخروج داخل المستقبل.
        fig.add_vline(
            x=forecast_entry_date,
            line_dash="dot",
            line_color="rgba(0,230,118,0.55)",
            annotation_text="دخول متوقع",
            annotation_position="bottom right",
            row=1,
            col=1,
        )
        fig.add_vline(
            x=forecast_exit_date,
            line_dash="dot",
            line_color="rgba(255,23,68,0.60)",
            annotation_text="خروج متوقع",
            annotation_position="top right",
            row=1,
            col=1,
        )

        y_min = min(y_min, forecast_entry_price - price_pad * 0.7, forecast_exit_price - price_pad * 0.45)
        y_max = max(y_max, forecast_entry_price + price_pad * 0.45, forecast_exit_price + price_pad * 0.7)

    rangebreaks = []
    if asset_type == "stock":
        rangebreaks = [dict(bounds=["sat", "mon"])]

    fig.update_layout(
        height=780,
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        font=dict(color="#d1d4dc", size=12),
        hovermode="x unified",
        dragmode="pan",
        margin=dict(l=8, r=62, t=42, b=36),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.015,
            xanchor="right",
            x=1,
            bgcolor="rgba(11,15,25,0.35)",
        ),
        xaxis=dict(
            range=[focus_start_date, range_end_date],
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.055)",
            tickformat="%b %Y",
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="rgba(255,255,255,0.25)",
            spikethickness=1,
            rangebreaks=rangebreaks,
            rangeselector=dict(
                x=0,
                y=1.08,
                bgcolor="rgba(255,255,255,0.08)",
                activecolor="rgba(47,128,237,0.55)",
                buttons=list([
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=2, label="2Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ]),
            ),
        ),
        xaxis2=dict(
            range=[focus_start_date, range_end_date],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.035)",
            tickformat="%Y-%m-%d",
            rangebreaks=rangebreaks,
        ),
        yaxis=dict(
            range=[y_min, y_max],
            side="right",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.055)",
            fixedrange=False,
            tickformat=",.2f",
        ),
        yaxis2=dict(
            side="right",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.035)",
            fixedrange=True,
            title="",
            showticklabels=True,
        ),
        bargap=0,
        uirevision=f"{symbol}-{asset_type}",
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(showline=True, linewidth=1, linecolor="rgba(255,255,255,0.12)")

    chart_config = {
        "scrollZoom": True,
        "displaylogo": False,
        "displayModeBar": True,
        "doubleClick": "reset+autosize",
        "responsive": True,
        "modeBarButtonsToRemove": [
            "select2d",
            "lasso2d",
            "autoScale2d",
            "zoom2d",
        ],
    }

    st.plotly_chart(fig, use_container_width=True, config=chart_config)
    st.caption(
        "الشارت يعرض الدورة الحالية، دورة التوقع، منطقة الدخول المثالية، ومثلث الدخول المتوقع وسهم الخروج المتوقع. "
        "استخدم عجلة الماوس للتكبير، واسحب يمين/يسار للتحريك. التوقع سيناريو بنيوي وليس ضماناً."
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


with tab_opportunities:
    st.subheader("صيد الانفجارات السعرية")
    st.caption("قائمة مستقلة تبحث عن أسهم/كريبتو عندها سيناريو انفجار سعري محتمل يبدأ من 50% ومفتوح للأعلى. ليست وعداً بالصعود ولا توصية شراء.")

    st.caption("طريقة التحديث: عند الضغط على زر الفحص يتم تحديث البيانات وإعادة بناء القائمة.")

    last_update = st.session_state.get("explosion_hunter_last_update")
    if last_update:
        st.caption(f"آخر فحص: {last_update}")

    if st.button("تحديث وفحص الآن", use_container_width=True):
        with st.spinner("جاري تحديث البيانات وفحص المرشحين... قد يستغرق قليلاً"):
            try:
                load_data.clear()
                scan_explosion_hunter.clear()
                scan_high_upside_opportunities.clear()
            except Exception:
                pass
            st.session_state["explosion_hunter"] = scan_explosion_hunter(EXPLOSION_MIN_UPSIDE_PCT)
            st.session_state["explosion_hunter_last_update"] = datetime.now().strftime("%Y-%m-%d %I:%M %p")

    explosion_df = st.session_state.get("explosion_hunter", pd.DataFrame())

    if explosion_df.empty:
        st.warning("لا يوجد مرشح انفجار سعري صالح الآن حسب فلتر 50%+. عدم ظهور أسماء أفضل من إظهار فرص وهمية.")
    else:
        display_cols = [
            "الرمز",
            "الاسم",
            "النوع",
            "نسبة الصعود المفتوحة",
            "منطقة الدخول",
            "هدف بنيوي مفتوح",
            "إلغاء السيناريو",
            "نافذة المراقبة",
            "مرحلة الدورة",
            "خطر فشل الاختراق",
            "سبب الظهور",
        ]
        st.dataframe(
            explosion_df[display_cols],
            use_container_width=True,
            hide_index=True,
        )
        top_row = explosion_df.iloc[0]
        st.success(
            f"أقوى مرشح الآن: {top_row['الرمز']} — صعود مفتوح {top_row['نسبة الصعود المفتوحة']}، "
            f"ومنطقة الدخول {top_row['منطقة الدخول']}، ونافذة المراقبة {top_row['نافذة المراقبة']}."
        )
        st.caption("اقرأ هذه القائمة كقائمة مراقبة. القرار النهائي يبقى من صفحة القرار والشارت لكل رمز، مع إلغاء السيناريو واضح.")
