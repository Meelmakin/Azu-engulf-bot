"""
Structure Engulf Scanner Bot
=============================
Live version of the 1H engulf + 5m retest strategy validated in backtesting
(structure_engulf_backtest.py). Scans the 10 pairs that showed net-positive
expectancy, and sends a Telegram alert when a fresh, high-quality setup
triggers.

Strategy:
  1. Find a 1H engulfing candle.
  2. Score the leg into it for cleanliness (0-6). Only act on MIN_SCORE+.
  3. Watch for price to retrace into that 1H candle's zone.
  4. Alert when a same-direction engulfing candle forms on the 5m inside
     the zone (fires on the latest completed 5m candle only, so alerts
     are timely rather than historical).

Also sends a "Scanner alive" heartbeat message every run, summarizing
each pair's active structure zones, 1H trend, and session status, so
you know the bot is running even when nothing fires.

Setup:
  - GitHub repo secrets required: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  - Runs on a schedule via GitHub Actions (see structure_scanner.yml)
  - Keeps state/alerted.json to avoid sending the same setup twice;
    the workflow commits this file back to the repo after each run.
"""

import os
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

PAIRS = {
    "USOIL": "CL=F",
    "XAUUSD": "GC=F",
    "GER40": "^GDAXI",
    "US30": "^DJI",
    "US100": "^NDX",
    "US500": "^GSPC",
    "JP225": "^N225",
    "BTCUSDT": "BTC-USD",
    "SOLUSDT": "SOL-USD",
    "BNBUSDT": "BNB-USD",
}

MIN_SCORE = 5          # only alert 5/6 or 6/6 setups
TARGET_R = 3            # take-profit multiple (backtested best net expectancy)
LOOKBACK_LEGS = 8
MIN_BODY_RATIO = 0.60
MAX_CONSOL_RATIO = 0.35
MAX_OVERLAP = 0.30
RECENT_1H_CANDLES = 12   # how far back to look for a fresh 1H engulf
RETEST_MAX_BARS_5M = 48  # how many 5m bars to allow for the retest to form
DRIFT_GATE_R = 0.35      # skip alert if live price has moved this many R from entry
TREND_EMA_SPAN = 50      # 1H EMA span used for the heartbeat trend tag

STATE_PATH = "state/alerted.json"
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


# ---------------------------------------------------------------------------
# CANDLE HELPERS
# ---------------------------------------------------------------------------

def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def body(c):
    return abs(c["Close"] - c["Open"])

def rng(c):
    return max(c["High"] - c["Low"], 1e-9)

def is_bull_engulf(prev, cur):
    return (cur["Close"] > cur["Open"] and prev["Close"] < prev["Open"]
            and cur["Close"] >= prev["Open"] and cur["Open"] <= prev["Close"])

def is_bear_engulf(prev, cur):
    return (cur["Close"] < cur["Open"] and prev["Close"] > prev["Open"]
            and cur["Open"] >= prev["Close"] and cur["Close"] <= prev["Open"])


def structure_score(df, i, direction):
    start = max(0, i - LOOKBACK_LEGS)
    leg = df.iloc[start:i + 1]
    if len(leg) < 3:
        return 0

    checks = {}
    body_ratios = [body(c) / rng(c) for _, c in leg.iterrows()]
    checks["body_ratio"] = np.mean(body_ratios) >= MIN_BODY_RATIO

    consol = leg.iloc[-3:-1] if len(leg) >= 4 else leg.iloc[:1]
    consol_range = consol["High"].max() - consol["Low"].min() if len(consol) else 0
    leg_range = leg["High"].max() - leg["Low"].min()
    checks["consolidation"] = (consol_range / max(leg_range, 1e-9)) <= MAX_CONSOL_RATIO

    overlaps = []
    rows = leg.reset_index(drop=True)
    for j in range(1, len(rows)):
        prev_c, cur_c = rows.iloc[j - 1], rows.iloc[j]
        prev_top, prev_bot = max(prev_c["Open"], prev_c["Close"]), min(prev_c["Open"], prev_c["Close"])
        cur_top, cur_bot = max(cur_c["Open"], cur_c["Close"]), min(cur_c["Open"], cur_c["Close"])
        overlap_amt = max(0, min(prev_top, cur_top) - max(prev_bot, cur_bot))
        overlaps.append(overlap_amt / max(body(prev_c), 1e-9))
    checks["low_overlap"] = (np.mean(overlaps) <= MAX_OVERLAP) if overlaps else True

    highs, lows = leg["High"].values, leg["Low"].values
    if direction == "bull":
        checks["swing_clarity"] = np.mean(np.diff(highs) >= 0) >= 0.6 and np.mean(np.diff(lows) >= 0) >= 0.6
    else:
        checks["swing_clarity"] = np.mean(np.diff(highs) <= 0) >= 0.6 and np.mean(np.diff(lows) <= 0) >= 0.6

    avg_body = np.mean([body(c) for _, c in df.iloc[max(0, i - 10):i].iterrows()])
    checks["strong_engulf"] = body(df.iloc[i]) >= avg_body

    pre_engulf = leg.iloc[:-1]
    if direction == "bull":
        checks["clean_leg_low"] = df.iloc[i]["Low"] >= pre_engulf["Low"].min()
    else:
        checks["clean_leg_low"] = df.iloc[i]["High"] <= pre_engulf["High"].max()

    return sum(checks.values())


def get_1h_trend(df_1h):
    """Simple EMA-based trend tag for the heartbeat message."""
    span = min(TREND_EMA_SPAN, max(len(df_1h) - 1, 2))
    ema = df_1h["Close"].ewm(span=span).mean()
    return "up" if df_1h["Close"].iloc[-1] > ema.iloc[-1] else "down"


def count_active_zones(df_1h):
    """Number of qualifying (score >= MIN_SCORE) 1H engulfs in the recent
    lookback window, i.e. zones currently being watched for a retest."""
    count = 0
    start_i = max(1, len(df_1h) - RECENT_1H_CANDLES)
    for i in range(start_i, len(df_1h)):
        prev, cur = df_1h.iloc[i - 1], df_1h.iloc[i]
        direction = None
        if is_bull_engulf(prev, cur):
            direction = "bull"
        elif is_bear_engulf(prev, cur):
            direction = "bear"
        if direction is None:
            continue
        if structure_score(df_1h, i, direction) >= MIN_SCORE:
            count += 1
    return count


# ---------------------------------------------------------------------------
# STATE (dedup across runs)
# ---------------------------------------------------------------------------

def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return set(json.load(f))
    return set()

def save_state(alerted):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(sorted(alerted), f, indent=2)


# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured, skipping send. Message would be:\n", message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, data={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
    }, timeout=10)
    if resp.status_code != 200:
        print(f"Telegram send failed: {resp.status_code} {resp.text}")


# ---------------------------------------------------------------------------
# SCAN LOGIC
# ---------------------------------------------------------------------------

def scan_pair(name, ticker, alerted, stats):
    """Scans one pair for alertable setups, and returns its heartbeat info
    (independent of whether an alert fired)."""
    df_1h = _flatten(yf.download(ticker, period="14d", interval="1h", progress=False))
    df_5m = _flatten(yf.download(ticker, period="5d", interval="5m", progress=False))
    if df_1h.empty or df_5m.empty or len(df_1h) < LOOKBACK_LEGS + 2:
        print(f"[{name}] insufficient data, skipping")
        return {"name": name, "status": "no data"}

    live_price = df_5m["Close"].iloc[-1]
    now_utc = datetime.now(timezone.utc)
    in_session = 7 <= now_utc.hour < 20

    heartbeat = {
        "name": name,
        "zones": count_active_zones(df_1h),
        "trend": get_1h_trend(df_1h),
        "in_session": in_session,
    }

    # look at the most recent 1H candles for a fresh engulf
    start_i = max(1, len(df_1h) - RECENT_1H_CANDLES)
    for i in range(start_i, len(df_1h)):
        prev, cur = df_1h.iloc[i - 1], df_1h.iloc[i]
        direction = None
        if is_bull_engulf(prev, cur):
            direction = "bull"
        elif is_bear_engulf(prev, cur):
            direction = "bear"
        if direction is None:
            continue

        score = structure_score(df_1h, i, direction)
        if score < MIN_SCORE:
            continue

        zone_low, zone_high = cur["Low"], cur["High"]
        engulf_time = df_1h.index[i]

        # look for the retest + 5m engulf entry, most recent 5m candle only
        window = df_5m[df_5m.index > engulf_time].iloc[:RETEST_MAX_BARS_5M]
        if len(window) < 2:
            continue

        tapped = False
        entry_candle = None
        for j in range(1, len(window)):
            wprev, wcur = window.iloc[j - 1], window.iloc[j]
            if direction == "bull" and wcur["Low"] < zone_low:
                tapped = False
                break
            if direction == "bear" and wcur["High"] > zone_high:
                tapped = False
                break
            in_zone = wcur["Low"] <= zone_high and wcur["High"] >= zone_low
            if in_zone:
                tapped = True
            if tapped:
                if direction == "bull" and is_bull_engulf(wprev, wcur):
                    entry_candle = wcur
                elif direction == "bear" and is_bear_engulf(wprev, wcur):
                    entry_candle = wcur

        if entry_candle is None:
            continue

        entry_time = entry_candle.name
        # only alert if the entry trigger is the LATEST completed 5m candle
        if entry_time != df_5m.index[-1]:
            continue

        key = f"{name}_{direction}_{engulf_time.isoformat()}_{entry_time.isoformat()}"
        if key in alerted:
            continue

        entry = entry_candle["Close"]
        stop = zone_low if direction == "bull" else zone_high
        risk = abs(entry - stop)
        if risk == 0:
            continue
        target = entry + TARGET_R * risk if direction == "bull" else entry - TARGET_R * risk

        drift_r = abs(live_price - entry) / risk
        if drift_r > DRIFT_GATE_R:
            print(f"[{name}] setup found but drift {drift_r:.2f}R exceeds gate, skipping alert")
            alerted.add(key)
            continue

        session_tag = "in-session" if in_session else "off-session"
        arrow = "🟢 BUY" if direction == "bull" else "🔴 SELL"
        message = (
            f"*{arrow} — {name}*\n"
            f"Structure Engulf | Score {score}/6 | {session_tag}\n\n"
            f"Entry: `{entry:.5f}`\n"
            f"Stop: `{stop:.5f}`\n"
            f"Target ({TARGET_R}R): `{target:.5f}`\n"
            f"Live: `{live_price:.5f}`\n\n"
            f"1H zone: {zone_low:.5f} - {zone_high:.5f}\n"
            f"Engulf: {engulf_time} UTC"
        )
        print(f"[{name}] ALERT — {arrow} score {score}/6")
        send_telegram(message)
        alerted.add(key)
        stats["signals"] += 1

    return heartbeat


# ---------------------------------------------------------------------------
# HEARTBEAT
# ---------------------------------------------------------------------------

def format_heartbeat_line(info):
    if "status" in info:
        return f"{info['name']}: {info['status']}"
    session_tag = "in-session" if info["in_session"] else "off-session"
    return f"{info['name']}: {info['zones']} zones, 1H trend {info['trend']}, {session_tag}"


def send_heartbeat(heartbeats, signal_count):
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"Scanner alive {now_str}. Signals: {signal_count}"]
    lines += [format_heartbeat_line(h) for h in heartbeats]
    send_telegram("\n".join(lines))


def main():
    alerted = load_state()
    stats = {"signals": 0}
    heartbeats = []
    for name, ticker in PAIRS.items():
        try:
            info = scan_pair(name, ticker, alerted, stats)
        except Exception as e:
            print(f"[{name}] error: {e}")
            info = {"name": name, "status": "error"}
        heartbeats.append(info)
    save_state(alerted)
    send_heartbeat(heartbeats, stats["signals"])


if __name__ == "__main__":
    main()
              
