"""
Structure Engulf Scanner Bot
=============================
Live version of the 1H engulf + 5m retest strategy validated in backtesting
(structure_engulf_backtest.py). Scans the 10 pairs that showed net-positive
expectancy, and sends a Telegram alert (with a per-rule checklist) when a
fresh, high-quality setup triggers.

Also:
  - Messages are sent ONLY when a signal fires (plus replies to commands you
    type). The "Scanner alive" heartbeat and daily/weekly summaries are
    switched off by default (SEND_HEARTBEAT / SEND_SUMMARIES below).
  - Polls for Telegram commands once per run (fits the existing scheduled
    GitHub Actions cadence rather than needing an always-on process):
      /calc <account> <risk%> <entry> <stop> [contract_size]  — position
        size calculator. Without a contract size it returns raw units;
        with one (e.g. your broker's oz/lot or barrels/lot) it returns
        lots directly.
      /help — command list

Setup:
  - GitHub repo secrets required: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  - Runs on a schedule via GitHub Actions (see structure_scanner.yml)
  - Keeps state/alerted.json (dedup) and state/last_update_id.json
    (command polling offset); the workflow commits these back to the
    repo after each run.
"""

import os
import json
import csv
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
SEND_HEARTBEAT = False   # True = "Scanner alive" message every run
SEND_SUMMARIES = False   # True = daily/weekly journal summaries

STATE_PATH = "state/alerted.json"
UPDATE_ID_PATH = "state/last_update_id.json"
JOURNAL_PATH = "state/journal.csv"
SUMMARY_STATE_PATH = "state/last_summary.json"
JOURNAL_FIELDS = ["id", "logged_at", "pair", "direction", "score", "entry",
                   "stop", "target", "session", "outcome", "actual_r"]
DAILY_SUMMARY_HOUR = 21   # ~9pm UTC, after most sessions close
WEEKLY_SUMMARY_WEEKDAY = 6  # Sunday (Monday=0 .. Sunday=6)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

CHECK_LABELS = {
    "body_ratio": "Strong candle bodies",
    "consolidation": "Tight consolidation",
    "low_overlap": "Low candle overlap",
    "swing_clarity": "Clear swing structure",
    "strong_engulf": "Strong engulf candle",
    "clean_leg_low": "Clean leg extreme",
}

HELP_TEXT = (
    "*Azu Engulf Bot*\n"
    "Scans USOIL, XAUUSD, GER40, US30, US100, US500, JP225, BTCUSDT, "
    "SOLUSDT, BNBUSDT for 1H engulf + 5m retest structure setups.\n\n"
    "*Commands*\n"
    "/calc <account> <risk%> <entry> <stop> [contract_size] — position "
    "size calculator. Example: /calc $2500 0.5% 1985.50 1980.00 100\n"
    "Add your broker's contract size (e.g. oz per lot, barrels per lot) "
    "as the 5th number to get lots directly.\n"
    "/outcome <pair> <win|loss|be> [actual_r] — record how a trade went. "
    "Updates your most recent open trade for that pair. Example: "
    "/outcome XAUUSD win  or  /outcome XAUUSD loss -1\n"
    "/help — this message\n\n"
    "Every alert is auto-logged to the journal as \"open\" until you record "
    "an outcome. Daily and weekly summaries (win rate, avg R) post "
    "automatically around 21:00 UTC.\n\n"
    "_Commands are checked once per scan run, so replies can take up to "
    "the scan interval to arrive._"
)


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


def get_structure_checks(df, i, direction):
    """Returns an ordered dict of rule_name -> bool for the leg into
    candle i, so the alert can show a per-rule checklist rather than
    just a bare score."""
    start = max(0, i - LOOKBACK_LEGS)
    leg = df.iloc[start:i + 1]
    checks = {}
    if len(leg) < 3:
        return {k: False for k in CHECK_LABELS}

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

    return checks


def structure_score(df, i, direction):
    return sum(get_structure_checks(df, i, direction).values())


def format_checklist(checks):
    return "\n".join(
        f"{CHECK_LABELS.get(k, k)} {'✅' if v else '❌'}" for k, v in checks.items()
    )


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

def get_last_update_id():
    if os.path.exists(UPDATE_ID_PATH):
        with open(UPDATE_ID_PATH) as f:
            return json.load(f).get("update_id", 0)
    return 0

def save_last_update_id(update_id):
    os.makedirs(os.path.dirname(UPDATE_ID_PATH), exist_ok=True)
    with open(UPDATE_ID_PATH, "w") as f:
        json.dump({"update_id": update_id}, f)


# ---------------------------------------------------------------------------
# JOURNAL
# ---------------------------------------------------------------------------

def log_trade(name, direction, score, entry, stop, target, session_tag, engulf_time, entry_time):
    trade_id = f"{name}_{direction}_{engulf_time.isoformat()}_{entry_time.isoformat()}"
    os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
    is_new = not os.path.exists(JOURNAL_PATH)
    with open(JOURNAL_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "id": trade_id,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "pair": name,
            "direction": direction,
            "score": score,
            "entry": entry,
            "stop": stop,
            "target": target,
            "session": session_tag,
            "outcome": "open",
            "actual_r": "",
        })


def load_journal():
    if not os.path.exists(JOURNAL_PATH):
        return []
    with open(JOURNAL_PATH, newline="") as f:
        return list(csv.DictReader(f))


def save_journal(rows):
    os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
    with open(JOURNAL_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def handle_outcome_command(text):
    parts = text.strip().split()
    if len(parts) < 3:
        return (
            "Usage: /outcome <pair> <win|loss|be> [actual_r]\n"
            "Example: /outcome XAUUSD win\n"
            "Example: /outcome XAUUSD loss -1\n"
            "Updates your most recent open trade for that pair."
        )
    pair = parts[1].upper()
    result = parts[2].lower()
    if result not in ("win", "loss", "be"):
        return "Result must be win, loss, or be."
    try:
        actual_r = _clean_number(parts[3]) if len(parts) > 3 else None
    except ValueError:
        return "actual_r must be a number, e.g. /outcome XAUUSD win 2.8"

    rows = load_journal()
    for row in reversed(rows):
        if row["pair"] == pair and row["outcome"] == "open":
            row["outcome"] = result
            row["actual_r"] = actual_r if actual_r is not None else {"win": TARGET_R, "loss": -1, "be": 0}[result]
            save_journal(rows)
            return f"Updated {pair} trade ({row['id']}) → {result}, {row['actual_r']}R"
    return f"No open trade found for {pair}."


def within_days(row, days, now):
    try:
        ts = datetime.fromisoformat(row["logged_at"])
    except (KeyError, ValueError):
        return False
    return (now - ts).days < days


def compute_stats(rows):
    closed = [r for r in rows if r["outcome"] in ("win", "loss", "be")]
    if not closed:
        return None
    wins = sum(1 for r in closed if r["outcome"] == "win")
    total_r = sum(float(r["actual_r"]) for r in closed if r["actual_r"] not in ("", None))
    return {
        "count": len(closed),
        "wins": wins,
        "win_rate": wins / len(closed) * 100,
        "avg_r": total_r / len(closed),
        "total_r": total_r,
    }


def format_stats_message(title, rows):
    stats = compute_stats(rows)
    open_count = sum(1 for r in rows if r["outcome"] == "open")
    if not stats:
        return f"*{title}*\nNo closed trades yet. {open_count} still open."
    return (
        f"*{title}*\n"
        f"Closed trades: {stats['count']} ({stats['wins']} wins, {stats['win_rate']:.0f}% win rate)\n"
        f"Avg R: {stats['avg_r']:.2f} | Total R: {stats['total_r']:.2f}\n"
        f"Still open: {open_count}"
    )


def get_summary_state():
    if os.path.exists(SUMMARY_STATE_PATH):
        with open(SUMMARY_STATE_PATH) as f:
            return json.load(f)
    return {}


def save_summary_state(state):
    os.makedirs(os.path.dirname(SUMMARY_STATE_PATH), exist_ok=True)
    with open(SUMMARY_STATE_PATH, "w") as f:
        json.dump(state, f)


def maybe_send_summaries():
    """Sends a daily summary once per day and a weekly summary once per
    week, both around DAILY_SUMMARY_HOUR UTC. Guards against duplicate
    sends within the same day/week using state/last_summary.json."""
    now = datetime.now(timezone.utc)
    if now.hour != DAILY_SUMMARY_HOUR:
        return
    state = get_summary_state()
    rows = load_journal()

    today_str = now.strftime("%Y-%m-%d")
    if state.get("daily") != today_str:
        todays_rows = [r for r in rows if within_days(r, 1, now)]
        send_telegram(format_stats_message(f"Daily Summary — {today_str}", todays_rows))
        state["daily"] = today_str
        save_summary_state(state)

    if now.weekday() == WEEKLY_SUMMARY_WEEKDAY:
        week_str = now.strftime("%Y-W%W")
        if state.get("weekly") != week_str:
            weekly_rows = [r for r in rows if within_days(r, 7, now)]
            send_telegram(format_stats_message(f"Weekly Summary — {week_str}", weekly_rows))
            state["weekly"] = week_str
            save_summary_state(state)


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


def _clean_number(s):
    """Strips $, %, and thousands commas so /calc accepts either
    '2500' or '$2,500', and either '0.5' or '0.5%'."""
    return float(s.replace("$", "").replace("%", "").replace(",", ""))


def handle_calc_command(text):
    parts = text.strip().split()
    try:
        account = _clean_number(parts[1])
        risk_pct = _clean_number(parts[2])
        entry = _clean_number(parts[3])
        stop = _clean_number(parts[4])
        contract_size = _clean_number(parts[5]) if len(parts) > 5 else None
    except (IndexError, ValueError):
        return (
            "Usage: /calc <account> <risk%> <entry> <stop> [contract_size]\n"
            "Example: /calc $2500 0.5% 95.75 95.40\n"
            "With lot sizing: /calc $2500 0.5% 1985.50 1980.00 100  (e.g. 100 oz/lot)"
        )

    distance = abs(entry - stop)
    if distance == 0:
        return "Entry and stop can't be equal."

    risk_amount = account * risk_pct / 100
    units = risk_amount / distance

    lines = [
        "*Risk Calculator*",
        f"Account: ${account:,.2f}",
        f"Risk: {risk_pct}% (${risk_amount:,.2f})",
        f"Entry: {entry}",
        f"Stop: {stop}",
        f"Distance: {distance:.5f}",
        f"Position size: {units:,.2f} units",
    ]
    if contract_size:
        lots = units / contract_size
        lines.append(f"Contract size: {contract_size}")
        lines.append(f"Lots: {lots:.3f}")
    else:
        lines.append(
            "_Add your broker's contract size as a 5th number to get lots "
            "directly, e.g. /calc 2500 0.5 95.75 95.40 1000_"
        )
    return "\n".join(lines)


def poll_commands():
    """Checks for new Telegram messages once per run and replies to
    recognized commands. Only responds in the configured chat."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    last_id = get_last_update_id()
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    try:
        resp = requests.get(url, params={"offset": last_id + 1, "timeout": 0}, timeout=10)
        data = resp.json()
    except Exception as e:
        print(f"getUpdates failed: {e}")
        return
    if not data.get("ok"):
        print(f"getUpdates error: {data}")
        return

    max_id = last_id
    for upd in data.get("result", []):
        max_id = max(max_id, upd["update_id"])
        msg = upd.get("message") or upd.get("edited_message")
        if not msg:
            continue
        chat_id = str(msg.get("chat", {}).get("id"))
        if chat_id != str(TELEGRAM_CHAT_ID):
            continue  # ignore commands from anyone but the configured chat
        text = msg.get("text", "").strip()
        if text.startswith("/calc"):
            send_telegram(handle_calc_command(text))
        elif text.startswith("/outcome"):
            send_telegram(handle_outcome_command(text))
        elif text.startswith("/start") or text.startswith("/help"):
            send_telegram(HELP_TEXT)

    if data.get("result"):
        save_last_update_id(max_id)


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

        checks = get_structure_checks(df_1h, i, direction)
        score = sum(checks.values())
        if score < MIN_SCORE:
            continue

        zone_low, zone_high = cur["Low"], cur["High"]
        engulf_time = df_1h.index[i]

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
            f"Structure Engulf | {session_tag}\n\n"
            f"{format_checklist(checks)}\n"
            f"Retest tapped ✅\n"
            f"5m confirmation ✅\n"
            f"Score: {score}/6\n\n"
            f"Entry: `{entry:.5f}`\n"
            f"Stop: `{stop:.5f}`\n"
            f"Target ({TARGET_R}R): `{target:.5f}`\n"
            f"Live: `{live_price:.5f}`\n\n"
            f"1H zone: {zone_low:.5f} - {zone_high:.5f}\n"
            f"Engulf: {engulf_time} UTC\n\n"
            f"_Reply /calc <account> <risk%> {entry:.5f} {stop:.5f} to size this trade._"
        )
        print(f"[{name}] ALERT — {arrow} score {score}/6")
        send_telegram(message)
        alerted.add(key)
        stats["signals"] += 1
        log_trade(name, direction, score, entry, stop, target, session_tag, engulf_time, entry_time)

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
    if SEND_HEARTBEAT:
        send_heartbeat(heartbeats, stats["signals"])
    elif heartbeats and all("status" in h for h in heartbeats):
        # every pair failed / had no data: say so once instead of staying silent
        send_telegram("Scanner ERROR: no pair could be scanned.\n"
                      + "\n".join(format_heartbeat_line(h) for h in heartbeats[:4]))
    poll_commands()
    if SEND_SUMMARIES:
        maybe_send_summaries()


if __name__ == "__main__":
    main()
