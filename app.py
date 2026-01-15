from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Optional autorefresh (recommended)
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

from src.predict_api import predict_game
from src.betting import (
    parse_american_odds,
    breakeven_prob_from_american,
    kelly_fraction,
    prob_over_under_from_mean_sd,
    prob_spread_cover_from_mean_sd,
    fmt_pct,
)

# -----------------------------
# Page config + light styling
# -----------------------------
st.set_page_config(
    page_title="PerryPicks 🕵️‍♂️",
    page_icon="🕵️‍♂️",
    layout="wide",
)

st.markdown(
    """
    <style>
      [data-testid="stSidebar"] { display: none !important; }
      [data-testid="stSidebarNav"] { display: none !important; }
      .block-container { padding-top: 1.0rem; padding-bottom: 2.5rem; }

      .pp-card {
        border-radius: 18px;
        padding: 16px 18px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 8px 26px rgba(0,0,0,0.15);
      }
      .pp-title {
        font-size: 26px; font-weight: 800; letter-spacing: 0.2px;
        margin: 0 0 2px 0;
      }
      .pp-sub { opacity: 0.85; margin: 0 0 10px 0; }
      .pp-kpi {
        border-radius: 16px;
        padding: 12px 14px;
        background: rgba(0,0,0,0.18);
        border: 1px solid rgba(255,255,255,0.10);
      }
      .pp-muted { opacity: 0.8; font-size: 13px; }
      .pp-bigtime { font-size: 28px; font-weight: 900; line-height: 1.0; }
      .pp-qtr { font-size: 13px; opacity: 0.85; margin-top: 2px; }
      .stButton>button { border-radius: 14px; font-weight: 650; }
      .stTextInput>div>div>input { border-radius: 14px; }
      .stNumberInput>div>div>input { border-radius: 14px; }
      .stSelectbox>div>div { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
GAMEID_RE = re.compile(r"(002\d{8})")

def extract_gid(s: str) -> str:
    m = GAMEID_RE.search(s or "")
    if not m:
        raise ValueError("Could not find a GAME_ID like 002######## in that input.")
    return m.group(1)

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def pt_clock_to_mmss(clock_str: Optional[str]) -> Optional[str]:
    """
    Convert NBA API clock like 'PT11M47.00S' -> '11:47'
    """
    if not clock_str:
        return None
    m = re.search(r"PT(\d+)M(\d+)(?:\.\d+)?S", clock_str)
    if not m:
        return None
    mm = int(m.group(1))
    ss = int(m.group(2))
    return f"{mm:02d}:{ss:02d}"

def is_clock_zero(clock_str: Optional[str]) -> bool:
    mmss = pt_clock_to_mmss(clock_str)
    return mmss == "00:00"

def mean_from_interval(lo: float, hi: float) -> float:
    return 0.5 * (float(lo) + float(hi))

def sd_from_80_interval(lo: float, hi: float, min_sd: float) -> float:
    """
    Assume (lo,hi) is central 80% interval => q10, q90 for a Normal.
    Then hi - lo = 2 * z * sd, z = 1.28155
    """
    z = 1.281551565545
    lo = float(lo); hi = float(hi)
    width = hi - lo
    if width <= 0:
        return float(min_sd)
    sd = width / (2.0 * z)
    return max(float(min_sd), float(sd))

def kelly_to_text(f: float) -> str:
    if f <= 0:
        return "0% (no bet)"
    return f"{min(0.25, f)*100:.1f}% of bankroll"

def init_state():
    st.session_state.setdefault("game_input", "https://www.nba.com/game/phx-vs-mia-0022500564")
    st.session_state.setdefault("auto_refresh", False)
    st.session_state.setdefault("refresh_mins", 3)
    st.session_state.setdefault("use_clock_shrink", True)

    st.session_state.setdefault("last_pred", None)
    st.session_state.setdefault("last_run_ts", None)
    st.session_state.setdefault("pred_history", [])  # list of {ts, game_id}

    # Tracking
    st.session_state.setdefault("tracked_bets", [])     # list of dicts
    st.session_state.setdefault("tracked_parlays", [])  # list of dicts
    st.session_state.setdefault("track_history", {})    # key -> list of {ts, p}

init_state()

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="pp-card">
      <div class="pp-title">PerryPicks 🕵️‍♂️</div>
      <div class="pp-sub">Paste an NBA game URL or GAME_ID. Add lines/odds. Get a projection + value bets + tracking.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# -----------------------------
# Inputs (top)
# -----------------------------
with st.container():
    c1, c2, c3 = st.columns([2.2, 1.2, 1.2], vertical_alignment="bottom")

    with c1:
        game_input = st.text_input(
            "Game URL or GAME_ID",
            value=st.session_state.game_input,
            help="Example: https://www.nba.com/game/nyk-vs-por-0022500551  or  0022500551",
        ).strip()
        st.session_state.game_input = game_input

    with c2:
        st.session_state.auto_refresh = st.toggle("Auto refresh", value=st.session_state.auto_refresh)
        if st.session_state.auto_refresh:
            st.session_state.refresh_mins = st.number_input(
                "Every N minutes", min_value=1, max_value=10, value=int(st.session_state.refresh_mins), step=1
            )

    with c3:
        st.session_state.use_clock_shrink = st.toggle(
            "Clock-aware confidence",
            value=st.session_state.use_clock_shrink,
            help="Shrinks uncertainty as time runs off in 2H (but never below Min SD).",
        )
        manual_refresh = st.button("🔄 Refresh now", use_container_width=True)

    if st.session_state.auto_refresh and HAS_AUTOREFRESH:
        st_autorefresh(interval=int(st.session_state.refresh_mins * 60_000), key="pp_autorefresh")
    elif st.session_state.auto_refresh and not HAS_AUTOREFRESH:
        st.info("Auto refresh needs `streamlit-autorefresh` (Streamlit Cloud supports it fine if in requirements.txt).")

st.write("")

# -----------------------------
# Betting Inputs + Probability tuning
# -----------------------------
with st.container():
    st.markdown('<div class="pp-card">', unsafe_allow_html=True)
    st.subheader("Betting lines (optional)")

    b1, b2, b3 = st.columns(3)

    with b1:
        total_line = st.number_input("Game Total (O/U)", value=0.0, step=0.5, help="Enter 0 to ignore")
        odds_over = st.text_input("Odds: Over", value="-110")
        odds_under = st.text_input("Odds: Under", value="-110")

    with b2:
        # We'll label this after we know the team names, but keep the value here.
        spread_line_home = st.number_input("Spread (Home team line)", value=0.0, step=0.5, help="Example: -3.5 means home is -3.5")
        odds_home = st.text_input("Odds: Home side", value="-110")
        odds_away = st.text_input("Odds: Away side", value="-110")

    with b3:
        bankroll = st.number_input("Bankroll (for Kelly sizing)", value=1000.0, step=50.0)
        kelly_mult = st.slider("Kelly fraction multiplier", 0.0, 1.0, 0.5, 0.05, help="0.5 = half-Kelly (recommended).")

        st.markdown("**Probability tuning**")
        min_sd_total = st.slider("Min SD (total)", 6.0, 30.0, 12.0, 0.5)
        min_sd_margin = st.slider("Min SD (margin)", 4.0, 24.0, 8.0, 0.5)
        widen = st.slider("Widen/Narrow uncertainty", 0.70, 2.00, 1.15, 0.05)

    st.markdown('<div class="pp-muted">Tip: Confirm bets below to track hit probability over time (including parlays).</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# -----------------------------
# Prediction runner (matches YOUR predict_api schema)
# -----------------------------
def run_prediction() -> Dict[str, Any]:
    # IMPORTANT: your predict_game() does NOT accept use_binned_intervals kwarg
    return predict_game(game_input)

should_run = manual_refresh or (st.session_state.last_pred is None) or st.session_state.auto_refresh

if should_run:
    try:
        pred = run_prediction()
        st.session_state.last_pred = pred
        st.session_state.last_run_ts = now_utc_iso()
        st.session_state.pred_history.append({"ts": st.session_state.last_run_ts, "game_id": pred.get("game_id")})
    except Exception as e:
        st.error(f"Prediction failed: {repr(e)}")
        st.stop()

pred = st.session_state.last_pred
if not isinstance(pred, dict) or not pred:
    st.warning("No prediction data returned from predict_game().")
    st.stop()

# -----------------------------
# Parse schema
# -----------------------------
home_name = pred.get("home_name", "HOME")
away_name = pred.get("away_name", "AWAY")

h1_home = pred.get("h1_home", None)
h1_away = pred.get("h1_away", None)

status = pred.get("status", {}) or {}
period = status.get("period")
clock_pt = status.get("gameClock")
clock_mmss = pt_clock_to_mmss(clock_pt) or "—:—"
status_text = status.get("gameStatusText", "")

bands80 = pred.get("bands80", {}) or {}

# Required intervals (lo, hi)
h2_total_lo, h2_total_hi = bands80.get("h2_total", [None, None])
h2_margin_lo, h2_margin_hi = bands80.get("h2_margin", [None, None])

final_total_lo, final_total_hi = bands80.get("final_total", [None, None])
final_margin_lo, final_margin_hi = bands80.get("final_margin", [None, None])

final_home_lo, final_home_hi = bands80.get("final_home", [None, None])
final_away_lo, final_away_hi = bands80.get("final_away", [None, None])

# Means from interval midpoints (robust, avoids parsing text)
def safe_mid(lo, hi):
    if lo is None or hi is None:
        return None
    return mean_from_interval(lo, hi)

h2_total_mu = safe_mid(h2_total_lo, h2_total_hi)
h2_margin_mu = safe_mid(h2_margin_lo, h2_margin_hi)

final_total_mu = safe_mid(final_total_lo, final_total_hi)
final_margin_mu = safe_mid(final_margin_lo, final_margin_hi)

final_home_mu = safe_mid(final_home_lo, final_home_hi)
final_away_mu = safe_mid(final_away_lo, final_away_hi)

# SDs from 80% intervals (central 80% => q10/q90)
base_sd_total = sd_from_80_interval(final_total_lo, final_total_hi, min_sd_total) if final_total_lo is not None else float(min_sd_total)
base_sd_margin = sd_from_80_interval(final_margin_lo, final_margin_hi, min_sd_margin) if final_margin_lo is not None else float(min_sd_margin)

# Optional clock shrink (only meaningful during 2H, not when final)
clock_zero = is_clock_zero(clock_pt)
if st.session_state.use_clock_shrink and (not clock_zero) and isinstance(period, int) and period in (3, 4) and clock_mmss != "—:—":
    # If we're in Q3/Q4, shrink based on remaining minutes in game (regulation only)
    # Remaining minutes = (periods left after this * 12) + minutes left in this period
    m = re.search(r"(\d+):(\d+)", clock_mmss)
    if m:
        mm = int(m.group(1)); ss = int(m.group(2))
        mins_left_in_period = mm + ss / 60.0
        periods_left_after = 4 - int(period)
        min_remaining = periods_left_after * 12.0 + mins_left_in_period
        # scale by sqrt(remaining / 24) relative to 2H remaining
        rem_2h = max(0.0, min(24.0, float(min_remaining)))
        scale = (rem_2h / 24.0) ** 0.5 if rem_2h > 0 else 0.0
        base_sd_total = max(float(min_sd_total), base_sd_total * scale)
        base_sd_margin = max(float(min_sd_margin), base_sd_margin * scale)

final_sd_total = max(float(min_sd_total), float(base_sd_total) * float(widen))
final_sd_margin = max(float(min_sd_margin), float(base_sd_margin) * float(widen))

# Labels (optional)
labels = pred.get("labels", {}) or {}
label_total = labels.get("total", "")
label_margin = labels.get("margin", "")

# -----------------------------
# Game / projection card
# -----------------------------
st.markdown('<div class="pp-card">', unsafe_allow_html=True)

g1, g2, g3, g4 = st.columns([1.45, 1.0, 1.0, 1.0])

with g1:
    st.markdown(f"**{away_name} @ {home_name}**")
    if h1_home is not None and h1_away is not None:
        st.markdown(f"**Halftime:** {home_name} {int(h1_home)} – {int(h1_away)} {away_name}")

    # Live clock block (big time + quarter below)
    q_txt = f"Q{period}" if period else "Q—"
    st.markdown(
        f"""
        <div style="margin-top:10px">
          <div class="pp-bigtime">{clock_mmss}</div>
          <div class="pp-qtr">{q_txt} • {status_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with g2:
    if h2_total_mu is not None and h2_total_lo is not None:
        st.markdown(
            "<div class='pp-kpi'>"
            "<div class='pp-muted'>2H Total</div>"
            f"<div style='font-size:20px;font-weight:800'>{h2_total_mu:.2f}</div>"
            f"<div class='pp-muted'>80%: {h2_total_lo:.1f} – {h2_total_hi:.1f}</div>"
            f"<div class='pp-muted'>{label_total}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

with g3:
    if h2_margin_mu is not None and h2_margin_lo is not None:
        st.markdown(
            "<div class='pp-kpi'>"
            "<div class='pp-muted'>2H Margin (home)</div>"
            f"<div style='font-size:20px;font-weight:800'>{h2_margin_mu:.2f}</div>"
            f"<div class='pp-muted'>80%: {h2_margin_lo:.1f} – {h2_margin_hi:.1f}</div>"
            f"<div class='pp-muted'>{label_margin}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

with g4:
    if final_total_mu is not None and final_total_lo is not None:
        st.markdown(
            "<div class='pp-kpi'>"
            "<div class='pp-muted'>Final Total</div>"
            f"<div style='font-size:20px;font-weight:800'>{final_total_mu:.2f}</div>"
            f"<div class='pp-muted'>80%: {final_total_lo:.1f} – {final_total_hi:.1f}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

st.write("")
st.markdown("### Predicted final score")
sc1, sc2 = st.columns(2)

with sc1:
    if final_home_mu is not None:
        st.metric(label=home_name, value=f"{final_home_mu:.1f}")
        st.caption(f"80% CI: {final_home_lo:.1f} – {final_home_hi:.1f}")

with sc2:
    if final_away_mu is not None:
        st.metric(label=away_name, value=f"{final_away_mu:.1f}")
        st.caption(f"80% CI: {final_away_lo:.1f} – {final_away_hi:.1f}")

st.write("")
st.markdown("### Predicted final margin")
m1, m2 = st.columns([1.2, 1.8])
with m1:
    if final_margin_mu is not None:
        st.metric(label=f"Margin (home: {home_name} - {away_name})", value=f"{final_margin_mu:+.1f}")
with m2:
    if final_margin_lo is not None:
        st.caption(f"80% CI: {final_margin_lo:+.1f} to {final_margin_hi:+.1f}")

st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# -----------------------------
# Betting evaluation + recommendation (sanity-safe)
# -----------------------------
def evaluate_bets() -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []

    if final_total_mu is None or final_margin_mu is None:
        return recs

    # IMPORTANT: Use final_sd_* derived from your model's 80% bands, with min clamps and widen.
    mu_total = float(final_total_mu)
    mu_margin = float(final_margin_mu)
    sd_total = float(final_sd_total)
    sd_margin = float(final_sd_margin)

    # Total O/U
    if float(total_line) > 0:
        o_odds = parse_american_odds(odds_over)
        u_odds = parse_american_odds(odds_under)

        p_over = prob_over_under_from_mean_sd(mu_total, sd_total, float(total_line))
        p_under = 1.0 - p_over

        be_over = breakeven_prob_from_american(o_odds)
        be_under = breakeven_prob_from_american(u_odds)

        recs.append({
            "type": "Total",
            "side": f"Over {float(total_line):.1f}",
            "odds": o_odds,
            "p": float(p_over),
            "breakeven": float(be_over),
            "edge": float(p_over - be_over),
            "kelly": float(kelly_fraction(p_over, o_odds) * float(kelly_mult)),
        })
        recs.append({
            "type": "Total",
            "side": f"Under {float(total_line):.1f}",
            "odds": u_odds,
            "p": float(p_under),
            "breakeven": float(be_under),
            "edge": float(p_under - be_under),
            "kelly": float(kelly_fraction(p_under, u_odds) * float(kelly_mult)),
        })

    # Spread (home line, but show team names clearly)
    # Convention: line is for HOME team (home - away). Example: -3.5 means home favored by 3.5
    if float(spread_line_home) != 0.0:
        h_odds = parse_american_odds(odds_home)
        a_odds = parse_american_odds(odds_away)

        # P(home covers home_line)
        p_home_cover = prob_spread_cover_from_mean_sd(mu_margin, sd_margin, float(spread_line_home))
        p_away_cover = 1.0 - p_home_cover

        be_h = breakeven_prob_from_american(h_odds)
        be_a = breakeven_prob_from_american(a_odds)

        # Display as: Home -6.5 or Home +6.5
        recs.append({
            "type": "Spread",
            "side": f"{home_name} {float(spread_line_home):+.1f}",
            "odds": h_odds,
            "p": float(p_home_cover),
            "breakeven": float(be_h),
            "edge": float(p_home_cover - be_h),
            "kelly": float(kelly_fraction(p_home_cover, h_odds) * float(kelly_mult)),
        })
        # Away gets opposite sign
        away_line = -float(spread_line_home)
        recs.append({
            "type": "Spread",
            "side": f"{away_name} {away_line:+.1f}",
            "odds": a_odds,
            "p": float(p_away_cover),
            "breakeven": float(be_a),
            "edge": float(p_away_cover - be_a),
            "kelly": float(kelly_fraction(p_away_cover, a_odds) * float(kelly_mult)),
        })

    recs.sort(key=lambda r: r["edge"], reverse=True)
    return recs

recs = evaluate_bets()

st.markdown('<div class="pp-card">', unsafe_allow_html=True)
st.subheader("Bet evaluation (value + probability)")

st.caption(
    f"Base SD(total)≈{base_sd_total:.2f}, Base SD(margin)≈{base_sd_margin:.2f} | "
    f"Final SD(total)={final_sd_total:.2f}, Final SD(margin)={final_sd_margin:.2f}"
)

# Only show a “clock zero” warning if we ALSO have current scores (otherwise it’s misleading)
if clock_zero and (pred.get("current_home") is not None) and (pred.get("current_away") is not None):
    st.warning("Game clock shows 0:00 — probabilities may be extreme because the result is decided.")

if not recs:
    st.info("Add a total and/or spread above to see bet evaluation.")
else:
    top = recs[0]

    # Guardrail: if top is a spread bet that contradicts the mean badly, demote it
    # (This prevents the “best bet is -11.5 when model margin is +2.8” type of confusion.)
    def spread_contradiction_penalty(r: Dict[str, Any]) -> float:
        if r["type"] != "Spread":
            return 0.0
        # Determine which team this spread is for:
        side = r["side"]
        # Extract numeric line from side string (last token)
        try:
            line = float(side.split()[-1])
        except Exception:
            return 0.0
        # For home bet, we compare margin vs line. For away bet, we convert to home line.
        if side.startswith(home_name):
            line_home = line
        else:
            # Away line is shown as +X means away gets points; home line would be -X
            line_home = -line
        # If mean margin is way on the other side of the spread, penalize
        # (threshold ~ 1.0 * sd)
        mu = float(final_margin_mu)
        sd = float(final_sd_margin)
        # distance from line in SD units:
        dist = abs(mu - line_home) / max(0.01, sd)
        # If dist huge, fine. Contradiction is if the bet implies needing a big win
        # opposite to the mean. We approximate by checking sign mismatch between (mu-line).
        if (mu - line_home) < 0:
            # this side is “against” the mean direction
            return min(0.08, 0.02 * dist)  # small penalty in probability points
        return 0.0

    # Apply a small penalty to spread recs if needed, re-rank for recommendation only
    scored = []
    for r in recs:
        adj_edge = r["edge"]
        if r["type"] == "Spread":
            adj_edge -= spread_contradiction_penalty(r)
        scored.append((adj_edge, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]

    if best["edge"] <= 0.0:
        st.markdown("**Recommendation:** No clear value bet from the lines entered (all edges are ≤ 0).")
    else:
        st.markdown(
            f"**Recommendation:** {best['side']} at **{best['odds']}** looks best "
            f"({fmt_pct(best['p'])} to hit, edge **{best['edge']*100:.1f} pts** vs break-even). "
            f"Suggested size (Kelly×{kelly_mult:.2f}): **{kelly_to_text(best['kelly'])}**."
        )

    # Table output
    cols = st.columns([1.0, 1.9, 0.7, 0.9, 0.9, 0.9])
    headers = ["Type", "Bet", "Odds", "P(hit)", "Break-even", "Edge"]
    for c, h in zip(cols, headers):
        c.markdown(f"**{h}**")

    for r in recs[:6]:
        cols = st.columns([1.0, 1.9, 0.7, 0.9, 0.9, 0.9])
        cols[0].write(r["type"])
        cols[1].write(r["side"])
        cols[2].write(str(r["odds"]))
        cols[3].write(fmt_pct(r["p"]))
        cols[4].write(fmt_pct(r["breakeven"]))
        cols[5].write(f"{r['edge']*100:.1f} pts")

st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# -----------------------------
# Tracking: individual bets + parlays + combined chart
# -----------------------------
st.markdown('<div class="pp-card">', unsafe_allow_html=True)
st.subheader("Track your bets (and parlays)")

if not recs:
    st.info("Enter lines above to enable bet tracking.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# Build stable option labels for selection
def rec_label(r: Dict[str, Any]) -> str:
    return f"{r['type']} | {r['side']} | {r['odds']}"

top_choices = recs[:6]
choice_labels = [rec_label(r) for r in top_choices]
label_to_rec = {rec_label(r): r for r in top_choices}

cA, cB = st.columns([1.4, 1.6])

with cA:
    st.markdown("**Add an individual bet to tracking**")
    pick = st.selectbox("Pick a bet", options=choice_labels, index=0, key="track_pick")
    stake = st.number_input("Stake (optional)", value=0.0, step=5.0, help="Only for your notes.")
    if st.button("✅ Track this bet", use_container_width=True):
        gid = extract_gid(game_input)
        key = f"{gid}::BET::{pick}"
        if all(b.get("key") != key for b in st.session_state.tracked_bets):
            st.session_state.tracked_bets.append({
                "key": key,
                "label": pick,
                "stake": float(stake),
                "created_ts": now_utc_iso(),
            })
            st.session_state.track_history.setdefault(key, [])
            st.success("Added bet to tracking.")

with cB:
    st.markdown("**Create a parlay (tracks combined probability)**")
    parlay_picks = st.multiselect("Select 2+ legs", options=choice_labels, default=[], key="parlay_picks")
    parlay_stake = st.number_input("Parlay stake (optional)", value=0.0, step=5.0, key="parlay_stake")
    if st.button("🎟️ Track parlay", use_container_width=True):
        if len(parlay_picks) < 2:
            st.warning("Pick at least 2 legs for a parlay.")
        else:
            gid = extract_gid(game_input)
            legs = sorted(parlay_picks)
            key = f"{gid}::PARLAY::" + " + ".join(legs)
            if all(p.get("key") != key for p in st.session_state.tracked_parlays):
                st.session_state.tracked_parlays.append({
                    "key": key,
                    "legs": legs,
                    "label": " + ".join(legs),
                    "stake": float(parlay_stake),
                    "created_ts": now_utc_iso(),
                })
                st.session_state.track_history.setdefault(key, [])
                st.success("Added parlay to tracking.")

st.write("")

# Record tracking each refresh
def record_tracking_points():
    # recompute current recs for the same labels
    current = evaluate_bets()
    current_by_label = {rec_label(r): r for r in current}

    # Individual bets
    for b in st.session_state.tracked_bets:
        key = b["key"]
        label = b["label"]
        r = current_by_label.get(label)
        if not r:
            continue
        st.session_state.track_history.setdefault(key, []).append({"ts": now_utc_iso(), "p": float(r["p"])})

    # Parlays: assume independence => P(parlay) = product of leg probabilities
    for p in st.session_state.tracked_parlays:
        key = p["key"]
        legs = p["legs"]
        probs = []
        for leg in legs:
            r = current_by_label.get(leg)
            if r:
                probs.append(float(r["p"]))
        if len(probs) != len(legs) or not probs:
            continue
        p_combo = 1.0
        for x in probs:
            p_combo *= max(0.0, min(1.0, x))
        st.session_state.track_history.setdefault(key, []).append({"ts": now_utc_iso(), "p": float(p_combo)})

record_tracking_points()

# Controls
cX, cY, cZ = st.columns([1.2, 1.2, 2.0])
with cX:
    if st.button("🧹 Clear tracking", use_container_width=True):
        st.session_state.tracked_bets = []
        st.session_state.tracked_parlays = []
        st.session_state.track_history = {}
        st.success("Tracking cleared.")
with cY:
    max_points = st.number_input("Keep last N points", min_value=10, max_value=500, value=120, step=10)
with cZ:
    st.caption("Tracking updates on every refresh (manual or auto).")

# Build combined chart with multiple lines
series = []  # (name, key)
for b in st.session_state.tracked_bets:
    series.append((b["label"], b["key"]))
for p in st.session_state.tracked_parlays:
    series.append(("PARLAY: " + p["label"], p["key"]))

if not series:
    st.info("Track a bet or parlay to see probability-over-time charts.")
else:
    # Create wide table: rows = time index, columns = series names
    # We'll align by index order (not perfect timestamp merge, but works well for Streamlit charts).
    wide_rows: List[Dict[str, Any]] = []
    max_len = 0
    trimmed_hist = {}

    for name, key in series:
        hist = st.session_state.track_history.get(key, [])
        hist = hist[-int(max_points):]
        trimmed_hist[key] = hist
        max_len = max(max_len, len(hist))

    for i in range(max_len):
        row = {"t": i}
        for name, key in series:
            hist = trimmed_hist.get(key, [])
            if i < len(hist):
                row[name] = hist[i]["p"]
        wide_rows.append(row)

    st.line_chart(wide_rows, x="t", height=280)

    # Show latest values
    st.markdown("**Latest tracked probabilities**")
    for name, key in series[-10:]:
        hist = st.session_state.track_history.get(key, [])
        if not hist:
            continue
        last = hist[-1]
        st.write(f"- {name}: **{fmt_pct(last['p'])}** (last update {last['ts']})")

st.markdown("</div>", unsafe_allow_html=True)

st.caption("Note: probabilities use a Normal approximation from your model’s calibrated 80% bands (with min SD clamps + optional clock shrink + widen).")
