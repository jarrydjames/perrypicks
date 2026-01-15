import re
import time
from datetime import datetime, timezone

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
# Page + Theme UX
# -----------------------------
st.set_page_config(
    page_title="PerryPicks 🕵️‍♂️",
    page_icon="🕵️‍♂️",
    layout="wide",
)

# Hide sidebar entirely (even if Streamlit tries to show it)
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
GAMEID_RE = re.compile(r"(002\d{7})")


def extract_gid_safe(s: str) -> str | None:
    """
    Returns GAME_ID (e.g., 0022500551) if present, else None.
    Never raises.
    """
    if not s:
        return None
    m = GAMEID_RE.search(str(s))
    return m.group(1) if m else None


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sd_from_q10_q90(lo: float, hi: float) -> float:
    # For Normal: q90 - q10 ~= 2*1.2816*sd
    denom = 2.0 * 1.281551565545
    return max(0.01, (hi - lo) / denom)


def parse_pt_clock(clock_str: str | None) -> str | None:
    """
    Converts NBA clock strings like 'PT11M47.00S' to '11:47'.
    """
    if not clock_str:
        return None
    m = re.search(r"PT(\d+)M(\d+)(?:\.\d+)?S", clock_str)
    if not m:
        return None
    mm = int(m.group(1))
    ss = int(m.group(2))
    return f"{mm}:{ss:02d}"


def minutes_remaining(period: int | None, clock_str: str | None) -> float | None:
    """
    period: 1-4, clock_str: 'PT##M##.##S'
    Returns minutes remaining in regulation game (max 48).
    OT not handled (returns None).
    """
    if not period or not clock_str:
        return None
    m = re.search(r"PT(\d+)M(\d+)(?:\.\d+)?S", clock_str)
    if not m:
        return None
    mm = int(m.group(1))
    ss = int(m.group(2))
    mins_left_in_period = mm + ss / 60.0
    if period < 1 or period > 4:
        return None
    periods_left_after_this = 4 - period
    return periods_left_after_this * 12.0 + mins_left_in_period


def shrink_sd_with_clock(sd: float, min_rem: float | None, min_total: float = 24.0) -> float:
    """
    Shrink SD as time elapses in 2H. Scale ~ sqrt(remaining / 24).
    """
    if min_rem is None:
        return sd
    rem_2h = min(24.0, max(0.0, float(min_rem)))
    scale = (rem_2h / min_total) ** 0.5 if min_total > 0 else 1.0
    return max(0.01, sd * scale)


def kelly_to_text(f: float) -> str:
    if f <= 0:
        return "0% (no bet)"
    return f"{min(0.25, f) * 100:.1f}% of bankroll"


def init_state():
    st.session_state.setdefault("last_pred", None)
    st.session_state.setdefault("pred_history", [])      # list of {ts, pred}
    st.session_state.setdefault("tracked_bets", [])      # list of bet dicts
    st.session_state.setdefault("tracked_parlays", [])   # list of parlay dicts
    st.session_state.setdefault("track_history", {})     # key -> list of {ts,p,edge}
    st.session_state.setdefault("auto_refresh", False)
    st.session_state.setdefault("refresh_mins", 3)
    st.session_state.setdefault("use_clock_shrink", True)


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
# Inputs (top, no sidebar)
# -----------------------------
DEFAULT_GAME = "https://www.nba.com/game/nyk-vs-por-0022500551"

with st.container():
    c1, c2, c3 = st.columns([2.2, 1.2, 1.2], vertical_alignment="bottom")

    with c1:
        game_input = st.text_input(
            "Game URL or GAME_ID",
            value=st.session_state.get("game_input", DEFAULT_GAME),
            help="Example: https://www.nba.com/game/nyk-vs-por-0022500551  or  0022500551",
        )
        st.session_state["game_input"] = game_input

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

    # Auto refresh hook
    if st.session_state.auto_refresh and HAS_AUTOREFRESH:
        st_autorefresh(interval=int(st.session_state.refresh_mins * 60_000), key="pp_autorefresh")
    elif st.session_state.auto_refresh and not HAS_AUTOREFRESH:
        st.info("Auto refresh needs `streamlit-autorefresh` (already in requirements.txt).")

st.write("")

# -----------------------------
# Betting Inputs (directly under URL area)
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
        # We'll fill these labels after we have a prediction (team names).
        spread_line_home = st.number_input("Home spread (home - away)", value=0.0, step=0.5, help="Example: -3.5 means home is -3.5")
        odds_home = st.text_input("Odds: Home side", value="-110")
        odds_away = st.text_input("Odds: Away side", value="-110")

    with b3:
        bankroll = st.number_input("Bankroll (for Kelly sizing)", value=1000.0, step=50.0)
        kelly_mult = st.slider("Kelly fraction multiplier", 0.0, 1.0, 0.5, 0.05, help="0.5 = half-Kelly (recommended).")

    st.markdown('<div class="pp-muted">Tip: Confirm bets below to track hit probability over time.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# -----------------------------
# Validate input before running
# -----------------------------
gid = extract_gid_safe(game_input)
if gid is None:
    st.warning("Paste a full nba.com game URL or a GAME_ID like **0022500551** to run predictions.")
    st.stop()

# -----------------------------
# Run prediction (manual refresh or initial load)
# -----------------------------
def run_prediction():
    pred = predict_game(game_input)

    status = pred.get("status", {}) or {}
    period = status.get("period")
    clock = status.get("gameClock")

    min_rem = minutes_remaining(period, clock)

    # Derive SD from bands80 (q10/q90-like)
    bands = pred.get("bands80", {}) or {}
    (t_lo, t_hi) = bands.get("final_total", (None, None))
    (m_lo, m_hi) = bands.get("final_margin", (None, None))

    # Fallback to normal if needed
    normal = pred.get("normal", {}) or {}
    if t_lo is None or t_hi is None:
        (t_lo, t_hi) = normal.get("final_total", (None, None))
    if m_lo is None or m_hi is None:
        (m_lo, m_hi) = normal.get("final_margin", (None, None))

    # If still missing, use sane defaults
    base_sd_total = 12.0
    base_sd_margin = 8.0
    if t_lo is not None and t_hi is not None:
        base_sd_total = sd_from_q10_q90(float(t_lo), float(t_hi))
    if m_lo is not None and m_hi is not None:
        base_sd_margin = sd_from_q10_q90(float(m_lo), float(m_hi))

    final_sd_total = base_sd_total
    final_sd_margin = base_sd_margin

    if st.session_state.use_clock_shrink:
        final_sd_total = shrink_sd_with_clock(final_sd_total, min_rem)
        final_sd_margin = shrink_sd_with_clock(final_sd_margin, min_rem)

    pred["_derived"] = {
        "min_remaining": min_rem,
        "base_sd_total": float(base_sd_total),
        "base_sd_margin": float(base_sd_margin),
        "sd_final_total": float(final_sd_total),
        "sd_final_margin": float(final_sd_margin),
        "clock_mmss": parse_pt_clock(clock),
        "period": period,
        "clock_str": clock,
    }

    st.session_state.last_pred = pred
    st.session_state.pred_history.append({"ts": now_utc_iso(), "pred": pred})


if manual_refresh or st.session_state.last_pred is None:
    try:
        run_prediction()
    except Exception as e:
        st.error(f"Prediction failed: {repr(e)}")
        st.stop()

pred = st.session_state.last_pred

# -----------------------------
# Display: game info + projections
# -----------------------------
home_name = pred.get("home_name", "HOME")
away_name = pred.get("away_name", "AWAY")
h1_home = float(pred.get("h1_home", 0))
h1_away = float(pred.get("h1_away", 0))

bands = pred.get("bands80", {}) or {}

st.markdown('<div class="pp-card">', unsafe_allow_html=True)
g1, g2, g3, g4 = st.columns([1.3, 1.0, 1.0, 1.0])

with g1:
    st.markdown(f"**{away_name} @ {home_name}**")
    st.markdown(f"**Halftime:** {home_name} {int(h1_home)} – {int(h1_away)} {away_name}")

    per = pred["_derived"].get("period")
    mmss = pred["_derived"].get("clock_mmss")
    if per and mmss:
        st.markdown(
            f"<div style='font-size:34px;font-weight:900;line-height:1.0'>{mmss}</div>"
            f"<div class='pp-muted'>Q{per}</div>",
            unsafe_allow_html=True,
        )
    elif pred["_derived"]["min_remaining"] is not None:
        st.markdown(f"<span class='pp-muted'>Minutes remaining: {pred['_derived']['min_remaining']:.1f}</span>", unsafe_allow_html=True)

with g2:
    # 2H total is in pred["text"], but we also have normal/bands. Use bands80 h2_total if present.
    h2t = pred.get("normal", {}).get("h2_total")
    b_h2t = bands.get("h2_total")
    if b_h2t:
        h2t_lo, h2t_hi = b_h2t
    elif h2t:
        h2t_lo, h2t_hi = h2t
    else:
        h2t_lo, h2t_hi = (None, None)

    # The model mean isn't explicitly in bands; use midpoint if no mean provided.
    h2_total_mean = pred.get("pred_2h_total", None)
    if h2_total_mean is None and h2t_lo is not None and h2t_hi is not None:
        h2_total_mean = (float(h2t_lo) + float(h2t_hi)) / 2.0

    if h2_total_mean is None:
        st.markdown("<div class='pp-kpi'><div class='pp-muted'>2H Total</div><div style='font-size:20px;font-weight:800'>—</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='pp-kpi'><div class='pp-muted'>2H Total</div>"
            f"<div style='font-size:20px;font-weight:800'>{float(h2_total_mean):.2f}</div>"
            + (f"<div class='pp-muted'>80%: {float(h2t_lo):.1f} – {float(h2t_hi):.1f}</div>" if h2t_lo is not None else "")
            + "</div>",
            unsafe_allow_html=True,
        )

with g3:
    h2m = pred.get("normal", {}).get("h2_margin")
    b_h2m = bands.get("h2_margin")
    if b_h2m:
        h2m_lo, h2m_hi = b_h2m
    elif h2m:
        h2m_lo, h2m_hi = h2m
    else:
        h2m_lo, h2m_hi = (None, None)

    h2_margin_mean = pred.get("pred_2h_margin", None)
    if h2_margin_mean is None and h2m_lo is not None and h2m_hi is not None:
        h2_margin_mean = (float(h2m_lo) + float(h2m_hi)) / 2.0

    if h2_margin_mean is None:
        st.markdown("<div class='pp-kpi'><div class='pp-muted'>2H Margin (home)</div><div style='font-size:20px;font-weight:800'>—</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='pp-kpi'><div class='pp-muted'>2H Margin (home)</div>"
            f"<div style='font-size:20px;font-weight:800'>{float(h2_margin_mean):.2f}</div>"
            + (f"<div class='pp-muted'>80%: {float(h2m_lo):.1f} – {float(h2m_hi):.1f}</div>" if h2m_lo is not None else "")
            + "</div>",
            unsafe_allow_html=True,
        )

with g4:
    ft_lo, ft_hi = bands.get("final_total", (None, None))
    final_total_mean = None
    if ft_lo is not None and ft_hi is not None:
        final_total_mean = (float(ft_lo) + float(ft_hi)) / 2.0
    st.markdown(
        "<div class='pp-kpi'><div class='pp-muted'>Final Total</div>"
        f"<div style='font-size:20px;font-weight:800'>{float(final_total_mean):.2f}</div>"
        + (f"<div class='pp-muted'>80%: {float(ft_lo):.1f} – {float(ft_hi):.1f}</div>" if ft_lo is not None else "")
        + "</div>",
        unsafe_allow_html=True,
    )

st.write("")
st.markdown("### Predicted final score")
sc1, sc2 = st.columns(2)

fh_lo, fh_hi = bands.get("final_home", (None, None))
fa_lo, fa_hi = bands.get("final_away", (None, None))

# derive means from intervals
final_home_mean = (float(fh_lo) + float(fh_hi)) / 2.0 if fh_lo is not None else 0.0
final_away_mean = (float(fa_lo) + float(fa_hi)) / 2.0 if fa_lo is not None else 0.0

with sc1:
    st.metric(label=home_name, value=f"{final_home_mean:.1f}", delta=None)
    if fh_lo is not None:
        st.caption(f"80% CI: {float(fh_lo):.1f} – {float(fh_hi):.1f}")

with sc2:
    st.metric(label=away_name, value=f"{final_away_mean:.1f}", delta=None)
    if fa_lo is not None:
        st.caption(f"80% CI: {float(fa_lo):.1f} – {float(fa_hi):.1f}")

st.write("")
st.markdown("### Predicted final margin")
fm_lo, fm_hi = bands.get("final_margin", (None, None))
final_margin_mean = (float(fm_lo) + float(fm_hi)) / 2.0 if fm_lo is not None else 0.0
m1, m2 = st.columns(2)
with m1:
    st.metric(label=f"Margin (home: {home_name} - {away_name})", value=f"{final_margin_mean:+.1f}")
with m2:
    if fm_lo is not None:
        st.metric(label="80% CI", value=f"{float(fm_lo):+.1f} to {float(fm_hi):+.1f}")

st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# -----------------------------
# Betting evaluation + recommendation
# -----------------------------
def evaluate_bets():
    derived = pred.get("_derived", {}) or {}

    final_total_mu = (float(ft_lo) + float(ft_hi)) / 2.0 if ft_lo is not None else None
    final_margin_mu = (float(fm_lo) + float(fm_hi)) / 2.0 if fm_lo is not None else None

    sd_total = float(derived.get("sd_final_total", 12.0))
    sd_margin = float(derived.get("sd_final_margin", 8.0))

    recs = []

    if final_total_mu is not None and float(total_line) > 0:
        o_odds = parse_american_odds(odds_over)
        u_odds = parse_american_odds(odds_under)

        p_over = prob_over_under_from_mean_sd(final_total_mu, sd_total, float(total_line))
        p_under = 1.0 - p_over

        be_over = breakeven_prob_from_american(o_odds)
        be_under = breakeven_prob_from_american(u_odds)

        recs.append({
            "type": "Total",
            "side": f"Over {float(total_line):.1f}",
            "odds": o_odds,
            "p": p_over,
            "breakeven": be_over,
            "edge": (p_over - be_over),
            "kelly": kelly_fraction(p_over, o_odds) * float(kelly_mult),
        })
        recs.append({
            "type": "Total",
            "side": f"Under {float(total_line):.1f}",
            "odds": u_odds,
            "p": p_under,
            "breakeven": be_under,
            "edge": (p_under - be_under),
            "kelly": kelly_fraction(p_under, u_odds) * float(kelly_mult),
        })

    if final_margin_mu is not None and float(spread_line_home) != 0.0:
        h_odds = parse_american_odds(odds_home)
        a_odds = parse_american_odds(odds_away)

        p_home_cover = prob_spread_cover_from_mean_sd(final_margin_mu, sd_margin, float(spread_line_home))
        p_away_cover = 1.0 - p_home_cover

        be_h = breakeven_prob_from_american(h_odds)
        be_a = breakeven_prob_from_american(a_odds)

        recs.append({
            "type": "Spread",
            "side": f"{home_name} {float(spread_line_home):+.1f}",
            "odds": h_odds,
            "p": p_home_cover,
            "breakeven": be_h,
            "edge": (p_home_cover - be_h),
            "kelly": kelly_fraction(p_home_cover, h_odds) * float(kelly_mult),
        })
        recs.append({
            "type": "Spread",
            "side": f"{away_name} {-float(spread_line_home):+.1f}",
            "odds": a_odds,
            "p": p_away_cover,
            "breakeven": be_a,
            "edge": (p_away_cover - be_a),
            "kelly": kelly_fraction(p_away_cover, a_odds) * float(kelly_mult),
        })

    recs.sort(key=lambda r: r["edge"], reverse=True)
    return recs, sd_total, sd_margin


recs, sd_total, sd_margin = evaluate_bets()

st.markdown('<div class="pp-card">', unsafe_allow_html=True)
st.subheader("Bet evaluation (value + probability)")
st.caption(f"Using SD(total)={sd_total:.2f}, SD(margin)={sd_margin:.2f} (derived from your calibrated 80% bands).")

if not recs:
    st.info("Add a total and/or spread above to see bet evaluation.")
else:
    top = recs[0]
    if top["edge"] <= 0.0:
        st.markdown("**Recommendation:** No clear value bet from the lines entered (all edges are ≤ 0).")
    else:
        st.markdown(
            f"**Recommendation:** {top['side']} at **{top['odds']}** looks best "
            f"({fmt_pct(top['p'])} to hit, edge **{top['edge']*100:.1f} pts** vs break-even). "
            f"Suggested size (Kelly×{kelly_mult:.2f}): **{kelly_to_text(top['kelly'])}**."
        )

    cols = st.columns([1.0, 1.7, 0.7, 0.9, 0.9, 0.9])
    headers = ["Type", "Bet", "Odds", "P(hit)", "Break-even", "Edge"]
    for c, h in zip(cols, headers):
        c.markdown(f"**{h}**")

    for r in recs[:6]:
        cols = st.columns([1.0, 1.7, 0.7, 0.9, 0.9, 0.9])
        cols[0].write(r["type"])
        cols[1].write(r["side"])
        cols[2].write(str(r["odds"]))
        cols[3].write(fmt_pct(r["p"]))
        cols[4].write(fmt_pct(r["breakeven"]))
        cols[5].write(f"{r['edge']*100:.1f} pts")

st.markdown("</div>", unsafe_allow_html=True)

