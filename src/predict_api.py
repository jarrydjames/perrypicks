import os
import sys
import re
import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict_from_gameid_v2 import (
    extract_game_id, fetch_box, fetch_pbp_df,
    first_half_score, behavior_counts_1h,
    team_totals_from_box_team, add_rate_features
)

from src.betting import normal_from_q10q90

MARGIN_INFLATE = 1.03  # keep consistent with your tuned backtest

def parse_iso_duration_clock(s: str):
    """
    Handles clocks like:
      - "PT09M51.00S"
      - "PT12M00.00S"
      - "09:51"
      - "12:00"
    Returns remaining seconds in the current period, or None.
    """
    if not s:
        return None
    s = str(s).strip()

    m = re.match(r"^PT(\d+)M(\d+)(?:\.\d+)?S$", s)
    if m:
        mm = int(m.group(1))
        ss = int(m.group(2))
        return mm * 60 + ss

    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if m:
        mm = int(m.group(1))
        ss = int(m.group(2))
        return mm * 60 + ss

    return None

def seconds_elapsed_since_halftime(period, clock_str):
    """
    Estimate "game seconds elapsed since halftime" based on period + clock remaining.
    Assumes 12-min quarters and 5-min OTs.
    Returns None if period/clock missing or before 2H.
    """
    try:
        p = int(period)
    except Exception:
        return None
    if p < 3:
        return None

    rem = parse_iso_duration_clock(clock_str)
    if rem is None:
        return None

    if p == 3:
        return max(0, 12 * 60 - rem)
    if p == 4:
        return max(0, 12 * 60 + (12 * 60 - rem))

    # OT: p=5 is OT1
    ot_index = p - 5
    ot_len = 5 * 60
    return max(0, 24 * 60 + ot_index * ot_len + (ot_len - rem))

def _safe_name(team: dict, fallback: str):
    nm = f"{team.get('teamCity','')} {team.get('teamName','')}".strip()
    if nm:
        return nm
    return team.get("teamTricode") or fallback

def pick_bin(x: float, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= x < edges[i + 1]:
            return i
    return len(edges) - 2

def get_band(obj, kind: str, bin_index: int):
    d = (obj or {}).get(kind, {}).get(bin_index)
    if not d:
        return None
    return float(d["q10"]), float(d["q90"]), int(d.get("n", 0))

def inflate_band(q10: float, q90: float, factor: float):
    c = (q10 + q90) / 2.0
    h = (q90 - q10) / 2.0
    h2 = h * factor
    return c - h2, c + h2

def fmt_ci(lo, hi, digits=1):
    return f"{lo:.{digits}f} – {hi:.{digits}f}"

def pbp_last_scores(pbp: pd.DataFrame):
    """
    Returns (homeScore, awayScore) from PBP, forward-filling if needed.
    """
    if pbp is None or pbp.empty:
        return None, None
    df = pbp.copy()
    for c in ["homeScore", "awayScore"]:
        if c not in df.columns:
            return None, None
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[["homeScore", "awayScore"]] = df[["homeScore", "awayScore"]].ffill()
    if df["homeScore"].isna().all() or df["awayScore"].isna().all():
        return None, None
    return int(df["homeScore"].iloc[-1]), int(df["awayScore"].iloc[-1])

def clock_aware_adjustment(
    pred_2h_total_mean: float,
    pred_2h_margin_mean: float,
    resid_total_q10: float,
    resid_total_q90: float,
    resid_margin_q10: float,
    resid_margin_q90: float,
    h1_home: int,
    h1_away: int,
    cur_home: int,
    cur_away: int,
    elapsed_2h_seconds: int,
    total_2h_seconds_nominal: int = 24 * 60
):
    """
    If you refresh after halftime, adjust prediction to the *remaining* portion of 2H
    using current score + elapsed time since halftime.

    Approach:
      - Compute observed 2H pace so far (points/min and margin/min)
      - Blend model pace (from halftime model) with observed pace
      - Convert blended pace into remaining 2H total + margin
      - Shrink interval widths ~ sqrt(remaining_fraction)
    """
    if elapsed_2h_seconds is None or elapsed_2h_seconds <= 0:
        return None  # no adjustment

    # If we are in OT, elapsed_2h_seconds > 24*60; in that case, "remaining 2H" is 0
    remaining_seconds = max(0, total_2h_seconds_nominal - elapsed_2h_seconds)
    if remaining_seconds <= 0:
        return {
            "remaining_2h_total_mean": 0.0,
            "remaining_2h_margin_mean": 0.0,
            "remaining_2h_total_ci": (0.0, 0.0),
            "remaining_2h_margin_ci": (0.0, 0.0),
            "blended": {"alpha_model": 0.0, "note": "2H finished (or OT started)"},
        }

    # Observed so far in 2H
    obs_2h_home = cur_home - h1_home
    obs_2h_away = cur_away - h1_away
    obs_2h_total = obs_2h_home + obs_2h_away
    obs_2h_margin = obs_2h_home - obs_2h_away

    elapsed_min = max(1e-6, elapsed_2h_seconds / 60.0)
    rem_min = remaining_seconds / 60.0

    model_pace_total = pred_2h_total_mean / 24.0
    model_pace_margin = pred_2h_margin_mean / 24.0

    obs_pace_total = obs_2h_total / elapsed_min
    obs_pace_margin = obs_2h_margin / elapsed_min

    # Alpha: trust model early, trust observed more as 2H progresses.
    # At 0: alpha=1. At 24 min: alpha=0.
    alpha = max(0.0, min(1.0, 1.0 - (elapsed_2h_seconds / float(total_2h_seconds_nominal))))

    blended_total_pace = alpha * model_pace_total + (1.0 - alpha) * obs_pace_total
    blended_margin_pace = alpha * model_pace_margin + (1.0 - alpha) * obs_pace_margin

    remaining_2h_total_mean = blended_total_pace * rem_min
    remaining_2h_margin_mean = blended_margin_pace * rem_min

    # Shrink interval widths with remaining fraction (sqrt)
    rem_frac = remaining_seconds / float(total_2h_seconds_nominal)
    shrink = rem_frac ** 0.5

    # residual bands around halftime prediction -> apply to remaining component
    # Convert residual q10/q90 on total/margin into half-widths, shrink, then apply around remaining mean
    total_half = ((pred_2h_total_mean + resid_total_q90) - (pred_2h_total_mean + resid_total_q10)) / 2.0
    total_ci_lo = remaining_2h_total_mean - total_half * shrink
    total_ci_hi = remaining_2h_total_mean + total_half * shrink

    margin_half = ((pred_2h_margin_mean + resid_margin_q90) - (pred_2h_margin_mean + resid_margin_q10)) / 2.0
    margin_ci_lo = remaining_2h_margin_mean - margin_half * shrink
    margin_ci_hi = remaining_2h_margin_mean + margin_half * shrink

    return {
        "remaining_2h_total_mean": float(remaining_2h_total_mean),
        "remaining_2h_margin_mean": float(remaining_2h_margin_mean),
        "remaining_2h_total_ci": (float(total_ci_lo), float(total_ci_hi)),
        "remaining_2h_margin_ci": (float(margin_ci_lo), float(margin_ci_hi)),
        "blended": {
            "alpha_model": float(alpha),
            "obs_pace_total": float(obs_pace_total),
            "model_pace_total": float(model_pace_total),
            "obs_pace_margin": float(obs_pace_margin),
            "model_pace_margin": float(model_pace_margin),
            "remaining_seconds": int(remaining_seconds),
        },
    }

def predict_game(gid_or_url: str) -> dict:
    gid = extract_game_id(gid_or_url)

    game = fetch_box(gid)
    pbp = fetch_pbp_df(gid)

    home = game["homeTeam"]
    away = game["awayTeam"]
    home_name = _safe_name(home, "HOME")
    away_name = _safe_name(away, "AWAY")

    status = {
        "gameStatus": game.get("gameStatus"),
        "gameStatusText": game.get("gameStatusText") or game.get("gameStatusString"),
        "period": game.get("period"),
        "gameClock": game.get("gameClock"),
        "gameTimeUTC": game.get("gameTimeUTC") or game.get("gameEtUtc") or game.get("gameDateTimeUTC"),
    }

    h1_home, h1_away = first_half_score(game)
    beh = behavior_counts_1h(pbp)

    ht = team_totals_from_box_team(home)
    at = team_totals_from_box_team(away)

    row = {
        "h1_home": h1_home,
        "h1_away": h1_away,
        "h1_total": h1_home + h1_away,
        "h1_margin": h1_home - h1_away,
    }
    row.update(beh)
    row.update(add_rate_features("home", ht, at))
    row.update(add_rate_features("away", at, ht))

    X = pd.DataFrame([row])

    total_obj = joblib.load(os.path.join(PROJECT_ROOT, "models/team_v2_2h_total.joblib"))
    margin_obj = joblib.load(os.path.join(PROJECT_ROOT, "models/team_v2_2h_margin.joblib"))
    global_int = joblib.load(os.path.join(PROJECT_ROOT, "models/v2_intervals.joblib"))

    binned = None
    try:
        binned = joblib.load(os.path.join(PROJECT_ROOT, "models/v2_intervals_binned.joblib"))
    except Exception:
        binned = None

    team_int = None
    try:
        team_int = joblib.load(os.path.join(PROJECT_ROOT, "models/v2_team_score_intervals.joblib"))
    except Exception:
        team_int = None

    pred_2h_total = float(total_obj["model"].predict(X[total_obj["features"]])[0])
    pred_2h_margin = float(margin_obj["model"].predict(X[margin_obj["features"]])[0])

    used_total = "global"
    used_margin = "global"

    resid_total_q10 = float(global_int["resid_total_q10"])
    resid_total_q90 = float(global_int["resid_total_q90"])
    resid_margin_q10 = float(global_int["resid_margin_q10"])
    resid_margin_q90 = float(global_int["resid_margin_q90"])

    if binned is not None:
        t_edges = binned.get("total_bins", [-1e9, 1e9])
        m_edges = binned.get("abs_margin_bins", [-1e9, 1e9])

        tb = pick_bin(pred_2h_total, t_edges)
        mb = pick_bin(abs(pred_2h_margin), m_edges)

        bt = get_band(binned, "total", tb)
        if bt and bt[2] >= 30:
            resid_total_q10, resid_total_q90 = bt[0], bt[1]
            used_total = f"binned(bin={tb},n={bt[2]})"

        bm = get_band(binned, "margin", mb)
        if bm and bm[2] >= 30:
            resid_margin_q10, resid_margin_q90 = bm[0], bm[1]
            used_margin = f"binned(bin={mb},n={bm[2]})"

    # inflate margin band (your tuned setting)
    resid_margin_q10, resid_margin_q90 = inflate_band(resid_margin_q10, resid_margin_q90, MARGIN_INFLATE)
    used_margin = used_margin + f"+infl({MARGIN_INFLATE:.2f}x)"

    # Halftime-based 2H bands
    h2_total_lo = pred_2h_total + resid_total_q10
    h2_total_hi = pred_2h_total + resid_total_q90
    h2_margin_lo = pred_2h_margin + resid_margin_q10
    h2_margin_hi = pred_2h_margin + resid_margin_q90

    # Halftime-based team means
    pred_h2_home = (pred_2h_total + pred_2h_margin) / 2.0
    pred_h2_away = (pred_2h_total - pred_2h_margin) / 2.0

    pred_final_home = h1_home + pred_h2_home
    pred_final_away = h1_away + pred_h2_away

    # Final total/margin means and bands
    fh_total = h1_home + h1_away
    fh_margin = h1_home - h1_away

    final_total_mean = fh_total + pred_2h_total
    final_total_lo = fh_total + h2_total_lo
    final_total_hi = fh_total + h2_total_hi

    final_margin_mean = fh_margin + pred_2h_margin
    final_margin_lo = fh_margin + h2_margin_lo
    final_margin_hi = fh_margin + h2_margin_hi

    # Team score bands (prefer calibrated)
    if team_int is not None:
        h2_home_lo = pred_h2_home + float(team_int["h2_home_q10"])
        h2_home_hi = pred_h2_home + float(team_int["h2_home_q90"])
        h2_away_lo = pred_h2_away + float(team_int["h2_away_q10"])
        h2_away_hi = pred_h2_away + float(team_int["h2_away_q90"])

        final_home_lo = pred_final_home + float(team_int["final_home_q10"])
        final_home_hi = pred_final_home + float(team_int["final_home_q90"])
        final_away_lo = pred_final_away + float(team_int["final_away_q10"])
        final_away_hi = pred_final_away + float(team_int["final_away_q90"])
        team_ci_note = "team-score calibrated"
    else:
        corners = [(h2_total_lo, h2_margin_lo), (h2_total_lo, h2_margin_hi), (h2_total_hi, h2_margin_lo), (h2_total_hi, h2_margin_hi)]
        h2_home_vals = [(t + m) / 2.0 for (t, m) in corners]
        h2_away_vals = [(t - m) / 2.0 for (t, m) in corners]
        h2_home_lo, h2_home_hi = min(h2_home_vals), max(h2_home_vals)
        h2_away_lo, h2_away_hi = min(h2_away_vals), max(h2_away_vals)

        final_home_lo, final_home_hi = h1_home + h2_home_lo, h1_home + h2_home_hi
        final_away_lo, final_away_hi = h1_away + h2_away_lo, h1_away + h2_away_hi
        team_ci_note = "conservative"

    # Normal approximations for betting (halftime-based)
    h2_total_mu, h2_total_sigma = normal_from_q10q90(pred_2h_total, h2_total_lo, h2_total_hi)
    h2_margin_mu, h2_margin_sigma = normal_from_q10q90(pred_2h_margin, h2_margin_lo, h2_margin_hi)
    final_total_mu, final_total_sigma = normal_from_q10q90(final_total_mean, final_total_lo, final_total_hi)
    final_margin_mu, final_margin_sigma = normal_from_q10q90(final_margin_mean, final_margin_lo, final_margin_hi)

    # --- Clock-aware adjustment (post-halftime refreshes) ---
    period = status.get("period")
    clock = status.get("gameClock")
    elapsed_2h = seconds_elapsed_since_halftime(period, clock)

    cur_home, cur_away = pbp_last_scores(pbp)
    clock_adj = None
    if elapsed_2h is not None and cur_home is not None and cur_away is not None:
        clock_adj = clock_aware_adjustment(
            pred_2h_total_mean=pred_2h_total,
            pred_2h_margin_mean=pred_2h_margin,
            resid_total_q10=resid_total_q10,
            resid_total_q90=resid_total_q90,
            resid_margin_q10=resid_margin_q10,
            resid_margin_q90=resid_margin_q90,
            h1_home=h1_home,
            h1_away=h1_away,
            cur_home=cur_home,
            cur_away=cur_away,
            elapsed_2h_seconds=int(elapsed_2h),
            total_2h_seconds_nominal=24 * 60,
        )

    # Build display text (shows current score if clock-aware)
    lines = []
    lines.append(f"GAME_ID: {gid}")
    lines.append(f"Matchup: {away_name} @ {home_name}")

    if cur_home is not None and cur_away is not None and elapsed_2h is not None:
        lines.append(f"Current: {home_name} {cur_home} – {cur_away} {away_name}  | Period={period} Clock={clock}")
    else:
        lines.append(f"Halftime: {home_name} {h1_home} – {h1_away} {away_name}")

    lines.append("")
    lines.append("2H Projection (halftime model):")
    lines.append(f"  Total:  {pred_2h_total:.2f}  (80% CI: {fmt_ci(h2_total_lo, h2_total_hi)})  [{used_total}]")
    lines.append(f"  Margin: {pred_2h_margin:.2f} (80% CI: {fmt_ci(h2_margin_lo, h2_margin_hi)})  [{used_margin}]  [home={home_name}]")

    if clock_adj is not None:
        rem = clock_adj["blended"]["remaining_seconds"]
        alpha = clock_adj["blended"]["alpha_model"]
        lines.append("")
        lines.append("Clock-aware remaining 2H projection:")
        lines.append(f"  Remaining seconds in 2H: {rem}")
        lines.append(f"  Blend alpha (model -> observed): {alpha:.2f}")
        rtm = clock_adj["remaining_2h_total_mean"]
        rml = clock_adj["remaining_2h_total_ci"][0]
        rmh = clock_adj["remaining_2h_total_ci"][1]
        rmm = clock_adj["remaining_2h_margin_mean"]
        rml2 = clock_adj["remaining_2h_margin_ci"][0]
        rmh2 = clock_adj["remaining_2h_margin_ci"][1]
        lines.append(f"  Remaining Total:  {rtm:.2f}  (80% CI: {fmt_ci(rml, rmh)})")
        lines.append(f"  Remaining Margin: {rmm:.2f} (80% CI: {fmt_ci(rml2, rmh2)})  [home={home_name}]")

        # Use clock-aware remaining mean to compute clock-aware final mean
        rem_home = (rtm + rmm) / 2.0
        rem_away = (rtm - rmm) / 2.0
        clock_final_home = cur_home + rem_home
        clock_final_away = cur_away + rem_away

        lines.append("")
        lines.append("Clock-aware final projection (from current score):")
        lines.append(f"  {home_name}: {clock_final_home:.1f}")
        lines.append(f"  {away_name}: {clock_final_away:.1f}")
        lines.append(f"  Total: {(clock_final_home + clock_final_away):.1f}")
        lines.append(f"  Margin ({home_name}): {(clock_final_home - clock_final_away):.1f}")

    lines.append("")
    lines.append(f"Final Projection ({team_ci_note}) [halftime-based]:")
    lines.append(f"  {home_name}: {pred_final_home:.1f}  (80% CI: {fmt_ci(final_home_lo, final_home_hi)})")
    lines.append(f"  {away_name}: {pred_final_away:.1f}  (80% CI: {fmt_ci(final_away_lo, final_away_hi)})")
    lines.append(f"  Total: {final_total_mean:.2f}  (80% CI: {fmt_ci(final_total_lo, final_total_hi)})")
    lines.append(f"  Margin ({home_name}): {final_margin_mean:.2f}")

    return {
        "game_id": gid,
        "home_name": home_name,
        "away_name": away_name,
        "h1_home": h1_home,
        "h1_away": h1_away,
        "status": status,
        "elapsed_since_halftime_seconds": elapsed_2h,
        "current_home": cur_home,
        "current_away": cur_away,
        "clock_adjustment": clock_adj,  # None if not applicable
        "text": "\n".join(lines),

        # Distributions (halftime-based, still useful)
        "normal": {
            "h2_total": (h2_total_mu, h2_total_sigma),
            "h2_margin": (h2_margin_mu, h2_margin_sigma),
            "final_total": (final_total_mu, final_total_sigma),
            "final_margin": (final_margin_mu, final_margin_sigma),
        },
        "bands80": {
            "h2_total": (h2_total_lo, h2_total_hi),
            "h2_margin": (h2_margin_lo, h2_margin_hi),
            "final_total": (final_total_lo, final_total_hi),
            "final_margin": (final_margin_lo, final_margin_hi),
            "final_home": (final_home_lo, final_home_hi),
            "final_away": (final_away_lo, final_away_hi),
        },
        "labels": {"total": used_total, "margin": used_margin},
    }
