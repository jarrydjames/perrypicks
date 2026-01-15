import sys
import joblib
import pandas as pd

from predict_from_gameid_v2 import (
    extract_game_id, fetch_box, fetch_pbp_df,
    first_half_score, behavior_counts_1h,
    team_totals_from_box_team, add_rate_features
)

# Small safety inflation for binned margin bands to recover nominal coverage
MARGIN_INFLATE = 1.03  # widen around center by 3% when using binned margin

def fmt_ci(lo, hi, digits=1):
    return f"{lo:.{digits}f} – {hi:.{digits}f}"

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
    # widen symmetrically around center
    c = (q10 + q90) / 2.0
    h = (q90 - q10) / 2.0
    h2 = h * factor
    return c - h2, c + h2

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 src/predict_from_gameid_v2_ci.py <GAME_ID or nba.com URL>")

    gid = extract_game_id(sys.argv[1])

    game = fetch_box(gid)
    pbp = fetch_pbp_df(gid)

    home = game["homeTeam"]
    away = game["awayTeam"]
    home_name = f"{home.get('teamCity','')} {home.get('teamName','')}".strip() or home.get("teamTricode", "HOME")
    away_name = f"{away.get('teamCity','')} {away.get('teamName','')}".strip() or away.get("teamTricode", "AWAY")

    # 1H truth + features
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

    # Models
    total_obj = joblib.load("models/team_v2_2h_total.joblib")
    margin_obj = joblib.load("models/team_v2_2h_margin.joblib")

    # Global residual bands (fallback)
    global_tm = joblib.load("models/v2_intervals.joblib")

    # Binned residual bands (preferred)
    binned = None
    try:
        binned = joblib.load("models/v2_intervals_binned.joblib")
    except Exception:
        binned = None

    # Team-score residual bands (already calibrated)
    team_int = None
    try:
        team_int = joblib.load("models/v2_team_score_intervals.joblib")
    except Exception:
        team_int = None

    pred_t = float(total_obj["model"].predict(X[total_obj["features"]])[0])
    pred_m = float(margin_obj["model"].predict(X[margin_obj["features"]])[0])

    # Default: global bands
    used_total = "global"
    used_margin = "global"

    resid_total_q10 = float(global_tm["resid_total_q10"])
    resid_total_q90 = float(global_tm["resid_total_q90"])
    resid_margin_q10 = float(global_tm["resid_margin_q10"])
    resid_margin_q90 = float(global_tm["resid_margin_q90"])

    used_binned_margin = False

    if binned is not None:
        t_edges = binned.get("total_bins", [-1e9, 1e9])
        m_edges = binned.get("abs_margin_bins", [-1e9, 1e9])

        tb = pick_bin(pred_t, t_edges)
        mb = pick_bin(abs(pred_m), m_edges)

        bt = get_band(binned, "total", tb)
        if bt and bt[2] >= 30:
            resid_total_q10, resid_total_q90 = bt[0], bt[1]
            used_total = f"binned(bin={tb},n={bt[2]})"

        bm = get_band(binned, "margin", mb)
        if bm and bm[2] >= 30:
            resid_margin_q10, resid_margin_q90 = bm[0], bm[1]
            used_margin = f"binned(bin={mb},n={bm[2]})"
            used_binned_margin = True

    # Inflate binned margin band slightly to restore nominal coverage
    if used_binned_margin and MARGIN_INFLATE and MARGIN_INFLATE > 1.0:
        resid_margin_q10, resid_margin_q90 = inflate_band(resid_margin_q10, resid_margin_q90, MARGIN_INFLATE)
        used_margin = used_margin + f"+infl({MARGIN_INFLATE:.2f}x)"

    # 2H CI (total/margin)
    t_lo = pred_t + resid_total_q10
    t_hi = pred_t + resid_total_q90
    m_lo = pred_m + resid_margin_q10
    m_hi = pred_m + resid_margin_q90

    # Point team splits
    pred_h2_home = (pred_t + pred_m) / 2.0
    pred_h2_away = (pred_t - pred_m) / 2.0

    pred_final_home = h1_home + pred_h2_home
    pred_final_away = h1_away + pred_h2_away
    pred_final_total = pred_final_home + pred_final_away
    pred_final_margin = pred_final_home - pred_final_away  # home perspective

    # Final total CI from 2H total CI
    fh_total = h1_home + h1_away
    final_total_lo = fh_total + t_lo
    final_total_hi = fh_total + t_hi

    # Team-score CIs (use calibrated team-score residuals if present; else conservative corners)
    if team_int is not None:
        h2_home_lo = pred_h2_home + float(team_int["h2_home_q10"])
        h2_home_hi = pred_h2_home + float(team_int["h2_home_q90"])
        h2_away_lo = pred_h2_away + float(team_int["h2_away_q10"])
        h2_away_hi = pred_h2_away + float(team_int["h2_away_q90"])

        final_home_lo = pred_final_home + float(team_int["final_home_q10"])
        final_home_hi = pred_final_home + float(team_int["final_home_q90"])
        final_away_lo = pred_final_away + float(team_int["final_away_q10"])
        final_away_hi = pred_final_away + float(team_int["final_away_q90"])
        team_ci_note = " (team-score calibrated)"
    else:
        corners = [(t_lo, m_lo), (t_lo, m_hi), (t_hi, m_lo), (t_hi, m_hi)]
        h2_home_vals = [(t + m) / 2.0 for (t, m) in corners]
        h2_away_vals = [(t - m) / 2.0 for (t, m) in corners]
        h2_home_lo, h2_home_hi = min(h2_home_vals), max(h2_home_vals)
        h2_away_lo, h2_away_hi = min(h2_away_vals), max(h2_away_vals)

        final_home_lo, final_home_hi = h1_home + h2_home_lo, h1_home + h2_home_hi
        final_away_lo, final_away_hi = h1_away + h2_away_lo, h1_away + h2_away_hi
        team_ci_note = " (conservative from total/margin bands)"

    print(f"\nGAME_ID: {gid}")
    print(f"Matchup: {away_name} @ {home_name}")
    print(f"Halftime: {home_name} {h1_home} – {h1_away} {away_name}")

    print("\n2H Projection:")
    print(f"  Total:  {pred_t:.2f}  (80% CI: {fmt_ci(t_lo, t_hi)})  [{used_total}]")
    print(f"  Margin: {pred_m:.2f} (80% CI: {fmt_ci(m_lo, m_hi)})  [{used_margin}]  [home={home_name}]")

    print(f"\n2H Team Projection{team_ci_note}:")
    print(f"  {home_name}: {pred_h2_home:.1f}  (80% CI: {fmt_ci(h2_home_lo, h2_home_hi)})")
    print(f"  {away_name}: {pred_h2_away:.1f}  (80% CI: {fmt_ci(h2_away_lo, h2_away_hi)})")

    print(f"\nFinal Projection{team_ci_note}:")
    print(f"  {home_name}: {pred_final_home:.1f}  (80% CI: {fmt_ci(final_home_lo, final_home_hi)})")
    print(f"  {away_name}: {pred_final_away:.1f}  (80% CI: {fmt_ci(final_away_lo, final_away_hi)})")
    print(f"  Total: {pred_final_total:.2f}  (80% CI: {fmt_ci(final_total_lo, final_total_hi)})")
    print(f"  Margin ({home_name}): {pred_final_margin:.2f}")

if __name__ == "__main__":
    main()
