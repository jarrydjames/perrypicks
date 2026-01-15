import json
import re
from datetime import datetime
import requests

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"

def parse_date(s: str):
    # scheduleLeagueV2 typically uses ISO timestamps like "2025-10-21T00:00:00Z"
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def season_from_game_id(game_id: str) -> int:
    # Example: 0022500562 -> season ends in 2026? Actually "25" means 2025-26 season.
    # pattern: 002YYxxxxx where YY is season end year last 2 digits (25 -> 2025-26).
    m = re.match(r"^002(\d{2})", game_id)
    return int(m.group(1)) if m else -1

def main(season_end_yy: int = 25):
    sched = requests.get(SCHEDULE_URL, timeout=30).json()

    games = sched.get("leagueSchedule", {}).get("gameDates", [])
    out = []
    for gd in games:
        for g in gd.get("games", []):
            gid = g.get("gameId")
            if not gid:
                continue
            if season_from_game_id(gid) != season_end_yy:
                continue

            out.append({
                "gameId": gid,
                "gameDate": g.get("gameDateTimeUTC") or g.get("gameDateTime") or g.get("gameDate"),
                "homeTeam": (g.get("homeTeam", {}) or {}).get("teamTricode"),
                "awayTeam": (g.get("awayTeam", {}) or {}).get("teamTricode"),
                "gameStatus": g.get("gameStatus"),
            })

    out_path = f"data/processed/game_ids_20{season_end_yy:02d}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved {len(out)} games -> {out_path}")

if __name__ == "__main__":
    # 2025-26 season corresponds to season_end_yy=25 in the 002YYxxxxx format
    main(season_end_yy=25)
