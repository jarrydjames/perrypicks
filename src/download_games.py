import json
from pathlib import Path
import time, random
import requests
from tqdm import tqdm

PBP_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gid}.json"
BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"

def fetch(url: str, timeout=30):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def main(game_ids_path: str):
    games = json.loads(Path(game_ids_path).read_text())

    pbp_dir = Path("data/raw/pbp"); pbp_dir.mkdir(parents=True, exist_ok=True)
    box_dir = Path("data/raw/box"); box_dir.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0

    for g in tqdm(games, desc="Downloading"):
        gid = g["gameId"]
        pbp_file = pbp_dir / f"{gid}.json"
        box_file = box_dir / f"{gid}.json"

        if pbp_file.exists() and box_file.exists():
            continue

        try:
            if not pbp_file.exists():
                pbp_file.write_text(json.dumps(fetch(PBP_URL.format(gid=gid))))
            if not box_file.exists():
                box_file.write_text(json.dumps(fetch(BOX_URL.format(gid=gid))))
            ok += 1
        except Exception as e:
            fail += 1
            # keep going; print a short line
            print(f"\nFailed {gid}: {type(e).__name__}: {e}")

        time.sleep(random.uniform(0.12, 0.30))

    print(f"\nDone. downloaded_ok={ok} failed={fail}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 src/download_games.py data/processed/game_ids_2025.json")
    main(sys.argv[1])
