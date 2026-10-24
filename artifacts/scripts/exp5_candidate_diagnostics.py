#!/usr/bin/env python3
"""Explain main-split strong-watch candidate misses; diagnostics only."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def main() -> None:
    inter = pd.read_parquet(DATA / "exp5_user_airings_rich.parquet",
                            columns=["user_id", "tv_show_id", "week", "user_watch_pct"])
    inter["user_id"] = inter["user_id"].astype(str)
    users = set(pd.read_csv(DATA / "submission.csv", dtype={"user_id": str})["user_id"])
    history = inter[inter.week.le(20)]
    target = inter[inter.week.between(21, 30) & inter.user_id.isin(users)].copy()
    target["tv_show_id"] = target["tv_show_id"].astype(int)
    schedule = pd.read_csv(DATA / "export_arh_21-30-final.csv", usecols=["tv_show_id", "start_time"])
    schedule["start_time"] = pd.to_datetime(schedule.start_time, dayfirst=True)
    eligible = set(schedule.loc[schedule.start_time.dt.isocalendar().week.between(21, 30)
                                & schedule.tv_show_id.ne(0), "tv_show_id"].astype(int))
    relevant = target[target.user_watch_pct.ge(0.8) & target.tv_show_id.isin(eligible)]
    relevant = relevant[["user_id", "tv_show_id"]].drop_duplicates()
    user_pairs = set(map(tuple, history[["user_id", "tv_show_id"]].drop_duplicates().itertuples(index=False, name=None)))
    global_items = set(history.tv_show_id.unique())
    relevant["seen_by_user"] = [(u, s) in user_pairs for u, s in relevant.itertuples(index=False)]
    relevant["seen_globally"] = relevant.tv_show_id.isin(global_items)
    print(f"eligible strong-watch user-show pairs={len(relevant):,}; users={relevant.user_id.nunique():,}")
    print(f"seen by same user={int(relevant.seen_by_user.sum()):,} ({relevant.seen_by_user.mean():.4%})")
    print(f"show seen by any user in history={int(relevant.seen_globally.sum()):,} ({relevant.seen_globally.mean():.4%})")

    candidate_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "exp5_candidates_main.parquet"
    if not candidate_path.is_absolute():
        candidate_path = ROOT / candidate_path
    candidates = pd.read_parquet(candidate_path)
    merged = relevant.merge(candidates, on=["user_id", "tv_show_id"], how="left")
    for source in ("repeat", "lightfm", "ppr", "popularity", "tv_slot"):
        count = int(merged[f"in_{source}"].fillna(0).sum())
        print(f"source {source}: {count:,} pairs ({count / len(relevant):.4%})")
    flags = [f"in_{x}" for x in ("repeat", "lightfm", "ppr", "popularity", "tv_slot")]
    covered = int(merged[flags].fillna(0).astype(bool).any(axis=1).sum())
    print(f"union covered={covered:,}/{len(relevant):,} ({covered / len(relevant):.4%})")


if __name__ == "__main__":
    main()
