#!/usr/bin/env python3
"""Compare sampled rich-table overlaps to the reference notebook's naive join."""
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RICH = DATA / "exp5_user_airings_rich.parquet"


def main() -> None:
    cols = ["user_id", "vsetv_id", "start_time", "stop_time"]
    views = pd.read_csv(DATA / "dataset11-30.csv", usecols=cols, low_memory=False)
    views["start_time"] = pd.to_datetime(views["start_time"])
    views["stop_time"] = pd.to_datetime(views["stop_time"])
    views = views.sample(n=5000, random_state=420).reset_index(drop=True)

    schedule_parts = []
    for name in ("export_arh_11-20-final.csv", "export_arh_21-30-final.csv"):
        s = pd.read_csv(DATA / name, usecols=["channel_id", "tv_show_id", "start_time", "duration"], low_memory=False)
        s["start_time"] = pd.to_datetime(s["start_time"], dayfirst=True)
        s["duration"] = pd.to_numeric(s["duration"], errors="coerce")
        s = s[s["tv_show_id"].ne(0) & s["duration"].gt(0)].copy()
        s["end_time"] = s["start_time"] + pd.to_timedelta(s["duration"], unit="s")
        schedule_parts.append(s[["channel_id", "tv_show_id", "start_time", "end_time"]])
    by_channel = {channel: group for channel, group in pd.concat(schedule_parts).groupby("channel_id", sort=False)}

    expected = []
    for view in views.itertuples(index=False):
        schedule = by_channel.get(view.vsetv_id)
        if schedule is None:
            continue
        hits = schedule[(schedule.start_time < view.stop_time) & (schedule.end_time > view.start_time)]
        for airing in hits.itertuples(index=False):
            seconds = (min(view.stop_time, airing.end_time) - max(view.start_time, airing.start_time)).total_seconds()
            if seconds > 0:
                expected.append((str(view.user_id), int(view.vsetv_id), view.start_time, view.stop_time,
                                 int(airing.tv_show_id), airing.start_time, airing.end_time, seconds))

    keys = views.rename(columns={"vsetv_id": "channel_id", "start_time": "view_start", "stop_time": "view_stop"}).copy()
    keys["user_id"] = keys["user_id"].astype(str)
    actual_df = pd.read_parquet(RICH)
    actual_df["user_id"] = actual_df["user_id"].astype(str)
    actual_df = actual_df.merge(keys, on=["user_id", "channel_id", "view_start", "view_stop"], how="inner")
    actual = [(r.user_id, int(r.channel_id), r.view_start, r.view_stop, int(r.tv_show_id),
               r.show_start, r.show_stop, float(r.user_watch_seconds)) for r in actual_df.itertuples(index=False)]

    expected_counts, actual_counts = Counter(expected), Counter(actual)
    print(f"sample views: {len(views)} reference pairs: {len(expected)} rich-table pairs: {len(actual)}")
    print(f"reference-only: {sum((expected_counts - actual_counts).values())} "
          f"table-only: {sum((actual_counts - expected_counts).values())} "
          f"exact multiset match: {expected_counts == actual_counts}")
    if expected_counts != actual_counts:
        raise AssertionError("Sampled overlap pairs differ from reference predicate")


if __name__ == "__main__":
    main()
