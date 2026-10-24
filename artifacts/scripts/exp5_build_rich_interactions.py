#!/usr/bin/env python3
"""Build threshold-free view × labelled-airing overlap table for exp5.

The interval predicate matches the reference notebook:
show_start < view_stop and show_stop > view_start, on matching channel.
Unlike join_interactions.py, this retains every positive overlap and applies
no watch threshold during table construction.
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUTPUT = DATA / "exp5_user_airings_rich.parquet"
SCHEDULE_FILES = ["export_arh_11-20-final.csv", "export_arh_21-30-final.csv"]


def load_schedule() -> pd.DataFrame:
    parts = []
    for name in SCHEDULE_FILES:
        df = pd.read_csv(DATA / name, low_memory=False)
        df["start_time"] = pd.to_datetime(df["start_time"], dayfirst=True)
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
        df = df[df["tv_show_id"].ne(0) & df["duration"].gt(0)].copy()
        df["end_time"] = df["start_time"] + pd.to_timedelta(df["duration"], unit="s")
        parts.append(df)
    schedule = pd.concat(parts, ignore_index=True)
    # Searchsorted on both starts and stops is valid only for non-overlapping
    # intervals. Refuse silent semantic drift if future source files change.
    for channel, group in schedule.sort_values("start_time").groupby("channel_id", sort=False):
        starts = group["start_time"].to_numpy(dtype="datetime64[ns]")
        stops = group["end_time"].to_numpy(dtype="datetime64[ns]")
        if len(starts) > 1 and np.any(starts[1:] < stops[:-1]):
            raise ValueError(f"Overlapping schedule intervals on channel {channel}; exact reference join needed")
    return schedule


def build() -> pd.DataFrame:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    t0 = time.time()
    log = pd.read_csv(DATA / "dataset11-30.csv", low_memory=False)
    log["start_time"] = pd.to_datetime(log["start_time"])
    log["stop_time"] = pd.to_datetime(log["stop_time"])
    schedule = load_schedule()
    common = set(log["vsetv_id"].unique()) & set(schedule["channel_id"].unique())
    result = []

    for channel in common:
        views = log.loc[log["vsetv_id"].eq(channel)].sort_values("start_time").reset_index(drop=True)
        shows = schedule.loc[schedule["channel_id"].eq(channel)].sort_values("start_time").reset_index(drop=True)
        vs = views["start_time"].to_numpy(dtype="datetime64[ns]")
        ve = views["stop_time"].to_numpy(dtype="datetime64[ns]")
        ss = shows["start_time"].to_numpy(dtype="datetime64[ns]")
        se = shows["end_time"].to_numpy(dtype="datetime64[ns]")
        # Since schedule intervals are non-overlapping, every intersecting
        # airing lies in the contiguous slice [lo, hi), computed per view.
        lo = np.searchsorted(se, vs, side="right")
        hi = np.searchsorted(ss, ve, side="left")
        counts = np.maximum(hi - lo, 0)
        mask = counts > 0
        if not mask.any():
            continue
        view_idx = np.repeat(np.flatnonzero(mask), counts[mask])
        show_idx = np.concatenate([np.arange(lo[i], hi[i]) for i in np.flatnonzero(mask)])
        overlap_ns = (np.minimum(ve[view_idx], se[show_idx]) - np.maximum(vs[view_idx], ss[show_idx])).astype("timedelta64[ns]").astype(np.int64)
        positive = overlap_ns > 0
        view_idx, show_idx, overlap_ns = view_idx[positive], show_idx[positive], overlap_ns[positive]

        v = views.iloc[view_idx].reset_index(drop=True)
        s = shows.iloc[show_idx].reset_index(drop=True)
        show_start = s["start_time"]
        overlap_seconds = overlap_ns / 1e9
        duration = s["duration"].to_numpy(dtype=float)
        iso = show_start.dt.isocalendar()
        out = pd.DataFrame({
            "user_id": v["user_id"].to_numpy(),
            "tv_show_id": s["tv_show_id"].to_numpy(),
            "channel_id": s["channel_id"].to_numpy(),
            "view_start": v["start_time"].to_numpy(),
            "view_stop": v["stop_time"].to_numpy(),
            "show_start": show_start.to_numpy(),
            "show_stop": s["end_time"].to_numpy(),
            "week": iso.week.to_numpy(dtype=np.int16),
            "day_of_week": show_start.dt.day_of_week.to_numpy(dtype=np.int8),
            "hour": show_start.dt.hour.to_numpy(dtype=np.int8),
            "slot_30m": (show_start.dt.hour * 2 + (show_start.dt.minute // 30)).to_numpy(dtype=np.int8),
            "user_watch_seconds": overlap_seconds,
            "show_duration": duration,
            "user_watch_pct": overlap_seconds / duration,
        })
        result.append(out)

    if not result:
        raise RuntimeError("No view × airing overlaps found")
    rich = pd.concat(result, ignore_index=True)
    # Every row must satisfy the reference strict interval predicate.
    if not ((rich["view_start"] < rich["show_stop"]) & (rich["view_stop"] > rich["show_start"])).all():
        raise AssertionError("Output contains non-overlapping intervals")
    rich.to_parquet(OUTPUT, index=False)
    print(f"[exp5] output={OUTPUT}")
    print(f"[exp5] rows={len(rich):,}; users={rich.user_id.nunique():,}; shows={rich.tv_show_id.nunique():,}")
    print(f"[exp5] overlap seconds min/median/max={rich.user_watch_seconds.min():.6f}/{rich.user_watch_seconds.median():.6f}/{rich.user_watch_seconds.max():.6f}")
    print(f"[exp5] watch_pct min/median/max={rich.user_watch_pct.min():.6f}/{rich.user_watch_pct.median():.6f}/{rich.user_watch_pct.max():.6f}")
    print(f"[exp5] target counts >=.3/.5/.8={(rich.user_watch_pct.ge(.3).sum()):,}/{(rich.user_watch_pct.ge(.5).sum()):,}/{(rich.user_watch_pct.ge(.8).sum()):,}")
    print(f"[exp5] elapsed_s={time.time() - t0:.1f}; parquet_bytes={OUTPUT.stat().st_size:,}")
    return rich


if __name__ == "__main__":
    build()
