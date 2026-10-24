"""E9 split loading and candidate-preserving, history/schedule-only features."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
KEYS = ["user_id", "tv_show_id"]
CATEGORIES = {"tv_show_category": "category", **{
    f"tv_show_genre_{i}": f"genre_{i}" for i in range(1, 4)}}
SCHEDULE_COLUMNS = ["channel_id", "tv_show_id", "start_time", "duration", *CATEGORIES]


def load_split(train_end: int, target_start: int, target_end: int):
    """Return rich history, past/future schedules, and separate >=.5 truth pairs."""
    if not 11 <= train_end < target_start <= target_end <= 42:
        raise ValueError("Expected 11 <= train_end < target_start <= target_end <= 42")
    rich = DATA / "exp5_user_airings_rich.parquet"
    history = pd.read_parquet(rich, filters=[("week", ">=", 11), ("week", "<=", train_end)])
    truth = pd.read_parquet(rich, columns=KEYS, filters=[
        ("week", ">=", target_start), ("week", "<=", target_end),
        ("user_watch_pct", ">=", 0.5)]).drop_duplicates()
    for frame in (history, truth):
        frame["user_id"] = frame.user_id.astype(str)
        frame["tv_show_id"] = frame.tv_show_id.astype(np.int64)
    history.attrs["train_end"] = train_end
    parts = []
    for lo, hi in ((11, 20), (21, 30), (31, 42)):
        if lo > train_end and (hi < target_start or lo > target_end):
            continue
        part = pd.read_csv(DATA / f"export_arh_{lo}-{hi}-final.csv",
                           usecols=SCHEDULE_COLUMNS, low_memory=False)
        part["start_time"] = pd.to_datetime(part.start_time, dayfirst=True, errors="coerce")
        part["duration"] = pd.to_numeric(part.duration, errors="coerce")
        part["tv_show_id"] = pd.to_numeric(part.tv_show_id, errors="coerce").fillna(0).astype(np.int64)
        parts.append(part.loc[part.tv_show_id.ne(0) & part.duration.gt(0)])
    schedule = pd.concat(parts, ignore_index=True)
    week = schedule.start_time.dt.isocalendar().week
    past = schedule.loc[week.between(11, train_end)].reset_index(drop=True)
    future = schedule.loc[week.between(target_start, target_end)].reset_index(drop=True)
    past.attrs["train_end"] = train_end
    return history, past, future, truth.reset_index(drop=True)


def build_features(history, past_schedule, future_schedule, candidates) -> pd.DataFrame:
    """Add only E9 features; preserve candidate order/scores and never join truth."""
    out = candidates.copy()
    out["user_id"] = out.user_id.astype(str)
    out["tv_show_id"] = out.tv_show_id.astype(np.int64)
    if out.duplicated(KEYS).any():
        raise ValueError("Candidates must contain unique user-show pairs")
    h = history.loc[history.user_watch_seconds.gt(0)].copy()
    h["user_id"] = h.user_id.astype(str)
    h["tv_show_id"] = h.tv_show_id.astype(np.int64)
    cutoff = history.attrs.get("train_end", past_schedule.attrs.get("train_end"))
    if cutoff is None:
        cutoff = past_schedule.start_time.dt.isocalendar().week.max()
        if pd.isna(cutoff):
            cutoff = h.week.max()
    counts = h[KEYS].copy()
    frequency = [f"user_show_freq_{suffix}" for suffix in ("03", "05", "08")]
    for name, threshold in zip(frequency, (0.3, 0.5, 0.8)):
        counts[name] = h.user_watch_pct.ge(threshold).astype(np.int32)
    counts["recent_freq_1w"] = h.week.eq(cutoff).astype(np.int32)
    counts["recent_freq_4w"] = h.week.between(cutoff - 3, cutoff).astype(np.int32)
    numeric = [*frequency, "recent_freq_1w", "recent_freq_4w"]
    counts = counts.groupby(KEYS, sort=False)[numeric].sum().reset_index()
    counts["user_show_rank"] = counts.groupby("user_id").user_show_freq_05.rank(
        method="min", ascending=False)
    unseen_rank = counts.groupby("user_id").user_show_rank.max().add(1)
    out = out.merge(counts, on=KEYS, how="left", sort=False, validate="one_to_one")
    out[numeric] = out[numeric].fillna(0).astype(np.int32)
    out["user_show_rank"] = out.user_show_rank.fillna(out.user_id.map(unseen_rank)).fillna(1)

    # One session may overlap multiple shows: count it once per user/channel.
    sessions = h[["user_id", "channel_id", "view_start", "view_stop"]].drop_duplicates()
    channels = sessions.groupby(["user_id", "channel_id"]).size().rename("user_channel_count").reset_index()
    totals = sessions.groupby("user_id").size()
    channels["user_channel_share"] = channels.user_channel_count / channels.user_id.map(totals).clip(lower=1)
    dominant = future_schedule.groupby(["tv_show_id", "channel_id"]).size().rename("_n").reset_index()
    dominant = dominant.sort_values(["tv_show_id", "_n", "channel_id"], ascending=[True, False, True])
    dominant = dominant.drop_duplicates("tv_show_id").drop(columns="_n")
    out = out.merge(dominant, on="tv_show_id", how="left", sort=False, validate="many_to_one")
    out = out.merge(channels, on=["user_id", "channel_id"], how="left", sort=False, validate="many_to_one")
    out = out.drop(columns="channel_id")
    out["user_channel_count"] = out.user_channel_count.fillna(0).astype(np.int32)
    out["user_channel_share"] = out.user_channel_share.fillna(0)

    metadata = future_schedule.sort_values(["start_time", "channel_id"], kind="stable")
    metadata = metadata.groupby("tv_show_id", sort=False)[list(CATEGORIES)].first().rename(columns=CATEGORIES)
    out = out.merge(metadata, on="tv_show_id", how="left", sort=False, validate="many_to_one")
    for name in CATEGORIES.values():
        out[name] = out[name].fillna("__MISSING__").astype(str)
    for name, schedule in (("past_airtime", past_schedule), ("future_airtime", future_schedule)):
        airtime = schedule.groupby("tv_show_id").duration.sum()
        out[name] = out.tv_show_id.map(airtime).fillna(0)
    ratio = out.future_airtime / out.past_airtime.clip(lower=1)
    out["future_past_airtime_ratio"] = ratio.replace([np.inf, -np.inf], 0).fillna(0)
    out.index = candidates.index
    return out
