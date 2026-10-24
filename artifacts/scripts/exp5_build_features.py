#!/usr/bin/env python3
"""Build leakage-safe user-show features for exp5 candidate tables.

The script materializes features only; it does not fit a binary LightGBM model
or LGBMRanker. Target-window views are read solely to create label columns.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

WINDOWS = {
    "main": (20, 21, 30, "export_arh_21-30-final.csv", "exp5_candidates_main_expanded9.parquet"),
    "early": (16, 17, 20, "export_arh_11-20-final.csv", "exp5_candidates_main_expanded9_early.parquet"),
}
FEATURES_OUT = {k: DATA / f"exp5_features_{k}.parquet" for k in WINDOWS}
LABEL_TARGETS = (0.5, 0.8)
SCHEDULE_COLUMNS = ["channel_id", "tv_show_id", "start_time", "duration", "tv_show_category",
                    "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3", "year_of_production"]


def load_schedule_window(filename, first_week, last_week):
    schedule = pd.read_csv(DATA / filename, usecols=SCHEDULE_COLUMNS, low_memory=False)
    schedule["start_time"] = pd.to_datetime(schedule.start_time, dayfirst=True)
    schedule["duration"] = pd.to_numeric(schedule.duration, errors="coerce").fillna(0.0)
    schedule["tv_show_id"] = pd.to_numeric(schedule.tv_show_id, errors="coerce").fillna(0).astype(np.int64)
    weeks = schedule.start_time.dt.isocalendar().week
    schedule = schedule[schedule.tv_show_id.ne(0) & weeks.between(first_week, last_week)].copy()
    schedule["day_of_week"] = schedule.start_time.dt.day_of_week.astype(np.int8)
    schedule["hour"] = schedule.start_time.dt.hour.astype(np.int8)
    schedule["slot_30m"] = (schedule.hour * 2 + schedule.start_time.dt.minute // 30).astype(np.int8)
    return schedule


def read_past_schedules(train_end):
    frames = []
    for name in ("export_arh_11-20-final.csv", "export_arh_21-30-final.csv"):
        part = pd.read_csv(DATA / name, usecols=SCHEDULE_COLUMNS, low_memory=False)
        part["start_time"] = pd.to_datetime(part.start_time, dayfirst=True)
        part["duration"] = pd.to_numeric(part.duration, errors="coerce").fillna(0.0)
        part["tv_show_id"] = pd.to_numeric(part.tv_show_id, errors="coerce").fillna(0).astype(np.int64)
        week = part.start_time.dt.isocalendar().week
        part = part[part.tv_show_id.ne(0) & week.le(train_end)].copy()
        frames.append(part)
    if not frames:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def build_candidate_features(split):
    if split not in WINDOWS:
        raise SystemExit("usage: exp5_build_features.py [main|early]")
    output = FEATURES_OUT[split]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    train_end, target_start, target_end, schedule_file, candidate_file = WINDOWS[split]

    candidates = pd.read_parquet(DATA / candidate_file)
    candidates["user_id"] = candidates.user_id.astype(str)
    candidates["tv_show_id"] = candidates.tv_show_id.astype(np.int64)
    history = pd.read_parquet(DATA / "exp5_user_airings_rich.parquet")
    history["user_id"] = history.user_id.astype(str)
    history = history[history.week.le(train_end)].copy()
    target = pd.read_parquet(DATA / "exp5_user_airings_rich.parquet",
                             columns=["user_id", "tv_show_id", "week", "user_watch_pct"])
    target["user_id"] = target.user_id.astype(str)
    target = target[target.week.between(target_start, target_end)].copy()
    future = load_schedule_window(schedule_file, target_start, target_end)
    past_schedule = read_past_schedules(train_end)
    if not len(history) or not len(future):
        raise RuntimeError(f"Empty history or target schedule for split={split}")

    # (user, show) history features; no target-window rows appear here.
    pairs = history.groupby(["user_id", "tv_show_id"], sort=False)
    hist_features = pairs.agg(
        freq_watch_pct_030=("user_watch_pct", lambda x: int(x.ge(0.3).sum())),
        freq_watch_pct_050=("user_watch_pct", lambda x: int(x.ge(0.5).sum())),
        freq_watch_pct_080=("user_watch_pct", lambda x: int(x.ge(0.8).sum())),
        freq_1w=("week", lambda x: int(x.ge(train_end).sum())),
        freq_2w=("week", lambda x: int(x.ge(train_end-1).sum())),
        freq_4w=("week", lambda x: int(x.ge(train_end-3).sum())),
        freq_8w=("week", lambda x: int(x.ge(train_end-7).sum())),
        distinct_weeks_watched=("week", "nunique"),
        sum_watch_seconds=("user_watch_seconds", "sum"),
        mean_watch_pct=("user_watch_pct", "mean"),
        max_watch_pct=("user_watch_pct", "max"),
        median_watch_pct=("user_watch_pct", "median"),
        last_watch=("show_start", "max"),
    ).reset_index()
    target_reference = future.start_time.min()
    hist_features["days_since_last_watch"] = (
        target_reference - hist_features.last_watch).dt.total_seconds() / 86400.0
    hist_features["rank_freq"] = hist_features.groupby("user_id").freq_watch_pct_050.rank(
        method="min", ascending=False).astype(np.float32)
    hist_features["rank_recent"] = hist_features.groupby("user_id").last_watch.rank(
        method="min", ascending=False).astype(np.float32)
    hist_features = hist_features.drop(columns="last_watch")

    # Session-level user statistics, deduplicating the repeated view×airing rows.
    history["session_seconds"] = (history.view_stop - history.view_start).dt.total_seconds()
    sessions = history.groupby(["user_id", "view_start", "view_stop"], sort=False).agg(
        session_seconds=("session_seconds", "first"), week=("week", "min"), channel_id=("channel_id", "first")
    ).reset_index()
    sessions["active_day"] = sessions.view_start.dt.date
    user_features = sessions.groupby("user_id", sort=False).agg(
        total_views=("view_start", "size"), total_seconds=("session_seconds", "sum"),
        active_days=("active_day", "nunique"), active_weeks=("week", "nunique"),
        mean_session_duration=("session_seconds", "mean"),
    ).reset_index()
    history["recent4w"] = history.week.ge(train_end-3)
    recent_total_views = sessions[sessions.week.ge(train_end-3)].groupby("user_id").size().rename("recent_total_views")

    # User×channel features, shares are based on distinct view sessions.
    session_channel = sessions.groupby(["user_id", "channel_id"], sort=False).agg(
        channel_views=("view_start", "size"), channel_seconds=("session_seconds", "sum"),
        recent_channel_views=("week", lambda x: int(x.ge(train_end-3).sum())),
    ).reset_index()
    session_channel = session_channel.merge(user_features[["user_id", "total_views"]], on="user_id", how="left")
    session_channel["channel_share"] = session_channel.channel_views / session_channel.total_views.clip(lower=1)
    session_channel = session_channel.merge(recent_total_views, on="user_id", how="left")
    session_channel["recent_channel_share"] = session_channel.recent_channel_views / session_channel.recent_total_views.fillna(0).clip(lower=1)
    session_channel = session_channel.drop(columns=["total_views", "recent_total_views"])

    # User time affinities; normalized interaction shares keyed by dominant future airing context.
    user_totals = history.groupby("user_id").size().rename("denom")
    def affinity_table(keys, value_name):
        tab = history.groupby(["user_id", *keys]).size().rename(value_name).reset_index()
        tab = tab.merge(user_totals, on="user_id", how="left")
        tab[value_name] = tab[value_name] / tab.denom.clip(lower=1)
        return tab.drop(columns="denom")
    affinities = [
        (affinity_table(["day_of_week"], "user_dow_affinity"), ["day_of_week"]),
        (affinity_table(["hour"], "user_hour_affinity"), ["hour"]),
        (affinity_table(["slot_30m"], "user_slot_affinity"), ["slot_30m"]),
        (affinity_table(["channel_id", "day_of_week"], "user_channel_dow_affinity"), ["channel_id", "day_of_week"]),
        (affinity_table(["channel_id", "hour"], "user_channel_hour_affinity"), ["channel_id", "hour"]),
        (affinity_table(["channel_id", "day_of_week", "slot_30m"], "user_channel_dow_slot_affinity"),
         ["channel_id", "day_of_week", "slot_30m"]),
    ]

    # Future/past schedule aggregates and a representative (most frequent) future context.
    future_stats = future.groupby("tv_show_id", sort=False).agg(
        future_airings_count=("tv_show_id", "size"), future_total_airtime=("duration", "sum"),
        mean_future_hour=("hour", "mean"),
    ).reset_index()
    context_counts = future.groupby(["tv_show_id", "channel_id", "day_of_week", "hour", "slot_30m"],
                                   sort=False).size().rename("_count").reset_index()
    context_counts = context_counts.sort_values(
        ["tv_show_id", "_count", "channel_id", "day_of_week", "hour", "slot_30m"],
        ascending=[True, False, True, True, True, True], kind="stable")
    dominant_context = context_counts.drop_duplicates("tv_show_id").drop(columns="_count")
    past_stats = past_schedule.groupby("tv_show_id", sort=False).agg(
        past_airings_count=("tv_show_id", "size"), past_total_airtime=("duration", "sum")
    ).reset_index()
    metadata = future.groupby("tv_show_id", sort=False)[
        ["tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3", "year_of_production"]
    ].first().reset_index()
    show_features = future_stats.merge(dominant_context, on="tv_show_id", how="left")
    show_features = show_features.merge(past_stats, on="tv_show_id", how="left")
    show_features = show_features.merge(metadata, on="tv_show_id", how="left")
    show_features[["past_airings_count", "past_total_airtime"]] = show_features[
        ["past_airings_count", "past_total_airtime"]].fillna(0)
    show_features["future_past_airtime_ratio"] = show_features.future_total_airtime / show_features.past_total_airtime.clip(lower=1)

    past_hour = history.groupby("user_id").hour.mean().rename("user_mean_past_hour").reset_index()
    user_features = user_features.merge(past_hour, on="user_id", how="left")
    # Merge all features into the candidate rows; target labels are joined last and never used above.
    features = candidates.merge(hist_features, on=["user_id", "tv_show_id"], how="left")
    features = features.merge(user_features, on="user_id", how="left")
    features = features.merge(show_features, on="tv_show_id", how="left")
    features["schedule_hour_shift"] = features.mean_future_hour - features.user_mean_past_hour
    for table, keys in affinities:
        features = features.merge(table, on=["user_id", *keys], how="left")
    features = features.merge(session_channel, on=["user_id", "channel_id"], how="left",
                              suffixes=("", "_channel"))

    # Labels are generated from target-window rows only and are not feature inputs.
    for threshold, label in ((0.5, "label_050"), (0.8, "label_080")):
        positives = target[target.user_watch_pct.ge(threshold)][["user_id", "tv_show_id"]].drop_duplicates()
        positives[label] = 1
        features = features.merge(positives, on=["user_id", "tv_show_id"], how="left")
        features[label] = features[label].fillna(0).astype(np.int8)

    features["year_of_production"] = pd.to_numeric(features.year_of_production, errors="coerce")
    categorical = ["tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3", "channel_id"]
    for col in categorical:
        if col in features:
            features[col] = features[col].fillna("__MISSING__").astype("category")
    if features[["user_id", "tv_show_id"]].duplicated().any():
        raise AssertionError("Candidate feature join multiplied user-show rows")
    features.to_parquet(output, index=False)
    print(f"[exp5] split={split}; output={output}")
    print(f"[exp5] rows={len(features):,}; columns={len(features.columns)}; users={features.user_id.nunique():,}")
    print(f"[exp5] labels .5 positives={int(features.label_050.sum()):,}; .8 positives={int(features.label_080.sum()):,}")
    print(f"[exp5] duplicate user-show rows=0; target cutoff={target_end}; history cutoff={train_end}")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "main"
    build_candidate_features(split)
