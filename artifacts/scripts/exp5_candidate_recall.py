#!/usr/bin/env python3
"""Temporal candidate-recall audit for the exp5 rich interaction table.

Main split: history weeks <=20, target weeks 21-30. No target-window view
information enters candidate generators. All recommendations are restricted
to labelled target-schedule shows.
"""
from collections import defaultdict
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from lightfm import LightFM
from exp3_e1_attr_ppr import build_graph_base, build_transition, ppr_tripartite

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TABLES = ROOT / "results" / "tables"
MODE = sys.argv[1] if len(sys.argv) > 1 else "baseline"
if MODE not in {"baseline", "baseline_corrected", "expanded", "expanded2", "expanded3", "expanded4", "expanded5", "expanded6", "expanded7", "expanded8", "expanded9", "expanded10", "expanded11"}:
    raise SystemExit("usage: exp5_candidate_recall.py [baseline|baseline_corrected|expanded|expanded2|expanded3|expanded4|expanded5|expanded6|expanded7|expanded8|expanded9|expanded10|expanded11]")
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "main"
WINDOWS = {
    "main": (20, 21, 30, "export_arh_21-30-final.csv"),
    "early": (16, 17, 20, "export_arh_11-20-final.csv"),
    "roll21_24": (20, 21, 24, "export_arh_21-30-final.csv"),
    "roll25_27": (24, 25, 27, "export_arh_21-30-final.csv"),
    "roll28_30": (27, 28, 30, "export_arh_21-30-final.csv"),
    "final": (30, 31, 42, "export_arh_31-42-final.csv"),
}
if SPLIT not in WINDOWS:
    raise SystemExit(f"split must be one of {', '.join(WINDOWS)}")
MODE_SUFFIX = {"baseline": "", "baseline_corrected": "_corrected", "expanded": "_expanded",
               "expanded2": "_expanded2", "expanded3": "_expanded3", "expanded4": "_expanded4",
               "expanded5": "_expanded5", "expanded6": "_expanded6", "expanded7": "_expanded7",
               "expanded8": "_expanded8", "expanded9": "_expanded9", "expanded10": "_expanded10",
               "expanded11": "_expanded11"}[MODE]
if SPLIT == "main":
    CANDIDATE_OUT = DATA / f"exp5_candidates_main{MODE_SUFFIX}.parquet"
    METRICS_OUT = TABLES / f"exp5_candidate_recall{MODE_SUFFIX}.csv"
elif SPLIT == "early":
    suffix = f"{MODE_SUFFIX or '_baseline'}_early"
    CANDIDATE_OUT = DATA / f"exp5_candidates_main{suffix}.parquet"
    METRICS_OUT = TABLES / f"exp5_candidate_recall{suffix}.csv"
elif SPLIT == "final":
    CANDIDATE_OUT = DATA / f"exp5_candidates_final{MODE_SUFFIX}.parquet"
    METRICS_OUT = None
else:
    CANDIDATE_OUT = DATA / f"exp5_candidates_{SPLIT}{MODE_SUFFIX}.parquet"
    METRICS_OUT = TABLES / f"exp5_candidate_recall_{SPLIT}{MODE_SUFFIX}.csv"


def top_ids(series, n):
    return series.sort_values(ascending=False, kind="stable").head(n).index.tolist()


def build_item_attrs(history_end, target_schedule):
    """Assemble tripartite PPR attributes from past and target-window schedules only."""
    cols = ["channel_id", "tv_show_id", "start_time", "tv_show_category",
            "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"]
    parts = []
    for name in ("export_arh_11-20-final.csv", "export_arh_21-30-final.csv"):
        frame = pd.read_csv(DATA / name, usecols=cols, low_memory=False)
        frame["start_time"] = pd.to_datetime(frame.start_time, dayfirst=True)
        weeks = frame.start_time.dt.isocalendar().week
        frame = frame[frame.tv_show_id.ne(0) & weeks.le(history_end)].copy()
        parts.append(frame.drop(columns="start_time"))
    future = target_schedule[cols].drop(columns="start_time")
    parts.append(future)
    schedule = pd.concat(parts, ignore_index=True)
    schedule["tv_show_id"] = pd.to_numeric(schedule.tv_show_id, errors="coerce")
    schedule = schedule[schedule.tv_show_id.notna() & schedule.tv_show_id.ne(0)]
    attrs = []
    channel = schedule[["tv_show_id", "channel_id"]].dropna()
    attrs.append(channel.assign(attr="ch:" + channel.channel_id.astype(str))[["tv_show_id", "attr"]])
    category = schedule[["tv_show_id", "tv_show_category"]].dropna()
    attrs.append(category.assign(attr="cat:" + category.tv_show_category.astype(str))[["tv_show_id", "attr"]])
    for col in ("tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"):
        genre = schedule[["tv_show_id", col]].dropna()
        attrs.append(genre.assign(attr="gen:" + genre[col].astype(str))[["tv_show_id", "attr"]])
    all_attrs = pd.concat(attrs, ignore_index=True).drop_duplicates()
    return all_attrs.groupby("tv_show_id").attr.apply(list).to_dict()


def projected_affinity_lists(history, target_schedule, user_keys, schedule_keys, top_n=500):
    """Project sparse user affinities onto shows scheduled in matching contexts."""
    future_by_key = {}
    for key, group in target_schedule.groupby(schedule_keys, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        future_by_key[key] = sorted(set(group.tv_show_id.astype(int)))
    stats = history.groupby(["user_id", *user_keys]).agg(
        views=("user_watch_pct", "size"), seconds=("user_watch_seconds", "sum")
    ).reset_index()
    result = {}
    for user, group in stats.groupby("user_id", sort=False):
        show_scores = defaultdict(list)
        for row in group.itertuples(index=False, name=None):
            u, *values, views, seconds = row
            key = tuple(values)
            shows = future_by_key.get(key, ())
            if not shows:
                continue
            score = np.log1p(views) + 0.1 * np.log1p(seconds)
            for show in shows:
                show_scores[show].append(score)
        scores = pd.Series({show: max(vals) + 0.1 * np.log1p(sum(vals))
                            for show, vals in show_scores.items()})
        result[user] = top_ids(scores, top_n) if len(scores) else []
    return result


def schedule_propensity_candidates(history, target_schedule, users, top_n=500):
    """User×channel×weekday×slot airing propensity with four show aggregations."""
    total = history.groupby("user_id").size().to_dict()
    channel_counts = history.groupby(["user_id", "channel_id"]).size().to_dict()
    dow_counts = history.groupby(["user_id", "day_of_week"]).size().to_dict()
    slot_counts = history.groupby(["user_id", "slot_30m"]).size().to_dict()
    exact = history.groupby(["user_id", "channel_id", "day_of_week", "slot_30m"]).size().rename("n").reset_index()
    exact_by_user = {user: group for user, group in exact.groupby("user_id", sort=False)}

    keys = ["channel_id", "day_of_week", "slot_30m"]
    future = target_schedule.groupby([*keys, "tv_show_id"], sort=False).size().rename("airings").reset_index()
    future_by_key = defaultdict(list)
    for row in future.itertuples(index=False):
        key = (row.channel_id, row.day_of_week, row.slot_30m)
        future_by_key[key].append((int(row.tv_show_id), int(row.airings)))

    values, lists_by_source = {}, {name: {} for name in
                                   ("prop_max", "prop_mean", "prop_sum", "prop_logsumexp")}
    for user in users:
        user_history = exact_by_user.get(user)
        per_show = defaultdict(list)
        n_total = max(int(total.get(user, 0)), 1)
        for row in (user_history.itertuples(index=False) if user_history is not None else ()):
            context = (row.channel_id, row.day_of_week, row.slot_30m)
            scheduled = future_by_key.get(context, ())
            if not scheduled:
                continue
            p_channel = channel_counts.get((user, row.channel_id), 0) / n_total
            p_dow = dow_counts.get((user, row.day_of_week), 0) / n_total
            p_slot = slot_counts.get((user, row.slot_30m), 0) / n_total
            p_exact = int(row.n) / n_total
            score = (np.log1p(p_channel) + np.log1p(p_dow) + np.log1p(p_slot)
                     + 2.0 * np.log1p(p_exact))
            for show, count in scheduled:
                per_show[show].append((float(score), count))

        features = {}
        for show, parts in per_show.items():
            scores = np.asarray([x[0] for x in parts], dtype=np.float64)
            counts = np.asarray([x[1] for x in parts], dtype=np.float64)
            maximum = float(scores.max())
            weighted_sum = float(np.sum(scores * counts))
            total_airings = float(counts.sum())
            mean = weighted_sum / total_airings
            logsumexp = maximum + float(np.log(np.sum(counts * np.exp(scores - maximum))))
            features[show] = {"prop_max": maximum, "prop_mean": mean,
                              "prop_sum": weighted_sum, "prop_logsumexp": logsumexp}
        values[user] = features
        for source in lists_by_source:
            ranked = sorted(features, key=lambda show: (-features[show][source], show))[:top_n]
            lists_by_source[source][user] = ranked
    return lists_by_source, values


def main():
    output_paths = (CANDIDATE_OUT,) if METRICS_OUT is None else (CANDIDATE_OUT, METRICS_OUT)
    for p in output_paths:
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite {p}")
    t0 = time.time()
    inter = pd.read_parquet(DATA / "exp5_user_airings_rich.parquet")
    inter["user_id"] = inter["user_id"].astype(str)
    inter["tv_show_id"] = inter["tv_show_id"].astype(int)
    users = pd.read_csv(DATA / "submission.csv", dtype={"user_id": str})["user_id"].tolist()
    train_end, target_start, target_end, schedule_file = WINDOWS[SPLIT]
    target_sched = pd.read_csv(DATA / schedule_file, low_memory=False)
    target_sched["start_time"] = pd.to_datetime(target_sched["start_time"], dayfirst=True)
    target_sched["duration"] = pd.to_numeric(target_sched["duration"], errors="coerce")
    target_sched = target_sched[target_sched.start_time.dt.isocalendar().week.between(target_start, target_end)].copy()
    target_sched = target_sched[target_sched["tv_show_id"].ne(0)].copy()
    target_sched["tv_show_id"] = target_sched["tv_show_id"].astype(int)
    target_sched["day_of_week"] = target_sched.start_time.dt.day_of_week
    target_sched["hour"] = target_sched.start_time.dt.hour
    target_sched["slot_30m"] = target_sched.start_time.dt.hour * 2 + target_sched.start_time.dt.minute // 30
    catalog = sorted(set(target_sched.tv_show_id.unique()))
    catalog_set = set(catalog)

    history = inter[inter.week.le(train_end)].copy()
    target = inter[inter.week.between(target_start, target_end) & inter.user_id.isin(users)]
    target = target[target.tv_show_id.isin(catalog)]
    relevant_by_threshold = {}
    for threshold in (0.5, 0.8):
        rel = target[target.user_watch_pct.ge(threshold)].groupby("user_id").tv_show_id.apply(lambda x: set(x))
        relevant_by_threshold[threshold] = rel.to_dict()
    eval_users = users if SPLIT == "final" else [u for u in users if relevant_by_threshold[0.5].get(u) or relevant_by_threshold[0.8].get(u)]
    print(f"[exp5] history rows={len(history):,}; users={history.user_id.nunique():,}; target catalog={len(catalog):,}")
    print(f"[exp5] eval users with strong target={len(eval_users):,}; relevance users >=.5={len(relevant_by_threshold[.5]):,}, >=.8={len(relevant_by_threshold[.8]):,}")

    # A: personal repeat history, ordered by watch frequency then recency.
    hist = history[history.user_id.isin(eval_users)].copy()
    hist["recent4w"] = hist.week.ge(train_end - 3)
    hist["recency_score"] = np.exp(-(train_end - hist.week.to_numpy()) / 4.0)
    agg = hist.groupby(["user_id", "tv_show_id"]).agg(
        freq=("user_watch_pct", "size"), recent=("recent4w", "sum"), recency=("recency_score", "sum")
    )
    repeat_lists, repeat_extended_lists = {}, {}
    for u, g in agg.groupby(level=0):
        g = g.droplevel(0)
        g = g[g.index.isin(catalog)]
        score = g["freq"] + 0.5 * g["recent"] + 0.1 * g["recency"]
        repeat_lists[u] = top_ids(score, 150)
        repeat_extended_lists[u] = top_ids(score, 500)

    # D: recent global popularity among the last four history weeks.
    recent = history[history.week.ge(train_end - 3)].groupby("tv_show_id").user_id.nunique().reindex(catalog).fillna(0)
    popularity = top_ids(recent, 50)
    popularity_lists = {u: popularity for u in eval_users}
    popularity_extended = top_ids(recent, 500)
    popularity_extended_lists = {u: popularity_extended for u in eval_users}

    # E: user × channel × weekday × 30-minute-slot affinity, scored per future airing.
    keys = ["user_id", "channel_id", "day_of_week", "slot_30m"]
    affinity = hist.groupby(keys).agg(n=("user_watch_pct", "size"), sec=("user_watch_seconds", "sum")).reset_index()
    affinity["airing_affinity"] = np.log1p(affinity.n) + 0.1 * np.log1p(affinity.sec)
    schedule_keys = (target_sched.groupby(["channel_id", "day_of_week", "slot_30m"], as_index=False)
                     .tv_show_id.agg(lambda x: list(set(map(int, x)))))
    air = affinity.merge(schedule_keys, on=["channel_id", "day_of_week", "slot_30m"], how="inner")
    air = air.explode("tv_show_id", ignore_index=True)
    affinity_by_user = defaultdict(list)
    for row in air.itertuples(index=False):
        affinity_by_user[row.user_id].append((int(row.tv_show_id), float(row.airing_affinity)))
    schedule_lists, schedule_extended_lists = {}, {}
    for u in eval_users:
        scores = defaultdict(list)
        for show, score in affinity_by_user.get(u, []):
            if show in catalog_set:
                scores[show].append(score)
        show_score = pd.Series({s: max(vals) + 0.1 * np.log1p(sum(vals)) for s, vals in scores.items()})
        schedule_lists[u] = top_ids(show_score, 100) if len(show_score) else []
        schedule_extended_lists[u] = top_ids(show_score, 500) if len(show_score) else []

    # B: LightFM WARP. The reference uses binary positive user-show interactions
    # from all strict overlaps; it does not discard short overlaps here.
    pairs = history[["user_id", "tv_show_id"]].drop_duplicates()
    user_ids = history.user_id.drop_duplicates().tolist()
    item_ids = history.tv_show_id.drop_duplicates().tolist()
    u_map = {u: i for i, u in enumerate(user_ids)}
    i_map = {s: i for i, s in enumerate(item_ids)}
    mat = csr_matrix((np.ones(len(pairs), dtype=np.float32),
                      (pairs.user_id.map(u_map), pairs.tv_show_id.map(i_map))),
                     shape=(len(user_ids), len(item_ids)))
    model = LightFM(random_state=42, loss="warp")
    model.fit(mat, epochs=15, num_threads=4, verbose=False)
    known_catalog = [s for s in catalog if s in i_map]
    lightfm_lists, lightfm_extended_lists, lightfm_scores, lightfm_ranks = {}, {}, {}, {}
    for u in eval_users:
        if u not in u_map or not known_catalog:
            lightfm_lists[u], lightfm_extended_lists[u], lightfm_scores[u], lightfm_ranks[u] = [], [], {}, {}
            continue
        ids = np.asarray(known_catalog, dtype=np.int32)
        scores = model.predict(u_map[u], np.asarray([i_map[s] for s in known_catalog], dtype=np.int32))
        order = np.argsort(-scores, kind="stable")[:500]
        lightfm_extended_lists[u] = [known_catalog[j] for j in order]
        lightfm_lists[u] = lightfm_extended_lists[u][:200]
        lightfm_scores[u] = {known_catalog[j]: float(scores[j]) for j in order}
        lightfm_ranks[u] = {known_catalog[j]: rank for rank, j in enumerate(order)}

    lightfm_threshold_lists, lightfm_threshold_scores, lightfm_threshold_ranks = {}, {}, {}
    if MODE == "expanded11":
        for threshold in (0.5, 0.8):
            name = f"lightfm_{int(threshold*100):03d}"
            events = history[history.user_watch_pct.ge(threshold)][["user_id", "tv_show_id"]].drop_duplicates()
            threshold_mat = csr_matrix((np.ones(len(events), dtype=np.float32),
                                        (events.user_id.map(u_map), events.tv_show_id.map(i_map))),
                                       shape=(len(user_ids), len(item_ids)))
            threshold_model = LightFM(random_state=42, loss="warp")
            threshold_model.fit(threshold_mat, epochs=15, num_threads=4, verbose=False)
            lists, scores_by_user, ranks_by_user = {}, {}, {}
            item_indices = np.asarray([i_map[show] for show in known_catalog], dtype=np.int32)
            for user in eval_users:
                if user not in u_map or not known_catalog:
                    lists[user], scores_by_user[user], ranks_by_user[user] = [], {}, {}
                    continue
                scores = threshold_model.predict(u_map[user], item_indices)
                order = np.argsort(-scores, kind="stable")[:500]
                lists[user] = [known_catalog[j] for j in order]
                scores_by_user[user] = {known_catalog[j]: float(scores[j]) for j in order}
                ranks_by_user[user] = {known_catalog[j]: rank for rank, j in enumerate(order)}
            lightfm_threshold_lists[name] = lists
            lightfm_threshold_scores[name] = scores_by_user
            lightfm_threshold_ranks[name] = ranks_by_user

    # C: existing tripartite PPR (user–show–channel/category/genre), attr_weight=.5.
    # Only history <= train_end and the current target-window schedule supply graph data.
    ppr_train = history[["user_id", "tv_show_id", "week", "user_watch_seconds"]].rename(
        columns={"user_watch_seconds": "overlap_seconds"})
    item_attrs = build_item_attrs(train_end, target_sched)
    ppr_base = build_graph_base(ppr_train, catalog, item_attrs)
    transition = build_transition(ppr_base, attr_weight=0.5)
    n_users_graph = ppr_base["nU"]
    n_nodes = n_users_graph + ppr_base["nI"] + ppr_base["nA"]
    pool_local = np.asarray([ppr_base["i_map"][s] for s in catalog], dtype=np.int64)
    ppr_lists, ppr_extended_lists, ppr_scores, ppr_ranks = {}, {}, {}, {}
    for start in range(0, len(eval_users), 48):
        batch = eval_users[start:start+48]
        restart = np.zeros((n_nodes, len(batch)), dtype=np.float64)
        for b, u in enumerate(batch):
            ui = ppr_base["u_map"].get(u)
            if ui is None:
                continue
            item_indices, item_weights = ppr_base["per_user"][ui]
            restart[n_users_graph + item_indices, b] = item_weights
        ppr_matrix = ppr_tripartite(transition, restart, n_iter=30, alpha=0.5)
        item_scores = ppr_matrix[n_users_graph + pool_local, :]
        for b, u in enumerate(batch):
            scores = {s: float(item_scores[i, b]) for i, s in enumerate(catalog)}
            ranked = sorted(scores, key=scores.get, reverse=True)[:500]
            ppr_extended_lists[u] = ranked
            ppr_lists[u] = ranked[:100]
            ppr_scores[u] = {s: scores[s] for s in ranked}
            ppr_ranks[u] = {s: rank for rank, s in enumerate(ranked)}
        for u in batch:
            ppr_lists.setdefault(u, [])

    generators = {"repeat": repeat_lists, "lightfm": lightfm_lists, "ppr": ppr_lists,
                  "popularity": popularity_lists, "tv_slot": schedule_lists}
    als_scores, als_ranks = {}, {}
    if MODE in {"expanded", "expanded2", "expanded3", "expanded4", "expanded5", "expanded6", "expanded7", "expanded8", "expanded9", "expanded10", "expanded11"}:
        # Broad channel affinity projects a user's watched channels onto every
        # labelled future show on those channels, even at different time slots.
        user_channels = hist.groupby(["user_id", "channel_id"]).agg(
            views=("user_watch_pct", "size"), seconds=("user_watch_seconds", "sum")
        ).reset_index()
        show_channels = target_sched.groupby(["channel_id", "tv_show_id"], as_index=False).agg(
            future_airings=("tv_show_id", "size"), future_airtime=("duration", "sum")
        )
        projected = user_channels.merge(show_channels, on="channel_id", how="inner")
        projected["channel_score"] = (np.log1p(projected.views) + 0.15 * np.log1p(projected.seconds)
                                       + 0.2 * np.log1p(projected.future_airings))
        projected = projected.groupby(["user_id", "tv_show_id"]).channel_score.max().reset_index()
        channel_lists = {}
        for u, g in projected.groupby("user_id", sort=False):
            channel_lists[u] = top_ids(g.set_index("tv_show_id").channel_score, 500)
        for u in eval_users:
            channel_lists.setdefault(u, [])
        generators = {"repeat": repeat_lists, "lightfm": lightfm_extended_lists,
                      "ppr": ppr_extended_lists, "popularity": popularity_extended_lists,
                      "tv_slot": schedule_extended_lists, "user_channel": channel_lists}
        if MODE in {"expanded5", "expanded6", "expanded7", "expanded8", "expanded9", "expanded10", "expanded11"}:
            time_slot_lists = projected_affinity_lists(
                hist, target_sched, ["day_of_week", "slot_30m"], ["day_of_week", "slot_30m"])
            channel_dow_lists = projected_affinity_lists(
                hist, target_sched, ["channel_id", "day_of_week"], ["channel_id", "day_of_week"])
            channel_hour_lists = projected_affinity_lists(
                hist, target_sched, ["channel_id", "hour"], ["channel_id", "hour"])
            for u in eval_users:
                time_slot_lists.setdefault(u, [])
                channel_dow_lists.setdefault(u, [])
                channel_hour_lists.setdefault(u, [])
            generators.update({"user_time_slot": time_slot_lists,
                               "user_channel_dow": channel_dow_lists,
                               "user_channel_hour": channel_hour_lists})
        if MODE in {"expanded6", "expanded7", "expanded8", "expanded9", "expanded10", "expanded11"}:
            future_airings = target_sched.groupby("tv_show_id").size().reindex(catalog).fillna(0)
            airing_pop_score = recent * np.log1p(future_airings)
            airing_pop_list = top_ids(airing_pop_score, 500)
            generators["airing_popularity"] = {u: airing_pop_list for u in eval_users}

    if MODE in {"expanded2", "expanded3", "expanded4", "expanded5", "expanded6", "expanded7", "expanded8", "expanded9", "expanded10", "expanded11"}:
        # Metadata affinity supplies candidates for globally new or user-cold shows.
        history_schedules = []
        meta_cols = ["tv_show_id", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"]
        for name in ("export_arh_11-20-final.csv",):
            m = pd.read_csv(DATA / name, usecols=meta_cols, low_memory=False)
            history_schedules.append(m[m.tv_show_id.ne(0)])
        history_meta = pd.concat(history_schedules, ignore_index=True).drop_duplicates("tv_show_id")
        target_meta = target_sched[meta_cols].drop_duplicates("tv_show_id")
        hist_meta = hist[["user_id", "tv_show_id", "user_watch_seconds"]].merge(history_meta, on="tv_show_id", how="left")
        attrs = meta_cols[1:]
        user_pref = hist_meta.melt(id_vars=["user_id"], value_vars=attrs, value_name="attribute")
        user_pref = user_pref.dropna(subset=["attribute"]).groupby(["user_id", "variable", "attribute"]).size().rename("pref").reset_index()
        item_attrs = target_meta.melt(id_vars=["tv_show_id"], value_vars=attrs, value_name="attribute").dropna(subset=["attribute"])
        metadata_scores = user_pref.merge(item_attrs, on=["variable", "attribute"], how="inner")
        metadata_scores = metadata_scores.groupby(["user_id", "tv_show_id"]).pref.sum().reset_index()
        metadata_lists = {}
        for u, g in metadata_scores.groupby("user_id", sort=False):
            metadata_lists[u] = top_ids(g.set_index("tv_show_id").pref, 500)
        for u in eval_users:
            metadata_lists.setdefault(u, [])

        # ALS is an additional candidate source and retained as a reranker signal.
        from implicit.als import AlternatingLeastSquares
        weighted_pairs = history.groupby(["user_id", "tv_show_id"]).user_watch_seconds.sum().reset_index()
        weighted_pairs["weight"] = np.log1p(weighted_pairs.user_watch_seconds).astype(np.float32)
        als_mat = csr_matrix((weighted_pairs.weight, (weighted_pairs.user_id.map(u_map),
                                                       weighted_pairs.tv_show_id.map(i_map))),
                             shape=(len(user_ids), len(item_ids)))
        als = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=20,
                                      random_state=42, num_threads=4)
        als.fit(als_mat, show_progress=False)
        als_lists, als_scores, als_ranks = {}, {}, {}
        catalog_indices = np.asarray([i_map[s] for s in known_catalog], dtype=np.int32)
        for u in eval_users:
            if u not in u_map:
                als_lists[u], als_scores[u], als_ranks[u] = [], {}, {}
                continue
            scores = als.user_factors[u_map[u]] @ als.item_factors[catalog_indices].T
            order = np.argsort(-scores, kind="stable")[:500]
            als_lists[u] = [known_catalog[j] for j in order]
            als_scores[u] = {known_catalog[j]: float(scores[j]) for j in order}
            als_ranks[u] = {known_catalog[j]: rank for rank, j in enumerate(order)}
        generators.update({"user_metadata": metadata_lists, "als": als_lists})

    assoc_scores, assoc_ranks = {}, {}
    if MODE == "expanded7":
        from exp_assoc_rules import build_transactions, count_itemsets, frequent_pairs, generate_rules
        train_rules = history[["user_id", "tv_show_id", "week", "user_watch_seconds"]].rename(
            columns={"user_watch_seconds": "overlap_seconds"})
        transactions = build_transactions(train_rules)
        item_count, pair_count = count_itemsets(transactions)
        frequent = frequent_pairs(pair_count, len(transactions), min_support=0.002)
        rules = generate_rules(frequent, item_count, len(transactions))
        rules_by_antecedent = defaultdict(list)
        for rule in rules:
            rules_by_antecedent[rule["antecedent"]].append((int(rule["consequent"]), float(rule["confidence"])))
        recent_user_items = train_rules[train_rules.week.ge(train_end - 3)].groupby("user_id").tv_show_id.apply(set)
        assoc_lists = {}
        global_tie = {show: rank for rank, show in enumerate(popularity_extended)}
        catalog_tie = {show: rank for rank, show in enumerate(catalog)}
        for user in eval_users:
            scores = defaultdict(float)
            for antecedent in recent_user_items.get(user, set()):
                for consequent, confidence in rules_by_antecedent.get(int(antecedent), ()):
                    if consequent in catalog_set:
                        scores[consequent] += confidence
            ranked = sorted(scores, key=lambda show: (-scores[show],
                                                       global_tie.get(show, len(global_tie) + catalog_tie[show])))[:500]
            assoc_lists[user] = ranked if ranked else popularity_extended[:500]
            assoc_scores[user] = {show: float(scores.get(show, 0.0)) for show in assoc_lists[user]}
            assoc_ranks[user] = {show: rank for rank, show in enumerate(assoc_lists[user])}
        generators["assoc_rules"] = assoc_lists
    if MODE == "expanded8":
        generators["repeat_long"] = repeat_extended_lists
    propensity_features = {}
    if MODE in {"expanded9", "expanded10", "expanded11"}:
        propensity_lists, propensity_features = schedule_propensity_candidates(hist, target_sched, eval_users)
        generators.update(propensity_lists)
    rp3_scores, rp3_ranks, ease_scores, ease_ranks = {}, {}, {}, {}
    if MODE == "expanded10":
        # Additional collaborative candidate sources from the existing RP3/EASE grid.
        from scipy.sparse import diags
        known_target = [show for show in catalog if show in i_map]
        target_item_idx = np.asarray([i_map[show] for show in known_target], dtype=np.int32)
        row_sum = np.asarray(mat.sum(axis=1)).ravel()
        row_inv = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
        P_ui = diags(row_inv, format="csr") @ mat
        Xt = mat.T.tocsr()
        item_sum = np.asarray(Xt.sum(axis=1)).ravel()
        item_inv = np.divide(1.0, item_sum, out=np.zeros_like(item_sum), where=item_sum > 0)
        P_iu = diags(item_inv, format="csr") @ Xt
        degree = np.asarray(mat.sum(axis=0)).ravel()
        rp3_matrix = np.power((P_iu @ P_ui).toarray(), 0.5)
        degree_scale = np.ones_like(degree)
        degree_scale[degree > 0] = degree[degree > 0] ** 0.3
        rp3_matrix /= degree_scale[None, :]

        gram = (mat.T @ mat).toarray()
        inverse = np.linalg.inv(gram + 1000.0 * np.eye(gram.shape[0]))
        ease_matrix = -inverse / np.diag(inverse)[None, :]
        np.fill_diagonal(ease_matrix, 0.0)

        rp3_lists, ease_lists = {}, {}
        for user in eval_users:
            if user not in u_map or not len(known_target):
                rp3_lists[user], ease_lists[user] = [], []
                continue
            user_row = u_map[user]
            rp3_all = np.asarray(P_ui[user_row, :] @ rp3_matrix).ravel()
            ease_all = np.asarray(mat[user_row, :].toarray().ravel() @ ease_matrix).ravel()
            rp3_target = rp3_all[target_item_idx]
            ease_target = ease_all[target_item_idx]
            rp3_order = np.argsort(-rp3_target, kind="stable")[:500]
            ease_order = np.argsort(-ease_target, kind="stable")[:500]
            rp3_lists[user] = [known_target[i] for i in rp3_order]
            ease_lists[user] = [known_target[i] for i in ease_order]
            rp3_scores[user] = {known_target[i]: float(rp3_target[i]) for i in rp3_order}
            ease_scores[user] = {known_target[i]: float(ease_target[i]) for i in ease_order}
            rp3_ranks[user] = {known_target[i]: rank for rank, i in enumerate(rp3_order)}
            ease_ranks[user] = {known_target[i]: rank for rank, i in enumerate(ease_order)}
        generators.update({"rp3": rp3_lists, "ease": ease_lists})
    if MODE == "expanded11":
        for source, lists in lightfm_threshold_lists.items():
            generators[source] = lists
    rows, out_rows = [], []
    rrf_k = 60.0
    for u in eval_users:
        rank_sum = defaultdict(float)
        per_source = {}
        source_ranks, source_sets = {}, {}
        for name, lists in generators.items():
            ids = lists.get(u, [])
            per_source[name] = ids
            source_ranks[name] = {show: rank for rank, show in enumerate(ids)}
            source_sets[name] = set(ids)
            for rank, show in enumerate(ids):
                rank_sum[show] += 1.0 / (rrf_k + rank + 1)
        ordered = sorted(rank_sum, key=rank_sum.get, reverse=True)
        for show in ordered:
            rows.append({"user_id": u, "tv_show_id": show, "candidate_score": rank_sum[show],
                         "lightfm_score": lightfm_scores.get(u, {}).get(show, np.nan),
                         "lightfm_rank": lightfm_ranks.get(u, {}).get(show, np.nan),
                         "user_bias": float(model.user_biases[u_map[u]]) if u in u_map else np.nan,
                         "item_bias": float(model.item_biases[i_map[show]]) if show in i_map else np.nan,
                         "ppr_score": ppr_scores.get(u, {}).get(show, np.nan),
                         "ppr_rank": ppr_ranks.get(u, {}).get(show, np.nan),
                         "rp3_score": rp3_scores.get(u, {}).get(show, np.nan),
                         "rp3_rank": rp3_ranks.get(u, {}).get(show, np.nan),
                         "ease_score": ease_scores.get(u, {}).get(show, np.nan),
                         "ease_rank": ease_ranks.get(u, {}).get(show, np.nan),
                         "als_score": als_scores.get(u, {}).get(show, np.nan),
                         "als_rank": als_ranks.get(u, {}).get(show, np.nan),
                         "lightfm_050_score": lightfm_threshold_scores.get("lightfm_050", {}).get(u, {}).get(show, np.nan),
                         "lightfm_050_rank": lightfm_threshold_ranks.get("lightfm_050", {}).get(u, {}).get(show, np.nan),
                         "lightfm_080_score": lightfm_threshold_scores.get("lightfm_080", {}).get(u, {}).get(show, np.nan),
                         "lightfm_080_rank": lightfm_threshold_ranks.get("lightfm_080", {}).get(u, {}).get(show, np.nan),
                         "assoc_rule_score": assoc_scores.get(u, {}).get(show, np.nan),
                         "assoc_rule_rank": assoc_ranks.get(u, {}).get(show, np.nan),
                         "airing_propensity_max": propensity_features.get(u, {}).get(show, {}).get("prop_max", np.nan),
                         "airing_propensity_mean": propensity_features.get(u, {}).get(show, {}).get("prop_mean", np.nan),
                         "airing_propensity_sum": propensity_features.get(u, {}).get(show, {}).get("prop_sum", np.nan),
                         "airing_propensity_logsumexp": propensity_features.get(u, {}).get(show, {}).get("prop_logsumexp", np.nan),
                         "repeat_rank": source_ranks["repeat"].get(show, np.nan),
                         "popularity_rank": source_ranks["popularity"].get(show, np.nan),
                         **{f"rank_{name}": source_ranks[name].get(show, np.nan) for name in generators},
                         **{f"in_{name}": int(show in source_sets[name]) for name in generators}})
        out_rows.append((u, ordered, per_source))

    candidate_df = pd.DataFrame(rows)
    candidate_df.to_parquet(CANDIDATE_OUT, index=False)
    if SPLIT == "final":
        print(f"[exp5] final candidates rows={len(candidate_df):,}; users={candidate_df.user_id.nunique():,}; mean pool/user={candidate_df.groupby('user_id').size().mean():.1f}")
        print(f"[exp5] saved={CANDIDATE_OUT}; target shows={len(catalog):,}; training history<=30; no target views were read")
        return
    metric_rows = []
    for threshold, rel in relevant_by_threshold.items():
        users_eval = [u for u in eval_users if rel.get(u)]
        for source, lists in generators.items():
            for k in (200, 500):
                got = sum(len(set(lists.get(u, [])[:k]) & rel.get(u, set())) for u in users_eval)
                total = sum(len(rel[u]) for u in users_eval)
                metric_rows.append({"threshold": threshold, "generator": source, "cutoff": k,
                                    "recall": got / total if total else 0.0, "hits": got,
                                    "relevant": total, "users": len(users_eval)})
        for k in (200, 500):
            got = total = 0
            for u, ordered, _ in out_rows:
                if u in rel and rel[u]:
                    got += len(set(ordered[:k]) & rel[u]); total += len(rel[u])
            metric_rows.append({"threshold": threshold, "generator": "rrf_union", "cutoff": k,
                                "recall": got / total if total else 0.0, "hits": got,
                                "relevant": total, "users": len(users_eval)})
        # Full-union candidate coverage (before truncation).
        got = total = 0
        for u, ordered, _ in out_rows:
            if u in rel and rel[u]: got += len(set(ordered) & rel[u]); total += len(rel[u])
        metric_rows.append({"threshold": threshold, "generator": "rrf_union", "cutoff": len(catalog),
                            "recall": got / total if total else 0.0, "hits": got,
                            "relevant": total, "users": len(users_eval)})
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(METRICS_OUT, index=False)
    print(f"[exp5] candidates rows={len(candidate_df):,}; users={candidate_df.user_id.nunique():,}; mean union/user={candidate_df.groupby('user_id').size().mean():.1f}")
    print(pd.DataFrame(metric_rows).query("generator == 'rrf_union'").to_string(index=False))
    print(f"[exp5] saved={CANDIDATE_OUT}; {METRICS_OUT}; elapsed_s={time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
