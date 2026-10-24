#!/usr/bin/env python3
"""Tune candidate-source RRF weights on early temporal validation only.

Train candidates/features: history <=16 -> target 17-20. The selected weights
are then evaluated, without retuning, on history <=20 -> target 21-30.
This is candidate-stage Recall@K tuning, not the eventual MAP@5 reranker.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA, TABLES = ROOT / "data", ROOT / "results" / "tables"
TAG = sys.argv[1] if len(sys.argv) > 1 else "bipartite"
if TAG not in {"bipartite", "tripartite", "expanded5", "expanded6", "expanded7", "expanded8", "expanded9", "expanded10", "expanded11"}:
    raise SystemExit("usage: exp5_tune_candidate_fusion.py [bipartite|tripartite|expanded5|expanded6|expanded7|expanded8|expanded9|expanded10|expanded11]")
SUFFIX = {"bipartite": "_all_sources", "tripartite": "_tripartite",
          "expanded5": "_expanded5", "expanded6": "_expanded6", "expanded7": "_expanded7",
          "expanded8": "_expanded8", "expanded9": "_expanded9", "expanded10": "_expanded10",
          "expanded11": "_expanded11"}[TAG]
WEIGHTS_OUT = TABLES / f"exp5_candidate_fusion_weights{SUFFIX}.csv"
METRICS_OUT = TABLES / f"exp5_candidate_fusion_tuned{SUFFIX}.csv"
TUNED_OUT = DATA / f"exp5_candidates_main_tuned{SUFFIX}.parquet"
SOURCES = ["repeat", "lightfm", "ppr", "popularity", "tv_slot", "user_channel", "user_metadata", "als"]
if TAG in {"expanded5", "expanded6", "expanded7", "expanded8", "expanded9", "expanded10", "expanded11"}:
    SOURCES += ["user_time_slot", "user_channel_dow", "user_channel_hour"]
if TAG in {"expanded6", "expanded7", "expanded8", "expanded9", "expanded10", "expanded11"}:
    SOURCES += ["airing_popularity"]
if TAG == "expanded7":
    SOURCES += ["assoc_rules"]
if TAG == "expanded8":
    SOURCES += ["repeat_long"]
if TAG in {"expanded9", "expanded10", "expanded11"}:
    SOURCES += ["prop_max", "prop_mean", "prop_sum", "prop_logsumexp"]
if TAG == "expanded10":
    SOURCES += ["rp3", "ease"]
if TAG == "expanded11":
    SOURCES += ["lightfm_050", "lightfm_080"]
RRF_K = 60.0


def load_truth(start_week, end_week, catalog):
    inter = pd.read_parquet(DATA / "exp5_user_airings_rich.parquet",
                            columns=["user_id", "tv_show_id", "week", "user_watch_pct"])
    inter["user_id"] = inter.user_id.astype(str)
    users = set(pd.read_csv(DATA / "submission.csv", dtype={"user_id": str}).user_id)
    relevant = inter[inter.week.between(start_week, end_week) & inter.user_id.isin(users)
                     & inter.tv_show_id.isin(catalog) & inter.user_watch_pct.ge(0.8)]
    return {u: set(g.tv_show_id) for u, g in relevant.groupby("user_id")}


def prepare(path, start_week, end_week, schedule_file):
    df = pd.read_parquet(path)
    schedule = pd.read_csv(DATA / schedule_file, usecols=["tv_show_id", "start_time"], low_memory=False)
    schedule["start_time"] = pd.to_datetime(schedule.start_time, dayfirst=True)
    schedule = schedule[schedule.start_time.dt.isocalendar().week.between(start_week, end_week)
                        & schedule.tv_show_id.ne(0)]
    catalog = set(schedule.tv_show_id.astype(int))
    truth = load_truth(start_week, end_week, catalog)
    rank_cols = [f"rank_{source}" for source in SOURCES]
    ranks = df[rank_cols].to_numpy(dtype=np.float64)
    components = np.divide(1.0, RRF_K + 1.0 + ranks, out=np.zeros_like(ranks), where=np.isfinite(ranks))
    # Recover the fixed contribution of the remaining candidate sources from
    # the original equal-weight RRF score, avoiding leakage from target labels.
    df["other_source_score"] = np.maximum(0.0, df.candidate_score.to_numpy() - components.sum(axis=1))
    # groupby.indices values are positions; retain dense arrays for repeated scoring.
    groups = [(np.asarray(idx, dtype=np.int64), str(user)) for user, idx in df.groupby("user_id", sort=False).indices.items()]
    return df, components, groups, truth


def recall_at(df, components, groups, truth, weights, k):
    w = np.asarray(weights, dtype=np.float64)
    scores = df.other_source_score.to_numpy() + components @ w
    hits = relevant = 0
    for idx, user in groups:
        actual = truth.get(user)
        if not actual:
            continue
        top = idx[np.argpartition(-scores[idx], min(k, len(idx)) - 1)[:min(k, len(idx))]]
        hits += sum(int(show in actual) for show in df.tv_show_id.to_numpy()[top])
        relevant += len(actual)
    return hits / relevant if relevant else 0.0, hits, relevant


def main():
    for p in (WEIGHTS_OUT, METRICS_OUT, TUNED_OUT):
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite {p}")
    mode = TAG if TAG in {"expanded5", "expanded6", "expanded7", "expanded8", "expanded9", "expanded10", "expanded11"} else ("expanded4" if TAG == "tripartite" else "expanded2")
    early, early_comp, early_groups, early_truth = prepare(
        DATA / f"exp5_candidates_main_{mode}_early.parquet", 17, 20, "export_arh_11-20-final.csv")
    main, main_comp, main_groups, main_truth = prepare(
        DATA / f"exp5_candidates_main_{mode}.parquet", 21, 30, "export_arh_21-30-final.csv")

    weights = np.ones(len(SOURCES), dtype=np.float64)
    grid = (0.0, 0.5, 1.0, 2.0, 4.0)
    # Coordinate search is fit only on early target labels; ties prefer weight 1.
    for _ in range(3):
        changed = False
        for j in range(len(SOURCES)):
            trials = []
            for value in grid:
                trial = weights.copy(); trial[j] = value
                score, _, _ = recall_at(early, early_comp, early_groups, early_truth, trial, 500)
                trials.append((score, -abs(value - 1.0), value))
            best = max(trials)
            if weights[j] != best[2]:
                weights[j] = best[2]; changed = True
        if not changed:
            break

    rows = []
    for split, df, components, groups, truth in (
        ("early_tuning", early, early_comp, early_groups, early_truth),
        ("main_holdout", main, main_comp, main_groups, main_truth),
    ):
        for label, w in (("equal_rrf", np.ones(len(SOURCES))), ("early_tuned", weights)):
            for threshold in (0.5, 0.8):
                # Candidate generators stay fixed; this compares two target
                # relevance definitions on the same out-of-window predictions.
                rel_inter = pd.read_parquet(DATA / "exp5_user_airings_rich.parquet",
                                            columns=["user_id", "tv_show_id", "week", "user_watch_pct"])
                rel_inter.user_id = rel_inter.user_id.astype(str)
                users = set(pd.read_csv(DATA / "submission.csv", dtype={"user_id": str}).user_id)
                start, end = (17, 20) if split == "early_tuning" else (21, 30)
                sched_name = "export_arh_11-20-final.csv" if split == "early_tuning" else "export_arh_21-30-final.csv"
                sched = pd.read_csv(DATA / sched_name, usecols=["tv_show_id", "start_time"], low_memory=False)
                sched["start_time"] = pd.to_datetime(sched.start_time, dayfirst=True)
                catalog = set(sched.loc[sched.start_time.dt.isocalendar().week.between(start, end)
                                        & sched.tv_show_id.ne(0), "tv_show_id"].astype(int))
                relevant_df = rel_inter[rel_inter.week.between(start, end) & rel_inter.user_id.isin(users)
                                        & rel_inter.tv_show_id.isin(catalog)
                                        & rel_inter.user_watch_pct.ge(threshold)]
                rel = relevant_df.groupby("user_id").tv_show_id.apply(set).to_dict()
                for k in (200, 500):
                    value, hits, total = recall_at(df, components, groups, rel, w, k)
                    rows.append({"split": split, "ranking": label, "threshold": threshold,
                                 "cutoff": k, "recall": value, "hits": hits,
                                 "relevant": total, "users": len(rel)})

    weights_df = pd.DataFrame({"source": SOURCES, "weight": weights})
    weights_df["tuning_split"] = "history<=16,target=17-20"
    weights_df.to_csv(WEIGHTS_OUT, index=False)
    pd.DataFrame(rows).to_csv(METRICS_OUT, index=False)
    main_scores = main.other_source_score.to_numpy() + main_comp @ weights
    main["candidate_score_tuned"] = main_scores
    main["rank_tuned"] = main.groupby("user_id", sort=False).candidate_score_tuned.rank(method="first", ascending=False).astype(np.int32) - 1
    main.to_parquet(TUNED_OUT, index=False)
    print("weights tuned on early split:", dict(zip(SOURCES, weights.tolist())))
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"saved {WEIGHTS_OUT}; {METRICS_OUT}; {TUNED_OUT}")


if __name__ == "__main__":
    main()
