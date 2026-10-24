#!/usr/bin/env python3
"""Tune source RRF weights for candidate ordering by early MAP@5.

Weights are optimized on history<=16 / target17-20 and evaluated frozen on
history<=20 / target21-30. This is candidate-ordering only; no learned reranker
is trained, and Recall@500 remains a separate generation gate.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA, TABLES = ROOT / "data", ROOT / "results" / "tables"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "base"
if VARIANT not in {"base", "widegrid", "propensity", "propensity_widegrid", "lightfm_threshold"}:
    raise SystemExit("usage: exp5_tune_candidate_map.py [base|widegrid|propensity|propensity_widegrid|lightfm_threshold]")
SUFFIX = {"base": "", "widegrid": "_widegrid", "propensity": "_propensity",
          "propensity_widegrid": "_propensity_widegrid", "lightfm_threshold": "_lightfm_threshold"}[VARIANT]
METRICS_OUT = TABLES / f"exp5_candidate_map_tuned{SUFFIX}.csv"
WEIGHTS_OUT = TABLES / f"exp5_candidate_map_weights{SUFFIX}.csv"
SOURCES = ["repeat", "lightfm", "ppr", "popularity", "tv_slot", "user_channel",
           "user_metadata", "als", "user_time_slot", "user_channel_dow",
           "user_channel_hour", "airing_popularity"]
if VARIANT in {"propensity", "propensity_widegrid"}:
    SOURCES += ["prop_max", "prop_mean", "prop_sum", "prop_logsumexp"]
if VARIANT == "lightfm_threshold":
    SOURCES += ["prop_max", "prop_mean", "prop_sum", "prop_logsumexp", "lightfm_050", "lightfm_080"]
RRF_K = 60.0


def load_window(path, start, end, schedule_file):
    candidates = pd.read_parquet(path)
    ranks = candidates[[f"rank_{s}" for s in SOURCES]].to_numpy(dtype=np.float64)
    components = np.divide(1.0, RRF_K + 1.0 + ranks, out=np.zeros_like(ranks), where=np.isfinite(ranks))
    group_indices = [(str(u), np.asarray(idx, dtype=np.int64))
                     for u, idx in candidates.groupby("user_id", sort=False).indices.items()]

    schedule = pd.read_csv(DATA / schedule_file, usecols=["tv_show_id", "start_time"], low_memory=False)
    schedule.start_time = pd.to_datetime(schedule.start_time, dayfirst=True)
    weeks = schedule.start_time.dt.isocalendar().week
    catalog = set(schedule.loc[weeks.between(start, end) & schedule.tv_show_id.ne(0), "tv_show_id"].astype(int))

    inter = pd.read_parquet(DATA / "exp5_user_airings_rich.parquet",
                            columns=["user_id", "tv_show_id", "week", "user_watch_pct"])
    inter.user_id = inter.user_id.astype(str)
    users = set(pd.read_csv(DATA / "submission.csv", dtype={"user_id": str}).user_id)
    target = inter[inter.week.between(start, end) & inter.user_id.isin(users)
                   & inter.tv_show_id.isin(catalog)]
    truths = {}
    for threshold in (0.5, 0.8):
        relevant = target[target.user_watch_pct.ge(threshold)]
        truths[threshold] = relevant.groupby("user_id").tv_show_id.apply(set).to_dict()
    return candidates, components, group_indices, truths


def evaluate(candidates, components, groups, truth, weights):
    scores = components @ np.asarray(weights, dtype=np.float64)
    show_ids = candidates.tv_show_id.to_numpy()
    aps, hit_count, relevant_count = [], 0, 0
    for user, idx in groups:
        actual = truth.get(user)
        if not actual:
            continue
        local_scores = scores[idx]
        k = min(5, len(idx))
        if len(idx) <= k:
            top_idx = idx
        else:
            partition = np.argpartition(-local_scores, k - 1)[:k]
            cutoff = local_scores[partition].min()
            above = idx[local_scores > cutoff]
            tied = idx[local_scores == cutoff]
            needed = k - len(above)
            if len(tied) > needed:
                tied = tied[np.argsort(show_ids[tied], kind="stable")[:needed]]
            top_idx = np.concatenate((above, tied))
        order = np.lexsort((show_ids[top_idx], -scores[top_idx]))
        top = show_ids[top_idx[order]]
        hits, ap = 0, 0.0
        for rank, show in enumerate(top, start=1):
            if show in actual:
                hits += 1
                ap += hits / rank
        aps.append(ap / min(len(actual), 5))
        hit_count += len(set(top) & actual)
        relevant_count += len(actual)
    map5 = float(np.mean(aps)) if aps else 0.0
    recall = hit_count / relevant_count if relevant_count else 0.0
    return map5, len(aps), recall


def recall_at_500(candidates, components, groups, truth, weights):
    scores = components @ np.asarray(weights, dtype=np.float64)
    hit_count, relevant_count = 0, 0
    shows_all = candidates.tv_show_id.to_numpy()
    for user, idx in groups:
        actual = truth.get(user)
        if not actual:
            continue
        if len(idx) <= 500:
            top_idx = idx
        else:
            pick = np.argpartition(-scores[idx], 499)[:500]
            local = np.lexsort((shows_all[idx[pick]], -scores[idx[pick]]))
            top_idx = idx[pick[local]]
        hit_count += len(set(shows_all[top_idx]) & actual)
        relevant_count += len(actual)
    return hit_count / relevant_count if relevant_count else 0.0


def main():
    for p in (METRICS_OUT, WEIGHTS_OUT):
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite {p}")
    candidate_mode = ("expanded9" if VARIANT in {"propensity", "propensity_widegrid"}
                      else "expanded11" if VARIANT == "lightfm_threshold" else "expanded6")
    early = load_window(DATA / f"exp5_candidates_main_{candidate_mode}_early.parquet", 17, 20,
                        "export_arh_11-20-final.csv")
    main = load_window(DATA / f"exp5_candidates_main_{candidate_mode}.parquet", 21, 30,
                       "export_arh_21-30-final.csv")
    weights_by_threshold = {}
    wide = VARIANT in {"widegrid", "propensity_widegrid"}
    grid = ((0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0) if wide
            else (0.0, 0.5, 1.0, 2.0, 4.0))
    n_passes = 4 if wide else 2
    for threshold in (0.5, 0.8):
        weights = np.ones(len(SOURCES), dtype=np.float64)
        early_df, early_comp, early_groups, early_truths = early
        truth = early_truths[threshold]
        for _ in range(n_passes):
            changed = False
            for j in range(len(SOURCES)):
                trials = []
                for value in grid:
                    trial = weights.copy(); trial[j] = value
                    map5, _, _ = evaluate(early_df, early_comp, early_groups, truth, trial)
                    trials.append((map5, -abs(value - 1.0), value))
                best = max(trials)
                if weights[j] != best[2]:
                    weights[j] = best[2]; changed = True
            if not changed:
                break
        weights_by_threshold[threshold] = weights

    metric_rows = []
    for split_name, data in (("early_tune", early), ("main_holdout", main)):
        candidates, components, groups, truths = data
        rankings = {"equal_rrf": np.ones(len(SOURCES))}
        if split_name == "early_tune":
            rankings.update({f"early_tuned_target{threshold:.1f}": w
                             for threshold, w in weights_by_threshold.items()})
        else:
            rankings.update({f"early_tuned_target{threshold:.1f}": w
                             for threshold, w in weights_by_threshold.items()})
        for threshold, truth in truths.items():
            for name, w in rankings.items():
                map5, n_users, _ = evaluate(candidates, components, groups, truth, w)
                recall500 = recall_at_500(candidates, components, groups, truth, w)
                metric_rows.append({"split": split_name, "target_threshold": threshold,
                                    "ranking": name, "MAP5": map5, "Recall500": recall500,
                                    "n_users": n_users})

    weight_rows = [{"target_threshold": threshold, "source": source, "weight": float(weight),
                    "tuning_split": "history<=16,target=17-20", "objective": "MAP@5"}
                   for threshold, weights in weights_by_threshold.items()
                   for source, weight in zip(SOURCES, weights)]
    pd.DataFrame(weight_rows).to_csv(WEIGHTS_OUT, index=False)
    result = pd.DataFrame(metric_rows)
    result.to_csv(METRICS_OUT, index=False)
    print("Early MAP@5-tuned candidate weights:")
    print(pd.DataFrame(weight_rows).to_string(index=False))
    print("\nMAP@5 and Recall@500 (Recall remains the candidate-generation gate):")
    print(result.to_string(index=False))
    print(f"saved {WEIGHTS_OUT}; {METRICS_OUT}")


if __name__ == "__main__":
    main()
