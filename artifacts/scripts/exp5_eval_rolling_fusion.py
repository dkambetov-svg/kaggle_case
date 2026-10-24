#!/usr/bin/env python3
"""Apply early-tuned tripartite candidate-fusion weights to rolling windows."""
from pathlib import Path

import numpy as np
import pandas as pd
from exp5_tune_candidate_fusion import RRF_K, SOURCES, prepare, recall_at

ROOT = Path(__file__).resolve().parents[2]
DATA, TABLES = ROOT / "data", ROOT / "results" / "tables"
OUTPUT = TABLES / "exp5_candidate_fusion_rolling_tripartite.csv"
WEIGHTS = TABLES / "exp5_candidate_fusion_weights_tripartite.csv"
SPLITS = {
    "roll21_24": (21, 24, "export_arh_21-30-final.csv"),
    "roll25_27": (25, 27, "export_arh_21-30-final.csv"),
    "roll28_30": (28, 30, "export_arh_21-30-final.csv"),
}


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    tuned = pd.read_csv(WEIGHTS).set_index("source").weight
    weights = np.asarray([tuned[s] for s in SOURCES], dtype=np.float64)
    submission_users = set(pd.read_csv(DATA / "submission.csv", dtype={"user_id": str}).user_id)
    inter = pd.read_parquet(DATA / "exp5_user_airings_rich.parquet",
                            columns=["user_id", "tv_show_id", "week", "user_watch_pct"])
    inter.user_id = inter.user_id.astype(str)
    rows = []
    for split, (start_week, end_week, sched_name) in SPLITS.items():
        candidates_path = DATA / f"exp5_candidates_{split}_expanded4.parquet"
        candidates, components, groups, _ = prepare(candidates_path, start_week, end_week, sched_name)
        schedule = pd.read_csv(DATA / sched_name, usecols=["tv_show_id", "start_time"], low_memory=False)
        schedule.start_time = pd.to_datetime(schedule.start_time, dayfirst=True)
        iso_week = schedule.start_time.dt.isocalendar().week
        catalog = set(schedule.loc[iso_week.between(start_week, end_week) & schedule.tv_show_id.ne(0),
                                   "tv_show_id"].astype(int))
        target = inter[inter.week.between(start_week, end_week) & inter.user_id.isin(submission_users)
                       & inter.tv_show_id.isin(catalog)]
        for threshold in (0.5, 0.8):
            relevant = target[target.user_watch_pct.ge(threshold)].groupby("user_id").tv_show_id.apply(set).to_dict()
            for name, source_weights in (("equal_rrf", np.ones(len(SOURCES))), ("early_tuned", weights)):
                for k in (200, 500):
                    value, hits, total = recall_at(candidates, components, groups, relevant, source_weights, k)
                    rows.append({"split": split, "ranking": name, "threshold": threshold,
                                 "cutoff": k, "recall": value, "hits": hits,
                                 "relevant": total, "users": sum(bool(v) for v in relevant.values())})
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT, index=False)
    print(out.to_string(index=False))
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
