#!/usr/bin/env python3
"""Финальные сабмиты RRF-гибридов (k=60): лучший гибрид из greedy ablation (pop×airings + PPR + ALS),
PPR-гибрид (pop×airings + PPR), AR-гибрид (pop×airings + AR).
Обучение: недели 11-30. Пул: топ-500 по pop(27-30)×log(1+airings в 31-42).
Ничего из текущего финала/прежних результатов не перезаписывает: новые файлы
data/submission_rrf_*.csv."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "artifacts" / "scripts"))
from exp_rrf_hybrid import (
    DATA, RRF_K, POOL,
    rank_list_from_scores, ppr_rankings, ar_rankings, als_rankings,
    fuse_rrf,
)


def main():
    inter = pd.read_parquet(DATA / "user_interactions.parquet")
    inter["user_id"] = inter["user_id"].astype(str)
    train = inter[inter["week"] <= 30]
    sub = pd.read_csv(DATA / "submission.csv", dtype={"user_id": str})
    test_users = sub["user_id"].tolist()

    sched42 = pd.read_csv(DATA / "export_arh_31-42-final.csv", low_memory=False)
    candidates = sched42[sched42["tv_show_id"] != 0]["tv_show_id"].unique()
    candidates_set = set(candidates)
    airings = sched42[sched42["tv_show_id"] != 0]["tv_show_id"].value_counts().to_dict()

    max_week = 30
    pop_recent = train[train["week"] >= max_week - 3].groupby("tv_show_id")["user_id"].nunique()
    score_b = {c: pop_recent.get(c, 0) * np.log1p(airings.get(c, 0)) for c in candidates}
    pool = rank_list_from_scores(score_b, score_b, k=POOL)
    rank_pop = {it: r for r, it in enumerate(pool)}
    print(f"[pool] топ-{len(pool)} по pop(27-30)×log(1+airings 31-42)")

    t0 = time.time()
    ppr_r = ppr_rankings(train, test_users, pool, rank_pop)
    print(f"[PPR] done ({time.time()-t0:.1f}s), warm={sum(1 for u in test_users if u in ppr_r)}")

    t0 = time.time()
    ar_r = ar_rankings(train, test_users, candidates, rank_pop, max_week)
    print(f"[AR] done ({time.time()-t0:.1f}s)")

    t0 = time.time()
    als_r = als_rankings(train, test_users, pool, rank_pop)
    print(f"[ALS] done ({time.time()-t0:.1f}s)")

    base_rank = {u: list(pool) for u in test_users}

    variants = {
        "best": ["pop_airings", "PPR", "ALS"],
        "ppr": ["pop_airings", "PPR"],
        "ar": ["pop_airings", "AR"],
    }
    rankings_by_component = {
        "pop_airings": base_rank,
        "PPR": ppr_r,
        "AR": ar_r,
        "ALS": als_r,
    }

    def write_submission(preds, path):
        out = sub.copy()
        out["tv_show_id"] = out["user_id"].map(lambda u: " ".join(str(int(x)) for x in preds[u][:5]))
        assert (out["user_id"].values == sub["user_id"].values).all()
        for ids in out["tv_show_id"]:
            parts = ids.split()
            assert len(parts) == 5
            for p in parts:
                assert int(p) in candidates_set
        out.to_csv(path, index=False)
        print(f"[saved] {path} rows={len(out)}")

    for name, selected in variants.items():
        preds = fuse_rrf(rankings_by_component, selected, pool, k=5)
        path = DATA / f"submission_rrf_{name}_{'_'.join(selected)}.csv".replace("pop_airings", "popair")
        write_submission(preds, path)


if __name__ == "__main__":
    main()
