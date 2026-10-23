#!/usr/bin/env python3
"""exp3: 5-й (последний) сабмит раунда — fusion k=20 w=(1,2,1) + PPR E1 без decay.

Обоснование: на Kaggle чистый E1 (0.32306) чуть лучше E2 с decay (0.32261),
локально fusion с PPR E1 (0.5365/0.4632) не хуже decayed-версии (0.5357/0.4626)
на обоих сплитах. Обучение 11–30, тот же пул и формат.
Выход: data/submission_exp3_e3b.csv (+ проверка формата).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "artifacts" / "scripts"))
from exp_rrf_hybrid import DATA, POOL, SEED, rank_list_from_scores, als_rankings
from exp3_e1_attr_ppr import load_item_attrs, build_graph_base
from exp3_e3_rrf_weights import tripartite_rankings, rank_matrix, fuse_weighted_top5
from exp3_submit import check_format

BEST_AW = 0.5
W = (1.0, 2.0, 1.0)
K_RRF = 20
MAX_WEEK = 30


def main():
    inter = pd.read_parquet(DATA / "user_interactions.parquet")
    inter["user_id"] = inter["user_id"].astype(str)
    train = inter[inter["week"] <= MAX_WEEK]
    sub = pd.read_csv(DATA / "submission.csv", dtype={"user_id": str})
    test_users = sub["user_id"].tolist()

    sched42 = pd.read_csv(DATA / "export_arh_31-42-final.csv", low_memory=False)
    candidates = sched42[sched42["tv_show_id"] != 0]["tv_show_id"].unique()
    candidates_set = set(int(c) for c in candidates)
    airings = sched42[sched42["tv_show_id"] != 0]["tv_show_id"].value_counts().to_dict()
    pop_recent = train[train["week"] >= MAX_WEEK - 3].groupby("tv_show_id")["user_id"].nunique()
    score_b = {c: pop_recent.get(c, 0) * np.log1p(airings.get(c, 0)) for c in candidates}
    pool_full = rank_list_from_scores(score_b, score_b, k=POOL)
    rank_pop = {it: r for r, it in enumerate(pool_full)}
    print(f"[pool] топ-{len(pool_full)}; train<= {MAX_WEEK}")

    item_attrs = load_item_attrs([DATA / "export_arh_11-20-final.csv",
                                  DATA / "export_arh_21-30-final.csv",
                                  DATA / "export_arh_31-42-final.csv"])
    t0 = time.perf_counter()
    base = build_graph_base(train, pool_full, item_attrs)  # без decay (E1)
    ppr_r, _ = tripartite_rankings(base, base["per_user"],
                                   pool_full, rank_pop, test_users, BEST_AW)
    print(f"[PPR E1 aw={BEST_AW} без decay] {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    als_r = als_rankings(train, test_users, pool_full, rank_pop)
    print(f"[ALS f=64] {time.perf_counter() - t0:.1f}s")

    M_pop = rank_matrix({u: list(pool_full) for u in test_users}, test_users, pool_full)
    M_ppr = rank_matrix(ppr_r, test_users, pool_full)
    M_als = rank_matrix(als_r, test_users, pool_full)
    order = fuse_weighted_top5([M_pop, M_ppr, M_als], W, K_RRF)
    preds = {u: [pool_full[j] for j in order[i]] for i, u in enumerate(test_users)}

    out = sub.copy()
    out["tv_show_id"] = out["user_id"].map(
        lambda u: " ".join(str(int(x)) for x in preds[u][:5]))
    path = DATA / "submission_exp3_e3b.csv"
    out.to_csv(path, index=False)
    print(f"[saved] {path.name} — RRF k={K_RRF} w=(pop1,ppr2,als1), PPR E1 без decay")
    check_format(path, DATA / "submission.csv", candidates_set)


if __name__ == "__main__":
    main()
