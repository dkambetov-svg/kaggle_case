#!/usr/bin/env python3
"""Сабмиты по лучшим вариантам этапа 1 (НЕ перезаписывает текущий финал).
Лучшая популярность (main split): window=2w, unique viewers, alpha=0 (без поправки на airings).
Лучший ALS-гибрид на её основе: pool=50, w=0.1 (см. results/tables/exp_stage1_sorted.csv)."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SEED = 42


def rank_from_scores(ids, scores, tie_breaker_rank):
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], tie_breaker_rank.get(ids[i], 10**9)))
    return {ids[o]: r for r, o in enumerate(order)}


def main():
    inter = pd.read_parquet(DATA / "user_interactions.parquet")
    inter["user_id"] = inter["user_id"].astype(str)
    train = inter[inter["week"] <= 30].copy()  # полный train для финального инференса
    sub = pd.read_csv(DATA / "submission.csv", dtype={"user_id": str})

    sched42 = pd.read_csv(DATA / "export_arh_31-42-final.csv", low_memory=False)
    candidates = sched42[sched42["tv_show_id"] != 0]["tv_show_id"].unique()
    candidates_set = set(candidates)

    # Лучшая популярность: window=2 недели (последние 2 недели трейна = 29-30), unique viewers, alpha=0
    max_week = 30
    window = 2
    recent = train[train["week"] >= max_week - window + 1]
    pop = recent.groupby("tv_show_id")["user_id"].nunique().reindex(candidates).fillna(0.0).sort_values(ascending=False)
    top5_pop = pop.index.tolist()[:5]
    print(f"[pop] window={window}w (недели {max_week-window+1}-{max_week}), unique, alpha=0; top5={top5_pop}")

    preds_pop = {u: top5_pop for u in sub["user_id"]}

    # ALS-гибрид: pool=50 по этой популярности, w=0.1
    pool = pop.index.tolist()[:50]
    rank_pop = {it: r for r, it in enumerate(pool)}

    train_items = train["tv_show_id"].unique()
    train_users = train["user_id"].unique()
    u_map = {u: i for i, u in enumerate(train_users)}
    i_map = {it: i for i, it in enumerate(train_items)}
    rows_u = train["user_id"].map(u_map).values
    cols_i = train["tv_show_id"].map(i_map).values
    data = np.log1p(train["overlap_seconds"].values)
    mat = csr_matrix((data, (rows_u, cols_i)), shape=(len(u_map), len(i_map)))
    als = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=20, random_state=SEED)
    als.fit(mat)

    pool_als_known = [it for it in pool if it in i_map]
    pool_als_idx = np.array([i_map[it] for it in pool_als_known])
    pool_als_factors = als.item_factors[pool_als_idx] if len(pool_als_idx) else np.zeros((0, 64))

    preds_hybrid = {}
    w = 0.1
    for u in sub["user_id"]:
        if u not in u_map or len(pool_als_known) == 0:
            rank_model = {}
        else:
            uf = als.user_factors[u_map[u]]
            scores = pool_als_factors @ uf
            rank_model = rank_from_scores(pool_als_known, scores, rank_pop)
        score = {it: (1 - w) * rank_pop[it] + w * rank_model.get(it, len(pool)) for it in pool}
        top = sorted(score.items(), key=lambda kv: (kv[1], rank_pop[kv[0]]))[:5]
        preds_hybrid[u] = [it for it, _ in top]

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

    write_submission(preds_pop, DATA / "submission_exp_stage1_pop_w2.csv")
    write_submission(preds_hybrid, DATA / "submission_exp_stage1_hybrid_als_pool50.csv")


if __name__ == "__main__":
    main()
