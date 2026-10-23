#!/usr/bin/env python3
"""Три сабмита RRF k=60 (обучение 11-30, пул топ-500 по pop(27-30)×log(1+airings в 31-42)):
(a) pop×airings + PPR restart=0.5 + ALS  (как текущий финал, но restart 0.5 вместо 0.15)
(b) pop×airings + PPR restart=0.15 + EASE(lambda=2000) + ALS
(c) чистый PPR restart=0.3
Ничего из текущего финала/прежних результатов не перезаписывает: data/submission_rrf2_*.csv."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "artifacts" / "scripts"))
from exp_rrf_hybrid import (
    DATA, RRF_K, POOL,
    rank_list_from_scores, ar_rankings, als_rankings,
)
from exp_grid_rp3_ease_ppr_extra import build_ppr_graph_local, ppr_batch


def fuse_rrf(rankings_by_component, selected, pool, k=5):
    preds = {}
    users = set()
    for name in selected:
        users.update(rankings_by_component[name].keys())
    for u in users:
        sc = {}
        for name in selected:
            for r, item in enumerate(rankings_by_component[name].get(u, [])[:POOL]):
                sc[item] = sc.get(item, 0.0) + 1.0 / (RRF_K + r)
        if not sc:
            preds[u] = list(pool[:k])
            continue
        items = list(sc.keys())
        scores = np.array([sc[i] for i in items])
        order = sorted(range(len(items)), key=lambda i: -scores[i])
        preds[u] = [items[o] for o in order[:k]]
    return preds


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

    # ---- PPR граф (один раз) ----
    u_map_g, i_map_g, Mu2i, Mi2u = build_ppr_graph_local(train)
    pool_items_known = [c for c in pool if c in i_map_g]
    pool_idx_g = np.array([i_map_g[c] for c in pool_items_known])
    pool_unknown = [c for c in pool if c not in i_map_g]
    tail_scores = {c: -1e9 for c in pool_unknown}

    Bb = len(test_users)
    Vu = np.zeros((len(u_map_g), Bb))
    Vi = np.zeros((len(i_map_g), Bb))
    for b, u in enumerate(test_users):
        if u in u_map_g:
            Vi[:, b] = Mu2i[u_map_g[u], :].toarray().ravel()

    def ppr_rank_list(restart):
        t0 = time.time()
        alpha = 1.0 - restart
        Pu, Pi = ppr_batch(Mu2i, Mi2u, Vu, Vi, alpha=alpha, n_iter=30)
        scores_pool = Pi[pool_idx_g, :]
        rankings = {}
        for b, u in enumerate(test_users):
            if u not in u_map_g:
                rankings[u] = list(pool)
                continue
            sc = {it: float(scores_pool[j, b]) for j, it in enumerate(pool_items_known)}
            sc.update(tail_scores)
            rankings[u] = rank_list_from_scores(sc, rank_pop, k=POOL)
        print(f"[PPR restart={restart}] done ({time.time()-t0:.1f}s)")
        return rankings

    ppr_015 = ppr_rank_list(0.15)
    ppr_05 = ppr_rank_list(0.5)
    ppr_03 = ppr_rank_list(0.3)

    # ---- ALS ----
    t0 = time.time()
    als_r = als_rankings(train, test_users, pool, rank_pop)
    print(f"[ALS] done ({time.time()-t0:.1f}s)")

    # ---- EASE (binary X, lambda=2000) ----
    t0 = time.time()
    agg = train.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum().reset_index()
    users = agg["user_id"].unique()
    items = agg["tv_show_id"].unique()
    u_map = {u: i for i, u in enumerate(users)}
    i_map = {it: i for i, it in enumerate(items)}
    rows_u = agg["user_id"].map(u_map).values
    cols_i = agg["tv_show_id"].map(i_map).values
    X_bin = csr_matrix((np.ones(len(agg)), (rows_u, cols_i)), shape=(len(u_map), len(i_map)))
    XtX = (X_bin.T @ X_bin).toarray()
    n_items = XtX.shape[0]
    lam = 2000
    P_inv = np.linalg.inv(XtX + lam * np.eye(n_items))
    diagP = np.diag(P_inv).copy()
    Bease = -P_inv / diagP[None, :]
    np.fill_diagonal(Bease, 0.0)
    ease_rankings = {}
    pool_known_ease = [c for c in pool if c in i_map]
    tail_ease = {c: -1e9 for c in pool if c not in i_map}
    for u in test_users:
        ui = u_map.get(u)
        if ui is None:
            ease_rankings[u] = list(pool)
            continue
        xu = X_bin[ui, :].toarray().ravel()
        scores = xu @ Bease
        sc = {it: float(scores[i_map[it]]) for it in pool_known_ease}
        sc.update(tail_ease)
        ease_rankings[u] = rank_list_from_scores(sc, rank_pop, k=POOL)
    print(f"[EASE lambda={lam}] done ({time.time()-t0:.1f}s)")

    base_rank = {u: list(pool) for u in test_users}
    rankings_by_component = {
        "pop_airings": base_rank,
        "PPR015": ppr_015,
        "PPR03": ppr_03,
        "PPR05": ppr_05,
        "EASE": ease_rankings,
        "ALS": als_r,
    }

    variants = {
        "a_ppr05_als": ["pop_airings", "PPR05", "ALS"],
        "b_ppr015_ease_als": ["pop_airings", "PPR015", "EASE", "ALS"],
        "c_ppr03": ["PPR03"],
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
        write_submission(preds, DATA / f"submission_rrf2_{name}.csv")


if __name__ == "__main__":
    main()
