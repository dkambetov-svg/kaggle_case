#!/usr/bin/env python3
"""Общие утилиты раунда exp4: параметризованный ALS, подготовка сплита/пула.

E5 требует перебора factors/reg/iterations — исходный als_rankings из
exp_rrf_hybrid.py зашит константами (64/0.05/20), поэтому здесь своя версия
als_rankings_p с параметрами и k-пулем из len(pool) (для E8 с большим пулом).

Никакие прежние таблицы не перезаписываются; выходные файлы раунда —
results/tables/exp4_*.csv.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

from exp_rrf_hybrid import (
    DATA, SEED, evaluate, build_split, rank_list_from_scores,
)


def als_rankings_p(train, val_users, pool, rank_pop, factors=64, reg=0.05,
                   iterations=20):
    """Как exp_rrf_hybrid.als_rankings, но factors/reg/iterations — параметры;
    длина ранжирования = len(pool) (не константа POOL)."""
    train_items = train["tv_show_id"].unique()
    train_users = train["user_id"].unique()
    u_map = {u: i for i, u in enumerate(train_users)}
    i_map = {it: i for i, it in enumerate(train_items)}
    rows_u = train["user_id"].map(u_map).values
    cols_i = train["tv_show_id"].map(i_map).values
    data = np.log1p(train["overlap_seconds"].values)
    mat = csr_matrix((data, (rows_u, cols_i)), shape=(len(u_map), len(i_map)))
    als = AlternatingLeastSquares(factors=factors, regularization=reg,
                                  iterations=iterations, random_state=SEED)
    als.fit(mat)
    pool_known = [c for c in pool if c in i_map]
    pool_idx = np.array([i_map[c] for c in pool_known])
    pool_factors = als.item_factors[pool_idx] if len(pool_idx) else np.zeros((0, factors))
    rankings = {}
    cold = []
    k = len(pool)
    for u in val_users:
        if u not in u_map or len(pool_known) == 0:
            cold.append(u)
            continue
        uf = als.user_factors[u_map[u]]
        scores = pool_factors @ uf
        sc = dict(zip(pool_known, scores))
        min_sc = min(sc.values()) - 1.0
        for c in pool:
            if c not in sc:
                sc[c] = min_sc
        rankings[u] = rank_list_from_scores(sc, rank_pop, k=k)
    for u in cold:
        rankings[u] = list(pool)
    return rankings


def prepare_split(inter, sub_users, cfg, pool_size=500):
    """(train, val_sched, candidates, relevant, val_users, pool_full, rank_pop)."""
    train, val_sched, candidates, relevant, val_users = build_split(
        inter, sub_users, cfg["train_end_week"], cfg["val_start_week"],
        cfg["val_end_week"], cfg["sched_val_path"])
    max_week = cfg["max_week"]
    airings = pd.read_csv(DATA / cfg["airings_file"], low_memory=False)
    airings = airings[airings["tv_show_id"] != 0]["tv_show_id"].value_counts().to_dict()
    pop_recent = train[train["week"] >= max_week - 3].groupby("tv_show_id")["user_id"].nunique()
    score_b = {c: pop_recent.get(c, 0) * np.log1p(airings.get(c, 0)) for c in candidates}
    pool_full = rank_list_from_scores(score_b, score_b, k=pool_size)
    rank_pop = {it: r for r, it in enumerate(pool_full)}
    return train, val_sched, candidates, relevant, val_users, pool_full, rank_pop


SPLITS = {
    "main": dict(train_end_week=20, val_start_week=21, val_end_week=30,
                 sched_val_path=DATA / "export_arh_21-30-final.csv", max_week=20,
                 airings_file="export_arh_21-30-final.csv",
                 attrs_files=[DATA / "export_arh_11-20-final.csv",
                              DATA / "export_arh_21-30-final.csv"]),
    "early": dict(train_end_week=16, val_start_week=17, val_end_week=20,
                  sched_val_path=DATA / "export_arh_11-20-final.csv", max_week=16,
                  airings_file="export_arh_11-20-final.csv",
                  attrs_files=[DATA / "export_arh_11-20-final.csv"]),
}


def load_inter():
    inter = pd.read_parquet(DATA / "user_interactions.parquet")
    inter["user_id"] = inter["user_id"].astype(str)
    sub = pd.read_csv(DATA / "submission.csv", dtype={"user_id": str})
    return inter, sub["user_id"].tolist()
