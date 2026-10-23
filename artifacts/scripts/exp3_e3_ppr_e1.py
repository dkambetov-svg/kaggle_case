#!/usr/bin/env python3
"""exp3: локальная оценка 5-го сабмита — E3-fusion (k=20, w=(1,2,1)) с PPR E1 (без decay).

Гипотеза по Kaggle: decay слегка вредит (E1 0.32306 > E2 0.32261 на public),
поэтому fusion-конфигурация лучшего E3, но с PPR без decay, может быть лучше.
Локальная проверка на обоих сплитах перед отправкой (фильтр: не хуже текущего
финала локально 0.5015/0.4318).

Таблица: results/tables/exp3_e3_ppr_e1.csv.
"""
import time

import numpy as np
import pandas as pd

from exp_rrf_hybrid import (
    DATA, RESULTS, POOL, SEED,
    evaluate, build_split, rank_list_from_scores, als_rankings,
)
from exp3_e1_attr_ppr import (
    RESTART, N_ITER,
    load_item_attrs, build_graph_base,
)
from exp3_e3_rrf_weights import tripartite_rankings, rank_matrix, fuse_weighted_top5

BEST_AW = 0.5
W = (1.0, 2.0, 1.0)
K_RRF = 20


def main():
    inter = pd.read_parquet(DATA / "user_interactions.parquet")
    inter["user_id"] = inter["user_id"].astype(str)
    sub = pd.read_csv(DATA / "submission.csv", dtype={"user_id": str})
    sub_users = sub["user_id"].tolist()

    splits = {
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
    acc = {}

    for split_name, cfg in splits.items():
        t0 = time.perf_counter()
        train, val_sched, candidates, relevant, val_users = build_split(
            inter, sub_users, cfg["train_end_week"], cfg["val_start_week"],
            cfg["val_end_week"], cfg["sched_val_path"])
        max_week = cfg["max_week"]
        airings = pd.read_csv(DATA / cfg["airings_file"], low_memory=False)
        airings = airings[airings["tv_show_id"] != 0]["tv_show_id"].value_counts().to_dict()
        pop_recent = train[train["week"] >= max_week - 3].groupby("tv_show_id")["user_id"].nunique()
        score_b = {c: pop_recent.get(c, 0) * np.log1p(airings.get(c, 0)) for c in candidates}
        pool_full = rank_list_from_scores(score_b, score_b, k=POOL)
        rank_pop = {it: r for r, it in enumerate(pool_full)}

        item_attrs = load_item_attrs(cfg["attrs_files"])
        base = build_graph_base(train, pool_full, item_attrs)  # без decay (E1)
        ppr_r, _ = tripartite_rankings(base, base["per_user"],
                                       pool_full, rank_pop, val_users, BEST_AW)
        als_r = als_rankings(train, val_users, pool_full, rank_pop)
        pop_rank = {u: list(pool_full) for u in val_users}

        M_pop = rank_matrix(pop_rank, val_users, pool_full)
        M_ppr = rank_matrix(ppr_r, val_users, pool_full)
        M_als = rank_matrix(als_r, val_users, pool_full)
        order = fuse_weighted_top5([M_pop, M_ppr, M_als], W, K_RRF)
        preds = {u: [pool_full[j] for j in order[i]] for i, u in enumerate(val_users)}
        m, n = evaluate(preds, relevant)
        m_ppr, _ = evaluate({u: ppr_r[u][:5] for u in val_users}, relevant)
        acc[split_name] = m
        print(f"[{split_name}] fusion k={K_RRF} w={W} + PPR E1 (без decay): "
              f"MAP@5={m:.4f} (n={n}), PPR отдельно={m_ppr:.4f}, "
              f"time={time.perf_counter() - t0:.1f}s")

    df = pd.DataFrame([{
        "variant": "E3-fusion k=20 w=(1,2,1) + PPR E1 (без decay)",
        "params": f"attr_weight={BEST_AW}; decay=none; restart={RESTART}; "
                  f"w_pop={W[0]}; w_ppr={W[1]}; w_als={W[2]}; k_rrf={K_RRF}; n_iter={N_ITER}",
        "MAP5_main": acc["main"], "MAP5_early": acc["early"],
        "MAP5_mean": (acc["main"] + acc["early"]) / 2,
        "time_s": None, "seed": SEED,
    }])
    out = RESULTS / "exp3_e3_ppr_e1.csv"
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
