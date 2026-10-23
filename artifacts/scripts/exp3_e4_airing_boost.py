#!/usr/bin/env python3
"""E4 (exp3): буст по эфирам поверх PPR как самостоятельный компонент.

score(item) = PPR_best(item) × (1 + airings)^a, a ∈ {0.25, 0.5} — ранжирование
заменяет ОТДЕЛЬНЫЕ компоненты pop×airings и PPR в fusion (как в финале их было
два: pop_airings + PPR; здесь один комбинированный). Оцениваем:
  1) компонент отдельно (MAP@5);
  2) RRF k=60: boost + ALS (замена пары pop×airings + PPR на boost).

airings: на валидации — из расписания val-окна (как в пуле), в финальном
инференсе — из 31–42 (только счётчик эфиров, просмотры не используются).
PPR_best = E1 aw=0.5 + E2 decay_tau=8, restart=raw.

Таблица: results/tables/exp3_e4.csv.
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
    load_item_attrs, build_graph_base, build_transition, ppr_tripartite,
)
from exp3_e3_rrf_weights import tripartite_rankings, rank_matrix, fuse_weighted_top5

A_POW = (0.25, 0.5)
BEST_AW = 0.5
BEST_TAU = 8.0


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

    acc = {(a, "solo"): dict(MAP5_main=None, MAP5_early=None, time_s=0.0) for a in A_POW}
    acc.update({(a, "rrf_als"): dict(MAP5_main=None, MAP5_early=None, time_s=0.0)
                for a in A_POW})
    acc[("ref", "pop_ppr_als")] = dict(MAP5_main=None, MAP5_early=None, time_s=0.0)

    for split_name, cfg in splits.items():
        t_split = time.perf_counter()
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
        print(f"\n[split {split_name}] pool={len(pool_full)}, val_users={len(val_users)}")

        item_attrs = load_item_attrs(cfg["attrs_files"])
        base = build_graph_base(train, pool_full, item_attrs,
                                decay_tau=BEST_TAU, max_week=max_week)
        ppr_r, ppr_scores = tripartite_rankings(base, base["per_user_raw"],
                                                pool_full, rank_pop, val_users, BEST_AW)
        # ppr_scores: (n_pool, B) в порядке pool_full
        als_r = als_rankings(train, val_users, pool_full, rank_pop)
        pop_rank = {u: list(pool_full) for u in val_users}

        # контроль: текущий финал локально (k=60, 1/1/1, bipartite PPR) — из exp3_e3
        # здесь: fusion pop+PPR_best+ALS k=60 как референс "популярная пара отдельно"
        M_pop = rank_matrix(pop_rank, val_users, pool_full)
        M_ppr = rank_matrix(ppr_r, val_users, pool_full)
        M_als = rank_matrix(als_r, val_users, pool_full)
        order_ref = fuse_weighted_top5([M_pop, M_ppr, M_als], (1.0, 1.0, 1.0), 60)
        preds_ref = {u: [pool_full[j] for j in order_ref[i]]
                     for i, u in enumerate(val_users)}
        m_ref, _ = evaluate(preds_ref, relevant)
        acc[("ref", "pop_ppr_als")][f"MAP5_{split_name}"] = m_ref
        print(f"  референс pop+PPR_best+ALS (k=60, 1/1/1): MAP@5={m_ref:.4f}")

        air_arr = np.array([airings.get(c, 0) for c in pool_full], dtype=np.float64)

        for a in A_POW:
            t0 = time.perf_counter()
            boost = (1.0 + air_arr) ** a  # (n_pool,)
            # solo: score = PPR × boost
            boosted_scores = ppr_scores * boost[:, None]  # (n_pool, B)
            preds_solo = {}
            for b, u in enumerate(val_users):
                sc = dict(zip(pool_full, boosted_scores[:, b]))
                preds_solo[u] = rank_list_from_scores(sc, rank_pop, k=POOL)
            m_solo, _ = evaluate(preds_solo, relevant)
            acc[(a, "solo")][f"MAP5_{split_name}"] = m_solo
            acc[(a, "solo")]["time_s"] += time.perf_counter() - t0

            # rrf_als: boost-ранжирование + ALS (k=60, w=1/1)
            t0 = time.perf_counter()
            rank_boost = {}
            for b, u in enumerate(val_users):
                sc = dict(zip(pool_full, boosted_scores[:, b]))
                rank_boost[u] = rank_list_from_scores(sc, rank_pop, k=POOL)
            M_boost = rank_matrix(rank_boost, val_users, pool_full)
            order = fuse_weighted_top5([M_boost, M_als], (1.0, 1.0), 60)
            preds = {u: [pool_full[j] for j in order[i]]
                     for i, u in enumerate(val_users)}
            m_f, _ = evaluate(preds, relevant)
            acc[(a, "rrf_als")][f"MAP5_{split_name}"] = m_f
            acc[(a, "rrf_als")]["time_s"] += time.perf_counter() - t0
            print(f"  a={a}: boost отдельно MAP@5={m_solo:.4f}; "
                  f"RRF k=60 boost+ALS MAP@5={m_f:.4f}")
        print(f"  [split завершён за {time.perf_counter() - t_split:.1f}s]")

    rows = []
    for (a, kind), v in acc.items():
        if kind == "pop_ppr_als":
            label = "референс: pop+PPR_best+ALS RRF k=60 (без буста)"
            params = "w_pop=1; w_ppr=1; w_als=1; k_rrf=60; ppr=E1aw0.5+E2tau8raw"
        else:
            label = (f"E4 a={a} boost отдельно" if kind == "solo"
                     else f"E4 a={a} RRF k=60 boost+ALS")
            params = (f"a={a}; ppr=E1aw0.5+E2tau8raw; airings=val-окно; "
                      f"component=PPR*(1+airings)^a"
                      + ("; fusion=+ALS k=60 w=1/1" if kind == "rrf_als" else ""))
        rows.append({
            "variant": label, "params": params,
            "MAP5_main": v["MAP5_main"], "MAP5_early": v["MAP5_early"],
            "MAP5_mean": (v["MAP5_main"] + v["MAP5_early"]) / 2,
            "time_s": round(v["time_s"], 3) if v["time_s"] else None, "seed": SEED,
        })
    df = pd.DataFrame(rows).sort_values("MAP5_mean", ascending=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "exp3_e4.csv"
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}")
    print("\n=== E4: PPR × (1+airings)^a (сортировка по среднему MAP@5) ===")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
