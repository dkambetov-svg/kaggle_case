#!/usr/bin/env python3
"""E3 (exp3): веса взвешенного RRF для лучшего PPR из E1/E2.

score(item) = w_pop/(k + rank_pop) + w_ppr/(k + rank_ppr) + w_als/(k + rank_als)
  w_pop ∈ {1, 1.5, 2}, w_ppr ∈ {1, 1.5, 2}, w_als ∈ {0, 0.5, 1}, k ∈ {20, 60}
= 54 комбинации. Абляция pop+PPR без ALS покрыта w_als=0 (лучшая отмечена).

Лучший PPR из E1/E2: трёхдольный граф attr_weight=0.5, decay_tau=8,
restart=raw (exp3_e1.csv / exp3_e2.csv). Контроль «текущий финал локально»:
RRF k=60, веса 1/1/1 из pop×airings + бипартитный PPR restart=0.5 (aw=0) + ALS.

Таблица: results/tables/exp3_e3.csv.
"""
import time
from itertools import product

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

W_POP = (1.0, 1.5, 2.0)
W_PPR = (1.0, 1.5, 2.0)
W_ALS = (0.0, 0.5, 1.0)
K_RRF = (20, 60)
BEST_AW = 0.5       # E1
BEST_TAU = 8.0      # E2, restart=raw


def tripartite_rankings(base, restart_src, pool_full, rank_pop, val_users, attr_weight,
                        decay_tau=None):
    """PPR-ранжирование по трёхдольному графу. Возвращает (rankings, scores_matrix)
    scores_matrix — (n_pool, B) в порядке pool_full."""
    nU = base["nU"]
    I_off = nU
    n_nodes = nU + base["nI"] + base["nA"]
    pool_local = np.array([base["i_map"][c] for c in pool_full])
    B = len(val_users)
    restart_mat = np.zeros((n_nodes, B))
    for b, u in enumerate(val_users):
        ui = base["u_map"].get(u)
        if ui is None:
            continue
        idxs, ws = restart_src[ui]
        restart_mat[I_off + idxs, b] = ws
    T = build_transition(base, attr_weight)
    P = ppr_tripartite(T, restart_mat)
    scores = P[I_off + pool_local, :]  # (n_pool, B)
    rankings = {}
    for b, u in enumerate(val_users):
        sc = dict(zip(pool_full, scores[:, b]))
        rankings[u] = rank_list_from_scores(sc, rank_pop, k=POOL)
    for u in val_users:  # cold: нулевые скоры -> порядок пула (tie_rank)
        if u not in rankings:
            rankings[u] = list(pool_full)
    return rankings, scores


def rank_matrix(rankings, val_users, pool_full):
    """(n_users, n_pool) матрица рангов; столбцы в порядке pool_full."""
    idx = {it: j for j, it in enumerate(pool_full)}
    M = np.zeros((len(val_users), len(pool_full)), dtype=np.int32)
    for i, u in enumerate(val_users):
        for r, it in enumerate(rankings[u]):
            M[i, idx[it]] = r
    return M


def fuse_weighted_top5(M_list, weights, k_rrf):
    """Топ-5 по Σ w/(k_rrf+rank); тай-брейк — порядок пула (столбцы)."""
    score = np.zeros_like(M_list[0], dtype=np.float64)
    for M, w in zip(M_list, weights):
        if w != 0.0:
            score += w / (k_rrf + M)
    order = np.argsort(-score, axis=1, kind="stable")[:, :5]
    return order


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

    grid = list(product(W_POP, W_PPR, W_ALS, K_RRF))
    acc = {g: dict(MAP5_main=None, MAP5_early=None, time_s=0.0) for g in grid}
    ctrl = dict(MAP5_main=None, MAP5_early=None)
    meta = dict(MAP5_ppr_main=None, MAP5_ppr_early=None, MAP5_als_main=None, MAP5_als_early=None)

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
        # PPR лучший из E1/E2: aw=0.5, tau=8, restart=raw (per_user_raw!)
        base_best = build_graph_base(train, pool_full, item_attrs,
                                     decay_tau=BEST_TAU, max_week=max_week)
        ppr_best, _ = tripartite_rankings(base_best, base_best["per_user_raw"],
                                          pool_full, rank_pop, val_users, BEST_AW)
        # PPR бипартитный (aw=0, restart=raw) — как в текущем финале
        base_bip = build_graph_base(train, pool_full, item_attrs)
        ppr_bip, _ = tripartite_rankings(base_bip, base_bip["per_user"],
                                         pool_full, rank_pop, val_users, 0.0)
        # ALS
        als_r = als_rankings(train, val_users, pool_full, rank_pop)

        pop_rank = {u: list(pool_full) for u in val_users}

        # отдельные комппоненты (справочно)
        m_ppr, _ = evaluate({u: ppr_best[u][:5] for u in val_users}, relevant)
        m_als, _ = evaluate({u: als_r[u][:5] for u in val_users}, relevant)
        meta[f"MAP5_ppr_{split_name}"] = m_ppr
        meta[f"MAP5_als_{split_name}"] = m_als
        print(f"  PPR_best отдельно MAP@5={m_ppr:.4f}; ALS отдельно MAP@5={m_als:.4f}")

        # контроль: текущий финал локально (k=60, w=1/1/1, bipartite PPR restart=0.5)
        M_pop = rank_matrix(pop_rank, val_users, pool_full)
        M_ppr_bip = rank_matrix(ppr_bip, val_users, pool_full)
        M_als = rank_matrix(als_r, val_users, pool_full)
        M_ppr_best = rank_matrix(ppr_best, val_users, pool_full)
        order = fuse_weighted_top5([M_pop, M_ppr_bip, M_als], (1.0, 1.0, 1.0), 60)
        preds_ctrl = {u: [pool_full[j] for j in order[i]] for i, u in enumerate(val_users)}
        m_ctrl, _ = evaluate(preds_ctrl, relevant)
        ctrl[f"MAP5_{split_name}"] = m_ctrl
        print(f"  [контроль: текущий финал локально, RRF k=60 1/1/1 с bipartite PPR] "
              f"MAP@5={m_ctrl:.4f}")

        # сетка 54 комбинации (лучший PPR как компонент)
        t0 = time.perf_counter()
        for w_pop, w_ppr, w_als, k_rrf in grid:
            order = fuse_weighted_top5([M_pop, M_ppr_best, M_als],
                                       (w_pop, w_ppr, w_als), k_rrf)
            preds = {u: [pool_full[j] for j in order[i]] for i, u in enumerate(val_users)}
            m, _ = evaluate(preds, relevant)
            g = (w_pop, w_ppr, w_als, k_rrf)
            acc[g][f"MAP5_{split_name}"] = m
            acc[g]["time_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
        print(f"  сетка {len(grid)} комбинаций посчитана "
              f"({time.perf_counter() - t_split:.1f}s на сплит)")
        # сохранить лучшие rank-матрицы для E4 не требуется — E4 считает отдельно
        del base_best, base_bip

    rows = []
    for g in grid:
        w_pop, w_ppr, w_als, k_rrf = g
        a = acc[g]
        label = f"E3 w_pop={w_pop} w_ppr={w_ppr} w_als={w_als} k={k_rrf}"
        rows.append({
            "variant": label,
            "params": (f"w_pop={w_pop}; w_ppr={w_ppr}; w_als={w_als}; k_rrf={k_rrf}; "
                       f"ppr=E1aw0.5+E2tau8raw; pool={POOL}"),
            "MAP5_main": a["MAP5_main"], "MAP5_early": a["MAP5_early"],
            "MAP5_mean": (a["MAP5_main"] + a["MAP5_early"]) / 2,
            "time_s": round(a["time_s"], 3), "seed": SEED,
            "ablation_pop_ppr_no_als": w_als == 0,
        })
    # контроль текущего финала и отдельные компоненты — в ту же таблицу
    rows.append({
        "variant": "КОНТРОЛЬ: текущий финал (k=60 w=1/1/1, bipartite PPR restart=0.5)",
        "params": "w_pop=1; w_ppr=1; w_als=1; k_rrf=60; ppr=bipartite restart=0.5",
        "MAP5_main": ctrl["MAP5_main"], "MAP5_early": ctrl["MAP5_early"],
        "MAP5_mean": (ctrl["MAP5_main"] + ctrl["MAP5_early"]) / 2,
        "time_s": None, "seed": SEED, "ablation_pop_ppr_no_als": False,
    })
    rows.append({
        "variant": "справочно: PPR_best отдельно (E1 aw=0.5 + E2 tau=8)",
        "params": "attr_weight=0.5; decay_tau=8; restart=raw; restart_p=0.5; n_iter=30",
        "MAP5_main": meta["MAP5_ppr_main"], "MAP5_early": meta["MAP5_ppr_early"],
        "MAP5_mean": (meta["MAP5_ppr_main"] + meta["MAP5_ppr_early"]) / 2,
        "time_s": None, "seed": SEED, "ablation_pop_ppr_no_als": False,
    })
    rows.append({
        "variant": "справочно: ALS отдельно",
        "params": "factors=64; reg=0.05; iterations=20",
        "MAP5_main": meta["MAP5_als_main"], "MAP5_early": meta["MAP5_als_early"],
        "MAP5_mean": (meta["MAP5_als_main"] + meta["MAP5_als_early"]) / 2,
        "time_s": None, "seed": SEED, "ablation_pop_ppr_no_als": False,
    })
    df = pd.DataFrame(rows).sort_values("MAP5_mean", ascending=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "exp3_e3.csv"
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}")
    print("\n=== E3: взвешенный RRF (топ-15 по среднему MAP@5) ===")
    print(df.head(15).to_string(index=False))
    print("\n=== абляция pop+PPR без ALS (w_als=0), топ-5 ===")
    abl = df[df["ablation_pop_ppr_no_als"]].head(5)
    print(abl[["variant", "MAP5_main", "MAP5_early", "MAP5_mean"]].to_string(index=False))
    return df


if __name__ == "__main__":
    main()
