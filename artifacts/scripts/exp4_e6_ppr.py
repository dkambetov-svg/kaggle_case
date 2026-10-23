#!/usr/bin/env python3
"""E6 (exp4): PPR-рефинемент — сетка restart × attr_weight + численная проверка сходимости.

Сетка: restart ∈ {0.3, 0.5, 0.7} × attr_weight ∈ {0.4, 0.5, 0.6, 0.7}
(12 точек, оба сплита), PPR без decay (E1-конфигурация победителя).
Для каждого PPR: отдельно MAP@5 и внутри fusion k=20 w=(1,2,1) с ALS 64/0.05/20
(если E5 выявит лучший ALS — точка пересчитается в E7).

Численная проверка сходимости: alpha = 1 − restart; при alpha=0.5 сжатие
0.5^30 ≈ 9.3e-10, при alpha=0.7 (restart=0.3) — 0.7^30 ≈ 2.3e-5. Считаем
score-матрицы при n_iter=30 и n_iter=100, берём max|Δ| — при сходимости
разность на уровне численного шума, и больше итераций не меняют скоры.
Проверка выполняется один раз на restart (при aw=CONV_AW): скорость сжатия
alpha^n зависит только от alpha, не от attr_weight (замер диагностикой:
PPR100 ≈ 46s, поэтому 6 проверок вместо 24).

Таблица: results/tables/exp4_ppr.csv, exp4_ppr_convergence.csv.
"""
import time

import numpy as np
import pandas as pd

from exp_rrf_hybrid import RESULTS, SEED, evaluate
from exp3_e1_attr_ppr import RESTART, N_ITER, load_item_attrs, build_graph_base, \
    build_transition, ppr_tripartite
from exp3_e3_rrf_weights import (
    tripartite_rankings, rank_matrix, fuse_weighted_top5,
)
from exp4_common import SPLITS, load_inter, prepare_split, als_rankings_p

RESTARTS = (0.3, 0.5, 0.7)
ATTR_WEIGHTS = (0.4, 0.5, 0.6, 0.7)
CONV_AW = 0.5          # проверка сходимости один раз на restart (rate зависит только от alpha)
W = (1.0, 2.0, 1.0)
K_RRF = 20
ALS_PARAMS = dict(factors=64, reg=0.05, iterations=20)


def main():
    inter, sub_users = load_inter()
    keys = [(r, aw) for r in RESTARTS for aw in ATTR_WEIGHTS]
    acc = {k: dict(MAP5_main=None, MAP5_early=None,
                   stand_main=None, stand_early=None, time_s=0.0) for k in keys}
    conv = []  # строки числовой проверки сходимости

    for split_name, cfg in SPLITS.items():
        t_split = time.perf_counter()
        train, val_sched, candidates, relevant, val_users, pool_full, rank_pop = \
            prepare_split(inter, sub_users, cfg, pool_size=500)
        item_attrs = load_item_attrs(cfg["attrs_files"])
        base = build_graph_base(train, pool_full, item_attrs)
        als_r = als_rankings_p(train, val_users, pool_full, rank_pop, **ALS_PARAMS)
        M_pop = rank_matrix({u: list(pool_full) for u in val_users}, val_users, pool_full)
        M_als = rank_matrix(als_r, val_users, pool_full)
        print(f"[split {split_name}] pool={len(pool_full)}, val_users={len(val_users)}",
              flush=True)

        # сетка: 12 точек × PPR30 (диагностика: 13.4s) ≈ 27 min на оба сплита
        nU = base["nU"]
        n_nodes = nU + base["nI"] + base["nA"]
        pool_local = np.array([base["i_map"][c] for c in pool_full])
        B = len(val_users)
        restart_mat = np.zeros((n_nodes, B))   # не зависит ни от restart, ни от aw
        for b, u in enumerate(val_users):
            ui = base["u_map"].get(u)
            if ui is None:
                continue
            idxs, ws = base["per_user"][ui]
            restart_mat[nU + idxs, b] = ws

        for restart in RESTARTS:
            alpha = 1.0 - restart
            for aw in ATTR_WEIGHTS:
                t0 = time.perf_counter()
                ppr_r, scores30 = tripartite_rankings(
                    base, base["per_user"], pool_full, rank_pop, val_users, aw,
                    alpha=alpha)
                # численная проверка (один aw на restart): 100 итераций против 30
                d_max = None
                if aw == CONV_AW:
                    T = build_transition(base, aw)
                    scores100 = ppr_tripartite(T, restart_mat, n_iter=100,
                                               alpha=alpha)[nU + pool_local, :]
                    d_max = float(np.abs(scores30 - scores100).max())
                    conv.append(dict(split=split_name, restart=restart, alpha=alpha,
                                     attr_weight=aw, n_iter_ref=30, n_iter_check=100,
                                     max_abs_score_diff=d_max,
                                     contraction_alpha_pow30=alpha ** 30))

                m_stand, _ = evaluate({u: ppr_r[u][:5] for u in val_users}, relevant)
                M_ppr = rank_matrix(ppr_r, val_users, pool_full)
                order = fuse_weighted_top5([M_pop, M_ppr, M_als], W, K_RRF)
                preds = {u: [pool_full[j] for j in order[i]]
                         for i, u in enumerate(val_users)}
                m_fus, _ = evaluate(preds, relevant)
                dt = time.perf_counter() - t0
                k = (restart, aw)
                acc[k][f"MAP5_{split_name}"] = m_fus
                acc[k][f"stand_{split_name}"] = m_stand
                acc[k]["time_s"] += dt
                conv_s = (f", max|Δ(30 vs 100 iter)|={d_max:.2e}"
                          if d_max is not None else "")
                print(f"  restart={restart} aw={aw}: PPR отдельно={m_stand:.4f}, "
                      f"fusion={m_fus:.4f}{conv_s} ({dt:.1f}s)", flush=True)
        print(f"  [split завершён за {time.perf_counter() - t_split:.1f}s]",
              flush=True)

    rows = []
    for restart, aw in keys:
        a = acc[(restart, aw)]
        rows.append({
            "variant": f"E6 restart={restart} attr_weight={aw}",
            "params": (f"restart={restart}; alpha={1.0 - restart:.1f}; attr_weight={aw}; "
                       f"decay=none; n_iter={N_ITER}; fusion=k{K_RRF} w={W}; "
                       f"als=64/0.05/20"),
            "MAP5_main": a["MAP5_main"], "MAP5_early": a["MAP5_early"],
            "MAP5_mean": (a["MAP5_main"] + a["MAP5_early"]) / 2,
            "PPR_standalone_main": a["stand_main"], "PPR_standalone_early": a["stand_early"],
            "PPR_standalone_mean": (a["stand_main"] + a["stand_early"]) / 2,
            "time_s": round(a["time_s"], 1), "seed": SEED,
        })
    df = pd.DataFrame(rows).sort_values("MAP5_mean", ascending=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "exp4_ppr.csv"
    df.to_csv(out, index=False)
    out_conv = RESULTS / "exp4_ppr_convergence.csv"
    pd.DataFrame(conv).to_csv(out_conv, index=False)
    print(f"\n[saved] {out}")
    print(f"[saved] {out_conv}")
    print("\n=== E6: restart × attr_weight (сортировка по среднему fusion MAP@5) ===")
    print(df.to_string(index=False))
    conv_df = pd.DataFrame(conv)
    print("\n=== E6: численная проверка сходимости (n_iter=30 против 100) ===")
    print(conv_df.groupby("restart")["max_abs_score_diff"].agg(["max", "mean"])
          .to_string())
    return df, conv_df


if __name__ == "__main__":
    main()
