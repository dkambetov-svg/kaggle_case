#!/usr/bin/env python3
"""Диагностика E6: реальная стоимость одного PPR-запуска (30 и 100 итераций).

Зачем: первый запуск E6 не уложился в 40 минут. Замеряем время build_transition,
PPR30, PPR100 на сплите main (aw=0.5, restart=0.5), чтобы оценить бюджет сетки
(12 точек × 2 сплита) и проверки сходимости.
"""
import time

from exp3_e1_attr_ppr import load_item_attrs, build_graph_base, build_transition, \
    ppr_tripartite
from exp4_common import SPLITS, load_inter, prepare_split, als_rankings_p
import numpy as np

AW = 0.5
ALPHA = 0.5


def main():
    inter, sub_users = load_inter()
    cfg = SPLITS["main"]
    t0 = time.perf_counter()
    train, _, _, _, val_users, pool_full, rank_pop = \
        prepare_split(inter, sub_users, cfg, pool_size=500)
    item_attrs = load_item_attrs(cfg["attrs_files"])
    base = build_graph_base(train, pool_full, item_attrs)
    print(f"prepare+graph: {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    T = build_transition(base, AW)
    print(f"build_transition: {time.perf_counter() - t0:.1f}s, nnz(T)={T.nnz}", flush=True)

    nU = base["nU"]
    n_nodes = nU + base["nI"] + base["nA"]
    B = len(val_users)
    restart_mat = np.zeros((n_nodes, B))
    for b, u in enumerate(val_users):
        ui = base["u_map"].get(u)
        if ui is None:
            continue
        idxs, ws = base["per_user"][ui]
        restart_mat[nU + idxs, b] = ws

    t0 = time.perf_counter()
    P30 = ppr_tripartite(T, restart_mat, n_iter=30, alpha=ALPHA)
    t30 = time.perf_counter() - t0
    print(f"PPR n_iter=30: {t30:.1f}s", flush=True)

    t0 = time.perf_counter()
    P100 = ppr_tripartite(T, restart_mat, n_iter=100, alpha=ALPHA)
    t100 = time.perf_counter() - t0
    print(f"PPR n_iter=100: {t100:.1f}s", flush=True)

    d = float(np.abs(P30 - P100).max())
    print(f"max|P30 - P100| (все узлы): {d:.3e}; alpha^30={ALPHA ** 30:.3e}", flush=True)
    print(f"\nОценка бюджета E6: сетка 12x2 PPR30 ≈ {24 * t30:.0f}s; "
          f"проверка сходимости (6 точек: 1 aw на restart) ≈ {6 * t100:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
