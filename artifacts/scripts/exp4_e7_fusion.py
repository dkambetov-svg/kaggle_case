#!/usr/bin/env python3
"""E7 (exp4): локальная доводка fusion вокруг победителя exp3_e3b.

Берёт лучший ALS из exp4_als.csv (E5) и лучший PPR из exp4_ppr.csv (E6),
затем сетка: k_rrf ∈ {10, 20, 40} × w_ppr ∈ {1.5, 2.0, 3.0} × w_als ∈ {0.5, 1.0}
при w_pop=1.0 (18 комбинаций, оба сплита). Точка победителя exp3
(k=20, w=(1,2,1)) входит в сетку как контроль.

Таблица: results/tables/exp4_fusion.csv.
"""
import time

import pandas as pd

from exp_rrf_hybrid import RESULTS, SEED, evaluate
from exp3_e1_attr_ppr import load_item_attrs, build_graph_base
from exp3_e3_rrf_weights import tripartite_rankings, rank_matrix, fuse_weighted_top5
from exp4_common import SPLITS, load_inter, prepare_split, als_rankings_p

K_GRID = (10, 20, 40)
W_PPR = (1.5, 2.0, 3.0)
W_ALS = (0.5, 1.0)
W_POP = 1.0


def _best_als_params():
    """Лучшая ALS-конфигурация из exp4_als.csv (E5), исключая строку-референс."""
    df = pd.read_csv(RESULTS / "exp4_als.csv")
    grid = df[~df["variant"].str.startswith("референс")].sort_values(
        "MAP5_mean", ascending=False).iloc[0]
    import re
    m = re.search(r"f=(\d+) it=(\d+) reg=([\d.]+)", grid["variant"])
    return dict(factors=int(m.group(1)), iterations=int(m.group(2)),
                reg=float(m.group(3))), grid["variant"], grid["MAP5_mean"]


def _best_ppr_params():
    """Лучшая PPR-конфигурация из exp4_ppr.csv (E6)."""
    df = pd.read_csv(RESULTS / "exp4_ppr.csv")
    best = df.sort_values("MAP5_mean", ascending=False).iloc[0]
    m = pd.read_csv(RESULTS / "exp4_ppr.csv")  # parse из params
    import re
    pm = re.search(r"restart=([\d.]+); alpha=[\d.]+; attr_weight=([\d.]+)",
                   best["params"])
    return dict(restart=float(pm.group(1)), attr_weight=float(pm.group(2))), \
        best["variant"], best["MAP5_mean"]


def main():
    als_params, als_src, als_map = _best_als_params()
    ppr_params, ppr_src, ppr_map = _best_ppr_params()
    alpha = 1.0 - ppr_params["restart"]
    print(f"[E7] ALS из E5: {als_src} (MAP5_mean={als_map:.4f}) -> {als_params}")
    print(f"[E7] PPR из E6: {ppr_src} (MAP5_mean={ppr_map:.4f}) -> {ppr_params}")

    inter, sub_users = load_inter()
    keys = [(k, wp, wa) for k in K_GRID for wp in W_PPR for wa in W_ALS]
    acc = {key: dict(MAP5_main=None, MAP5_early=None, time_s=0.0) for key in keys}

    for split_name, cfg in SPLITS.items():
        t_split = time.perf_counter()
        train, val_sched, candidates, relevant, val_users, pool_full, rank_pop = \
            prepare_split(inter, sub_users, cfg, pool_size=500)
        item_attrs = load_item_attrs(cfg["attrs_files"])
        base = build_graph_base(train, pool_full, item_attrs)
        ppr_r, _ = tripartite_rankings(base, base["per_user"], pool_full, rank_pop,
                                       val_users, ppr_params["attr_weight"],
                                       alpha=alpha)
        als_r = als_rankings_p(train, val_users, pool_full, rank_pop, **als_params)
        M_pop = rank_matrix({u: list(pool_full) for u in val_users}, val_users, pool_full)
        M_ppr = rank_matrix(ppr_r, val_users, pool_full)
        M_als = rank_matrix(als_r, val_users, pool_full)
        print(f"[split {split_name}] pool={len(pool_full)}, val_users={len(val_users)}")

        for k_rrf in K_GRID:
            for w_ppr in W_PPR:
                for w_als in W_ALS:
                    t0 = time.perf_counter()
                    order = fuse_weighted_top5([M_pop, M_ppr, M_als],
                                               (W_POP, w_ppr, w_als), k_rrf)
                    preds = {u: [pool_full[j] for j in order[i]]
                             for i, u in enumerate(val_users)}
                    m, _ = evaluate(preds, relevant)
                    dt = time.perf_counter() - t0
                    key = (k_rrf, w_ppr, w_als)
                    acc[key][f"MAP5_{split_name}"] = m
                    acc[key]["time_s"] += dt
        print(f"  сетка {len(keys)} посчитана "
              f"({time.perf_counter() - t_split:.1f}s на сплит)")

    rows = []
    for k_rrf, w_ppr, w_als in keys:
        a = acc[(k_rrf, w_ppr, w_als)]
        is_ctrl = (k_rrf, w_ppr, w_als) == (20, 2.0, 1.0)
        rows.append({
            "variant": (f"E7 k={k_rrf} w_pop={W_POP} w_ppr={w_ppr} w_als={w_als}"
                        + (" (контроль: точка победителя exp3)" if is_ctrl else "")),
            "params": (f"k_rrf={k_rrf}; w_pop={W_POP}; w_ppr={w_ppr}; w_als={w_als}; "
                       f"ppr={ppr_src}; als={als_src}"),
            "MAP5_main": a["MAP5_main"], "MAP5_early": a["MAP5_early"],
            "MAP5_mean": (a["MAP5_main"] + a["MAP5_early"]) / 2,
            "time_s": round(a["time_s"], 3), "seed": SEED,
            "exp3_winner_point": is_ctrl,
        })
    df = pd.DataFrame(rows).sort_values("MAP5_mean", ascending=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "exp4_fusion.csv"
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}")
    print("\n=== E7: fusion-доводка (топ-10 по среднему MAP@5) ===")
    print(df.head(10).to_string(index=False))
    return df


if __name__ == "__main__":
    main()
