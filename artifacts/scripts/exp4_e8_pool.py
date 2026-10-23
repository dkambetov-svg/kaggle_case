#!/usr/bin/env python3
"""E8 (exp4): размер пула кандидатов — 500 против 1000 и 2000.

Гипотеза: за пределами топ-500 по pop×log(1+airings) есть релевантные
программы, которые fusion сейчас не видит (пул — жёсткая шапка списка
кандидатов всех компонентов). Считаем конфигурацию-победителя (лучшие ALS
из E5, PPR из E6, fusion из E7) при pool ∈ {500, 1000, 2000} на обоих сплитах.

Таблица: results/tables/exp4_pool.csv.
"""
import re
import time

import pandas as pd

from exp_rrf_hybrid import RESULTS, SEED, evaluate
from exp3_e1_attr_ppr import load_item_attrs, build_graph_base
from exp3_e3_rrf_weights import tripartite_rankings, rank_matrix, fuse_weighted_top5
from exp4_common import SPLITS, load_inter, prepare_split, als_rankings_p

POOL_SIZES = (500, 1000, 2000)


def _winners():
    """(als_params, ppr_params, fusion_params) из таблиц E5/E6/E7."""
    df_als = pd.read_csv(RESULTS / "exp4_als.csv")
    best_als = df_als[~df_als["variant"].str.startswith("референс")].sort_values(
        "MAP5_mean", ascending=False).iloc[0]
    m = re.search(r"f=(\d+) it=(\d+) reg=([\d.]+)", best_als["variant"])
    als_params = dict(factors=int(m.group(1)), iterations=int(m.group(2)),
                      reg=float(m.group(3)))

    df_ppr = pd.read_csv(RESULTS / "exp4_ppr.csv")
    best_ppr = df_ppr.sort_values("MAP5_mean", ascending=False).iloc[0]
    pm = re.search(r"restart=([\d.]+); alpha=[\d.]+; attr_weight=([\d.]+)",
                   best_ppr["params"])
    ppr_params = dict(restart=float(pm.group(1)),
                      attr_weight=float(pm.group(2)))

    df_fus = pd.read_csv(RESULTS / "exp4_fusion.csv")
    best_fus = df_fus.sort_values("MAP5_mean", ascending=False).iloc[0]
    fm = re.search(r"k_rrf=(\d+); w_pop=([\d.]+); w_ppr=([\d.]+); w_als=([\d.]+)",
                   best_fus["params"])
    fus_params = dict(k_rrf=int(fm.group(1)), w_pop=float(fm.group(2)),
                      w_ppr=float(fm.group(3)), w_als=float(fm.group(4)))
    return als_params, ppr_params, fus_params, \
        (best_als["variant"], best_ppr["variant"], best_fus["variant"])


def main():
    als_params, ppr_params, fus_params, srcs = _winners()
    alpha = 1.0 - ppr_params["restart"]
    print(f"[E8] ALS (E5): {srcs[0]} -> {als_params}")
    print(f"[E8] PPR (E6): {srcs[1]} -> {ppr_params}")
    print(f"[E8] fusion (E7): {srcs[2]} -> {fus_params}")

    inter, sub_users = load_inter()
    acc = {ps: dict(MAP5_main=None, MAP5_early=None, time_s=0.0) for ps in POOL_SIZES}

    for split_name, cfg in SPLITS.items():
        t_split = time.perf_counter()
        # prepare_split без пула: нужен один train/val для всех размеров пула
        train, val_sched, candidates, relevant, val_users, _, _ = \
            prepare_split(inter, sub_users, cfg, pool_size=500)
        item_attrs = load_item_attrs(cfg["attrs_files"])
        print(f"[split {split_name}] val_users={len(val_users)}")

        for ps in POOL_SIZES:
            t0 = time.perf_counter()
            # пул: топ-ps по pop×airings (prepare_split при pool_size=ps)
            _, _, _, _, _, pool_full, rank_pop = \
                prepare_split(inter, sub_users, cfg, pool_size=ps)
            base = build_graph_base(train, pool_full, item_attrs)
            ppr_r, _ = tripartite_rankings(base, base["per_user"], pool_full,
                                           rank_pop, val_users,
                                           ppr_params["attr_weight"], alpha=alpha,
                                           k_pool=ps)
            als_r = als_rankings_p(train, val_users, pool_full, rank_pop,
                                   **als_params)
            M_pop = rank_matrix({u: list(pool_full) for u in val_users},
                                val_users, pool_full)
            M_ppr = rank_matrix(ppr_r, val_users, pool_full)
            M_als = rank_matrix(als_r, val_users, pool_full)
            order = fuse_weighted_top5([M_pop, M_ppr, M_als],
                                       (fus_params["w_pop"], fus_params["w_ppr"],
                                        fus_params["w_als"]), fus_params["k_rrf"])
            preds = {u: [pool_full[j] for j in order[i]]
                     for i, u in enumerate(val_users)}
            m, _ = evaluate(preds, relevant)
            dt = time.perf_counter() - t0
            acc[ps][f"MAP5_{split_name}"] = m
            acc[ps]["time_s"] += dt
            print(f"  pool={ps}: MAP@5={m:.4f} ({dt:.1f}s)")
        print(f"  [split завершён за {time.perf_counter() - t_split:.1f}s]")

    rows = []
    for ps in POOL_SIZES:
        a = acc[ps]
        rows.append({
            "variant": f"E8 pool={ps}",
            "params": (f"pool={ps}; ppr={ppr_params}; als={als_params}; "
                       f"fusion={fus_params}"),
            "MAP5_main": a["MAP5_main"], "MAP5_early": a["MAP5_early"],
            "MAP5_mean": (a["MAP5_main"] + a["MAP5_early"]) / 2,
            "time_s": round(a["time_s"], 1), "seed": SEED,
            "baseline_pool500": ps == 500,
        })
    df = pd.DataFrame(rows).sort_values("MAP5_mean", ascending=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "exp4_pool.csv"
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}")
    print("\n=== E8: размер пула ===")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
