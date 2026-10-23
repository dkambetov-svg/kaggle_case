#!/usr/bin/env python3
"""E5 (exp4): сетка гиперпараметров ALS — «больше эпох».

Текущий ALS (64 factors, reg=0.05, 20 итераций) — самый слабый компонент
(отдельно 0.1399/0.0919), но он входит в Kaggle-победителя exp3_e3b
(RRF k=20 w=(1,2,1) + PPR E1 aw=0.5 без decay). Идея: больше итераций /
факторов может усилить ALS-вклад в fusion.

Сетка: iterations ∈ {20,40,80,160} × factors ∈ {64,128} при reg=0.05
(8 конфигураций) + проба reg ∈ {0.01,0.2} при iterations=40, factors=64.
Для каждого конфига: ALS отдельно MAP@5 и внутри fusion k=20 w=(1,2,1)
с фиксированным PPR E1 (aw=0.5, без decay) на обоих сплитах.

Таблица: results/tables/exp4_als.csv.
"""
import time

import pandas as pd

from exp_rrf_hybrid import RESULTS, SEED
from exp3_e1_attr_ppr import load_item_attrs, build_graph_base
from exp3_e3_rrf_weights import tripartite_rankings, rank_matrix, fuse_weighted_top5
from exp4_common import (
    SPLITS, load_inter, prepare_split, als_rankings_p,
)
from exp_rrf_hybrid import evaluate

BEST_AW = 0.5   # E1
W = (1.0, 2.0, 1.0)
K_RRF = 20

EPOCH_GRID = [(f, it, 0.05) for f in (64, 128) for it in (20, 40, 80, 160)]
REG_GRID = [(64, 40, 0.01), (64, 40, 0.2)]  # проба регуляризации
GRID = EPOCH_GRID + REG_GRID


def main():
    inter, sub_users = load_inter()
    acc = {(f, it, r): dict(MAP5_main=None, MAP5_early=None,
                            stand_main=None, stand_early=None, time_s=0.0)
           for f, it, r in GRID}

    for split_name, cfg in SPLITS.items():
        t_split = time.perf_counter()
        train, val_sched, candidates, relevant, val_users, pool_full, rank_pop = \
            prepare_split(inter, sub_users, cfg, pool_size=500)
        item_attrs = load_item_attrs(cfg["attrs_files"])
        base = build_graph_base(train, pool_full, item_attrs)  # без decay (E1)
        ppr_r, _ = tripartite_rankings(base, base["per_user"], pool_full, rank_pop,
                                       val_users, BEST_AW)
        M_pop = rank_matrix({u: list(pool_full) for u in val_users}, val_users, pool_full)
        M_ppr = rank_matrix(ppr_r, val_users, pool_full)
        print(f"[split {split_name}] pool={len(pool_full)}, val_users={len(val_users)}")

        for f, it, reg in GRID:
            t0 = time.perf_counter()
            als_r = als_rankings_p(train, val_users, pool_full, rank_pop,
                                   factors=f, reg=reg, iterations=it)
            m_als, _ = evaluate({u: als_r[u][:5] for u in val_users}, relevant)
            M_als = rank_matrix(als_r, val_users, pool_full)
            order = fuse_weighted_top5([M_pop, M_ppr, M_als], W, K_RRF)
            preds = {u: [pool_full[j] for j in order[i]] for i, u in enumerate(val_users)}
            m_fus, _ = evaluate(preds, relevant)
            dt = time.perf_counter() - t0
            key = (f, it, reg)
            acc[key][f"MAP5_{split_name}"] = m_fus
            acc[key][f"stand_{split_name}"] = m_als
            acc[key]["time_s"] += dt
            print(f"  ALS f={f} it={it} reg={reg}: отдельно MAP@5={m_als:.4f}, "
                  f"fusion MAP@5={m_fus:.4f} ({dt:.1f}s)")
        print(f"  [split завершён за {time.perf_counter() - t_split:.1f}s]")

    rows = []
    for f, it, reg in GRID:
        a = acc[(f, it, reg)]
        label = f"E5 ALS f={f} it={it} reg={reg}"
        rows.append({
            "variant": label,
            "params": f"factors={f}; iterations={it}; reg={reg}; fusion=k{K_RRF} w={W}; ppr=E1aw0.5",
            "MAP5_main": a["MAP5_main"], "MAP5_early": a["MAP5_early"],
            "MAP5_mean": (a["MAP5_main"] + a["MAP5_early"]) / 2,
            "ALS_standalone_main": a["stand_main"], "ALS_standalone_early": a["stand_early"],
            "ALS_standalone_mean": (a["stand_main"] + a["stand_early"]) / 2,
            "time_s": round(a["time_s"], 1), "seed": SEED,
        })
    # референсные числа — из exp3_e3_ppr_e1.csv:2 (fusion) и exp3_e3.csv:58 (ALS отдельно)
    rows.append({
        "variant": "референс: fusion с текущим ALS (64/0.05/20) = exp3_e3b локально",
        "params": "factors=64; iterations=20; reg=0.05; fusion=k20 w=(1,2,1); ppr=E1aw0.5",
        "MAP5_main": 0.5365245321873032, "MAP5_early": 0.46317616959064295,
        "MAP5_mean": (0.5365245321873032 + 0.46317616959064295) / 2,
        "ALS_standalone_main": 0.13985719401916, "ALS_standalone_early": 0.09187659904970757,
        "ALS_standalone_mean": (0.13985719401916 + 0.09187659904970757) / 2,
        "time_s": None, "seed": SEED,
    })
    df = pd.DataFrame(rows).sort_values("MAP5_mean", ascending=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "exp4_als.csv"
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}")
    print("\n=== E5: ALS-гиперпараметры (сортировка по среднему MAP@5 fusion) ===")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
