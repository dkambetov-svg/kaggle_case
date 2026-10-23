#!/usr/bin/env python3
"""exp4: генератор сабмитов — параметризованный по варианту конфигурации.

Запуск: uv run python artifacts/scripts/exp4_submit.py <variant>
  winner  — полный победитель exp4: ALS f=128/it=20/reg=0.05 (E5),
            PPR restart=0.7 aw=0.7 alpha=0.3 (E6), RRF k=10 w=(1,2,1) (E7),
            пул-500 (E8: пул 500 не хуже 1000/2000)  [локально 0.5479/0.4715]
  ppr     — аблация: всё как exp3_e3b, но PPR restart=0.7 aw=0.7
            (изолирует эффект PPR)
  k10     — аблация: всё как exp3_e3b, но RRF k=10 (изолирует эффект k)
  als128  — аблация: всё как exp3_e3b, но ALS f=128/it=20
            (изолирует эффект ALS)

Обучение 11–30, пул топ-500 по pop(нед. 27–30) × log(1+эфиров 31–42),
атрибуты из всех трёх расписаний. Формат — check_format из exp3_submit.
Выход: data/submission_exp4_<variant>.csv; существующий файл не
перезаписывается (правило раунда).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "artifacts" / "scripts"))
from exp_rrf_hybrid import DATA, POOL, rank_list_from_scores
from exp3_e1_attr_ppr import load_item_attrs, build_graph_base
from exp3_e3_rrf_weights import tripartite_rankings, rank_matrix, fuse_weighted_top5
from exp3_submit import check_format
from exp4_common import als_rankings_p

MAX_WEEK = 30
W = (1.0, 2.0, 1.0)

# variant -> (ppr_aw, ppr_alpha, als, k_rrf); PPR restart = 1 - alpha
# все конфигурации — из таблиц exp4_{als,ppr,fusion,pool}.csv (E5-E8)
VARIANTS = {
    "winner":  dict(aw=0.7, alpha=0.3,
                    als=dict(factors=128, iterations=20, reg=0.05), k_rrf=10),
    "ppr":     dict(aw=0.7, alpha=0.3,
                    als=dict(factors=64, iterations=20, reg=0.05), k_rrf=20),
    "k10":     dict(aw=0.5, alpha=0.5,
                    als=dict(factors=64, iterations=20, reg=0.05), k_rrf=10),
    "als128":  dict(aw=0.5, alpha=0.5,
                    als=dict(factors=128, iterations=20, reg=0.05), k_rrf=20),
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in VARIANTS:
        sys.exit(f"usage: exp4_submit.py <{'|'.join(VARIANTS)}>")
    variant = sys.argv[1]
    cfg = VARIANTS[variant]
    out_path = DATA / f"submission_exp4_{variant}.csv"
    if out_path.exists():
        sys.exit(f"[refuse] {out_path.name} уже существует — не перезаписываю")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    inter = pd.read_parquet(DATA / "user_interactions.parquet")
    inter["user_id"] = inter["user_id"].astype(str)
    train = inter[inter["week"] <= MAX_WEEK]
    sub = pd.read_csv(DATA / "submission.csv", dtype={"user_id": str})
    test_users = sub["user_id"].tolist()

    # пул: топ-500 по pop(нед. MAX_WEEK-3..30) × log(1+эфиров 31-42)
    sched42 = pd.read_csv(DATA / "export_arh_31-42-final.csv", low_memory=False)
    candidates = sched42[sched42["tv_show_id"] != 0]["tv_show_id"].unique()
    candidates_set = set(int(c) for c in candidates)
    airings = sched42[sched42["tv_show_id"] != 0]["tv_show_id"].value_counts().to_dict()
    pop_recent = train[train["week"] >= MAX_WEEK - 3].groupby("tv_show_id")["user_id"].nunique()
    score_b = {c: pop_recent.get(c, 0) * np.log1p(airings.get(c, 0)) for c in candidates}
    pool_full = rank_list_from_scores(score_b, score_b, k=POOL)
    rank_pop = {it: r for r, it in enumerate(pool_full)}
    print(f"[pool] топ-{len(pool_full)}; train <= {MAX_WEEK}; variant={variant}")

    item_attrs = load_item_attrs([DATA / "export_arh_11-20-final.csv",
                                  DATA / "export_arh_21-30-final.csv",
                                  DATA / "export_arh_31-42-final.csv"])
    t0 = time.perf_counter()
    base = build_graph_base(train, pool_full, item_attrs)
    ppr_r, _ = tripartite_rankings(base, base["per_user"], pool_full, rank_pop,
                                   test_users, cfg["aw"], alpha=cfg["alpha"])
    print(f"[PPR aw={cfg['aw']} restart={1 - cfg['alpha']:.1f}] "
          f"{time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    als_r = als_rankings_p(train, test_users, pool_full, rank_pop, **cfg["als"])
    print(f"[ALS {cfg['als']}] {time.perf_counter() - t0:.1f}s")

    M_pop = rank_matrix({u: list(pool_full) for u in test_users}, test_users, pool_full)
    M_ppr = rank_matrix(ppr_r, test_users, pool_full)
    M_als = rank_matrix(als_r, test_users, pool_full)
    order = fuse_weighted_top5([M_pop, M_ppr, M_als], W, cfg["k_rrf"])
    preds = {u: [pool_full[j] for j in order[i]] for i, u in enumerate(test_users)}

    out = sub.copy()
    out["tv_show_id"] = out["user_id"].map(
        lambda u: " ".join(str(int(x)) for x in preds[u][:5]))
    out.to_csv(out_path, index=False)
    print(f"[saved] {out_path.name} — RRF k={cfg['k_rrf']} w={W}, "
          f"PPR aw={cfg['aw']}, ALS {cfg['als']}")
    check_format(out_path, DATA / "submission.csv", candidates_set)


if __name__ == "__main__":
    main()
