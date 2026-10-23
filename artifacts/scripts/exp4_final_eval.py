#!/usr/bin/env python3
"""exp4: локальная оценка ИТОГОВОЙ конфигурации финала — RRF k=10 w=(1,2,1)
над pop×airings + PPR E1 (restart=0.5, aw=0.5, без decay) + ALS 64/0.05/20,
пул-500. Конфигурация = exp3_e3b с единственным отличием k: 20 → 10
(Kaggle 0.34586/0.32383, run-id 57554720).

Точка не считалась в E7 (там k перебирался поверх новых PPR/ALS из E6/E5),
поэтому здесь отдельный прогон на обоих сплитах для §6 отчёта.

Выход: results/tables/exp4_final_local.csv + самозаписываемый ноутбук
experiments/*-exp4-final-eval.ipynb (вывод фиксируется при создании).
"""
import io
import json
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "artifacts" / "scripts"))
from exp4_common import SPLITS, prepare_split, load_inter, als_rankings_p
from exp3_e1_attr_ppr import load_item_attrs, build_graph_base
from exp3_e3_rrf_weights import tripartite_rankings, rank_matrix, fuse_weighted_top5
from exp_rrf_hybrid import evaluate

K_RRF = 10
W = (1.0, 2.0, 1.0)
AW = 0.5
ALPHA = 0.5
ALS = dict(factors=64, reg=0.05, iterations=20)
# референс exp3_e3b на этих сплитах (results/tables/exp4_als.csv:6, exp4_ppr.csv:8)
REF = dict(main=0.5365245321873032, early=0.46317616959064295)
POOL_SIZE = 500


def run_split(inter, sub_users, name, cfg):
    train, _, _, relevant, val_users, pool_full, rank_pop = \
        prepare_split(inter, sub_users, cfg, pool_size=POOL_SIZE)
    item_attrs = load_item_attrs(cfg["attrs_files"])
    base = build_graph_base(train, pool_full, item_attrs)
    ppr_r, _ = tripartite_rankings(base, base["per_user"], pool_full, rank_pop,
                                   val_users, AW, alpha=ALPHA)
    als_r = als_rankings_p(train, val_users, pool_full, rank_pop, **ALS)
    M_pop = rank_matrix({u: list(pool_full) for u in val_users}, val_users, pool_full)
    M_ppr = rank_matrix(ppr_r, val_users, pool_full)
    M_als = rank_matrix(als_r, val_users, pool_full)
    order = fuse_weighted_top5([M_pop, M_ppr, M_als], W, K_RRF)
    preds = {u: [pool_full[j] for j in order[i]] for i, u in enumerate(val_users)}
    score, nrel = evaluate(preds, relevant)
    return score, len(val_users)


def main():
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(str(s))

    emit(f"# exp4: финальная конфигурация (RRF k={K_RRF} w={W}, PPR aw={AW} "
         f"restart=0.5, ALS 64/0.05/20, пул-{POOL_SIZE})")
    emit(f"Дата: {datetime.now():%Y-%m-%d %H:%M}. Seed=42. Оба сплита, строгая релевантность.")
    emit()

    inter, sub_users = load_inter()
    rows = []
    t_all = time.perf_counter()
    for name in ("main", "early"):
        t0 = time.perf_counter()
        score, n_eval = run_split(inter, sub_users, name, SPLITS[name])
        rows.append(dict(split=name, MAP5=score, n_users_eval=n_eval,
                         MAP5_ref_exp3_e3b=REF[name], delta_vs_ref=None, seed=42,
                         time_s=round(time.perf_counter() - t0, 1)))
        emit(f"[{name}] MAP@5={score:.6f} (users={n_eval}) "
             f"за {time.perf_counter() - t0:.1f}s")
    for r in rows:
        r["delta_vs_ref"] = r["MAP5"] - r["MAP5_ref_exp3_e3b"]
    emit()
    for r in rows:
        emit(f"[{r['split']}] референс exp3_e3b {r['MAP5_ref_exp3_e3b']:.6f} → "
             f"дельта {r['delta_vs_ref']:+.6f}")
    emit(f"Итого: {time.perf_counter() - t_all:.1f}s")

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "tables" / "exp4_final_local.csv"
    df.to_csv(out, index=False)
    emit(f"[saved] {out}")

    # самозаписываемый ноутбук (вывод фиксируется при создании)
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "execution_count": None,
             "outputs": [],
             "source": (f"# exp4: финальная конфигурация (k={K_RRF}) — локальная оценка\n\n"
                        f"**Дата:** {datetime.now():%Y-%m-%d}. **Seed:** 42. "
                        f"Зачем: Kaggle-финал exp4_k10 (run-id 57554720) отличается от "
                        f"exp3_e3b только k (20→10); точка не была посчитана в E7 "
                        f"(там k перебирался поверх новых компонентов E5/E6). "
                        f"Оба сплита, строгая релевантность, пул-500.\n\n"
                        f"Скрипт: `artifacts/scripts/exp4_final_eval.py`. "
                        f"Таблица: `results/tables/exp4_final_local.csv`.")},
            {"cell_type": "code", "metadata": {"execution_count": 1,
                                               "execution_status": "ok"},
             "execution_count": 1,
             "outputs": [{"output_type": "stream", "name": "stdout",
                          "text": lines}],
             "source": (f"# полный вывод прогона (встраивается при создании)\n"
                        f"# таблица: results/tables/exp4_final_local.csv\n")},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python",
                                       "version": sys.version.split()[0]}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    nb_path = ROOT / "experiments" / f"{datetime.now():%Y-%m-%d-%H%M}-exp4-final-eval.ipynb"
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] {nb_path}")


if __name__ == "__main__":
    main()
