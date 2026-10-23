#!/usr/bin/env python3
"""exp3: генерация четырёх сабмитов (обучение 11–30, пул топ-500 pop×log(1+airings 31-42)).

Все варианты прошли локальный фильтр «не хуже текущего финала на обоих сплитах»
(контроль финала локально: main 0.5015 / early 0.4318, exp3_e3.csv):

1. exp3_e1   — чистый трёхдольный PPR: attr_weight=0.5, restart=0.5, без decay
               (лучший E1; локально 0.5427/0.4643).
2. exp3_e2   — E1 + decay_tau=8, restart=raw (лучший E1+E2; локально 0.5433/0.4645).
3. exp3_e3   — взвешенный RRF k=20: w_pop=1, w_ppr=2, w_als=1
               (лучший E3; локально 0.5357/0.4626).
4. exp3_disc — структура текущего финала RRF k=60, веса 1/1/1, но компонент PPR
               заменён на лучший (E1+E2): «рекорд + улучшенный PPR»
               (локально 0.5125/0.4371; обоснование: изолировать эффект PPR-компонента
               при неизменной fusion-структуре рекорда).

Атрибуты графа (программа–канал–категория–жанр) — из расписаний 11-20/21-30/31-42
(для пула — 31–42). Тестовые просмотры 31–42 не используются: только расписание
(связи и счётчик эфиров). Выход: data/submission_exp3_*.csv + проверка формата.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "artifacts" / "scripts"))
from exp_rrf_hybrid import (
    DATA, POOL, SEED,
    rank_list_from_scores, als_rankings,
)
from exp3_e1_attr_ppr import load_item_attrs, build_graph_base, ppr_tripartite
from exp3_e3_rrf_weights import tripartite_rankings, rank_matrix, fuse_weighted_top5

BEST_AW = 0.5
BEST_TAU = 8.0
MAX_WEEK = 30
N_TEST = 1260


def check_format(path, template_path, candidates_set):
    """Проверка формата сабмита перед отправкой (как в шаблоне)."""
    out = pd.read_csv(path, dtype={"user_id": str})
    tpl = pd.read_csv(template_path, dtype={"user_id": str})
    assert list(out.columns) == ["user_id", "tv_show_id"], f"колонки: {list(out.columns)}"
    assert len(out) == N_TEST == len(tpl), f"строк: {len(out)} (нужно {N_TEST})"
    assert (out["user_id"].values == tpl["user_id"].values).all(), "порядок user_id != шаблон"
    n_unique_total = 0
    for ids in out["tv_show_id"]:
        parts = str(ids).split()
        assert len(parts) == 5, f"не 5 id: {ids}"
        ints = [int(p) for p in parts]
        assert len(set(ints)) == 5, f"дубликаты в топ-5: {ids}"
        assert all(i in candidates_set for i in ints), f"id вне расписания 31-42: {ids}"
        n_unique_total += len(set(ints))
    print(f"  [format OK] {Path(path).name}: 1260 строк, 5 уникальных id, "
          f"все из расписания 31-42, порядок как в шаблоне")
    return True


def main():
    inter = pd.read_parquet(DATA / "user_interactions.parquet")
    inter["user_id"] = inter["user_id"].astype(str)
    train = inter[inter["week"] <= MAX_WEEK]
    sub = pd.read_csv(DATA / "submission.csv", dtype={"user_id": str})
    test_users = sub["user_id"].tolist()

    sched42 = pd.read_csv(DATA / "export_arh_31-42-final.csv", low_memory=False)
    candidates = sched42[sched42["tv_show_id"] != 0]["tv_show_id"].unique()
    candidates_set = set(int(c) for c in candidates)
    airings = sched42[sched42["tv_show_id"] != 0]["tv_show_id"].value_counts().to_dict()

    pop_recent = train[train["week"] >= MAX_WEEK - 3].groupby("tv_show_id")["user_id"].nunique()
    score_b = {c: pop_recent.get(c, 0) * np.log1p(airings.get(c, 0)) for c in candidates}
    pool_full = rank_list_from_scores(score_b, score_b, k=POOL)
    rank_pop = {it: r for r, it in enumerate(pool_full)}
    print(f"[pool] топ-{len(pool_full)} по pop(27-30)×log(1+airings 31-42); "
          f"train недель<= {MAX_WEEK}, строк={len(train)}")

    # атрибуты: все три расписания (пул получает связи из 31–42)
    item_attrs = load_item_attrs([DATA / "export_arh_11-20-final.csv",
                                  DATA / "export_arh_21-30-final.csv",
                                  DATA / "export_arh_31-42-final.csv"])

    # --- PPR E1 (без decay) ---
    t0 = time.perf_counter()
    base_e1 = build_graph_base(train, pool_full, item_attrs)
    ppr_e1, _ = tripartite_rankings(base_e1, base_e1["per_user"],
                                    pool_full, rank_pop, test_users, BEST_AW)
    print(f"[PPR E1 aw={BEST_AW} без decay] {time.perf_counter() - t0:.1f}s")

    # --- PPR E2 (tau=8, restart=raw) ---
    t0 = time.perf_counter()
    base_e2 = build_graph_base(train, pool_full, item_attrs,
                               decay_tau=BEST_TAU, max_week=MAX_WEEK)
    ppr_e2, _ = tripartite_rankings(base_e2, base_e2["per_user_raw"],
                                    pool_full, rank_pop, test_users, BEST_AW)
    print(f"[PPR E2 tau={BEST_TAU:.0f} restart=raw] {time.perf_counter() - t0:.1f}s")

    # --- ALS ---
    t0 = time.perf_counter()
    als_r = als_rankings(train, test_users, pool_full, rank_pop)
    print(f"[ALS f=64] {time.perf_counter() - t0:.1f}s")

    pop_rank = {u: list(pool_full) for u in test_users}
    M_pop = rank_matrix(pop_rank, test_users, pool_full)
    M_ppr = rank_matrix(ppr_e2, test_users, pool_full)
    M_als = rank_matrix(als_r, test_users, pool_full)

    def top5_from_rankings(rankings):
        return {u: rankings[u][:5] for u in test_users}

    def top5_from_order(order):
        return {u: [pool_full[j] for j in order[i]] for i, u in enumerate(test_users)}

    variants = {
        # name: (preds, описание)
        "exp3_e1": (top5_from_rankings(ppr_e1),
                    "tripartite PPR aw=0.5 restart=0.5 (E1, без decay)"),
        "exp3_e2": (top5_from_rankings(ppr_e2),
                    "tripartite PPR aw=0.5 + decay tau=8 restart=raw (E1+E2)"),
        "exp3_e3": (top5_from_order(
            fuse_weighted_top5([M_pop, M_ppr, M_als], (1.0, 2.0, 1.0), 20)),
            "RRF k=20 w=(pop=1, ppr=2, als=1); PPR=E1+E2"),
        "exp3_disc": (top5_from_order(
            fuse_weighted_top5([M_pop, M_ppr, M_als], (1.0, 1.0, 1.0), 60)),
            "RRF k=60 w=(1,1,1) как финал, но PPR=E1+E2 вместо бипартитного"),
    }

    for name, (preds, desc) in variants.items():
        out = sub.copy()
        out["tv_show_id"] = out["user_id"].map(
            lambda u: " ".join(str(int(x)) for x in preds[u][:5]))
        path = DATA / f"submission_{name}.csv"
        out.to_csv(path, index=False)
        print(f"[saved] {path.name} — {desc}")
        check_format(path, DATA / "submission.csv", candidates_set)

    print("\nГотово. Отправка: kaggle competitions submit -c sweettv-tv-program-recommender "
          "-f data/submission_<name>.csv -m '<msg>'")


if __name__ == "__main__":
    main()
