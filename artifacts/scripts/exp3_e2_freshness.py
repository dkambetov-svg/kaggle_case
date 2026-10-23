#!/usr/bin/env python3
"""E2 (exp3): свежесть в трёхдольном графе PPR поверх лучшего E1 (attr_weight=0.5).

Вес ребра пользователь–программа (агрегированный по просмотрам):
    w_tau(u,i) = log1p( Σ_r overlap_r · exp(−Δ_r/τ) ),  Δ_r = max_week − week_r
(недель от просмотра до конца обучения). При τ→∞ вес совпадает с E1
(log1p Σ overlap_r) — базовая строка таблицы это подтверждает.

τ ∈ {4, 8}. Два варианта restart-вектора:
  - restart=raw:      рестарт по СЫРЫМ весам (лог-нормализация без затухания);
  - restart=decayed:  рестарт по тем же затухающими весами, что и граф.
Атрибуты/attr_weight/рёбра программа–атрибут — как в лучшем E1 (aw=0.5,
restart 0.5, 30 итераций). Диагностика холодных программ — как в E1.

Таблица: results/tables/exp3_e2.csv.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from exp_rrf_hybrid import (
    DATA, RESULTS, POOL, SEED,
    evaluate, build_split, rank_list_from_scores,
)
from exp3_e1_attr_ppr import (
    ATTR_WEIGHTS, RESTART, N_ITER,
    load_item_attrs, build_graph_base, build_transition, ppr_tripartite,
)

BEST_AW = max(ATTR_WEIGHTS, key=lambda a: a)  # E1: 0.5 (лучший по среднему MAP@5)
TAUS = (4.0, 8.0)
RESTART_MODES = ("raw", "decayed")


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

    # variant-key -> acc
    variants = [("E2 baseline (без decay, как E1 aw=0.5)", None, "raw")]
    for tau in TAUS:
        for mode in RESTART_MODES:
            variants.append((f"E2 tau={tau:.0f} restart={mode}", tau, mode))
    acc = {v[0]: dict(MAP5_main=None, MAP5_early=None, time_s=0.0,
                      uu_main=None, uur_main=None, uu_early=None, uur_early=None)
           for v in variants}

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
        print(f"\n[split {split_name}] pool={len(pool_full)}, val_users={len(val_users)}, "
              f"train недель<={cfg['train_end_week']}, конец обучения=нед.{max_week}")

        item_attrs = load_item_attrs(cfg["attrs_files"])
        user_train_items = train.groupby("user_id")["tv_show_id"].apply(set).to_dict()

        # база графа пер tau (baseline tau=None -> без затухания); прогоняем по одному tau
        for label, tau, rmode in variants:
            t0 = time.perf_counter()
            base = build_graph_base(train, pool_full, item_attrs,
                                    decay_tau=tau, max_week=max_week)
            nU = base["nU"]
            I_off = nU
            n_nodes = nU + base["nI"] + base["nA"]
            pool_local = np.array([base["i_map"][c] for c in pool_full])

            restart_src = base["per_user_raw"] if rmode == "raw" else base["per_user"]
            B = len(val_users)
            restart_mat = np.zeros((n_nodes, B))
            for b, u in enumerate(val_users):
                ui = base["u_map"].get(u)
                if ui is None:
                    continue
                idxs, ws = restart_src[ui]
                restart_mat[I_off + idxs, b] = ws

            T = build_transition(base, BEST_AW)
            P = ppr_tripartite(T, restart_mat)
            scores = P[I_off + pool_local, :]
            preds = {}
            for b, u in enumerate(val_users):
                sc = dict(zip(pool_full, scores[:, b]))
                preds[u] = rank_list_from_scores(sc, rank_pop, k=POOL)
            m, n = evaluate(preds, relevant)
            dt = time.perf_counter() - t0

            unseen_u_slots = unseen_u_rel = 0
            for u in val_users:
                u_hist = user_train_items.get(u, set())
                for p in preds[u][:5]:
                    if p not in u_hist:
                        unseen_u_slots += 1
                        if p in relevant.get(u, ()):
                            unseen_u_rel += 1
            print(f"  {label}: MAP@5={m:.4f} (n={n}) time={dt:.1f}s | "
                  f"пер-юзерно не видел: слотов={unseen_u_slots}, из них релевантных={unseen_u_rel}")

            a = acc[label]
            a[f"MAP5_{split_name}"] = m
            a["time_s"] += dt
            a[f"uu_{split_name}"] = unseen_u_slots
            a[f"uur_{split_name}"] = unseen_u_rel
        print(f"  [split завершён за {time.perf_counter() - t_split:.1f}s]")

    rows = []
    for label, tau, rmode in variants:
        a = acc[label]
        if tau is None:
            params = f"attr_weight={BEST_AW}; decay=none; restart={rmode}; restart_p={RESTART}; n_iter={N_ITER}"
        else:
            params = (f"attr_weight={BEST_AW}; decay_tau={tau:.0f}; restart={rmode}; "
                      f"restart_p={RESTART}; n_iter={N_ITER}")
        rows.append({
            "variant": label, "params": params,
            "MAP5_main": a["MAP5_main"], "MAP5_early": a["MAP5_early"],
            "MAP5_mean": (a["MAP5_main"] + a["MAP5_early"]) / 2,
            "time_s": round(a["time_s"], 1), "seed": SEED,
            "usernew_top5_main": a["uu_main"], "usernew_rel_main": a["uur_main"],
            "usernew_top5_early": a["uu_early"], "usernew_rel_early": a["uur_early"],
        })
    df = pd.DataFrame(rows).sort_values("MAP5_mean", ascending=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "exp3_e2.csv"
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}")
    print("\n=== E2: свежесть в графе (сортировка по среднему MAP@5) ===")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
