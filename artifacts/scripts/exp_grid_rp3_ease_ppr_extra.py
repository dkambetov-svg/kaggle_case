#!/usr/bin/env python3
"""Дешёвое расширение grid-а: EASE lambda ∈ {2000, 5000} и PPR restart=0.7.
Только MAP@5 на обеих сплитах (те же пул/сплиты/fallback), добавление в ту же таблицу
results/tables/exp_grid_rp3_ease_ppr_sorted.csv (union, сортировка по среднему).
Не перезаписывает текущий финал/прежние результаты."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from exp_rrf_hybrid import (
    DATA, RESULTS, POOL,
    evaluate, build_split, rank_list_from_scores,
)


def row_normalize_sparse(X):
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sums)
    nz = row_sums > 0
    inv[nz] = 1.0 / row_sums[nz]
    d = csr_matrix((inv, (np.arange(len(inv)), np.arange(len(inv)))), shape=(len(inv), len(inv)))
    return d @ X


def build_ppr_graph_local(train):
    agg = train.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum().reset_index()
    agg["weight"] = np.log1p(agg["overlap_seconds"])
    users = agg["user_id"].unique()
    items = agg["tv_show_id"].unique()
    u_map = {u: i for i, u in enumerate(users)}
    i_map = {it: i for i, it in enumerate(items)}
    rows_u = agg["user_id"].map(u_map).values
    cols_i = agg["tv_show_id"].map(i_map).values
    X = csr_matrix((agg["weight"].values, (rows_u, cols_i)), shape=(len(u_map), len(i_map)))
    Mu2i = row_normalize_sparse(X)
    Mi2u = row_normalize_sparse(X.T.tocsr())
    return u_map, i_map, Mu2i, Mi2u


def ppr_batch(Mu2i, Mi2u, Vu, Vi, alpha, n_iter=30):
    Pu = Vu.copy()
    Pi = Vi.copy()
    Mu2iT = Mu2i.T.tocsr()
    Mi2uT = Mi2u.T.tocsr()
    for _ in range(n_iter):
        Pi = alpha * (Mu2iT @ Pu) + (1 - alpha) * Vi
        Pu = alpha * (Mi2uT @ Pi) + (1 - alpha) * Vu
    return Pu, Pi


def main():
    inter = pd.read_parquet(DATA / "user_interactions.parquet")
    inter["user_id"] = inter["user_id"].astype(str)
    sub = pd.read_csv(DATA / "submission.csv", dtype={"user_id": str})
    sub_users = sub["user_id"].tolist()

    splits = {
        "main (train<=20, val21-30)": dict(train_end_week=20, val_start_week=21, val_end_week=30,
                                            sched_val_path=DATA / "export_arh_21-30-final.csv", max_week=20,
                                            airings_file="export_arh_21-30-final.csv"),
        "early (train<=16, val17-20)": dict(train_end_week=16, val_start_week=17, val_end_week=20,
                                             sched_val_path=DATA / "export_arh_11-20-final.csv", max_week=16,
                                             airings_file="export_arh_11-20-final.csv"),
    }

    new_rows = []  # (variant, split, MAP5, n_eval, time_s)

    for split_name, cfg in splits.items():
        train, val_sched, candidates, relevant, val_users = build_split(
            inter, sub_users, cfg["train_end_week"], cfg["val_start_week"], cfg["val_end_week"], cfg["sched_val_path"]
        )
        max_week = cfg["max_week"]
        airings = pd.read_csv(DATA / cfg["airings_file"], low_memory=False)
        airings = airings[airings["tv_show_id"] != 0]["tv_show_id"].value_counts().to_dict()
        pop_recent = train[train["week"] >= max_week - 3].groupby("tv_show_id")["user_id"].nunique()
        score_b = {c: pop_recent.get(c, 0) * np.log1p(airings.get(c, 0)) for c in candidates}
        pool_full = rank_list_from_scores(score_b, score_b, k=POOL)
        top5_pop = pool_full[:5]
        print(f"\n[split {split_name}] pool={len(pool_full)}, val_users={len(val_users)}")

        agg = train.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum().reset_index()
        users = agg["user_id"].unique()
        items = agg["tv_show_id"].unique()
        u_map = {u: i for i, u in enumerate(users)}
        i_map = {it: i for i, it in enumerate(items)}
        rows_u = agg["user_id"].map(u_map).values
        cols_i = agg["tv_show_id"].map(i_map).values
        X_bin = csr_matrix((np.ones(len(agg)), (rows_u, cols_i)), shape=(len(u_map), len(i_map)))

        warm_users = [u for u in val_users if u in u_map]
        pool_items_known = [c for c in pool_full if c in i_map]
        pool_unknown = [c for c in pool_full if c not in i_map]
        tail_scores = {c: -1e9 for c in pool_unknown}

        # ---- EASE lambda ∈ {2000, 5000} ----
        XtX = (X_bin.T @ X_bin).toarray()
        n_items = XtX.shape[0]
        X_val_dense = np.zeros((len(warm_users), n_items))
        for r, u in enumerate(warm_users):
            X_val_dense[r, :] = X_bin[u_map[u], :].toarray().ravel()
        for lam in [2000, 5000]:
            t0 = time.perf_counter()
            P_inv = np.linalg.inv(XtX + lam * np.eye(n_items))
            diagP = np.diag(P_inv).copy()
            B = -P_inv / diagP[None, :]
            np.fill_diagonal(B, 0.0)
            scores_val = X_val_dense @ B
            preds = {}
            for u in val_users:
                ui = u_map.get(u)
                if ui is None:
                    preds[u] = top5_pop
                    continue
                r = warm_users.index(u)
                sc = {it: float(scores_val[r, i_map[it]]) for it in pool_items_known}
                sc.update(tail_scores)
                preds[u] = rank_list_from_scores(sc, {it: i for i, it in enumerate(pool_full)}, k=POOL)[:5]
            m, n = evaluate(preds, relevant)
            dt = time.perf_counter() - t0
            print(f"  EASE binary lambda={lam:5d}        MAP@5={m:.4f} (n={n}) time={dt:.1f}s")
            new_rows.append((f"EASE binary lambda={lam}", split_name, m, n, dt))
        del XtX, X_val_dense

        # ---- PPR restart=0.7 ----
        u_map_g, i_map_g, Mu2i, Mi2u = build_ppr_graph_local(train)
        pool_idx_g = np.array([i_map_g[c] for c in pool_items_known])
        Bb = len(warm_users)
        Vu = np.zeros((len(u_map_g), Bb))
        Vi = np.zeros((len(i_map_g), Bb))
        for b, u in enumerate(warm_users):
            Vi[:, b] = Mu2i[u_map_g[u], :].toarray().ravel()
        t0 = time.perf_counter()
        alpha = 1.0 - 0.7
        Pu, Pi = ppr_batch(Mu2i, Mi2u, Vu, Vi, alpha=alpha, n_iter=30)
        scores_pool = Pi[pool_idx_g, :]
        preds = {}
        for b, u in enumerate(warm_users):
            sc = {it: float(scores_pool[j, b]) for j, it in enumerate(pool_items_known)}
            sc.update(tail_scores)
            preds[u] = rank_list_from_scores(sc, {it: i for i, it in enumerate(pool_full)}, k=POOL)[:5]
        for u in val_users:
            if u not in u_map_g:
                preds[u] = top5_pop
        m, n = evaluate(preds, relevant)
        dt = time.perf_counter() - t0
        print(f"  PPR restart=0.7                    MAP@5={m:.4f} (n={n}) time={dt:.1f}s")
        new_rows.append(("PPR restart=0.7", split_name, m, n, dt))

    # ---- объединение с прежней таблицей (union по variant, сортировка по среднему) ----
    old_sorted = RESULTS / "exp_grid_rp3_ease_ppr_sorted.csv"
    old = pd.read_csv(old_sorted)
    records = old.to_dict("records")
    seen = {(r["variant"], r.get("MAP5_main") is not None): r for r in records}
    by_variant = {r["variant"]: r for r in records}
    for variant, split_name, m, n, dt in new_rows:
        if variant not in by_variant:
            by_variant[variant] = {"variant": variant, "MAP5_main": None, "n_main": None, "time_main_s": None,
                                    "MAP5_early": None, "n_early": None, "time_early_s": None, "MAP5_mean": None}
        rec = by_variant[variant]
        if "main" in split_name:
            rec["MAP5_main"] = m; rec["n_main"] = n; rec["time_main_s"] = round(dt, 2)
        else:
            rec["MAP5_early"] = m; rec["n_early"] = n; rec["time_early_s"] = round(dt, 2)
        rec["MAP5_mean"] = (rec["MAP5_main"] + rec["MAP5_early"]) / 2 if (
            rec["MAP5_main"] is not None and rec["MAP5_early"] is not None) else None

    df = pd.DataFrame(by_variant.values()).sort_values("MAP5_mean", ascending=False)
    df.to_csv(RESULTS / "exp_grid_rp3_ease_ppr_sorted.csv", index=False)
    df.sort_values("variant").to_csv(RESULTS / "exp_grid_rp3_ease_ppr.csv", index=False)
    print(f"\n[updated] {RESULTS/'exp_grid_rp3_ease_ppr_sorted.csv'}")

    print("\n=== Grid (обновлённая таблица, отсортировано по среднему MAP@5) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
