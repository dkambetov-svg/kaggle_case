#!/usr/bin/env python3
"""Grid: RP3beta (Christoffel et al., RecSys 2015), EASE (Steck, WWW 2019), PPR с разными restart.
Тот же пул (топ-500 по pop×airings в целевом окне), те же сплиты (main train<=20/val21-30,
early train<=16/val17-20), повторы разрешены, fallback популярности тот же, строгая валидация.
Для каждого варианта: MAP@5 на обоих сплитах + среднее. Таблица, отсортированная по среднему.

Ничего из текущего финала/прежних результатов не перезаписывает:
results/tables/exp_grid_rp3_ease_ppr.csv, exp_grid_rp3_ease_ppr_sorted.csv.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from exp_rrf_hybrid import (
    DATA, RESULTS, POOL,
    evaluate, strict_relevance, build_split,
    rank_list_from_scores, build_ppr_graph, ppr_batch,
)


def build_pool(train, candidates, airings, max_week):
    pop_recent = train[train["week"] >= max_week - 3].groupby("tv_show_id")["user_id"].nunique()
    score_b = {c: pop_recent.get(c, 0) * np.log1p(airings.get(c, 0)) for c in candidates}
    pool_full = rank_list_from_scores(score_b, score_b, k=POOL)
    rank_pop = {it: r for r, it in enumerate(pool_full)}
    return pool_full, rank_pop


def preds_from_scores(S, val_users, pool_full, rank_pop, top5_pop, k=5):
    """S: dict user -> dict item->score. Ранжирует пул, топ-k; холодные/пустые -> top5_pop."""
    preds = {}
    for u in val_users:
        sc = S.get(u, None)
        if not sc:
            preds[u] = top5_pop
            continue
        # все кандидаты пула обязаны иметь скор (пул фиксирован); хвост ранжирования по rank_pop
        for c in pool_full:
            if c not in sc:
                sc[c] = -1e9
        top = rank_list_from_scores(sc, rank_pop, k=POOL)[:k]
        preds[u] = top
    return preds


def main():
    rows = []
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

    variant_maps = {}  # {variant_name: {split: (MAP5, n_eval, time_s)}}

    for split_name, cfg in splits.items():
        t_split = time.perf_counter()
        train, val_sched, candidates, relevant, val_users = build_split(
            inter, sub_users, cfg["train_end_week"], cfg["val_start_week"], cfg["val_end_week"], cfg["sched_val_path"]
        )
        max_week = cfg["max_week"]
        airings = pd.read_csv(DATA / cfg["airings_file"], low_memory=False)
        airings = airings[airings["tv_show_id"] != 0]["tv_show_id"].value_counts().to_dict()
        pool_full, rank_pop = build_pool(train, candidates, airings, max_week)
        pop_recent = train[train["week"] >= max_week - 3].groupby("tv_show_id")["user_id"].nunique()
        top5_pop = pool_full[:5]  # топ-5 по pop×airings -- тот же fallback, что и раньше
        print(f"\n[split {split_name}] pool={len(pool_full)}, val_users={len(val_users)}")

        # ---- общий граф (user×item) из train, веса log(1+сек) и binary ----
        agg = train.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum().reset_index()
        users = agg["user_id"].unique()
        items = agg["tv_show_id"].unique()
        u_map = {u: i for i, u in enumerate(users)}
        i_map = {it: i for i, it in enumerate(items)}
        rows_u = agg["user_id"].map(u_map).values
        cols_i = agg["tv_show_id"].map(i_map).values
        X_log = csr_matrix((np.log1p(agg["overlap_seconds"].values), (rows_u, cols_i)),
                           shape=(len(u_map), len(i_map)))
        X_bin = csr_matrix((np.ones(len(agg)), (rows_u, cols_i)),
                           shape=(len(u_map), len(i_map)))

        # валидационные пользователи, присутствующие в train-графе (cold -> fallback popularity)
        warm_users = [u for u in val_users if u in u_map]
        user_row = {u: u_map[u] for u in warm_users}
        pool_items_idx = np.array([i_map[c] for c in pool_full if c in i_map])
        pool_items_known = [c for c in pool_full if c in i_map]
        pool_unknown = [c for c in pool_full if c not in i_map]
        # для всех кандидатов пула, неизвестных графу, складываем -inf (ранг по rank_pop в хвосте)
        tail_scores = {c: -1e9 for c in pool_unknown}

        # ================== RP3beta ==================
        # P_ui = row-normalized X (user->item), P_iu = row-normalized X^T (item->user)
        for weight_name, X in [("log", X_log), ("binary", X_bin)]:
            row_sums_ui = np.asarray(X.sum(axis=1)).ravel()
            inv_ui = np.zeros_like(row_sums_ui); nz = row_sums_ui > 0; inv_ui[nz] = 1 / row_sums_ui[nz]
            P_ui = csr_matrix((inv_ui, (np.arange(len(inv_ui)), np.arange(len(inv_ui)))),
                              shape=(len(inv_ui), len(inv_ui))) @ X  # users x items

            Xt = X.T.tocsr()
            row_sums_iu = np.asarray(Xt.sum(axis=1)).ravel()
            inv_iu = np.zeros_like(row_sums_iu); nz = row_sums_iu > 0; inv_iu[nz] = 1 / row_sums_iu[nz]
            P_iu = csr_matrix((inv_iu, (np.arange(len(inv_iu)), np.arange(len(inv_iu)))),
                              shape=(len(inv_iu), len(inv_iu))) @ Xt  # items x users

            # W = P_iu @ P_ui: item x item, элементы в степени alpha, столбцы делим на deg(j)^beta
            W_base = (P_iu @ P_ui).toarray()  # items x items dense (2900x2900 -> приемлемо)
            deg_j = np.asarray(X.sum(axis=0)).ravel()  # deg(j) = число пользователей, смотревших j

            for alpha in [0.5, 1.0]:
                W_pow = np.power(W_base, alpha)
                for beta in [0.0, 0.1, 0.2, 0.3]:
                    t0 = time.perf_counter()
                    W = W_pow.copy()
                    # деление столбцов на deg(j)^beta (deg=0 -> без изменения, такого не должно быть)
                    scale = np.ones_like(deg_j)
                    nzdeg = deg_j > 0
                    scale[nzdeg] = deg_j[nzdeg] ** beta
                    W = W / scale[None, :]  # столбец j

                    # Скор: P_ui(user) @ W -> item-скор; берём только известные графом элементы пула
                    S = {}
                    for u in warm_users:
                        pu = P_ui[u_map[u], :].toarray().ravel()
                        scores_all = pu @ W  # (items,)
                        sc = {it: float(scores_all[i_map[it]]) for it in pool_items_known}
                        sc.update(tail_scores)
                        S[u] = sc
                    preds = preds_from_scores(S, val_users, pool_full, rank_pop, top5_pop)
                    m, n = evaluate(preds, relevant)
                    dt = time.perf_counter() - t0
                    variant = f"RP3beta weight={weight_name} alpha={alpha} beta={beta}"
                    print(f"  {variant:45s} MAP@5={m:.4f} (n={n}) time={dt:.1f}s")
                    variant_maps.setdefault(variant, {})[split_name] = (m, n, dt)
            del W_base

        # ================== EASE ==================
        # B = -P/diag(P), P = (X^T X + lambda I)^-1, X binary, diag(B)=0
        XtX = (X_bin.T @ X_bin).toarray()  # items x items
        n_items = XtX.shape[0]
        X_bin_dense_val = np.zeros((len(warm_users), n_items))
        for r, u in enumerate(warm_users):
            X_bin_dense_val[r, :] = X_bin[u_map[u], :].toarray().ravel()
        for lam in [50, 200, 500, 1000]:
            t0 = time.perf_counter()
            P_inv = np.linalg.inv(XtX + lam * np.eye(n_items))
            diagP = np.diag(P_inv).copy()
            B = -P_inv / diagP[None, :]  # деление столбцов на diag(P)
            np.fill_diagonal(B, 0.0)
            scores_val = X_bin_dense_val @ B  # warm_users x items
            S = {}
            for r, u in enumerate(warm_users):
                sc = {it: float(scores_val[r, i_map[it]]) for it in pool_items_known}
                sc.update(tail_scores)
                S[u] = sc
            preds = preds_from_scores(S, val_users, pool_full, rank_pop, top5_pop)
            m, n = evaluate(preds, relevant)
            dt = time.perf_counter() - t0
            variant = f"EASE binary lambda={lam}"
            print(f"  {variant:45s} MAP@5={m:.4f} (n={n}) time={dt:.1f}s")
            variant_maps.setdefault(variant, {})[split_name] = (m, n, dt)
        del XtX, X_bin_dense_val

        # ================== PPR (персонализированный) с разными restart ==================
        u_map_g, i_map_g, Mu2i, Mi2u = build_ppr_graph(train)
        pool_idx_g = np.array([i_map_g[c] for c in pool_items_known])
        B = len(warm_users)
        Vu = np.zeros((len(u_map_g), B))
        Vi = np.zeros((len(i_map_g), B))
        for b, u in enumerate(warm_users):
            Vi[:, b] = Mu2i[u_map_g[u], :].toarray().ravel()
        for restart in [0.15, 0.3, 0.5]:
            t0 = time.perf_counter()
            alpha = 1.0 - restart
            Pu, Pi = ppr_batch(Mu2i, Mi2u, Vu, Vi, alpha=alpha, n_iter=30)
            scores_pool = Pi[pool_idx_g, :]
            S = {}
            for b, u in enumerate(warm_users):
                sc = {it: float(scores_pool[j, b]) for j, it in enumerate(pool_items_known)}
                sc.update(tail_scores)
                S[u] = sc
            preds = preds_from_scores(S, val_users, pool_full, rank_pop, top5_pop)
            m, n = evaluate(preds, relevant)
            dt = time.perf_counter() - t0
            variant = f"PPR restart={restart}"
            print(f"  {variant:45s} MAP@5={m:.4f} (n={n}) time={dt:.1f}s")
            variant_maps.setdefault(variant, {})[split_name] = (m, n, dt)

        print(f"  split завершён за {time.perf_counter()-t_split:.1f}s")

    # ---- итоговая таблица ----
    out_rows = []
    for variant, per_split in variant_maps.items():
        m_main = per_split.get("main (train<=20, val21-30)", (None, None, 0))
        m_early = per_split.get("early (train<=16, val17-20)", (None, None, 0))
        mean_map = (m_main[0] + m_early[0]) / 2 if (m_main[0] is not None and m_early[0] is not None) else None
        out_rows.append({
            "variant": variant,
            "MAP5_main": m_main[0], "n_main": m_main[1], "time_main_s": round(m_main[2], 2),
            "MAP5_early": m_early[0], "n_early": m_early[1], "time_early_s": round(m_early[2], 2),
            "MAP5_mean": mean_map,
        })
    df = pd.DataFrame(out_rows).sort_values("MAP5_mean", ascending=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "exp_grid_rp3_ease_ppr_sorted.csv", index=False)
    df_unsorted = df.sort_values("variant")
    df_unsorted.to_csv(RESULTS / "exp_grid_rp3_ease_ppr.csv", index=False)
    print(f"\n[saved] {RESULTS/'exp_grid_rp3_ease_ppr_sorted.csv'}")
    print(f"[saved] {RESULTS/'exp_grid_rp3_ease_ppr.csv'}")

    print("\n=== Grid: RP3beta / EASE / PPR (отсортировано по среднему MAP@5) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
