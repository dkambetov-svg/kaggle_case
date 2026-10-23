#!/usr/bin/env python3
"""Personalized PageRank (по образцу HW6: github.com/dryjins/RecSys-LLMs/tree/main/week6),
адаптированный под двудольный граф пользователь-программа этого проекта.

Граф: узлы = {пользователи из train} ∪ {программы из train}. Рёбра только user<->item,
вес = log1p(сумма overlap_seconds за train). Нет user-user / item-item рёбер.

Транзишены (по аналогии с HW6 §3.2):
  Mu2i[u, i] = w(u,i) / sum_i w(u,i)   (row-normalized, user -> item)
  Mi2u[i, u] = w(u,i) / sum_u w(u,i)   (row-normalized, item -> user)
Нет "padding"-узлов; узлы без исходящих рёбер в этом графе не встречаются по построению
(узел попадает в граф только если у него есть хотя бы одно обучающее взаимодействие).

Обновление (симметричный по блокам аналог p_next = alpha*P^T*p + (1-alpha)*v из HW6):
  p_item_next = alpha * Mu2i^T @ p_user + (1-alpha) * v_item
  p_user_next = alpha * Mi2u^T @ p_item + (1-alpha) * v_user
30 итераций, alpha=0.85 (restart prob 0.15), БАТЧЕМ для всех eval-пользователей сразу
через scipy.sparse (без python-цикла по пользователям в самой PPR-итерации).

Personalized restart: v_item = Mu2i[u, :] (программы пользователя из train, пропорционально
весам), v_user = 0. Global PageRank: v uniform по всем узлам графа (как в HW6 §3.4).

Скоры программ берутся из пула кандидатов (тот же пул, что у остальных методов), повторы
разрешены, холодным (нет в train-графе или все скоры по пулу нулевые) — fallback популярности.

Не перезаписывает финал/результаты: добавляет строки в results/tables/exp_stage1_full.csv
и exp_stage1_sorted.csv (это уже экспериментальные таблицы этапа 1/плана, не финальные).
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "tables"
ALPHA = 0.85  # вероятность перехода по ссылке (1 - restart prob 0.15), как задано в задаче
N_ITER = 30


def evaluate(predictions, relevant_sets, k=5):
    ap_sum = 0.0
    users = [u for u in predictions if u in relevant_sets and relevant_sets[u]]
    if not users:
        return 0.0, 0
    for u in users:
        preds = predictions[u][:k]
        actuals = relevant_sets[u]
        hits = ap = 0
        for i, p in enumerate(preds):
            if p in actuals:
                hits += 1
                ap += hits / (i + 1)
        ap_sum += ap / min(len(actuals), k)
    return ap_sum / len(users), len(users)


def strict_relevance(val_sub, val_sched):
    user_show_sum = val_sub.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum()
    dur_lookup = val_sched[val_sched["tv_show_id"] != 0].groupby("tv_show_id")["duration"].max()
    relevant = {}
    for u, s in user_show_sum.groupby(level=0):
        s = s.droplevel(0).sort_values(ascending=False)
        rel_ids = {show for show, sec in s.items() if pd.notna(dur_lookup.get(show, np.nan))
                   and dur_lookup[show] > 0 and sec >= 0.5 * dur_lookup[show]}
        relevant[u] = rel_ids
    return relevant


def build_split(inter, sub_users, train_end_week, val_start_week, val_end_week, sched_val_path):
    train = inter[inter["week"] <= train_end_week].copy()
    val_raw = inter[(inter["week"] >= val_start_week) & (inter["week"] <= val_end_week)].copy()
    val_sched = pd.read_csv(sched_val_path, low_memory=False)
    candidates = val_sched[val_sched["tv_show_id"] != 0]["tv_show_id"].unique()
    val_sub = val_raw[val_raw["user_id"].isin(sub_users)]
    relevant = strict_relevance(val_sub, val_sched)
    val_users = [u for u in sub_users if u in relevant and relevant[u]]
    return train, val_sched, candidates, relevant, val_users


def row_normalize(mat: csr_matrix) -> csr_matrix:
    """Row-normalize a sparse matrix; rows are guaranteed nonzero by construction here."""
    row_sums = np.asarray(mat.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sums)
    nonzero = row_sums > 0
    inv[nonzero] = 1.0 / row_sums[nonzero]
    d = csr_matrix((inv, (np.arange(len(inv)), np.arange(len(inv)))), shape=(len(inv), len(inv)))
    return d @ mat


def build_graph(train):
    """Строит R (users x items, вес=log1p(sum overlap_seconds)), Mu2i, Mi2u."""
    agg = train.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum().reset_index()
    agg["weight"] = np.log1p(agg["overlap_seconds"])

    graph_users = agg["user_id"].unique()
    graph_items = agg["tv_show_id"].unique()
    u_map = {u: i for i, u in enumerate(graph_users)}
    i_map = {it: i for i, it in enumerate(graph_items)}

    rows = agg["user_id"].map(u_map).values
    cols = agg["tv_show_id"].map(i_map).values
    R = csr_matrix((agg["weight"].values, (rows, cols)), shape=(len(u_map), len(i_map)))

    # Проверка: по построению (граф строится только из train-взаимодействий) нулевых
    # степеней быть не должно -- узел появляется в графе только если есть хотя бы одно ребро.
    zero_deg_users = int((np.asarray(R.sum(axis=1)).ravel() == 0).sum())
    zero_deg_items = int((np.asarray(R.sum(axis=0)).ravel() == 0).sum())

    Mu2i = row_normalize(R)          # user -> item, row-stochastic
    Mi2u = row_normalize(R.T.tocsr())  # item -> user, row-stochastic

    return u_map, i_map, R, Mu2i, Mi2u, zero_deg_users, zero_deg_items


def ppr_batch(Mu2i, Mi2u, Vu, Vi, alpha=ALPHA, n_iter=N_ITER):
    """Батчевая степенная итерация. Vu: (n_users,B), Vi: (n_items,B) -- restart-матрицы
    (каждый столбец -- своя restart-дистрибуция, суммируется в 1 по (Vu[:,b]+Vi[:,b])).
    Возвращает (Pu, Pi) после n_iter итераций и финальный L1-residual между последними
    двумя итерациями (для диагностики сходимости)."""
    Pu = Vu.copy()
    Pi = Vi.copy()
    Mu2iT = Mu2i.T.tocsr()
    Mi2uT = Mi2u.T.tocsr()
    residual = None
    for it in range(n_iter):
        Pi_next = alpha * (Mu2iT @ Pu) + (1 - alpha) * Vi
        Pu_next = alpha * (Mi2uT @ Pi) + (1 - alpha) * Vu
        if it == n_iter - 1:
            residual = float(np.abs(Pu_next - Pu).sum(axis=0).mean() + np.abs(Pi_next - Pi).sum(axis=0).mean())
        Pu, Pi = Pu_next, Pi_next
    # проверка сохранения массы (сумма по каждому столбцу должна остаться ~1)
    mass = (Pu.sum(axis=0) + Pi.sum(axis=0))
    return Pu, Pi, residual, mass


def rank_from_scores(ids, scores, tie_breaker_rank):
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], tie_breaker_rank.get(ids[i], 10**9)))
    return [ids[o] for o in order]


def run_split(split_name, inter, sub_users, cfg, rows_out):
    train, val_sched, candidates, relevant, val_users = build_split(
        inter, sub_users, cfg["train_end_week"], cfg["val_start_week"], cfg["val_end_week"], cfg["sched_val_path"]
    )
    max_week = cfg["max_week"]
    print(f"\n[split {split_name}] train<= {cfg['train_end_week']}, val {cfg['val_start_week']}-{cfg['val_end_week']}: "
          f"users_with_rel={len(val_users)}/{len(sub_users)}, pool={len(candidates)}")

    # Popularity fallback (recent 4w, unique viewers) -- тот же критерий, что и в cold-start fallback других методов
    pop_recent = train[train["week"] >= max_week - 3].groupby("tv_show_id")["user_id"].nunique()
    pop_recent = pop_recent.reindex(candidates).fillna(0.0).sort_values(ascending=False)
    pop_rank = {it: r for r, it in enumerate(pop_recent.index)}
    top5_pop = pop_recent.index.tolist()[:5]

    # --- граф из train ---
    u_map, i_map, R, Mu2i, Mi2u, zero_deg_users, zero_deg_items = build_graph(train)
    n_users_g, n_items_g = len(u_map), len(i_map)
    n_nodes = n_users_g + n_items_g
    print(f"  граф: {n_users_g} пользователей, {n_items_g} программ, {R.nnz} рёбер; "
          f"zero-degree users={zero_deg_users}, items={zero_deg_items}")
    rows_out.append({"stage": "PPR_meta", "split": split_name, "variant": "graph_users", "value": n_users_g})
    rows_out.append({"stage": "PPR_meta", "split": split_name, "variant": "graph_items", "value": n_items_g})
    rows_out.append({"stage": "PPR_meta", "split": split_name, "variant": "graph_edges", "value": int(R.nnz)})
    rows_out.append({"stage": "PPR_meta", "split": split_name, "variant": "zero_degree_users", "value": zero_deg_users})
    rows_out.append({"stage": "PPR_meta", "split": split_name, "variant": "zero_degree_items", "value": zero_deg_items})

    cand_in_graph = [c for c in candidates if c in i_map]
    cand_in_graph_idx = np.array([i_map[c] for c in cand_in_graph])
    print(f"  кандидатов в графе: {len(cand_in_graph)}/{len(candidates)}")

    # ---------------- Global PageRank (uniform restart, B=1) ----------------
    Vu_g = np.full((n_users_g, 1), 1.0 / n_nodes)
    Vi_g = np.full((n_items_g, 1), 1.0 / n_nodes)
    t0 = time.perf_counter()
    Pu_g, Pi_g, resid_g, mass_g = ppr_batch(Mu2i, Mi2u, Vu_g, Vi_g)
    t_global = time.perf_counter() - t0
    print(f"  Global PageRank: residual(L1, last iter)={resid_g:.2e}, mass_check={mass_g[0]:.6f}, time={t_global:.2f}s")
    rows_out.append({"stage": "PPR_meta", "split": split_name, "variant": "global_residual_L1", "value": resid_g})
    rows_out.append({"stage": "PPR_meta", "split": split_name, "variant": "global_mass_check", "value": float(mass_g[0])})

    global_scores = Pi_g[cand_in_graph_idx, 0] if len(cand_in_graph) else np.array([])
    global_score_dict = dict(zip(cand_in_graph, global_scores))
    global_ranked = rank_from_scores(cand_in_graph, global_scores, pop_rank) if len(cand_in_graph) else []
    preds_global = {}
    for u in val_users:
        top = global_ranked[:5] if global_ranked else []
        for p in top5_pop:
            if p not in top:
                top.append(p)
        preds_global[u] = top[:5]
    m_global, n_global = evaluate(preds_global, relevant)
    print(f"  Global PageRank  MAP@5={m_global:.4f} (n={n_global})")
    rows_out.append({"stage": "PPR_global", "split": split_name, "variant": "global_pagerank_alpha=0.85",
                      "MAP5": m_global, "n_eval": n_global})

    # ---------------- Personalized PageRank (батчем для всех eval-пользователей) ----------------
    eval_in_graph = [u for u in val_users if u in u_map]
    cold_users = [u for u in val_users if u not in u_map]
    print(f"  PPR: {len(eval_in_graph)}/{len(val_users)} eval-пользователей есть в train-графе "
          f"({len(cold_users)} холодных -> fallback популярности)")

    B = len(eval_in_graph)
    Vu_p = np.zeros((n_users_g, B))
    Vi_p = np.zeros((n_items_g, B))
    for b, u in enumerate(eval_in_graph):
        uidx = u_map[u]
        Vu_p[uidx, b] = 0.0  # restart только на программы пользователя, не на самого пользователя
        Vi_p[:, b] = Mu2i[uidx, :].toarray().ravel()  # нормированные веса программ пользователя

    t0 = time.perf_counter()
    Pu_p, Pi_p, resid_p, mass_p = ppr_batch(Mu2i, Mi2u, Vu_p, Vi_p)
    t_ppr = time.perf_counter() - t0
    print(f"  PPR batch: B={B}, residual(L1, last iter, mean)={resid_p:.2e}, "
          f"mass_check(min/max)={mass_p.min():.6f}/{mass_p.max():.6f}, time={t_ppr:.2f}s")
    rows_out.append({"stage": "PPR_meta", "split": split_name, "variant": "ppr_residual_L1_mean", "value": resid_p})
    rows_out.append({"stage": "PPR_meta", "split": split_name, "variant": "ppr_batch_time_s", "value": round(t_ppr, 2)})
    rows_out.append({"stage": "PPR_meta", "split": split_name, "variant": "ppr_n_cold_users", "value": len(cold_users)})

    item_scores_pool = Pi_p[cand_in_graph_idx, :] if len(cand_in_graph) else np.zeros((0, B))  # (n_cand, B)

    preds_ppr = {}
    for b, u in enumerate(eval_in_graph):
        scores = item_scores_pool[:, b] if len(cand_in_graph) else np.array([])
        if len(cand_in_graph) == 0 or scores.sum() == 0:
            preds_ppr[u] = top5_pop
            continue
        top = rank_from_scores(cand_in_graph, scores, pop_rank)[:5]
        for p in top5_pop:
            if len(top) >= 5:
                break
            if p not in top:
                top.append(p)
        preds_ppr[u] = top[:5]
    for u in cold_users:
        preds_ppr[u] = top5_pop

    m_ppr, n_ppr = evaluate(preds_ppr, relevant)
    print(f"  Personalized PageRank (restart=user's own items, alpha=0.85)  MAP@5={m_ppr:.4f} (n={n_ppr})")
    rows_out.append({"stage": "PPR_personalized", "split": split_name, "variant": "ppr_alpha=0.85_restart=own_items",
                      "MAP5": m_ppr, "n_eval": n_ppr})

    # Справочно: популярность на этом же сплите (для прямого сравнения в таблице)
    preds_pop = {u: top5_pop for u in val_users}
    m_pop, n_pop = evaluate(preds_pop, relevant)
    rows_out.append({"stage": "PPR_reference", "split": split_name, "variant": "popularity_recent4w_unique",
                      "MAP5": m_pop, "n_eval": n_pop})
    print(f"  (справочно) Popularity recent4w  MAP@5={m_pop:.4f} (n={n_pop})")


def main():
    rows_out = []
    inter = pd.read_parquet(DATA / "user_interactions.parquet")
    inter["user_id"] = inter["user_id"].astype(str)
    sub = pd.read_csv(DATA / "submission.csv", dtype={"user_id": str})
    sub_users = sub["user_id"].tolist()

    splits = {
        "main (train<=20, val21-30)": dict(train_end_week=20, val_start_week=21, val_end_week=30,
                                            sched_val_path=DATA / "export_arh_21-30-final.csv", max_week=20),
        "early (train<=16, val17-20)": dict(train_end_week=16, val_start_week=17, val_end_week=20,
                                             sched_val_path=DATA / "export_arh_11-20-final.csv", max_week=16),
    }

    for name, cfg in splits.items():
        run_split(name, inter, sub_users, cfg, rows_out)

    df_new = pd.DataFrame(rows_out)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(RESULTS / "exp_ppr.csv", index=False)
    print(f"\n[saved] {RESULTS/'exp_ppr.csv'}")

    # Добавляем в общую таблицу этапа 1 (append, не перезаписываем старые строки)
    full_path = RESULTS / "exp_stage1_full.csv"
    if full_path.exists():
        df_full_old = pd.read_csv(full_path)
        df_full_new = pd.concat([df_full_old, df_new], ignore_index=True)
    else:
        df_full_new = df_new
    df_full_new.to_csv(full_path, index=False)
    print(f"[updated] {full_path} (append PPR rows)")

    sorted_path = RESULTS / "exp_stage1_sorted.csv"
    df_map = df_full_new[df_full_new["MAP5"].notna()].sort_values("MAP5", ascending=False)
    df_map.to_csv(sorted_path, index=False)
    print(f"[updated] {sorted_path}")

    print("\n=== Таблица PPR + справочные строки (отсортирована по MAP@5) ===")
    df_map_ppr_rel = df_new[df_new["MAP5"].notna()].sort_values("MAP5", ascending=False)
    print(df_map_ppr_rel[["stage", "split", "variant", "MAP5", "n_eval"]].to_string(index=False))


if __name__ == "__main__":
    main()
