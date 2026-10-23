#!/usr/bin/env python3
"""Этап 1 плана (docs/improve_plan.md §4A, §4B): улучшение популярности + candidate recall.
Ничего из текущих финалов/results не перезаписывает. Выход: results/tables/exp_*.csv.

Структура (не полный перебор):
  Шаг A1: окно популярности (2/4/6/8 недель, unique viewers) при alpha=0 -> выбрать лучшее окно.
  Шаг A2: exponential decay (half-life 2 и 4 недели) vs лучшее окно из A1.
  Шаг A3: unique viewers vs sum-of-seconds, на лучшем базисе (окно или decay) из A1/A2.
  Шаг A4: alpha in {0, 0.5, 1, 1.5} для score = pop * (1+airings)^alpha, на лучшем базисе.
  Шаг B: candidate pool size 50/100/200 для ALS-гибрида, recall@50/100/200.
  Валидация: 2 временных окна (forward-chaining), строгое определение релевантности, одни и те же
  submission-пользователи и пул кандидатов для всех вариантов внутри каждого окна.
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "tables"
SEED = 42


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


def recall_at_k(pool, relevant_sets, k):
    """Доля релевантных программ пользователя, попавших в пул длины k (топ-k пула)."""
    pool_k = set(pool[:k])
    vals = []
    for u, rel in relevant_sets.items():
        if not rel:
            continue
        hit = len(rel & pool_k)
        vals.append(hit / len(rel))
    return float(np.mean(vals)) if vals else 0.0


def strict_relevance(val_sub, val_sched):
    """Строгое определение: сумма overlap_seconds пользователя по программе >= 50% max(duration) программы."""
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


def pop_score(train, candidates, window_weeks=None, half_life=None, max_week=None,
              weighting="unique", alpha=0.0, airing_count=None):
    """Считает score для всех candidates. Ровно одна из {window_weeks, half_life}."""
    df = train
    if window_weeks is not None:
        df = df[df["week"] >= max_week - window_weeks + 1]
        if weighting == "unique":
            pop = df.groupby("tv_show_id")["user_id"].nunique()
        else:
            pop = df.groupby("tv_show_id")["overlap_seconds"].sum()
    else:  # exponential decay по half_life
        age = (max_week - df["week"]).clip(lower=0)
        decay = 0.5 ** (age / half_life)
        if weighting == "unique":
            # вес на уникального пользователя: берём максимум decay по его просмотрам программы (простое и устойчивое правило)
            tmp = df.assign(decay=decay)
            pop = tmp.groupby(["tv_show_id", "user_id"])["decay"].max().groupby("tv_show_id").sum()
        else:
            tmp = df.assign(w=df["overlap_seconds"] * decay)
            pop = tmp.groupby("tv_show_id")["w"].sum()
    pop = pop.reindex(candidates).fillna(0.0)
    if alpha != 0.0 and airing_count is not None:
        ac = airing_count.reindex(candidates).fillna(0.0)
        pop = pop * (1.0 + ac) ** alpha
    return pop.sort_values(ascending=False)


def main():
    rows = []
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

    built = {}
    for name, cfg in splits.items():
        train, val_sched, candidates, relevant, val_users = build_split(
            inter, sub_users, cfg["train_end_week"], cfg["val_start_week"], cfg["val_end_week"], cfg["sched_val_path"]
        )
        built[name] = (train, val_sched, candidates, relevant, val_users, cfg["max_week"])
        print(f"[split {name}] train_weeks<= {cfg['train_end_week']}, val {cfg['val_start_week']}-{cfg['val_end_week']}: "
              f"users_with_rel={len(val_users)}/{len(sub_users)}, pool={len(candidates)}")
        rows.append({"stage": "meta", "split": name, "variant": "n_users_with_rel", "value": len(val_users)})
        rows.append({"stage": "meta", "split": name, "variant": "pool_size", "value": len(candidates)})

    # ---------------- Шаг A1: окно популярности, alpha=0, unique viewers ----------------
    print("\n=== A1: окно популярности (unique viewers, alpha=0) ===")
    a1_results = {}
    for name, (train, val_sched, candidates, relevant, val_users, max_week) in built.items():
        for window in [2, 4, 6, 8]:
            pop = pop_score(train, candidates, window_weeks=window, weighting="unique", max_week=max_week)
            top5 = pop.index.tolist()[:5]
            preds = {u: top5 for u in val_users}
            m, n = evaluate(preds, relevant)
            print(f"  [{name}] window={window}w unique  MAP@5={m:.4f} (n={n})")
            rows.append({"stage": "A1_window", "split": name, "variant": f"window={window}w_unique", "MAP5": m, "n_eval": n})
            a1_results[(name, window)] = m

    best_window_main = max([2, 4, 6, 8], key=lambda w: a1_results[("main (train<=20, val21-30)", w)])
    print(f"Лучшее окно на main split (unique, alpha=0): {best_window_main}w "
          f"(MAP@5={a1_results[('main (train<=20, val21-30)', best_window_main)]:.4f})")

    # ---------------- Шаг A2: exponential decay vs лучшее окно ----------------
    print("\n=== A2: exponential decay (half-life 2 и 4 недели) vs лучшее окно ===")
    a2_results = {}
    for name, (train, val_sched, candidates, relevant, val_users, max_week) in built.items():
        for hl in [2, 4]:
            pop = pop_score(train, candidates, half_life=hl, weighting="unique", max_week=max_week)
            top5 = pop.index.tolist()[:5]
            preds = {u: top5 for u in val_users}
            m, n = evaluate(preds, relevant)
            print(f"  [{name}] half_life={hl}w unique  MAP@5={m:.4f} (n={n})")
            rows.append({"stage": "A2_decay", "split": name, "variant": f"half_life={hl}w_unique", "MAP5": m, "n_eval": n})
            a2_results[(name, hl)] = m

    main_candidates_a1a2 = {
        f"window={best_window_main}w": a1_results[("main (train<=20, val21-30)", best_window_main)],
        "half_life=2w": a2_results[("main (train<=20, val21-30)", 2)],
        "half_life=4w": a2_results[("main (train<=20, val21-30)", 4)],
    }
    best_basis_name = max(main_candidates_a1a2, key=main_candidates_a1a2.get)
    print(f"Лучший базис (окно vs decay) на main: {best_basis_name} (MAP@5={main_candidates_a1a2[best_basis_name]:.4f})")

    def basis_pop(train, candidates, max_week, weighting):
        if best_basis_name.startswith("window"):
            return pop_score(train, candidates, window_weeks=best_window_main, weighting=weighting, max_week=max_week)
        else:
            hl = int(best_basis_name.split("=")[1].rstrip("w"))
            return pop_score(train, candidates, half_life=hl, weighting=weighting, max_week=max_week)

    # ---------------- Шаг A3: unique viewers vs sum seconds на лучшем базисе ----------------
    print("\n=== A3: unique viewers vs sum-of-seconds (на лучшем базисе) ===")
    a3_results = {}
    for name, (train, val_sched, candidates, relevant, val_users, max_week) in built.items():
        for weighting in ["unique", "sum_seconds"]:
            pop = basis_pop(train, candidates, max_week, weighting)
            top5 = pop.index.tolist()[:5]
            preds = {u: top5 for u in val_users}
            m, n = evaluate(preds, relevant)
            print(f"  [{name}] basis={best_basis_name} weighting={weighting}  MAP@5={m:.4f} (n={n})")
            rows.append({"stage": "A3_weighting", "split": name, "variant": f"{best_basis_name}_{weighting}", "MAP5": m, "n_eval": n})
            a3_results[(name, weighting)] = m

    best_weighting = max(["unique", "sum_seconds"], key=lambda w: a3_results[("main (train<=20, val21-30)", w)])
    print(f"Лучшее взвешивание на main: {best_weighting}")

    # ---------------- Шаг A4: alpha in {0, 0.5, 1, 1.5} ----------------
    print("\n=== A4: score = pop * (1+airings)^alpha ===")
    a4_results = {}
    for name, (train, val_sched, candidates, relevant, val_users, max_week) in built.items():
        airing_count = val_sched[val_sched["tv_show_id"] != 0]["tv_show_id"].value_counts()
        for alpha in [0.0, 0.5, 1.0, 1.5]:
            pop = basis_pop(train, candidates, max_week, best_weighting)
            if alpha != 0.0:
                ac = airing_count.reindex(candidates).fillna(0.0)
                pop = (pop * (1.0 + ac) ** alpha).sort_values(ascending=False)
            top5 = pop.index.tolist()[:5]
            preds = {u: top5 for u in val_users}
            m, n = evaluate(preds, relevant)
            print(f"  [{name}] alpha={alpha}  MAP@5={m:.4f} (n={n})")
            rows.append({"stage": "A4_alpha", "split": name, "variant": f"{best_basis_name}_{best_weighting}_alpha={alpha}", "MAP5": m, "n_eval": n})
            a4_results[(name, alpha)] = m

    best_alpha = max([0.0, 0.5, 1.0, 1.5], key=lambda a: a4_results[("main (train<=20, val21-30)", a)])
    print(f"\nЛучшая популярность на main: basis={best_basis_name}, weighting={best_weighting}, alpha={best_alpha}  "
          f"MAP@5={a4_results[('main (train<=20, val21-30)', best_alpha)]:.4f}")

    # ---------------- Шаг B: candidate pool size для ALS-гибрида: recall@50/100/200 ----------------
    print("\n=== B: candidate pool size (recall@K, строгая валидация, main split) ===")
    name_main = "main (train<=20, val21-30)"
    train, val_sched, candidates, relevant, val_users, max_week = built[name_main]
    airing_count = val_sched[val_sched["tv_show_id"] != 0]["tv_show_id"].value_counts()
    pop_final = basis_pop(train, candidates, max_week, best_weighting)
    if best_alpha != 0.0:
        ac = airing_count.reindex(candidates).fillna(0.0)
        pop_final = (pop_final * (1.0 + ac) ** best_alpha).sort_values(ascending=False)
    pool_full = pop_final.index.tolist()

    val_sub_relevant_only = {u: relevant[u] for u in val_users}
    for k in [50, 100, 200]:
        r = recall_at_k(pool_full, val_sub_relevant_only, k)
        print(f"  recall@{k} = {r:.4f}")
        rows.append({"stage": "B_recall", "split": name_main, "variant": f"recall@{k}", "MAP5": None, "n_eval": len(val_users), "recall": r})

    # ---------------- ALS-гибрид на пулах 50/100/200 (w фиксирован = 0.1, как в текущем финале) ----------------
    print("\n=== ALS-гибрид (w=0.1) на пулах 50/100/200, main split ===")
    train_items = train["tv_show_id"].unique()
    train_users_u = train["user_id"].unique()
    u_map = {u: i for i, u in enumerate(train_users_u)}
    i_map = {it: i for i, it in enumerate(train_items)}
    rows_u = train["user_id"].map(u_map).values
    cols_i = train["tv_show_id"].map(i_map).values
    data = np.log1p(train["overlap_seconds"].values)
    mat = csr_matrix((data, (rows_u, cols_i)), shape=(len(u_map), len(i_map)))
    als = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=20, random_state=SEED)
    t0 = time.perf_counter()
    als.fit(mat)
    t_fit = time.perf_counter() - t0

    def rank_from_scores(ids, scores, tie_breaker_rank):
        order = sorted(range(len(ids)), key=lambda i: (-scores[i], tie_breaker_rank.get(ids[i], 10**9)))
        return {ids[o]: r for r, o in enumerate(order)}

    for pool_size in [50, 100, 200]:
        pool = pool_full[:pool_size]
        rank_pop = {it: r for r, it in enumerate(pool)}
        pool_als_known = [it for it in pool if it in i_map]
        pool_als_idx = np.array([i_map[it] for it in pool_als_known])
        pool_als_factors = als.item_factors[pool_als_idx] if len(pool_als_idx) else np.zeros((0, 64))

        preds = {}
        for u in val_users:
            if u not in u_map or len(pool_als_known) == 0:
                rank_model = {}
            else:
                uf = als.user_factors[u_map[u]]
                scores = pool_als_factors @ uf
                rank_model = rank_from_scores(pool_als_known, scores, rank_pop)
            w = 0.1
            score = {it: (1 - w) * rank_pop[it] + w * rank_model.get(it, len(pool)) for it in pool}
            top = sorted(score.items(), key=lambda kv: (kv[1], rank_pop[kv[0]]))[:5]
            preds[u] = [it for it, _ in top]

        m, n = evaluate(preds, relevant)
        print(f"  pool_size={pool_size}  MAP@5(ALS hybrid w=0.1)={m:.4f} (n={n})")
        rows.append({"stage": "B_als_hybrid", "split": name_main, "variant": f"pool={pool_size}_w=0.1", "MAP5": m, "n_eval": n})

    rows.append({"stage": "meta", "split": name_main, "variant": "als_fit_time_s", "value": round(t_fit, 2)})
    rows.append({"stage": "meta", "split": name_main, "variant": "als_matrix_shape", "value": f"{mat.shape}"})

    # ---------------- Таблица, отсортированная по MAP5 ----------------
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "exp_stage1_full.csv", index=False)

    df_map = df[df["MAP5"].notna()].sort_values("MAP5", ascending=False)
    df_map.to_csv(RESULTS / "exp_stage1_sorted.csv", index=False)
    print("\n=== Итоговая таблица (отсортирована по MAP@5) ===")
    print(df_map[["stage", "split", "variant", "MAP5", "n_eval"]].to_string(index=False))

    print(f"\n[saved] {RESULTS/'exp_stage1_full.csv'}")
    print(f"[saved] {RESULTS/'exp_stage1_sorted.csv'}")

    return {
        "best_basis": best_basis_name, "best_weighting": best_weighting, "best_alpha": best_alpha,
        "pool_full_main": pool_full,
    }


if __name__ == "__main__":
    main()
