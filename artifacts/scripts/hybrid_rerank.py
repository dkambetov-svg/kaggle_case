#!/usr/bin/env python3
"""Гибрид: кандидаты = топ-50 по pop x log(1+airings), переранжирование CBF или ALS
через rank fusion score = (1-w)*rank_pop + w*rank_model. Подбор w на строгой валидации."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "tables"
SEED = 42
W_GRID = [0.1, 0.2, 0.3, 0.5]


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


def build_cbf_features(sched_path):
    sched = pd.read_csv(sched_path)
    feat = sched[sched["tv_show_id"] != 0].drop_duplicates("tv_show_id").copy()
    genre_cols = []
    for col in ["tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"]:
        genre_cols.append(pd.get_dummies(feat[col].fillna(""), prefix=col))
    feat["year"] = pd.to_numeric(feat["year_of_production"], errors="coerce").fillna(2000)
    feat["decade"] = (feat["year"] // 10 * 10).astype(int).astype(str)
    decade_dummies = pd.get_dummies(feat["decade"], prefix="decade")
    channel_dummies = pd.get_dummies(feat["channel_id"], prefix="channel")
    feats = pd.concat(genre_cols + [decade_dummies, channel_dummies], axis=1).values.astype(float)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    id_to_idx = {it: i for i, it in enumerate(feat["tv_show_id"].values)}
    return feats, id_to_idx


def rank_from_scores(ids, scores, tie_breaker_rank):
    """ids, scores -> dict id -> rank (0=best). Ties broken by tie_breaker_rank (lower=better)."""
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], tie_breaker_rank.get(ids[i], 10**9)))
    return {ids[o]: r for r, o in enumerate(order)}


def hybrid_predict(pool, rank_pop, rank_model, w, k=5):
    preds = {}
    for u, rm in rank_model.items():
        score = {it: (1 - w) * rank_pop[it] + w * rm.get(it, len(pool)) for it in pool}
        top = sorted(score.items(), key=lambda kv: (kv[1], rank_pop[kv[0]]))[:k]
        preds[u] = [it for it, _ in top]
    return preds


def main():
    rows_out = []
    inter = pd.read_parquet(DATA / "user_interactions.parquet")
    inter["user_id"] = inter["user_id"].astype(str)
    sub = pd.read_csv(DATA / "submission.csv", dtype={"user_id": str})
    sub_users = sub["user_id"].tolist()

    train = inter[inter["week"] <= 20].copy()
    val_raw = inter[(inter["week"] > 20) & (inter["week"] <= 30)].copy()
    val_sched = pd.read_csv(DATA / "export_arh_21-30-final.csv")
    candidates = val_sched[val_sched["tv_show_id"] != 0]["tv_show_id"].unique()

    # релевантность (строгая и top20), submission-пользователи
    val_sub = val_raw[val_raw["user_id"].isin(sub_users)]
    user_show_sum = val_sub.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum()
    dur_lookup = val_sched[val_sched["tv_show_id"] != 0].groupby("tv_show_id")["duration"].max()

    relevant_strict, relevant_top20 = {}, {}
    for u, s in user_show_sum.groupby(level=0):
        s = s.droplevel(0).sort_values(ascending=False)
        strict_ids = {show for show, sec in s.items() if pd.notna(dur_lookup.get(show, np.nan))
                      and dur_lookup[show] > 0 and sec >= 0.5 * dur_lookup[show]}
        relevant_strict[u] = strict_ids
        relevant_top20[u] = set(s.index[:20].tolist())

    val_users_sub = [u for u in sub_users if u in relevant_strict]

    # --- пул кандидатов: топ-50 по pop(train 17-20) x log(1+airings в holdout 21-30) ---
    pop_recent4w = train[train["week"] >= 17].groupby("tv_show_id")["user_id"].nunique()
    airing_count_val = val_sched[val_sched["tv_show_id"] != 0]["tv_show_id"].value_counts()
    score_b = {show: pop_recent4w.get(show, 0) * np.log1p(airing_count_val.get(show, 0)) for show in candidates}
    pool = [it for it, _ in sorted(score_b.items(), key=lambda kv: -kv[1])[:50]]
    rank_pop = {it: r for r, it in enumerate(pool)}
    print(f"[pool] топ-50 по pop x log(airings), holdout: {pool[:5]} ... (всего {len(pool)})")

    # --- CBF фичи (train-расписание 11-20) ---
    feats_cbf, cbf_id_to_idx = build_cbf_features(DATA / "export_arh_11-20-final.csv")
    train_w = train.copy()
    train_w["weight"] = np.log1p(train_w["overlap_seconds"])

    pool_cbf_known = [it for it in pool if it in cbf_id_to_idx]
    pool_cbf_idx = np.array([cbf_id_to_idx[it] for it in pool_cbf_known])
    pool_cbf_feats = feats_cbf[pool_cbf_idx] if len(pool_cbf_idx) else np.zeros((0, feats_cbf.shape[1]))

    rank_cbf_per_user = {}
    for u in val_users_sub:
        g = train_w[train_w["user_id"] == u]
        idxs = [cbf_id_to_idx[sid] for sid in g["tv_show_id"] if sid in cbf_id_to_idx]
        if not idxs:
            rank_cbf_per_user[u] = {}
            continue
        mask = g["tv_show_id"].isin(cbf_id_to_idx)
        weights = g.loc[mask, "weight"].values
        profile = (feats_cbf[idxs].T @ weights) / weights.sum()
        profile = profile / (np.linalg.norm(profile) + 1e-9)
        scores = pool_cbf_feats @ profile if len(pool_cbf_known) else np.array([])
        r = rank_from_scores(pool_cbf_known, scores, rank_pop) if len(pool_cbf_known) else {}
        rank_cbf_per_user[u] = r

    # --- ALS (train 11-20) ---
    train_items = train["tv_show_id"].unique()
    train_users = train["user_id"].unique()
    u_map = {u: i for i, u in enumerate(train_users)}
    i_map = {it: i for i, it in enumerate(train_items)}
    rows_u = train["user_id"].map(u_map).values
    cols_i = train["tv_show_id"].map(i_map).values
    data = np.log1p(train["overlap_seconds"].values)
    mat = csr_matrix((data, (rows_u, cols_i)), shape=(len(u_map), len(i_map)))
    als = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=20, random_state=SEED)
    als.fit(mat)

    pool_als_known = [it for it in pool if it in i_map]
    pool_als_idx = np.array([i_map[it] for it in pool_als_known])
    pool_als_factors = als.item_factors[pool_als_idx] if len(pool_als_idx) else np.zeros((0, 64))

    rank_als_per_user = {}
    for u in val_users_sub:
        if u not in u_map or len(pool_als_known) == 0:
            rank_als_per_user[u] = {}
            continue
        uf = als.user_factors[u_map[u]]
        scores = pool_als_factors @ uf
        rank_als_per_user[u] = rank_from_scores(pool_als_known, scores, rank_pop)

    # --- grid w, таблица ---
    best = {"cbf": None, "als": None}
    for kind, rank_model in [("cbf", rank_cbf_per_user), ("als", rank_als_per_user)]:
        for w in W_GRID:
            preds = hybrid_predict(pool, rank_pop, rank_model, w)
            preds = {u: preds[u] for u in val_users_sub}
            m_s, n_s = evaluate(preds, relevant_strict)
            m_t, n_t = evaluate(preds, relevant_top20)
            print(f"Hybrid pop+{kind} w={w:.1f}  strict MAP@5={m_s:.4f} (n={n_s})  top20 MAP@5={m_t:.4f} (n={n_t})")
            rows_out.append({"method": f"Hybrid pop+{kind}", "w": w, "MAP5_strict": m_s,
                              "MAP5_top20": m_t, "n_strict": n_s, "n_top20": n_t})
            if best[kind] is None or m_s > best[kind][1]:
                best[kind] = (w, m_s, m_t)

    pure_pop_pool = {u: pool[:5] for u in val_users_sub}
    m_s, n_s = evaluate(pure_pop_pool, relevant_strict)
    m_t, n_t = evaluate(pure_pop_pool, relevant_top20)
    print(f"Pure pop (top5 of pool, w=0 reference)  strict MAP@5={m_s:.4f} (n={n_s})  top20 MAP@5={m_t:.4f} (n={n_t})")
    rows_out.append({"method": "Popularity (pool top5, w=0)", "w": 0.0, "MAP5_strict": m_s,
                      "MAP5_top20": m_t, "n_strict": n_s, "n_top20": n_t})

    print(f"\nЛучший w для CBF-гибрида (по strict): {best['cbf']}")
    print(f"Лучший w для ALS-гибрида (по strict): {best['als']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(RESULTS / "hybrid_validation.csv", index=False)
    print(f"[saved] {RESULTS/'hybrid_validation.csv'}")

    return best


if __name__ == "__main__":
    main()
