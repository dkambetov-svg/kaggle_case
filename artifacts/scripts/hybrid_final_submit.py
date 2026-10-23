#!/usr/bin/env python3
"""Финальные сабмиты: гибрид pop+cbf (w=0.2) и pop+als (w=0.1).
Обучение: недели 11-30. Пул: топ-50 по pop(27-30) x log(1+airings в 31-42)."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SEED = 42


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
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], tie_breaker_rank.get(ids[i], 10**9)))
    return {ids[o]: r for r, o in enumerate(order)}


def hybrid_top5(pool, rank_pop, rank_model, w, k=5):
    score = {it: (1 - w) * rank_pop[it] + w * rank_model.get(it, len(pool)) for it in pool}
    top = sorted(score.items(), key=lambda kv: (kv[1], rank_pop[kv[0]]))[:k]
    return [it for it, _ in top]


def main():
    inter = pd.read_parquet(DATA / "user_interactions.parquet")
    inter["user_id"] = inter["user_id"].astype(str)
    train = inter[inter["week"] <= 30].copy()
    sub = pd.read_csv(DATA / "submission.csv", dtype={"user_id": str})

    sched42 = pd.read_csv(DATA / "export_arh_31-42-final.csv")
    candidates = sched42[sched42["tv_show_id"] != 0]["tv_show_id"].unique()
    candidates_set = set(candidates)

    pop_recent4w = train[train["week"] >= 27].groupby("tv_show_id")["user_id"].nunique()
    airing_count_42 = sched42[sched42["tv_show_id"] != 0]["tv_show_id"].value_counts()
    score_b = {show: pop_recent4w.get(show, 0) * np.log1p(airing_count_42.get(show, 0)) for show in candidates}
    pool = [it for it, _ in sorted(score_b.items(), key=lambda kv: -kv[1])[:50]]
    rank_pop = {it: r for r, it in enumerate(pool)}
    print(f"[pool] top-50: {pool[:5]} ...")

    # --- CBF ---
    feats_cbf, cbf_id_to_idx = build_cbf_features(DATA / "export_arh_21-30-final.csv")
    train_w = train.copy()
    train_w["weight"] = np.log1p(train_w["overlap_seconds"])
    pool_cbf_known = [it for it in pool if it in cbf_id_to_idx]
    pool_cbf_idx = np.array([cbf_id_to_idx[it] for it in pool_cbf_known])
    pool_cbf_feats = feats_cbf[pool_cbf_idx] if len(pool_cbf_idx) else np.zeros((0, feats_cbf.shape[1]))

    def cbf_rank_for_user(u):
        g = train_w[train_w["user_id"] == u]
        idxs = [cbf_id_to_idx[sid] for sid in g["tv_show_id"] if sid in cbf_id_to_idx]
        if not idxs or len(pool_cbf_known) == 0:
            return {}
        mask = g["tv_show_id"].isin(cbf_id_to_idx)
        weights = g.loc[mask, "weight"].values
        profile = (feats_cbf[idxs].T @ weights) / weights.sum()
        profile = profile / (np.linalg.norm(profile) + 1e-9)
        scores = pool_cbf_feats @ profile
        return rank_from_scores(pool_cbf_known, scores, rank_pop)

    # --- ALS ---
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

    def als_rank_for_user(u):
        if u not in u_map or len(pool_als_known) == 0:
            return {}
        uf = als.user_factors[u_map[u]]
        scores = pool_als_factors @ uf
        return rank_from_scores(pool_als_known, scores, rank_pop)

    preds_cbf, preds_als = {}, {}
    for u in sub["user_id"]:
        preds_cbf[u] = hybrid_top5(pool, rank_pop, cbf_rank_for_user(u), w=0.2)
        preds_als[u] = hybrid_top5(pool, rank_pop, als_rank_for_user(u), w=0.1)

    def write_submission(preds, path):
        out = sub.copy()
        out["tv_show_id"] = out["user_id"].map(lambda u: " ".join(str(int(x)) for x in preds[u][:5]))
        assert (out["user_id"].values == sub["user_id"].values).all()
        for ids in out["tv_show_id"]:
            parts = ids.split()
            assert len(parts) == 5
            for p in parts:
                assert int(p) in candidates_set
        out.to_csv(path, index=False)
        print(f"[saved] {path} rows={len(out)}")

    write_submission(preds_cbf, DATA / "submission_hybrid_pop_cbf_w02.csv")
    write_submission(preds_als, DATA / "submission_hybrid_pop_als_w01.csv")


if __name__ == "__main__":
    main()
