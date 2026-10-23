#!/usr/bin/env python3
"""ALS, CBF re-rank и гибрид (rank fusion) с валидацией MAP@5 и анализом новизны."""
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


def evaluate(predictions, ground_truth, k=5):
    ap_sum = 0.0
    gt = ground_truth.groupby("user_id")["tv_show_id"].apply(set)
    users = set(predictions) & set(gt.index)
    if not users:
        return 0.0, 0
    for u in users:
        preds = predictions[u][:k]
        actuals = gt[u]
        hits = ap = 0
        for i, p in enumerate(preds):
            if p in actuals:
                hits += 1
                ap += hits / (i + 1)
        ap_sum += ap / min(len(actuals), k)
    return ap_sum / len(users), len(users)


def rank_fusion(rank_a, rank_b, w, candidates):
    """rank_a/b: dict item -> rank (0 = top). Score = -(w*rank_a + (1-w)*rank_b)."""
    n = len(candidates)
    scores = {}
    for c in candidates:
        ra = rank_a.get(c, n)
        rb = rank_b.get(c, n)
        scores[c] = -(w * ra + (1 - w) * rb)
    return scores


def build_top5(score_dict, pop_rank, k=5):
    # ties -> break by popularity (higher popularity first == lower pop_rank)
    items = sorted(score_dict.items(), key=lambda kv: (-kv[1], pop_rank.get(kv[0], 10**9)))
    return [i for i, _ in items[:k]]


def run():
    t_all = time.perf_counter()
    inter = pd.read_parquet(DATA / "user_interactions.parquet")
    inter["user_id"] = inter["user_id"].astype(str)

    train = inter[inter["week"] <= 20].copy()
    val = inter[(inter["week"] > 20) & (inter["week"] <= 30)].copy()
    val_users = val["user_id"].unique()

    val_sched = pd.read_csv(DATA / "export_arh_21-30-final.csv")
    candidates = val_sched[val_sched["tv_show_id"] != 0]["tv_show_id"].unique()

    gt = val.groupby("user_id")["tv_show_id"].apply(set)
    n_rel = [len(s) for s in gt]
    print(f"avg_rel={np.mean(n_rel):.2f} med_rel={np.median(n_rel):.2f} pool={len(candidates)}")

    # --- маппинги строго по train (без утечки) ---
    train_items = train["tv_show_id"].unique()
    train_users = train["user_id"].unique()
    u_map = {u: i for i, u in enumerate(train_users)}
    i_map = {it: i for i, it in enumerate(train_items)}
    train_item_set = set(train_items)

    # --- новизна: доля "новых" программ (не встречавшихся в train) ---
    new_in_pool = sum(1 for c in candidates if c not in train_item_set)
    new_in_rel = sum(1 for s in gt for it in s if it not in train_item_set)
    total_rel = sum(len(s) for s in gt)
    print(f"Новых программ в пуле кандидатов: {new_in_pool}/{len(candidates)} = {new_in_pool/len(candidates)*100:.1f}%")
    print(f"Новых программ среди релевантных (val): {new_in_rel}/{total_rel} = {new_in_rel/total_rel*100:.1f}%")

    # --- популярность (бейзлайны) ---
    pop_recent = train[train["week"] >= 17].groupby("tv_show_id")["user_id"].nunique().sort_values(ascending=False)
    pop_global = train.groupby("tv_show_id")["user_id"].nunique().sort_values(ascending=False)
    pop_rank = {it: r for r, it in enumerate(pop_recent.index)}
    preds_pop_recent = {u: pop_recent.index.tolist()[:5] for u in val_users}
    preds_pop_global = {u: pop_global.index.tolist()[:5] for u in val_users}

    # --- ALS ---
    t0 = time.perf_counter()
    rows = train["user_id"].map(u_map).values
    cols = train["tv_show_id"].map(i_map).values
    data = np.log1p(train["overlap_seconds"].values)
    mat = csr_matrix((data, (rows, cols)), shape=(len(u_map), len(i_map)))

    def fit_als(factors, reg, iters, alpha=1.0):
        model = AlternatingLeastSquares(
            factors=factors, regularization=reg, iterations=iters,
            random_state=SEED,
        )
        model.fit(mat * alpha)
        return model

    als_main = fit_als(64, 0.05, 20)
    t_als = time.perf_counter() - t0

    map_i_to_item = {i: it for it, i in i_map.items()}
    cand_known = [c for c in candidates if c in i_map]
    cand_unknown = [c for c in candidates if c not in i_map]
    cand_known_idx = np.array([i_map[c] for c in cand_known])

    def als_predict(model, cand_known, cand_known_idx, cand_unknown, pop_rank_local):
        item_factors = model.item_factors[cand_known_idx]
        preds = {}
        for u in val_users:
            if u not in u_map:
                preds[u] = pop_recent.index.tolist()[:5]
                continue
            uf = model.user_factors[u_map[u]]
            scores = item_factors @ uf
            min_score = scores.min() - 1.0  # новые/неизвестные программы получают скор ниже всех известных
            score_dict = dict(zip(cand_known, scores))
            for c in cand_unknown:
                score_dict[c] = min_score
            preds[u] = build_top5(score_dict, pop_rank_local, k=5)
        return preds

    preds_als = als_predict(als_main, cand_known, cand_known_idx, cand_unknown, pop_rank)

    # --- CBF (категории + 3 жанра one-hot + год по десятилетиям), профиль пользователя взвешен log(1+сек) ---
    sched_tr = pd.concat([pd.read_csv(DATA / f) for f in ["export_arh_11-20-final.csv"]])
    feat = sched_tr[sched_tr["tv_show_id"] != 0].drop_duplicates("tv_show_id").copy()
    genre_cols = []
    for col in ["tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"]:
        dummies = pd.get_dummies(feat[col].fillna(""), prefix=col)
        genre_cols.append(dummies)
    feat["year"] = pd.to_numeric(feat["year_of_production"], errors="coerce").fillna(2000)
    feat["decade"] = (feat["year"] // 10 * 10).astype(int).astype(str)
    decade_dummies = pd.get_dummies(feat["decade"], prefix="decade")
    feats = pd.concat(genre_cols + [decade_dummies], axis=1).values.astype(float)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    cbf_id_to_idx = {it: i for i, it in enumerate(feat["tv_show_id"].values)}

    train_w = train.copy()
    train_w["weight"] = np.log1p(train_w["overlap_seconds"])
    user_profiles = {}
    for uid, g in train_w.groupby("user_id"):
        idxs = [cbf_id_to_idx[sid] for sid in g["tv_show_id"] if sid in cbf_id_to_idx]
        if not idxs:
            continue
        mask = g["tv_show_id"].isin(cbf_id_to_idx)
        weights = g.loc[mask, "weight"].values
        profile = (feats[idxs].T @ weights) / weights.sum()
        norm = np.linalg.norm(profile)
        user_profiles[uid] = profile / (norm + 1e-9)

    cand_cbf_known = [c for c in candidates if c in cbf_id_to_idx]
    cand_cbf_idx = np.array([cbf_id_to_idx[c] for c in cand_cbf_known])
    cand_cbf_feats = feats[cand_cbf_idx]

    # вариант A: чистый CBF по всему пулу
    preds_cbf_full = {}
    for u in val_users:
        if u not in user_profiles:
            preds_cbf_full[u] = pop_recent.index.tolist()[:5]
            continue
        scores = cand_cbf_feats @ user_profiles[u]
        score_dict = dict(zip(cand_cbf_known, scores))
        preds_cbf_full[u] = build_top5(score_dict, pop_rank, k=5)

    # вариант B: CBF re-rank топ-50 популярных
    top50_pop = pop_recent.index.tolist()[:50]
    top50_known_mask = [c in cbf_id_to_idx for c in top50_pop]
    top50_known = [c for c, m in zip(top50_pop, top50_known_mask) if m]
    top50_idx = np.array([cbf_id_to_idx[c] for c in top50_known])
    top50_feats = feats[top50_idx] if len(top50_idx) else np.zeros((0, feats.shape[1]))

    preds_cbf_rerank = {}
    for u in val_users:
        if u not in user_profiles or len(top50_known) == 0:
            preds_cbf_rerank[u] = pop_recent.index.tolist()[:5]
            continue
        scores = top50_feats @ user_profiles[u]
        score_dict = dict(zip(top50_known, scores))
        preds_cbf_rerank[u] = build_top5(score_dict, pop_rank, k=5)

    # --- гибрид: rank fusion ALS + Popularity recent, вес подбираем на валидации ---
    als_rank_per_user = {}
    item_factors_full = als_main.item_factors[cand_known_idx]
    for u in val_users:
        if u not in u_map:
            als_rank_per_user[u] = {}
            continue
        uf = als_main.user_factors[u_map[u]]
        scores = item_factors_full @ uf
        order = np.argsort(-scores)
        als_rank_per_user[u] = {cand_known[o]: r for r, o in enumerate(order)}

    best_hybrid = None
    hybrid_results = []
    for w in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
        preds_hybrid = {}
        for u in val_users:
            if u not in u_map:
                preds_hybrid[u] = pop_recent.index.tolist()[:5]
                continue
            ra = als_rank_per_user[u]
            score_dict = rank_fusion(ra, pop_rank, w, candidates)
            preds_hybrid[u] = build_top5(score_dict, pop_rank, k=5)
        m, n = evaluate(preds_hybrid, val)
        hybrid_results.append((w, m, n))
        if best_hybrid is None or m > best_hybrid[1]:
            best_hybrid = (w, m, n, preds_hybrid)

    # --- ALS вариации гиперпараметров (только для сравнения на валидации) ---
    als_variants = []
    for factors, reg, iters in [(32, 0.05, 20), (64, 0.1, 20), (128, 0.05, 15)]:
        t0v = time.perf_counter()
        m_model = fit_als(factors, reg, iters)
        preds_v = als_predict(m_model, cand_known, cand_known_idx, cand_unknown, pop_rank)
        mv, nv = evaluate(preds_v, val)
        als_variants.append((f"ALS f={factors} reg={reg} it={iters}", mv, nv, time.perf_counter() - t0v))

    # --- сбор итоговой таблицы ---
    rows_out = []
    for name, preds in [
        ("Popularity (Global)", preds_pop_global),
        ("Popularity (Recent 4w)", preds_pop_recent),
        ("CBF (full pool)", preds_cbf_full),
        ("CBF (re-rank top-50 pop)", preds_cbf_rerank),
    ]:
        m, n = evaluate(preds, val)
        rows_out.append({"method": name, "MAP@5": m, "users_evaluated": n, "seed": SEED, "time_s": None})

    rows_out.append({"method": "ALS f=64 reg=0.05 it=20", "MAP@5": evaluate(preds_als, val)[0],
                      "users_evaluated": evaluate(preds_als, val)[1], "seed": SEED, "time_s": round(t_als, 2)})

    for name, mv, nv, tv in als_variants:
        rows_out.append({"method": name, "MAP@5": mv, "users_evaluated": nv, "seed": SEED, "time_s": round(tv, 2)})

    for w, m, n in hybrid_results:
        rows_out.append({"method": f"Hybrid ALS+Pop (rank fusion, w={w})", "MAP@5": m,
                          "users_evaluated": n, "seed": SEED, "time_s": None})

    df_out = pd.DataFrame(rows_out)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(RESULTS / "validation.csv", index=False)
    print(df_out.to_string(index=False))
    print(f"\nTotal time: {time.perf_counter() - t_all:.1f}s")

    return {
        "best_hybrid": best_hybrid,
        "als_main": als_main,
        "u_map": u_map, "i_map": i_map,
        "pop_recent": pop_recent,
        "candidates_val": candidates,
    }


if __name__ == "__main__":
    run()
