#!/usr/bin/env python3
"""Строгая валидация (только 1260 submission-пользователей):
- Определение A (strict): релевантна программа, если сумма overlap_seconds пользователя
  по этой программе в окне >= 50% длительности (берём MAX duration среди записей
  расписания для этого tv_show_id в целевом окне — явное допущение, т.к. duration
  одного конкретного airing-события в parquet не хранится).
- Определение B (top20): релевантны top-20 программ пользователя по сумме секунд
  в целевом окне (без порога).
Пересчёт методов: Popularity(Recent4w), ALS, Repeat, CBF(full pool) + новые (a)-(d).
"""
import time
import numpy as np
import pandas as pd
import implicit
from pathlib import Path
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "tables"
SEED = 42


def evaluate(predictions, relevant_sets, k=5):
    ap_sum = 0.0
    users = set(predictions) & set(relevant_sets)
    if not users:
        return 0.0, 0
    for u in users:
        preds = predictions[u][:k]
        actuals = relevant_sets[u]
        if not actuals:
            continue
        hits = ap = 0
        for i, p in enumerate(preds):
            if p in actuals:
                hits += 1
                ap += hits / (i + 1)
        ap_sum += ap / min(len(actuals), k)
    n_eff = len([u for u in users if relevant_sets[u]])
    return (ap_sum / n_eff if n_eff else 0.0), n_eff


def build_top5(score_dict, pop_rank, k=5):
    items = sorted(score_dict.items(), key=lambda kv: (-kv[1], pop_rank.get(kv[0], 10**9)))
    return [i for i, _ in items[:k]]


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

    # --- релевантность: строгая (A) и top-20 (B), только для submission-пользователей ---
    val_sub = val_raw[val_raw["user_id"].isin(sub_users)]
    user_show_sum = val_sub.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum()

    dur_lookup = val_sched[val_sched["tv_show_id"] != 0].groupby("tv_show_id")["duration"].max()

    relevant_strict = {}
    relevant_top20 = {}
    for u, s in user_show_sum.groupby(level=0):
        s = s.droplevel(0).sort_values(ascending=False)
        strict_ids = set()
        for show_id, sec in s.items():
            dur = dur_lookup.get(show_id, np.nan)
            if pd.notna(dur) and dur > 0 and sec >= 0.5 * dur:
                strict_ids.add(show_id)
        relevant_strict[u] = strict_ids
        relevant_top20[u] = set(s.index[:20].tolist())

    n_rel_strict = [len(v) for v in relevant_strict.values()]
    n_rel_top20 = [len(v) for v in relevant_top20.values()]
    print(f"[strict] пользователей с >=1 релевантной: {sum(1 for v in n_rel_strict if v>0)}/{len(relevant_strict)}; "
          f"mean={np.mean(n_rel_strict):.2f} median={np.median(n_rel_strict):.1f}")
    print(f"[top20] пользователей с >=1 релевантной: {sum(1 for v in n_rel_top20 if v>0)}/{len(relevant_top20)}; "
          f"mean={np.mean(n_rel_top20):.2f} median={np.median(n_rel_top20):.1f}")
    rows_out.append({"method": "_meta", "definition": "strict", "MAP@5": None,
                      "n_users_evaluated": sum(1 for v in n_rel_strict if v > 0),
                      "n_rel_mean": np.mean(n_rel_strict), "n_rel_median": np.median(n_rel_strict)})
    rows_out.append({"method": "_meta", "definition": "top20", "MAP@5": None,
                      "n_users_evaluated": sum(1 for v in n_rel_top20 if v > 0),
                      "n_rel_mean": np.mean(n_rel_top20), "n_rel_median": np.median(n_rel_top20)})

    val_users_sub = [u for u in sub_users if u in relevant_strict]

    # --- базовые ресурсы ---
    pop_recent = train[train["week"] >= 17].groupby("tv_show_id")["user_id"].nunique().sort_values(ascending=False)
    pop_rank = {it: r for r, it in enumerate(pop_recent.index)}
    top5_pop = pop_recent.index.tolist()[:5]

    # --- ALS (как раньше, f=64) ---
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
    cand_known = [c for c in candidates if c in i_map]
    cand_unknown = [c for c in candidates if c not in i_map]
    cand_known_idx = np.array([i_map[c] for c in cand_known])
    item_factors = als.item_factors[cand_known_idx]

    def als_predict():
        preds = {}
        for u in val_users_sub:
            if u not in u_map:
                preds[u] = top5_pop
                continue
            uf = als.user_factors[u_map[u]]
            scores = item_factors @ uf
            min_score = scores.min() - 1.0
            score_dict = dict(zip(cand_known, scores))
            for c in cand_unknown:
                score_dict[c] = min_score
            preds[u] = build_top5(score_dict, pop_rank)
        return preds

    # --- Repeat (train full 11-20) ---
    own_sec_full = train.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum()

    def repeat_predict(own_sec):
        preds = {}
        own_idx0 = set(own_sec.index.get_level_values(0))
        for u in val_users_sub:
            if u in own_idx0:
                own_u = own_sec.loc[u]
                own_u = own_u[own_u.index.isin(candidates)].sort_values(ascending=False)
                top = own_u.index.tolist()
                for p in top5_pop:
                    if p not in top:
                        top.append(p)
                preds[u] = top[:5]
            else:
                preds[u] = top5_pop
        return preds

    # --- CBF full pool (категории+жанры+десятилетие) ---
    sched_tr = pd.read_csv(DATA / "export_arh_11-20-final.csv")
    feat = sched_tr[sched_tr["tv_show_id"] != 0].drop_duplicates("tv_show_id").copy()
    genre_cols = []
    for col in ["tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"]:
        genre_cols.append(pd.get_dummies(feat[col].fillna(""), prefix=col))
    feat["year"] = pd.to_numeric(feat["year_of_production"], errors="coerce").fillna(2000)
    feat["decade"] = (feat["year"] // 10 * 10).astype(int).astype(str)
    decade_dummies = pd.get_dummies(feat["decade"], prefix="decade")
    feats = pd.concat(genre_cols + [decade_dummies], axis=1).values.astype(float)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    cbf_id_to_idx = {it: i for i, it in enumerate(feat["tv_show_id"].values)}

    train_w = train.copy()
    train_w["weight"] = np.log1p(train_w["overlap_seconds"])
    cand_cbf_known = [c for c in candidates if c in cbf_id_to_idx]
    cand_cbf_idx = np.array([cbf_id_to_idx[c] for c in cand_cbf_known])
    cand_cbf_feats = feats[cand_cbf_idx]

    def cbf_predict():
        preds = {}
        for u in val_users_sub:
            g = train_w[train_w["user_id"] == u]
            idxs = [cbf_id_to_idx[sid] for sid in g["tv_show_id"] if sid in cbf_id_to_idx]
            if not idxs:
                preds[u] = top5_pop
                continue
            mask = g["tv_show_id"].isin(cbf_id_to_idx)
            weights = g.loc[mask, "weight"].values
            profile = (feats[idxs].T @ weights) / weights.sum()
            profile = profile / (np.linalg.norm(profile) + 1e-9)
            scores = cand_cbf_feats @ profile
            score_dict = dict(zip(cand_cbf_known, scores))
            preds[u] = build_top5(score_dict, pop_rank)
        return preds

    # --- (a) Популярность по сумме секунд ---
    pop_sum = train.groupby("tv_show_id")["overlap_seconds"].sum().sort_values(ascending=False)
    top5_pop_sum = pop_sum.index.tolist()[:5]
    preds_pop_sum = {u: top5_pop_sum for u in val_users_sub}

    # --- (b) Популярность x log(1+airings в целевом окне) ---
    airing_count_val = val_sched[val_sched["tv_show_id"] != 0]["tv_show_id"].value_counts()
    score_b = {}
    for show in set(pop_recent.index) | set(airing_count_val.index):
        pr = pop_recent.get(show, 0)
        ac = airing_count_val.get(show, 0)
        score_b[show] = pr * np.log1p(ac)
    pop_rank_b = {it: r for r, (it, _) in enumerate(sorted(score_b.items(), key=lambda kv: -kv[1]))}
    top5_b = build_top5(score_b, pop_rank)
    preds_b = {u: top5_b for u in val_users_sub}

    # --- (c) CBF по каналам: профиль = доля секунд по каналам, скор = pop_recent4w(show) * channel_fraction(канал показа) ---
    # Приближение: канал программы = мода channel_id среди эфиров tv_show_id в train-расписании (per-view channel не хранится в parquet).
    sched_tr_full = pd.read_csv(DATA / "export_arh_11-20-final.csv")
    show_to_channel = (
        sched_tr_full[sched_tr_full["tv_show_id"] != 0]
        .groupby("tv_show_id")["channel_id"]
        .agg(lambda x: x.value_counts().idxmax())
    )
    train_ch = train.copy()
    train_ch["channel_id"] = train_ch["tv_show_id"].map(show_to_channel)
    user_channel_sec = train_ch.dropna(subset=["channel_id"]).groupby(["user_id", "channel_id"])["overlap_seconds"].sum()

    cand_channel = pd.Series(candidates).map(show_to_channel)
    cand_channel_map = dict(zip(candidates, cand_channel))

    def cbf_channel_predict():
        preds = {}
        user_totals = train_ch.dropna(subset=["channel_id"]).groupby("user_id")["overlap_seconds"].sum()
        for u in val_users_sub:
            if u not in user_totals.index or user_totals[u] == 0:
                preds[u] = top5_pop
                continue
            u_ch = user_channel_sec.loc[u] if u in user_channel_sec.index.get_level_values(0) else pd.Series(dtype=float)
            total = user_totals[u]
            score_dict = {}
            for show in candidates:
                ch = cand_channel_map.get(show)
                frac = (u_ch.get(ch, 0) / total) if (ch is not None and len(u_ch)) else 0.0
                score_dict[show] = pop_recent.get(show, 0) * frac
            preds[u] = build_top5(score_dict, pop_rank)
        return preds

    # --- (d) Repeat + (b) как добивка ---
    def repeat_plus_b_predict(own_sec):
        preds = {}
        own_idx0 = set(own_sec.index.get_level_values(0))
        for u in val_users_sub:
            if u in own_idx0:
                own_u = own_sec.loc[u]
                own_u = own_u[own_u.index.isin(candidates)].sort_values(ascending=False)
                top = own_u.index.tolist()
                for p in top5_b:
                    if p not in top:
                        top.append(p)
                preds[u] = top[:5]
            else:
                preds[u] = top5_b
        return preds

    methods = {
        "Popularity (Recent 4w)": {u: top5_pop for u in val_users_sub},
        "ALS f=64": als_predict(),
        "Repeat (train full 11-20)": repeat_predict(own_sec_full),
        "CBF (full pool)": cbf_predict(),
        "(a) Popularity by sum seconds": preds_pop_sum,
        "(b) Popularity x log(1+airings target)": preds_b,
        "(c) CBF by channels": cbf_channel_predict(),
        "(d) Repeat + (b) fill": repeat_plus_b_predict(own_sec_full),
    }

    for name, preds in methods.items():
        m_strict, n_strict = evaluate(preds, relevant_strict)
        m_top20, n_top20 = evaluate(preds, relevant_top20)
        print(f"{name:40s} strict MAP@5={m_strict:.4f} (n={n_strict})  top20 MAP@5={m_top20:.4f} (n={n_top20})")
        rows_out.append({"method": name, "definition": "strict", "MAP@5": m_strict,
                          "n_users_evaluated": n_strict, "n_rel_mean": None, "n_rel_median": None})
        rows_out.append({"method": name, "definition": "top20", "MAP@5": m_top20,
                          "n_users_evaluated": n_top20, "n_rel_mean": None, "n_rel_median": None})

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(RESULTS / "diagnostics_strict.csv", index=False)
    print(f"\n[saved] {RESULTS/'diagnostics_strict.csv'}")


if __name__ == "__main__":
    main()
