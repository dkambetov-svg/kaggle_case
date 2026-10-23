#!/usr/bin/env python3
"""Диагностика расхождения valid (0.62) vs public (0.28)."""
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


def build_top5(score_dict, pop_rank, k=5):
    items = sorted(score_dict.items(), key=lambda kv: (-kv[1], pop_rank.get(kv[0], 10**9)))
    return [i for i, _ in items[:k]]


def main():
    rows_out = []

    inter = pd.read_parquet(DATA / "user_interactions.parquet")
    inter["user_id"] = inter["user_id"].astype(str)
    sub = pd.read_csv(DATA / "submission.csv", dtype={"user_id": str})
    sub_users = set(sub["user_id"])

    train = inter[inter["week"] <= 20].copy()
    val = inter[(inter["week"] > 20) & (inter["week"] <= 30)].copy()
    val_users_all = val["user_id"].unique()
    val_users_sub = [u for u in val_users_all if u in sub_users]

    val_sched = pd.read_csv(DATA / "export_arh_21-30-final.csv")
    candidates = val_sched[val_sched["tv_show_id"] != 0]["tv_show_id"].unique()

    # --- п.2: релевантность по двум срезам пользователей ---
    gt_all = val.groupby("user_id")["tv_show_id"].apply(set)
    gt_sub = gt_all[gt_all.index.isin(sub_users)]
    n_rel_all = [len(s) for s in gt_all]
    n_rel_sub = [len(s) for s in gt_sub]
    print(f"[п.2] Релевантные (4799 всех): mean={np.mean(n_rel_all):.2f} median={np.median(n_rel_all):.1f}")
    print(f"[п.2] Релевантные (1260 submission-пересечение, n={len(gt_sub)}): mean={np.mean(n_rel_sub):.2f} median={np.median(n_rel_sub):.1f}")
    rows_out.append({"check": "n_relevant_mean_all4799", "value": np.mean(n_rel_all)})
    rows_out.append({"check": "n_relevant_median_all4799", "value": np.median(n_rel_all)})
    rows_out.append({"check": "n_relevant_mean_sub1260", "value": np.mean(n_rel_sub)})
    rows_out.append({"check": "n_relevant_median_sub1260", "value": np.median(n_rel_sub)})
    rows_out.append({"check": "n_users_val_all", "value": len(gt_all)})
    rows_out.append({"check": "n_users_val_sub_intersection", "value": len(gt_sub)})

    # Сколько submission-пользователей вообще не попали в val (нет релевантных программ в 21-30)
    n_sub_no_val = len(sub_users - set(gt_all.index))
    print(f"[п.2] Пользователей submission БЕЗ релевантных программ в 21-30 вообще: {n_sub_no_val}/{len(sub_users)}")
    rows_out.append({"check": "sub_users_without_val_relevant", "value": n_sub_no_val})

    pop_recent = train[train["week"] >= 17].groupby("tv_show_id")["user_id"].nunique().sort_values(ascending=False)
    pop_rank = {it: r for r, it in enumerate(pop_recent.index)}
    top5_pop = pop_recent.index.tolist()[:5]

    preds_pop_all = {u: top5_pop for u in val_users_all}
    preds_pop_sub = {u: top5_pop for u in val_users_sub}
    m_all, n_all = evaluate(preds_pop_all, val)
    m_sub, n_sub = evaluate(preds_pop_sub, val)
    print(f"[п.2] Popularity(Recent4w) MAP@5: all4799={m_all:.4f} (n={n_all})  sub1260={m_sub:.4f} (n={n_sub})")
    rows_out.append({"check": "MAP5_pop_recent4w_all4799", "value": m_all})
    rows_out.append({"check": "MAP5_pop_recent4w_sub_intersection", "value": m_sub})
    rows_out.append({"check": "n_eval_pop_all4799", "value": n_all})
    rows_out.append({"check": "n_eval_pop_sub_intersection", "value": n_sub})

    # --- п.3: ALS матрица и топ-5 по пользователям ---
    print(f"\n[п.3] implicit version: {implicit.__version__}")
    rows_out.append({"check": "implicit_version", "value": implicit.__version__})

    train_items = train["tv_show_id"].unique()
    train_users = train["user_id"].unique()
    u_map = {u: i for i, u in enumerate(train_users)}
    i_map = {it: i for i, it in enumerate(train_items)}
    rows_u = train["user_id"].map(u_map).values
    cols_i = train["tv_show_id"].map(i_map).values
    data = np.log1p(train["overlap_seconds"].values)
    mat = csr_matrix((data, (rows_u, cols_i)), shape=(len(u_map), len(i_map)))
    print(f"[п.3] Матрица в fit(): shape={mat.shape} (users={len(u_map)}, items={len(i_map)}) nnz={mat.nnz}")
    rows_out.append({"check": "als_matrix_shape_users", "value": mat.shape[0]})
    rows_out.append({"check": "als_matrix_shape_items", "value": mat.shape[1]})
    rows_out.append({"check": "als_matrix_nnz", "value": mat.nnz})

    als = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=20, random_state=SEED)
    als.fit(mat)

    cand_known = [c for c in candidates if c in i_map]
    cand_unknown = [c for c in candidates if c not in i_map]
    cand_known_idx = np.array([i_map[c] for c in cand_known])
    item_factors = als.item_factors[cand_known_idx]

    sched_tr = pd.read_csv(DATA / "export_arh_11-20-final.csv")
    title_lookup = sched_tr[sched_tr["tv_show_id"] != 0].drop_duplicates("tv_show_id").set_index("tv_show_id")["tv_show_title"]

    sample_users = list(train["user_id"].value_counts().index[:5])
    print("\n[п.3] Для 5 пользователей: top-5 собственных программ (по секундам, train) vs top-5 ALS:")
    for u in sample_users:
        own = train[train["user_id"] == u].groupby("tv_show_id")["overlap_seconds"].sum().sort_values(ascending=False)
        own_top5 = own.index[:5].tolist()
        own_titles = [title_lookup.get(i, "?") for i in own_top5]

        uf = als.user_factors[u_map[u]]
        scores = item_factors @ uf
        als_order = np.argsort(-scores)[:5]
        als_top5 = [cand_known[i] for i in als_order]
        als_titles = [title_lookup.get(i, "?") for i in als_top5]

        print(f"  user={u}")
        print(f"    own_top5 (sec): {list(zip(own_top5, own_titles))}")
        print(f"    als_top5       : {list(zip(als_top5, als_titles))}")

    def als_predict_full():
        preds = {}
        for u in val_users_all:
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

    preds_als = als_predict_full()
    preds_als_all = {u: preds_als[u] for u in val_users_all}
    preds_als_sub = {u: preds_als[u] for u in val_users_sub}
    m_als_all, _ = evaluate(preds_als_all, val)
    m_als_sub, _ = evaluate(preds_als_sub, val)
    print(f"[п.3] ALS MAP@5: all4799={m_als_all:.4f}  sub1260={m_als_sub:.4f}")
    rows_out.append({"check": "MAP5_als_f64_all4799", "value": m_als_all})
    rows_out.append({"check": "MAP5_als_f64_sub_intersection", "value": m_als_sub})

    # --- п.4: "repeat" сигнал: ранг по собственным секундам + добивка популярностью ---
    def repeat_model(train_df, label):
        own_sec = train_df.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum()
        preds = {}
        for u in val_users_all:
            if u in own_sec.index.get_level_values(0):
                own_u = own_sec.loc[u]
                own_u = own_u[own_u.index.isin(candidates)].sort_values(ascending=False)
                top = own_u.index.tolist()
                for p in top5_pop:
                    if p not in top:
                        top.append(p)
                preds[u] = top[:5]
            else:
                preds[u] = top5_pop
        preds_all = {u: preds[u] for u in val_users_all}
        preds_sub = {u: preds[u] for u in val_users_sub}
        m_a, _ = evaluate(preds_all, val)
        m_s, _ = evaluate(preds_sub, val)
        print(f"[п.4] Repeat({label}) MAP@5: all4799={m_a:.4f}  sub1260={m_s:.4f}")
        rows_out.append({"check": f"MAP5_repeat_{label}_all4799", "value": m_a})
        rows_out.append({"check": f"MAP5_repeat_{label}_sub_intersection", "value": m_s})
        return m_a, m_s

    m_full_a, m_full_s = repeat_model(train, "train_full_11_20")
    train_last4 = train[train["week"] >= 17]
    m_l4_a, m_l4_s = repeat_model(train_last4, "train_last4w_17_20")

    best_candidates = {
        "Popularity (Recent 4w)": m_all,
        "Repeat (train full 11-20)": m_full_a,
        "Repeat (train last4w 17-20)": m_l4_a,
        "ALS f=64": m_als_all,
    }
    best_name = max(best_candidates, key=best_candidates.get)
    print(f"\n[итог] Лучший метод на all4799 holdout: {best_name} = {best_candidates[best_name]:.4f} (бейзлайн Popularity={m_all:.4f})")
    rows_out.append({"check": "best_method_name", "value": best_name})
    rows_out.append({"check": "best_method_map5", "value": best_candidates[best_name]})
    rows_out.append({"check": "baseline_pop_map5", "value": m_all})

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(RESULTS / "diagnostics.csv", index=False)
    print(f"\n[saved] {RESULTS/'diagnostics.csv'}")

    return best_name, best_candidates[best_name], m_all


if __name__ == "__main__":
    main()
