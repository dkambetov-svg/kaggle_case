#!/usr/bin/env python3
"""§7: два пользователя (тяжёлый/лёгкий), Top-5 каждого метода + что реально смотрели."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
SEED = 42


def build_top5(score_dict, pop_rank, k=5):
    items = sorted(score_dict.items(), key=lambda kv: (-kv[1], pop_rank.get(kv[0], 10**9)))
    return [i for i, _ in items[:k]]


def main():
    inter = pd.read_parquet(DATA / "user_interactions.parquet")
    inter["user_id"] = inter["user_id"].astype(str)
    train = inter[inter["week"] <= 20].copy()
    val = inter[(inter["week"] > 20) & (inter["week"] <= 30)].copy()

    val_sched = pd.read_csv(DATA / "export_arh_21-30-final.csv")
    sched_lookup = val_sched[val_sched["tv_show_id"] != 0].drop_duplicates("tv_show_id").set_index("tv_show_id")
    candidates = sched_lookup.index.unique().tolist()

    # Выбор пользователей: тяжёлый (много просмотров в train) и лёгкий (мало, но >=1 в val)
    user_counts = train.groupby("user_id")["overlap_seconds"].agg(["count", "sum"])
    val_users_set = set(val["user_id"].unique())
    eligible = user_counts[user_counts.index.isin(val_users_set)]
    heavy_user = eligible.sort_values("sum", ascending=False).index[0]
    light_user = eligible[eligible["count"] >= 1].sort_values("count").index[0]

    print(f"Heavy user: {heavy_user} (sum_sec={eligible.loc[heavy_user,'sum']:.0f}, n={eligible.loc[heavy_user,'count']})")
    print(f"Light user: {light_user} (sum_sec={eligible.loc[light_user,'sum']:.0f}, n={eligible.loc[light_user,'count']})")

    # --- популярность ---
    pop_recent = train[train["week"] >= 17].groupby("tv_show_id")["user_id"].nunique().sort_values(ascending=False)
    pop_rank = {it: r for r, it in enumerate(pop_recent.index)}
    top5_pop = pop_recent.index.tolist()[:5]

    # --- ALS (f=64, как основной заявленный вариант) ---
    train_items = train["tv_show_id"].unique()
    train_users = train["user_id"].unique()
    u_map = {u: i for i, u in enumerate(train_users)}
    i_map = {it: i for i, it in enumerate(train_items)}
    rows = train["user_id"].map(u_map).values
    cols = train["tv_show_id"].map(i_map).values
    data = np.log1p(train["overlap_seconds"].values)
    mat = csr_matrix((data, (rows, cols)), shape=(len(u_map), len(i_map)))
    als = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=20, random_state=SEED)
    als.fit(mat)

    cand_known = [c for c in candidates if c in i_map]
    cand_unknown = [c for c in candidates if c not in i_map]
    cand_known_idx = np.array([i_map[c] for c in cand_known])
    item_factors = als.item_factors[cand_known_idx]

    def als_top5(user):
        if user not in u_map:
            return top5_pop
        uf = als.user_factors[u_map[user]]
        scores = item_factors @ uf
        min_score = scores.min() - 1.0
        score_dict = dict(zip(cand_known, scores))
        for c in cand_unknown:
            score_dict[c] = min_score
        return build_top5(score_dict, pop_rank)

    # --- CBF (full pool) ---
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

    def cbf_profile(user):
        g = train_w[train_w["user_id"] == user]
        idxs = [cbf_id_to_idx[sid] for sid in g["tv_show_id"] if sid in cbf_id_to_idx]
        if not idxs:
            return None
        mask = g["tv_show_id"].isin(cbf_id_to_idx)
        weights = g.loc[mask, "weight"].values
        profile = (feats[idxs].T @ weights) / weights.sum()
        return profile / (np.linalg.norm(profile) + 1e-9)

    cand_cbf_known = [c for c in candidates if c in cbf_id_to_idx]
    cand_cbf_idx = np.array([cbf_id_to_idx[c] for c in cand_cbf_known])
    cand_cbf_feats = feats[cand_cbf_idx]

    def cbf_top5(user):
        profile = cbf_profile(user)
        if profile is None:
            return top5_pop
        scores = cand_cbf_feats @ profile
        score_dict = dict(zip(cand_cbf_known, scores))
        return build_top5(score_dict, pop_rank)

    def lookup(show_id):
        if show_id in sched_lookup.index:
            row = sched_lookup.loc[show_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row["tv_show_title"], row["tv_show_category"]
        return "(неизвестно, вне пула 21-30)", "-"

    lines = ["# §7 Примеры рекомендаций: два пользователя\n"]
    lines.append(
        "Выбор: **тяжёлый** пользователь — максимум секунд просмотра в train (11-20); "
        "**лёгкий** — минимум просмотров (≥1) из тех, у кого есть хотя бы одно взаимодействие в проверке (21-30). "
        "Одинаковый пул кандидатов и fallback (популярность) для всех методов."
    )

    for label, user in [("Тяжёлый", heavy_user), ("Лёгкий", light_user)]:
        lines.append(f"\n## {label} пользователь `{user}`")
        lines.append(f"- train (11-20): {eligible.loc[user,'count']:.0f} просмотров, {eligible.loc[user,'sum']:.0f} сек суммарно")

        watched = val[val["user_id"] == user]["tv_show_id"].unique().tolist()
        lines.append(f"- реально смотрел в проверке (21-30): {len(watched)} программ")
        lines.append("\n**Факт: что реально смотрел (21-30)**\n")
        lines.append("| id | title | category |")
        lines.append("|---|---|---|")
        for sid in watched[:10]:
            t, c = lookup(sid)
            lines.append(f"| {sid} | {t} | {c} |")
        if len(watched) > 10:
            lines.append(f"| … | ещё {len(watched)-10} программ | |")

        for method_name, top5 in [
            ("Popularity (Recent 4w)", top5_pop),
            ("CBF (full pool)", cbf_top5(user)),
            ("ALS f=64 reg=0.05 it=20", als_top5(user)),
        ]:
            lines.append(f"\n**{method_name} — Top-5**\n")
            lines.append("| rank | id | title | category | в фактическом списке? |")
            lines.append("|---|---|---|---|---|")
            for rank, sid in enumerate(top5, 1):
                t, c = lookup(sid)
                hit = "да" if sid in watched else "нет"
                lines.append(f"| {rank} | {sid} | {t} | {c} | {hit} |")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[saved] {REPORTS/'examples.md'}")


if __name__ == "__main__":
    main()
