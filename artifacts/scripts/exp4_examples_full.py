#!/usr/bin/env python3
"""§7 (exp4): полные примеры рекомендаций — все методы для тех же двух пользователей.

Расширяет reports/examples.md (Popularity/CBF/ALS): добавляет Association Rules,
PPR (бипартитный aw=0 и трёхдольный aw=0.5) и гибрид exp3_e3b
(RRF k=20 w=(1,2,1) поверх pop×airings + PPR + ALS 64/0.05/20).

Пули различаются по дизайну методов и это указано в шапке файла:
- Popularity/CBF/ALS/AR — полный пул кандидатов (программы расписания 21–30 с id≠0);
- PPR/гибрид — пул топ-500 по pop×log(1+эфиров) (пространство их валидации).
Fallback одинаковый: популярность Recent-4w.

Выбор пользователей идентичен examples.py (детерминированно) — те же два id.
Выход: reports/examples_full.md (прежний examples.md не перезаписывается).
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

from exp_rrf_hybrid import DATA, SEED
from exp_assoc_rules import (
    MIN_SUPPORT, build_transactions, count_itemsets, frequent_pairs,
    generate_rules, rank_from_scores,
)
from exp3_e1_attr_ppr import load_item_attrs, build_graph_base
from exp3_e3_rrf_weights import tripartite_rankings, rank_matrix, fuse_weighted_top5
from exp4_common import SPLITS, prepare_split, als_rankings_p

REPORTS = Path(__file__).resolve().parents[2] / "reports"

# конфигурация гибрида (exp3_e3b — Kaggle 0.34487/0.32361)
HYBRID = dict(k_rrf=20, w=(1.0, 2.0, 1.0), ppr_aw=0.5, ppr_alpha=0.5,
              als=dict(factors=64, reg=0.05, iterations=20))
POOL_SIZE = 500


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

    # выбор пользователей — идентичен examples.py (тяжёлый/лёгкий)
    user_counts = train.groupby("user_id")["overlap_seconds"].agg(["count", "sum"])
    val_users_set = set(val["user_id"].unique())
    eligible = user_counts[user_counts.index.isin(val_users_set)]
    heavy_user = eligible.sort_values("sum", ascending=False).index[0]
    light_user = eligible[eligible["count"] >= 1].sort_values("count").index[0]
    two_users = [heavy_user, light_user]
    print(f"users: heavy={heavy_user}, light={light_user}", flush=True)

    # --- популярность Recent-4w (как examples.py) ---
    pop_recent = train[train["week"] >= 17].groupby("tv_show_id")["user_id"].nunique().sort_values(ascending=False)
    pop_rank_train = {it: r for r, it in enumerate(pop_recent.index)}
    top5_pop = pop_recent.index.tolist()[:5]

    # ранги популярности по кандидатам (для ничьих AR/гибрида, как в exp_assoc/exp_rrf)
    pop_cand = train[train["week"] >= 17].groupby("tv_show_id")["user_id"].nunique()
    pop_cand = pop_cand.reindex(candidates).fillna(0.0).sort_values(ascending=False)
    pop_rank_cand = {it: r for r, it in enumerate(pop_cand.index)}

    # --- ALS (как examples.py: полный пул кандидатов) ---
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
    item_factors = als.item_factors[np.array([i_map[c] for c in cand_known])]

    def als_top5(user):
        if user not in u_map:
            return top5_pop
        uf = als.user_factors[u_map[user]]
        scores = item_factors @ uf
        min_score = scores.min() - 1.0
        score_dict = dict(zip(cand_known, scores))
        for c in cand_unknown:
            score_dict[c] = min_score
        return build_top5(score_dict, pop_rank_train)

    # --- CBF (как examples.py) ---
    sched_tr = pd.read_csv(DATA / "export_arh_11-20-final.csv")
    feat = sched_tr[sched_tr["tv_show_id"] != 0].drop_duplicates("tv_show_id").copy()
    genre_cols = []
    for col in ["tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"]:
        genre_cols.append(pd.get_dummies(feat[col].fillna(""), prefix=col))
    feat["year"] = pd.to_numeric(feat["year_of_production"], errors="coerce").fillna(2000)
    feat["decade"] = (feat["year"] // 10 * 10).astype(int).astype(str)
    feats = pd.concat(genre_cols + [pd.get_dummies(feat["decade"], prefix="decade")], axis=1).values.astype(float)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    cbf_id_to_idx = {it: i for i, it in enumerate(feat["tv_show_id"].values)}
    train_w = train.copy()
    train_w["weight"] = np.log1p(train_w["overlap_seconds"])

    def cbf_top5(user):
        g = train_w[train_w["user_id"] == user]
        mask = g["tv_show_id"].isin(cbf_id_to_idx)
        if not mask.any():
            return top5_pop
        idxs = [cbf_id_to_idx[sid] for sid in g.loc[mask, "tv_show_id"]]
        weights = g.loc[mask, "weight"].values
        profile = (feats[idxs].T @ weights) / weights.sum()
        profile = profile / (np.linalg.norm(profile) + 1e-9)
        known = [c for c in candidates if c in cbf_id_to_idx]
        scores = feats[np.array([cbf_id_to_idx[c] for c in known])] @ profile
        return build_top5(dict(zip(known, scores)), pop_rank_train)

    # --- Association Rules (как exp_assoc_rules: транзакции user×week, support>=0.2%, lift>1) ---
    txns = build_transactions(train)
    n_txn = len(txns)
    item_count, pair_count = count_itemsets(txns)
    freq_pairs = frequent_pairs(pair_count, n_txn, MIN_SUPPORT)
    rules = generate_rules(freq_pairs, item_count, n_txn)
    rules_by_ant = defaultdict(list)
    for r in rules:
        rules_by_ant[r["antecedent"]].append((r["consequent"], r["confidence"]))
    print(f"AR: транзакций={n_txn}, частых пар={len(freq_pairs)}, правил lift>1={len(rules)}", flush=True)

    recent4w = train[(train["week"] >= 17) & (train["overlap_seconds"] > 0)]
    user_recent_items = recent4w.groupby("user_id")["tv_show_id"].apply(lambda s: set(s.unique()))

    def ar_top5(user):
        A_set = user_recent_items.get(user, set())
        if not A_set:
            return top5_pop
        score = defaultdict(float)
        for a in A_set:
            for c, conf in rules_by_ant.get(a, []):
                if c in candidates:
                    score[c] += conf
        if not score:
            return top5_pop
        ids = list(score.keys())
        scores = np.array([score[i] for i in ids])
        top = rank_from_scores(ids, scores, pop_rank_cand)
        for p in top5_pop:
            if len(top) >= 5:
                break
            if p not in top:
                top.append(p)
        return top[:5]

    # --- PPR и гибрид (пространство: пул топ-500, как в их валидации exp3) ---
    cfg = SPLITS["main"]
    train_s, _, _, _, _, pool_full, rank_pop = \
        prepare_split(inter, list(eligible.index), cfg, pool_size=POOL_SIZE)
    item_attrs = load_item_attrs(cfg["attrs_files"])
    base = build_graph_base(train_s, pool_full, item_attrs)

    def ppr_pool_rankings(aw, alpha):
        r, _ = tripartite_rankings(base, base["per_user"], pool_full, rank_pop,
                                   two_users, aw, alpha=alpha, k_pool=POOL_SIZE)
        return r

    ppr_bip = ppr_pool_rankings(aw=0.0, alpha=HYBRID["ppr_alpha"])   # бипартитный (aw=0 ≡ контроль)
    ppr_tri = ppr_pool_rankings(aw=HYBRID["ppr_aw"], alpha=HYBRID["ppr_alpha"])

    als_pool = als_rankings_p(train_s, two_users, pool_full, rank_pop, **HYBRID["als"])

    def rrf_top5():
        M_pop = rank_matrix({u: list(pool_full) for u in two_users}, two_users, pool_full)
        M_ppr = rank_matrix(ppr_tri, two_users, pool_full)
        M_als = rank_matrix(als_pool, two_users, pool_full)
        order = fuse_weighted_top5([M_pop, M_ppr, M_als], HYBRID["w"], HYBRID["k_rrf"])
        return {u: [pool_full[j] for j in order[i]] for i, u in enumerate(two_users)}

    hybrid_preds = rrf_top5()

    def lookup(show_id):
        if show_id in sched_lookup.index:
            row = sched_lookup.loc[show_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row["tv_show_title"], row["tv_show_category"]
        return "(неизвестно, вне пула 21-30)", "-"

    lines = ["# §7 Примеры рекомендаций (полные): все методы, те же два пользователя\n"]
    lines.append(
        "Выбор пользователей идентичен `reports/examples.md`: **тяжёлый** — максимум секунд "
        "в train (11–20); **лёгкий** — минимум просмотров (≥1) при наличии просмотров в проверке (21–30). "
        "Fallback одинаковый для всех методов — популярность Recent-4w.\n\n"
        "**Пул кандидатов различается по дизайну методов:** Popularity/CBF/ALS/AR ранжируют полный пул "
        "(программы расписания 21–30 с `tv_show_id != 0`); PPR и гибрид работают в пуле топ-500 по "
        "pop×log(1+эфиров) — пространстве, на котором они валидировались (exp3). "
        "Фолбэк-заполнение до 5 слотов — популярность."
    )
    lines.append(
        f"\nКонфигурация гибрида (exp3_e3b): RRF k={HYBRID['k_rrf']}, w={HYBRID['w']}, "
        f"PPR трёхдольный attr_weight={HYBRID['ppr_aw']} (restart=0,5, alpha={HYBRID['ppr_alpha']}), "
        f"ALS {HYBRID['als']}."
    )

    for label, user in [("Тяжёлый", heavy_user), ("Лёгкий", light_user)]:
        lines.append(f"\n## {label} пользователь `{user}`")
        lines.append(f"- train (11–20): {eligible.loc[user,'count']:.0f} просмотров, "
                     f"{eligible.loc[user,'sum']:.0f} сек суммарно")
        watched = val[val["user_id"] == user]["tv_show_id"].unique().tolist()
        lines.append(f"- реально смотрел в проверке (21–30): {len(watched)} программ")
        lines.append("\n**Факт: что реально смотрел (21–30)**\n")
        lines.append("| id | title | category |")
        lines.append("|---|---|---|")
        for sid in watched[:10]:
            t, c = lookup(sid)
            lines.append(f"| {sid} | {t} | {c} |")
        if len(watched) > 10:
            lines.append(f"| … | ещё {len(watched)-10} программ | |")

        blocks = [
            ("Popularity (Recent 4w)", top5_pop),
            ("CBF (full pool)", cbf_top5(user)),
            ("Association Rules (Σ confidence, last-4w)", ar_top5(user)),
            ("ALS f=64 reg=0.05 it=20 (full pool)", als_top5(user)),
            ("PPR бипартитный user–item (restart=0,5; пул-500)", ppr_bip[user][:5]),
            ("PPR трёхдольный user–item–attr aw=0,5 (restart=0,5; пул-500)", ppr_tri[user][:5]),
            ("Гибрид exp3_e3b: RRF k=20 w=(1,2,1) (пул-500)", hybrid_preds[user]),
        ]
        for method_name, top5 in blocks:
            lines.append(f"\n**{method_name} — Top-5**\n")
            lines.append("| rank | id | title | category | в фактическом списке? |")
            lines.append("|---|---|---|---|---|")
            for rank, sid in enumerate(top5, 1):
                t, c = lookup(sid)
                hit = "да" if sid in watched else "нет"
                lines.append(f"| {rank} | {sid} | {t} | {c} | {hit} |")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "examples_full.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
