#!/usr/bin/env python3
"""RRF-гибрид (k=60) из пяти сигналов: pop×log(airings), PPR, AR, ALS, CBF-new.
Веса подбираются на строгой валидации (greedy forward selection): сначала по одному
добавляем компонент к базе pop×airings и оставляем только те, что реально помогают.

Не перезаписывает текущий финал и прежние exp_* таблицы: выход results/tables/exp_rrf_hybrid*.csv.
Максимум 3 сабмита: лучший гибрид, PPR-гибрид, AR-гибрид (см. exp_rrf_submit.py).
"""
import time
from itertools import combinations
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "tables"
SEED = 42
RRF_K = 60
MIN_SUPPORT = 0.002
POOL = 500  # общий пул кандидатов для всех пяти сигналов (rank fusion работает по рангам в общем пуле)


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


def rank_list_from_scores(cand_scores, tie_rank, k=POOL):
    """cand_scores: dict item->score. Возвращает список item (длина <=k) -- ранжирование (0=best)."""
    items = list(cand_scores.keys())
    scores = np.array([cand_scores[i] for i in items])
    order = sorted(range(len(items)), key=lambda i: (-scores[i], tie_rank.get(items[i], 10**9)))
    return [items[o] for o in order[:k]]


def build_pop_airings_rank(train, candidates, val_sched, max_week, airings):
    """Пул кандидатов и базовое ранжирование pop×log(1+airings) + rank_pop для тай-брейка."""
    pop_recent = train[train["week"] >= max_week - 3].groupby("tv_show_id")["user_id"].nunique()
    score_b = {}
    for c in candidates:
        pr = pop_recent.get(c, 0)
        ac = airings.get(c, 0)
        score_b[c] = pr * np.log1p(ac)
    pool_ranked = rank_list_from_scores(score_b, score_b)
    rank_pop = {it: r for r, it in enumerate(pool_ranked)}
    return pool_ranked, rank_pop, pop_recent


# --------------------- PPR (персонализированный, batch, по exp_ppr.py) ---------------------
def row_normalize(mat: csr_matrix) -> csr_matrix:
    row_sums = np.asarray(mat.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sums)
    nz = row_sums > 0
    inv[nz] = 1.0 / row_sums[nz]
    d = csr_matrix((inv, (np.arange(len(inv)), np.arange(len(inv)))), shape=(len(inv), len(inv)))
    return d @ mat


def build_ppr_graph(train):
    agg = train.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum().reset_index()
    agg["weight"] = np.log1p(agg["overlap_seconds"])
    graph_users = agg["user_id"].unique()
    graph_items = agg["tv_show_id"].unique()
    u_map = {u: i for i, u in enumerate(graph_users)}
    i_map = {it: i for i, it in enumerate(graph_items)}
    rows = agg["user_id"].map(u_map).values
    cols = agg["tv_show_id"].map(i_map).values
    R = csr_matrix((agg["weight"].values, (rows, cols)), shape=(len(u_map), len(i_map)))
    Mu2i = row_normalize(R)
    Mi2u = row_normalize(R.T.tocsr())
    return u_map, i_map, Mu2i, Mi2u


def ppr_batch(Mu2i, Mi2u, Vu, Vi, alpha=0.85, n_iter=30):
    Pu = Vu.copy()
    Pi = Vi.copy()
    Mu2iT = Mu2i.T.tocsr()
    Mi2uT = Mi2u.T.tocsr()
    for _ in range(n_iter):
        Pi = alpha * (Mu2iT @ Pu) + (1 - alpha) * Vi
        Pu = alpha * (Mi2uT @ Pi) + (1 - alpha) * Vu
    return Pu, Pi


def ppr_rankings(train, val_users, pool, rank_pop):
    u_map, i_map, Mu2i, Mi2u = build_ppr_graph(train)
    warm_users = [u for u in val_users if u in u_map]
    cold = set(u for u in val_users if u not in u_map)
    pool_idx = np.array([i_map[c] for c in pool if c in i_map])
    pool_known = [c for c in pool if c in i_map]
    B = len(warm_users)
    Vu = np.zeros((len(u_map), B))
    Vi = np.zeros((len(i_map), B))
    for b, u in enumerate(warm_users):
        Vi[:, b] = Mu2i[u_map[u], :].toarray().ravel()
    Pu, Pi = ppr_batch(Mu2i, Mi2u, Vu, Vi)
    scores_pool = Pi[pool_idx, :]  # (n_pool_known, B)
    rankings = {}
    for b, u in enumerate(warm_users):
        sc = dict(zip(pool_known, scores_pool[:, b]))
        # кандидаты вне PPR-графа получают минимальный балл (ранг в хвосте)
        min_sc = min(sc.values()) - 1.0 if sc else 0.0
        for c in pool:
            if c not in sc:
                sc[c] = min_sc
        rankings[u] = rank_list_from_scores(sc, rank_pop, k=POOL)
    for u in cold:
        rankings[u] = list(pool)  # rank_pop порядок, пул уже по убыванию pop×airings
    return rankings


# --------------------- AR (association rules, по exp_assoc_rules.py) ---------------------
def ar_rankings(train, val_users, candidates, rank_pop, max_week):
    g = train[train["overlap_seconds"] > 0].groupby(["user_id", "week"])["tv_show_id"].apply(lambda s: frozenset(s.unique()))
    txns = [s for s in g.values if len(s) >= 1]
    n_txn = len(txns)
    item_count = defaultdict(int)
    pair_count = defaultdict(int)
    for txn in txns:
        for a in txn:
            item_count[a] += 1
        items = sorted(txn)
        for a, b in combinations(items, 2):
            pair_count[(a, b)] += 1
    freq = {}
    for pair, cnt in pair_count.items():
        sup = cnt / n_txn
        if sup >= MIN_SUPPORT:
            freq[pair] = (cnt, sup)
    rules_by_ant = defaultdict(list)
    for (a, b), (cnt_ab, _) in freq.items():
        cnt_a, cnt_b = item_count.get(a, 0), item_count.get(b, 0)
        if cnt_a == 0 or cnt_b == 0:
            continue
        conf_ab = cnt_ab / cnt_a
        lift_ab = conf_ab / (cnt_b / n_txn)
        if lift_ab > 1:
            rules_by_ant[a].append((b, conf_ab))
        conf_ba = cnt_ab / cnt_b
        lift_ba = conf_ba / (cnt_a / n_txn)
        if lift_ba > 1:
            rules_by_ant[b].append((a, conf_ba))

    recent4w = train[(train["week"] >= max_week - 3) & (train["overlap_seconds"] > 0)]
    user_recent_items = recent4w.groupby("user_id")["tv_show_id"].apply(lambda s: set(s.unique()))

    rankings = {}
    cand_set = set(candidates)
    for u in val_users:
        A_set = user_recent_items.get(u, set())
        if not A_set:
            rankings[u] = list(candidates[:POOL])
            continue
        score = defaultdict(float)
        for a in A_set:
            for c, conf in rules_by_ant.get(a, []):
                if c in cand_set:
                    score[c] += conf
        if not score:
            rankings[u] = list(candidates[:POOL])
            continue
        rankings[u] = rank_list_from_scores(dict(score), rank_pop, k=POOL)
    return rankings


# --------------------- ALS (по hybrid_final_submit.py) ---------------------
def als_rankings(train, val_users, pool, rank_pop):
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
    pool_known = [c for c in pool if c in i_map]
    pool_idx = np.array([i_map[c] for c in pool_known])
    pool_factors = als.item_factors[pool_idx] if len(pool_idx) else np.zeros((0, 64))
    rankings = {}
    cold = []
    for u in val_users:
        if u not in u_map or len(pool_known) == 0:
            cold.append(u)
            continue
        uf = als.user_factors[u_map[u]]
        scores = pool_factors @ uf
        sc = dict(zip(pool_known, scores))
        min_sc = min(sc.values()) - 1.0
        for c in pool:
            if c not in sc:
                sc[c] = min_sc
        rankings[u] = rank_list_from_scores(sc, rank_pop, k=POOL)
    for u in cold:
        rankings[u] = list(pool)
    return rankings


# --------------------- CBF-new (как в exp_cbf_hw2.py, cold = не смотрел в train) ---------------------
def load_train_schedule(train_end_week):
    sched = pd.read_csv(DATA / "export_arh_11-20-final.csv", low_memory=False)
    sched["start_time"] = pd.to_datetime(sched["start_time"], dayfirst=True)
    sched["week"] = sched["start_time"].dt.isocalendar().week.astype(int)
    sched["hour"] = sched["start_time"].dt.hour.astype(int)
    sched["day_of_week"] = sched["start_time"].dt.day_of_week.astype(int)
    return sched[sched["week"] <= train_end_week].copy()


def build_item_tokens(train_sched):
    train_sched = train_sched[train_sched["tv_show_id"] != 0].copy()
    item_tokens = {}
    for _, row in train_sched.iterrows():
        toks = []
        if pd.notna(row["tv_show_category"]):
            toks.append("cat:" + str(row["tv_show_category"]))
        for g in ["tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"]:
            if pd.notna(row[g]):
                toks.append("gen:" + str(row[g]))
        toks.append("ch:" + str(int(row["channel_id"])))
        item_tokens.setdefault(row["tv_show_id"], set()).update(toks)
    vocab = {}
    for toks in item_tokens.values():
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
    item_mat = {}
    for sid, toks in item_tokens.items():
        vec = np.zeros(len(vocab))
        for t in toks:
            vec[vocab[t]] = 1.0
        item_mat[sid] = vec
    return item_mat, vocab


def build_slot_popularity(train, train_sched, pop_train):
    sched_air = train_sched[train_sched["tv_show_id"] != 0].copy()
    sched_air["show_pop"] = sched_air["tv_show_id"].map(pop_train).fillna(0.0)
    slot_pop = sched_air.groupby(["channel_id", "hour", "day_of_week"])["show_pop"].mean()
    channel_pop = sched_air.groupby("channel_id")["show_pop"].mean()
    global_mean = float(sched_air["show_pop"].mean())
    return slot_pop, channel_pop, global_mean


def lookup_slot_pop(row, slot_pop, channel_pop, global_mean):
    key = (int(row["channel_id"]), int(row["hour"]), int(row["day_of_week"]))
    if key in slot_pop.index:
        return float(slot_pop.loc[key])
    ch = int(row["channel_id"])
    if ch in channel_pop.index:
        return float(channel_pop.loc[ch])
    return global_mean


def cbf_rankings(train, val_users, pool, rank_pop, max_week, candidates):
    pop_train = train.groupby("tv_show_id")["user_id"].nunique()
    train_sched = load_train_schedule(max_week)
    item_mat, vocab = build_item_tokens(train_sched)
    V = len(vocab)
    slot_pop, channel_pop, global_mean = build_slot_popularity(train, train_sched, pop_train)
    slot_max = float(slot_pop.max()) if len(slot_pop) else 1.0

    val_sched = pd.read_csv(DATA / ("export_arh_21-30-final.csv" if max_week == 20 else "export_arh_11-20-final.csv"), low_memory=False)
    val_sched_air = val_sched[val_sched["tv_show_id"] != 0].copy()
    val_sched_air["start_time"] = pd.to_datetime(val_sched_air["start_time"], dayfirst=True)
    val_sched_air["hour"] = val_sched_air["start_time"].dt.hour.astype(int)
    val_sched_air["day_of_week"] = val_sched_air["start_time"].dt.day_of_week.astype(int)
    val_sched_air["slot_raw"] = val_sched_air.apply(
        lambda r: lookup_slot_pop(r, slot_pop, channel_pop, global_mean), axis=1)
    new_slot_norm = {}
    for sid, g in val_sched_air.groupby("tv_show_id")["slot_raw"]:
        new_slot_norm[sid] = float(g.mean()) / slot_max if slot_max > 0 else 0.0

    all_vec_cands = [c for c in candidates if c in item_mat]
    known_cands = [c for c in all_vec_cands if c in pop_train.index]
    known_mat = np.stack([item_mat[c] for c in known_cands]) if known_cands else np.zeros((0, V))
    known_norm = np.linalg.norm(known_mat, axis=1); known_norm[known_norm == 0] = 1.0
    all_mat = np.stack([item_mat[c] for c in all_vec_cands]) if all_vec_cands else np.zeros((0, V))
    all_card = all_mat.sum(axis=1) if all_mat.size else np.array([]); all_card[all_card == 0] = 1.0
    all_slot_vec = np.array([new_slot_norm.get(c, 0.0) for c in all_vec_cands])

    agg = train.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum().reset_index()
    agg["weight"] = np.log1p(agg["overlap_seconds"])
    agg = agg[agg["tv_show_id"].isin(item_mat)]
    warm_users = [u for u in val_users if u in set(train["user_id"].unique())]
    cold_users = [u for u in val_users if u not in set(train["user_id"].unique())]
    user_idx = {u: i for i, u in enumerate(warm_users)}
    profile = np.zeros((len(warm_users), V))
    for (u, sid), w in zip(zip(agg["user_id"], agg["tv_show_id"]), agg["weight"]):
        ui = user_idx.get(u)
        if ui is not None:
            profile[ui] += w * item_mat[sid]
    prof_norm = np.linalg.norm(profile, axis=1); prof_norm[prof_norm == 0] = 1.0
    pmax = profile.max(axis=1, keepdims=True); pmax[pmax == 0] = 1.0
    prof_unit = profile / pmax

    user_train_items = train.groupby("user_id")["tv_show_id"].apply(set).to_dict()
    all_vec_idx = {c: i for i, c in enumerate(all_vec_cands)}

    rankings = {}
    pool_known = [c for c in pool if c in pop_train.index and c in item_mat]
    known_mat_pool = np.stack([item_mat[c] for c in pool_known]) if pool_known else np.zeros((0, V))
    known_norm_pool = np.linalg.norm(known_mat_pool, axis=1); known_norm_pool[known_norm_pool == 0] = 1.0
    pool_all_vec = [c for c in pool if c in item_mat]
    all_mat_pool = np.stack([item_mat[c] for c in pool_all_vec]) if pool_all_vec else np.zeros((0, V))
    all_card_pool = all_mat_pool.sum(axis=1) if all_mat_pool.size else np.array([]); all_card_pool[all_card_pool == 0] = 1.0
    all_slot_pool = np.array([new_slot_norm.get(c, 0.0) for c in pool_all_vec])

    for b, u in enumerate(warm_users):
        # известные (pool): косинус
        if pool_known:
            cos = (profile[b] @ known_mat_pool.T) / (prof_norm[b] * known_norm_pool)
            known_scores = dict(zip(pool_known, cos))
        else:
            known_scores = {}
        # cold (в пуле, без просмотров этим пользователем): Jaccard*slotpop
        cold_pool = [c for c in pool_all_vec if c not in user_train_items.get(u, set())]
        cold_scores = {}
        if cold_pool:
            cold_idx = np.array([pool_all_vec.index(c) for c in cold_pool])
            jac_num = prof_unit[b] @ all_mat_pool[cold_idx].T
            jac = jac_num / all_card_pool[cold_idx]
            cold_scores = dict(zip(cold_pool, jac * all_slot_pool[cold_idx]))
        # два страта: сначала известные по косинусу, затем cold (как в exp_cbf_hw2.py)
        if known_scores:
            top_known = rank_list_from_scores(known_scores, rank_pop, k=POOL)
        else:
            top_known = []
        if cold_scores:
            top_cold = rank_list_from_scores(cold_scores, rank_pop, k=POOL)
        else:
            top_cold = []
        merged = top_known[:]
        for c in top_cold:
            if len(merged) >= POOL:
                break
            if c not in merged:
                merged.append(c)
        for c in pool:
            if len(merged) >= POOL:
                break
            if c not in merged:
                merged.append(c)
        rankings[u] = merged[:POOL]
    for u in cold_users:
        rankings[u] = list(pool[:POOL])
    return rankings


# --------------------- RRF fusion + greedy ablation ---------------------
def fuse_rrf(rankings_by_component, selected, pool, weights=None, k=5):
    """rankings_by_component: {name: {user: [items ranked, 0=best]}}. selected: list of component names.
    RRF score(item) = sum 1/(RRF_K + rank). weights: необязательный dict name->float (по умолчанию 1)."""
    preds = {}
    users = set()
    for name in selected:
        users.update(rankings_by_component[name].keys())
    for u in users:
        sc = defaultdict(float)
        for name in selected:
            rank_list = rankings_by_component[name].get(u, [])
            wgt = 1.0 if weights is None else weights.get(name, 1.0)
            for r, item in enumerate(rank_list[:POOL]):
                sc[item] += wgt / (RRF_K + r)
        if not sc:
            preds[u] = list(pool[:k])
            continue
        items = list(sc.keys())
        scores = np.array([sc[i] for i in items])
        order = sorted(range(len(items)), key=lambda i: -scores[i])
        preds[u] = [items[o] for o in order[:k]]
    return preds


def greedy_ablation(rankings_by_component, base_name, other_names, relevant, val_users):
    """Forward greedy: начинаем с base_name, по одному пробуем добавить каждый из other_names,
    оставляем компонент с лучшим MAP@5, повторяем, пока прирост есть."""
    selected = [base_name]
    log = []
    cur_map, _ = evaluate(fuse_rrf(rankings_by_component, selected, [], k=5), relevant)
    log.append((tuple(selected), cur_map))
    while True:
        best_gain = 0.0
        best_comp = None
        best_map = cur_map
        for comp in other_names:
            if comp in selected:
                continue
            trial = selected + [comp]
            m, _ = evaluate(fuse_rrf(rankings_by_component, trial, [], k=5), relevant)
            if m - cur_map > best_gain:
                best_gain = m - cur_map
                best_comp = comp
                best_map = m
        if best_comp is not None:
            selected.append(best_comp)
            cur_map = best_map
            log.append((tuple(selected), cur_map))
        else:
            break
    return selected, cur_map, log


def main():
    rows_out = []
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

    all_results = {}
    for split_name, cfg in splits.items():
        t0 = time.time()
        train, val_sched, candidates, relevant, val_users = build_split(
            inter, sub_users, cfg["train_end_week"], cfg["val_start_week"], cfg["val_end_week"], cfg["sched_val_path"]
        )
        max_week = cfg["max_week"]
        airings = pd.read_csv(DATA / cfg["airings_file"], low_memory=False)
        airings = airings[airings["tv_show_id"] != 0]["tv_show_id"].value_counts().to_dict()

        pool_full = []
        score_b = {}
        pop_recent = train[train["week"] >= max_week - 3].groupby("tv_show_id")["user_id"].nunique()
        for c in candidates:
            score_b[c] = pop_recent.get(c, 0) * np.log1p(airings.get(c, 0))
        pool_full = rank_list_from_scores(score_b, score_b, k=POOL)
        rank_pop = {it: r for r, it in enumerate(pool_full)}

        print(f"\n[split {split_name}] pool(топ-{POOL} по pop×airings)={len(pool_full)}, val_users={len(val_users)}")
        # База
        base_rank = {u: list(pool_full) for u in val_users}
        m_base, _ = evaluate({u: base_rank[u][:5] for u in val_users}, relevant)
        print(f"  base pop×airings MAP@5={m_base:.4f}")
        rows_out.append({"stage": "RRF_base", "split": split_name, "variant": "pop×airings (только база)",
                          "MAP5": m_base, "n_eval": len(val_users)})

        # PPR
        t1 = time.time()
        ppr_r = ppr_rankings(train, val_users, pool_full, rank_pop)
        m_ppr, _ = evaluate({u: ppr_r[u][:5] for u in val_users}, relevant)
        print(f"  PPR отдельно MAP@5={m_ppr:.4f} ({time.time()-t1:.1f}s)")
        rows_out.append({"stage": "RRF_component", "split": split_name, "variant": "PPR отдельно",
                          "MAP5": m_ppr, "n_eval": len(val_users)})

        # AR
        t1 = time.time()
        ar_r = ar_rankings(train, val_users, candidates, rank_pop, max_week)
        m_ar, _ = evaluate({u: ar_r[u][:5] for u in val_users}, relevant)
        print(f"  AR отдельно MAP@5={m_ar:.4f} ({time.time()-t1:.1f}s)")
        rows_out.append({"stage": "RRF_component", "split": split_name, "variant": "AR отдельно",
                          "MAP5": m_ar, "n_eval": len(val_users)})

        # ALS
        t1 = time.time()
        als_r = als_rankings(train, val_users, pool_full, rank_pop)
        m_als, _ = evaluate({u: als_r[u][:5] for u in val_users}, relevant)
        print(f"  ALS отдельно MAP@5={m_als:.4f} ({time.time()-t1:.1f}s)")
        rows_out.append({"stage": "RRF_component", "split": split_name, "variant": "ALS отдельно",
                          "MAP5": m_als, "n_eval": len(val_users)})

        # CBF-new
        t1 = time.time()
        cbf_r = cbf_rankings(train, val_users, pool_full, rank_pop, max_week, candidates)
        m_cbf, _ = evaluate({u: cbf_r[u][:5] for u in val_users}, relevant)
        print(f"  CBF-new отдельно MAP@5={m_cbf:.4f} ({time.time()-t1:.1f}s)")
        rows_out.append({"stage": "RRF_component", "split": split_name, "variant": "CBF-new отдельно",
                          "MAP5": m_cbf, "n_eval": len(val_users)})

        rankings_by_component = {
            "pop_airings": base_rank,
            "PPR": ppr_r,
            "AR": ar_r,
            "ALS": als_r,
            "CBF_new": cbf_r,
        }

        # greedy ablation: по одному добавляем к базе pop×airings
        selected, best_map, log = greedy_ablation(rankings_by_component, "pop_airings",
                                                   ["PPR", "AR", "ALS", "CBF_new"], relevant, val_users)
        print(f"  greedy ablation log:")
        for sel, m in log:
            print(f"    {' + '.join(sel)} -> MAP@5={m:.4f}")
        print(f"  ЛУЧШИЙ ГИБРИД [{', '.join(selected)}] MAP@5={best_map:.4f}")
        rows_out.append({"stage": "RRF_greedy", "split": split_name, "variant": " + ".join(selected),
                          "MAP5": best_map, "n_eval": len(val_users)})
        rows_out.append({"stage": "RRF_meta", "split": split_name, "variant": "greedy_path",
                          "value": "; ".join(f"{' + '.join(s)}={m:.4f}" for s, m in log)})

        all_results[split_name] = dict(selected=selected, best_map=best_map, log=log,
                                        rankings_by_component=rankings_by_component, relevant=relevant,
                                        val_users=val_users)
        print(f"  split завершён за {time.time()-t0:.1f}s")

    df_new = pd.DataFrame(rows_out)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(RESULTS / "exp_rrf_hybrid.csv", index=False)
    print(f"\n[saved] {RESULTS/'exp_rrf_hybrid.csv'}")

    df_map = df_new[df_new["MAP5"].notna()].sort_values("MAP5", ascending=False)
    df_map.to_csv(RESULTS / "exp_rrf_hybrid_sorted.csv", index=False)
    print(f"[saved] {RESULTS/'exp_rrf_hybrid_sorted.csv'}")

    print("\n=== Итоговая таблица RRF-гибридов (отсортирована по MAP@5) ===")
    print(df_map[["stage", "split", "variant", "MAP5", "n_eval"]].to_string(index=False))

    # Сводка по лучшим компонентам на каждом сплите (для решения о сабмитах)
    for split_name, res in all_results.items():
        print(f"\n[выбор для сабмитов] {split_name}: лучший гибрид = {res['selected']}, MAP@5={res['best_map']:.4f}")

    return all_results


if __name__ == "__main__":
    main()
