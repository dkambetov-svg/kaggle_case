#!/usr/bin/env python3
"""E1 (exp3): трёхдольный граф PPR — пользователь, программа, атрибут.

Атрибуты программы: канал (ch:), категория (cat:), жанры (gen:) — из расписания.
Для программ валидационного пула атрибуты берутся из расписания валидационного
окна (для финального инференса — из 31–42; тестовые просмотры 31–42 не
используются, расписание — только для связей программа–канал–жанр и числа эфиров).

Рёбра программа–атрибут двусторонние. attr_weight — какая доля вероятности
перехода из программы уходит в атрибуты, а не к пользователям: item→attr
масштабируется на attr_weight (равномерно по атрибутам программы),
item→user на (1−attr_weight) (пропорционально log(1+сек)), затем строковая
нормализация всего переходного оператора T (user→item без масштабирования).
Программы пула без просмотров в обучении имеют только атрибутные рёбра —
после нормализации вся их исходящая вероятность уходит в атрибуты (это и
позволяет «холодным» программам получать массу и попадать в топ-5).

Рестарт 0.5 (alpha = 1 − restart = 0.5), 30 итераций, батч scipy.sparse,
SEED=42. Restart-вектор — программы пользователя пропорционально весам
(как в бипартитном PPR).

attr_weight=0 — контрольный прогон: должен воспроизвести бипартитный
PPR restart=0.5 (main 0.5333 / early 0.4577;
results/tables/exp_grid_rp3_ease_ppr_sorted.csv).

Диагностика: сколько программ, не встречавшихся в обучении, попало в топ-5
(слотов и уникальных) и сколько из них релевантны.

Таблица: results/tables/exp3_e1.csv (variant, params, MAP5_main, MAP5_early,
MAP5_mean, time_s, seed + диагностика холодных программ).
"""
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from exp_rrf_hybrid import (
    DATA, RESULTS, POOL, SEED,
    evaluate, build_split, rank_list_from_scores, row_normalize,
)

ATTR_WEIGHTS = (0.0, 0.1, 0.3, 0.5)  # 0.0 = контроль (бипартитный граф)
RESTART = 0.5
N_ITER = 30
USECOLS = ["channel_id", "tv_show_id", "tv_show_category",
           "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"]


def load_item_attrs(sched_paths):
    """tv_show_id -> список атрибутов (ch:/cat:/gen:) из расписания(й)."""
    frames = [pd.read_csv(p, usecols=USECOLS, dtype=str, low_memory=False)
              for p in sched_paths]
    sched = pd.concat(frames, ignore_index=True)
    sched["tv_show_id"] = sched["tv_show_id"].astype(np.int64)
    sched = sched[sched["tv_show_id"] != 0]
    parts = []
    ch = sched[["tv_show_id", "channel_id"]].dropna()
    parts.append(ch.assign(attr="ch:" + ch["channel_id"].astype(str))[["tv_show_id", "attr"]])
    cat = sched[["tv_show_id", "tv_show_category"]].dropna()
    parts.append(cat.assign(attr="cat:" + cat["tv_show_category"].astype(str))[["tv_show_id", "attr"]])
    for g in ["tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"]:
        gg = sched[["tv_show_id", g]].dropna()
        parts.append(gg.assign(attr="gen:" + gg[g].astype(str))[["tv_show_id", "attr"]])
    attrs = pd.concat(parts, ignore_index=True).drop_duplicates()
    item_attrs = attrs.groupby("tv_show_id")["attr"].apply(list).to_dict()
    print(f"  [attrs] файлы={[Path(p).name for p in sched_paths]}: "
          f"программ с атрибутами={len(item_attrs)}, уникальных атрибутов={attrs['attr'].nunique()}")
    return item_attrs


def _restart_from_weights(eu, ei, ew):
    """per-user (item_idx, norm_weights) для restart-вектора."""
    per_user = defaultdict(lambda: ([], []))
    for u_i, i_i, w in zip(eu, ei, ew):
        lst = per_user[u_i]
        lst[0].append(i_i)
        lst[1].append(w)
    user_restart = {}
    for u_i, (idxs, ws) in per_user.items():
        ws = np.asarray(ws, dtype=float)
        tot = ws.sum()
        if tot > 0:
            user_restart[u_i] = (np.asarray(idxs, dtype=np.int64), ws / tot)
        else:  # все веса нулевые -> равномерно
            user_restart[u_i] = (np.asarray(idxs, dtype=np.int64),
                                 np.full(len(idxs), 1.0 / len(idxs)))
    return user_restart


def build_graph_base(train, pool_full, item_attrs, decay_tau=None, max_week=None):
    """База графа: рёбра user–item (агрегированные), словари узлов, атрибуты программ.

    decay_tau (E2): вес программы = log1p(Σ_r overlap_r · exp(−Δ_r/τ)),
    Δ_r = max_week − week_r (недель от просмотра до конца обучения).
    Без decay_tau — как в E1: log1p(Σ_r overlap_r)."""
    raw_sum = train.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum()
    if decay_tau is not None:
        # week: dtype UInt32 -> явно int64, иначе унарный минус переполняет unsigned
        delta = max_week - train["week"].to_numpy(dtype=np.int64)  # недель до конца обучения
        assert delta.min() >= 0, "просмотры позже конца обучения"
        dec = train.assign(_d=train["overlap_seconds"] * np.exp(-delta / decay_tau))
        dec_sum = dec.groupby(["user_id", "tv_show_id"])["_d"].sum().reindex(raw_sum.index)
        ew_series = np.log1p(dec_sum.to_numpy(dtype=float))
    else:
        ew_series = np.log1p(raw_sum.to_numpy(dtype=float))
    ew_raw_series = np.log1p(raw_sum.to_numpy(dtype=float))

    agg = raw_sum.reset_index()
    users = agg["user_id"].unique()
    train_items = agg["tv_show_id"].unique()
    items = np.unique(np.concatenate([train_items, np.asarray(pool_full, dtype=train_items.dtype)]))
    u_map = {u: i for i, u in enumerate(users)}
    i_map = {it: i for i, it in enumerate(items)}
    eu = agg["user_id"].map(u_map).to_numpy()
    ei = agg["tv_show_id"].map(i_map).to_numpy()
    ew = ew_series
    ew_raw = ew_raw_series
    assert np.isfinite(ew).all(), "не-конечные веса рёбер (log1p inf/NaN)"

    # restart: по применённым весам (ew) и по сырым (ew_raw) — для E2-вариантов
    user_restart = _restart_from_weights(eu, ei, ew)
    user_restart_raw = _restart_from_weights(eu, ei, ew_raw)

    # атрибуты: item local idx -> list of attr local idx
    attr_index = {}
    per_item_attrs = []
    n_attrs_missing = 0
    for it in items:
        toks = item_attrs.get(int(it), [])
        if not toks:
            n_attrs_missing += 1
        idxs = []
        for t in toks:
            if t not in attr_index:
                attr_index[t] = len(attr_index)
            idxs.append(attr_index[t])
        per_item_attrs.append(idxs)
    print(f"  [graph] пользователей={len(u_map)}, программ={len(i_map)} "
          f"(train={len(train_items)}, пул без просмотров в train="
          f"{sum(1 for c in pool_full if int(c) not in set(int(x) for x in train_items))})", end="")
    print(f", атрибутов={len(attr_index)}, программ без атрибутов={n_attrs_missing}")
    return dict(eu=eu, ei=ei, ew=ew, nU=len(u_map), nI=len(i_map),
                nA=len(attr_index), u_map=u_map, i_map=i_map,
                per_user=user_restart, per_user_raw=user_restart_raw,
                per_item_attrs=per_item_attrs, train_item_set=set(int(x) for x in train_items))


def build_transition(base, attr_weight):
    """T (csr, row-normalized): доля attr_weight из программы уходит в атрибуты,
    (1−attr_weight) — к пользователям. Нормировка ВНУТРИ групп ДО смешивания:
    item→user = (1−aw)·w/Σ_u w, item→attr = aw/|attrs(item)| (сумма ряда = 1,
    если оба типа рёбер есть). user→item = w (row_normalize), attr→item = 1
    (row_normalize -> равномерно). aw=0 даёт бипартитный граф бит-в-бит."""
    nU, nI, nA = base["nU"], base["nI"], base["nA"]
    n = nU + nI + nA
    I_off, A_off = nU, nU + nI
    eu, ei, ew = base["eu"], base["ei"], base["ew"]
    item_w_sum = np.bincount(ei, weights=ew, minlength=nI)
    safe_sum = np.where(item_w_sum > 0, item_w_sum, 1.0)
    rows = [eu, ei + I_off]
    cols = [ei + I_off, eu]
    data = [ew, (1.0 - attr_weight) * ew / safe_sum[ei]]
    counts = np.array([len(a) for a in base["per_item_attrs"]], dtype=np.int64)
    total = int(counts.sum())
    if total > 0:
        item_ids = np.repeat(np.arange(nI, dtype=np.int64), counts)
        attr_ids = np.concatenate([np.asarray(a, dtype=np.int64)
                                   for a in base["per_item_attrs"] if a])
        # item -> attr
        rows.append(item_ids + I_off)
        cols.append(attr_ids + A_off)
        data.append(np.full(total, attr_weight) / np.repeat(counts, counts))
        # attr -> item
        rows.append(attr_ids + A_off)
        cols.append(item_ids + I_off)
        data.append(np.ones(total))
    T = csr_matrix((np.concatenate(data),
                    (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return row_normalize(T)


def ppr_tripartite(T, restart_mat, n_iter=N_ITER):
    """P = alpha*(T^T P) + (1-alpha)*restart; alpha = 1 − RESTART.
    restart_mat (n_nodes×B): ненулевой только блок программ (или нули для cold)."""
    alpha = 1.0 - RESTART
    Tt = T.T.tocsr()
    P = restart_mat.copy()
    for _ in range(n_iter):
        P = alpha * (Tt @ P) + (1.0 - alpha) * restart_mat
    return P


def main():
    inter = pd.read_parquet(DATA / "user_interactions.parquet")
    inter["user_id"] = inter["user_id"].astype(str)
    sub = pd.read_csv(DATA / "submission.csv", dtype={"user_id": str})
    sub_users = sub["user_id"].tolist()

    splits = {
        "main": dict(train_end_week=20, val_start_week=21, val_end_week=30,
                     sched_val_path=DATA / "export_arh_21-30-final.csv", max_week=20,
                     airings_file="export_arh_21-30-final.csv",
                     attrs_files=[DATA / "export_arh_11-20-final.csv",
                                  DATA / "export_arh_21-30-final.csv"]),
        "early": dict(train_end_week=16, val_start_week=17, val_end_week=20,
                      sched_val_path=DATA / "export_arh_11-20-final.csv", max_week=16,
                      airings_file="export_arh_11-20-final.csv",
                      attrs_files=[DATA / "export_arh_11-20-final.csv"]),
    }

    acc = {aw: dict(MAP5_main=None, MAP5_early=None, time_s=0.0,
                    new_main=None, newu_main=None, newr_main=None,
                    new_early=None, newu_early=None, newr_early=None,
                    uu_main=None, uur_main=None, uu_early=None, uur_early=None)
           for aw in ATTR_WEIGHTS}

    for split_name, cfg in splits.items():
        t_split = time.perf_counter()
        train, val_sched, candidates, relevant, val_users = build_split(
            inter, sub_users, cfg["train_end_week"], cfg["val_start_week"],
            cfg["val_end_week"], cfg["sched_val_path"])
        max_week = cfg["max_week"]
        airings = pd.read_csv(DATA / cfg["airings_file"], low_memory=False)
        airings = airings[airings["tv_show_id"] != 0]["tv_show_id"].value_counts().to_dict()
        pop_recent = train[train["week"] >= max_week - 3].groupby("tv_show_id")["user_id"].nunique()
        score_b = {c: pop_recent.get(c, 0) * np.log1p(airings.get(c, 0)) for c in candidates}
        pool_full = rank_list_from_scores(score_b, score_b, k=POOL)
        rank_pop = {it: r for r, it in enumerate(pool_full)}
        print(f"\n[split {split_name}] pool={len(pool_full)}, val_users={len(val_users)}, "
              f"train недель<={cfg['train_end_week']}")

        item_attrs = load_item_attrs(cfg["attrs_files"])
        base = build_graph_base(train, pool_full, item_attrs)
        nU, nI = base["nU"], base["nI"]
        I_off = nU
        pool_local = np.array([base["i_map"][c] for c in pool_full])
        n_nodes = nU + nI + base["nA"]
        # пер-юзерные просмотренные программы (для диагностики "новых" в топ-5)
        user_train_items = train.groupby("user_id")["tv_show_id"].apply(set).to_dict()

        # restart-матрица (n_nodes × B): блок программ = программы пользователя
        # (как в бипартитном PPR: Vi = Mu2i[u]); cold-пользователи -> нули,
        # их скоры по пулю нулевые -> ранг по tie_rank (pop×airings)
        B = len(val_users)
        restart_mat = np.zeros((n_nodes, B))
        for b, u in enumerate(val_users):
            ui = base["u_map"].get(u)
            if ui is None:
                continue
            idxs, ws = base["per_user"][ui]
            restart_mat[I_off + idxs, b] = ws

        for aw in ATTR_WEIGHTS:
            t0 = time.perf_counter()
            T = build_transition(base, aw)
            P = ppr_tripartite(T, restart_mat)
            scores = P[I_off + pool_local, :]
            preds = {}
            for b, u in enumerate(val_users):
                sc = dict(zip(pool_full, scores[:, b]))
                preds[u] = rank_list_from_scores(sc, rank_pop, k=POOL)
            m, n = evaluate(preds, relevant)
            dt = time.perf_counter() - t0

            # холодные программы в топ-5
            # (а) глобально: программа вообще не встречалась ни в одном train-программе
            # (б) пер-юзерно: программа не встречалась ЭТОМУ пользователю в обучении
            new_slots = new_rel = 0
            seen, rel_seen = set(), set()
            unseen_u_slots = unseen_u_rel = 0
            for u in val_users:
                u_hist = user_train_items.get(u, set())
                for p in preds[u][:5]:
                    if p not in base["train_item_set"]:
                        new_slots += 1
                        seen.add(p)
                        if p in relevant.get(u, ()):
                            new_rel += 1
                            rel_seen.add(p)
                    if p not in u_hist:
                        unseen_u_slots += 1
                        if p in relevant.get(u, ()):
                            unseen_u_rel += 1
            print(f"  attr_weight={aw}: MAP@5={m:.4f} (n={n}) time={dt:.1f}s | "
                  f"глобально нетренированных в топ-5: слотов={new_slots}, уникальных={len(seen)}, "
                  f"релевантных слотов={new_rel} | пер-юзерно не видел: слотов={unseen_u_slots}, "
                  f"из них релевантных={unseen_u_rel}")

            key = f"MAP5_{split_name}"
            acc[aw][key] = m
            acc[aw][f"time_s"] += dt
            acc[aw][f"new_{split_name}"] = new_slots
            acc[aw][f"newu_{split_name}"] = len(seen)
            acc[aw][f"newr_{split_name}"] = new_rel
            acc[aw][f"uu_{split_name}"] = unseen_u_slots
            acc[aw][f"uur_{split_name}"] = unseen_u_rel
        print(f"  [split завершён за {time.perf_counter() - t_split:.1f}s]")

    rows = []
    for aw in ATTR_WEIGHTS:
        a = acc[aw]
        label = (f"E1 attr_weight={aw}" + (" (контроль: бипартитный)" if aw == 0 else ""))
        rows.append({
            "variant": label,
            "params": f"attr_weight={aw}; restart={RESTART}; n_iter={N_ITER}; attrs=ch/cat/gen; graph=tripartite",
            "MAP5_main": a["MAP5_main"], "MAP5_early": a["MAP5_early"],
            "MAP5_mean": (a["MAP5_main"] + a["MAP5_early"]) / 2,
            "time_s": round(a["time_s"], 1), "seed": SEED,
            "new_top5_main": a["new_main"], "new_top5_unique_main": a["newu_main"],
            "new_top5_rel_main": a["newr_main"],
            "new_top5_early": a["new_early"], "new_top5_unique_early": a["newu_early"],
            "new_top5_rel_early": a["newr_early"],
            "usernew_top5_main": a["uu_main"], "usernew_rel_main": a["uur_main"],
            "usernew_top5_early": a["uu_early"], "usernew_rel_early": a["uur_early"],
        })
    df = pd.DataFrame(rows).sort_values("MAP5_mean", ascending=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "exp3_e1.csv"
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}")
    print("\n=== E1: трёхдольный граф PPR (сортировка по среднему MAP@5) ===")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
