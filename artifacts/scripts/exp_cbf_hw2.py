#!/usr/bin/env python3
"""CBF по образцу HW2 (week2: Jaccard content-based recommender), адаптированный под ТВ-программы.

Вектор программы = категория + жанры + канал (множество токенов из расписания, построение
по обучающим эфирам). Профиль пользователя = взвешенная по секундам сумма векторов
программ, которые он смотрел в train (вес = log1p(сумма overlap_seconds)).

Два класса кандидатов:
  1) Программы с историей просмотров в train: косинус между профилем и вектором программы.
  2) Программы БЕЗ истории просмотров в train (новые): score = Jaccard(профиль, программа)
     x средняя популярность канала в этот слот (час и день недели) по обучению.
     Jaccard берётся как взвешенный (Ruzicka): профиль нормируется в [0,1] по максимуму
     токена, программа бинарна -> Jaccard = sum(p_t over t in item) / |item|.
     Популярность слота нормируется к [0,1] (делится на максимум по всем слотам), чтобы
     шкала сравнима с косинусом (тоже [0,1]). Оба определения -- явные допущения дизайна,
     зафиксированы здесь честно, в HW2 точной рецептуры для ТВ-слотов нет.

Холодные (без train-истории) пользователи -> fallback популярности (везде одинаково).
Тай-брейк по популярности. Валидация: строгая, оба сплита. Репетиция: кандидаты свежие
(без истории) -- считаем, сколько из них попало в топ-5 и сколько из них релевантны.
Выход: results/tables/exp_cbf_hw2.csv + append в exp_stage1_full.csv/exp_stage1_sorted.csv.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "tables"


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


def load_train_schedule(train_end_week):
    """Расписание обучающих недель (11-20 c фильтром по ISO-неделе <= train_end_week)
    с колонками week/hour/day_of_week для построения токенов и слотов."""
    sched = pd.read_csv(DATA / "export_arh_11-20-final.csv", low_memory=False)
    sched["start_time"] = pd.to_datetime(sched["start_time"], dayfirst=True)
    sched["week"] = sched["start_time"].dt.isocalendar().week.astype(int)
    sched["hour"] = sched["start_time"].dt.hour.astype(int)
    sched["day_of_week"] = sched["start_time"].dt.day_of_week.astype(int)
    return sched[sched["week"] <= train_end_week].copy()


def build_item_tokens(train_sched):
    """Токены программы: категория + жанры + канал (объединение по всем train-эфирам программы)."""
    train_sched = train_sched[train_sched["tv_show_id"] != 0].copy()

    def tokens_of_row(row):
        toks = []
        if pd.notna(row["tv_show_category"]):
            toks.append("cat:" + str(row["tv_show_category"]))
        for g in ["tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"]:
            if pd.notna(row[g]):
                toks.append("gen:" + str(row[g]))
        toks.append("ch:" + str(int(row["channel_id"])))
        return toks

    item_tokens = {}
    for _, row in train_sched.iterrows():
        sid = row["tv_show_id"]
        item_tokens.setdefault(sid, set()).update(tokens_of_row(row))
    vocab = {}
    for toks in item_tokens.values():
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
    item_mat = {}
    for sid, toks in item_tokens.items():
        idx = [vocab[t] for t in toks]
        vec = np.zeros(len(vocab), dtype=np.float64)
        vec[idx] = 1.0
        item_mat[sid] = vec
    return item_mat, vocab, item_tokens


def build_slot_popularity(train, train_sched, pop_train):
    """Средняя популярность канала в слот (час, день недели) по обучению.
    Популярность эфира = nunique зрителей программы в train (0, если не смотрели)."""
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


def main():
    rows_out = []
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

    for split_name, cfg in splits.items():
        train, val_sched, candidates, relevant, val_users = build_split(
            inter, sub_users, cfg["train_end_week"], cfg["val_start_week"], cfg["val_end_week"], cfg["sched_val_path"]
        )
        max_week = cfg["max_week"]
        print(f"\n[split {split_name}] train<= {cfg['train_end_week']}, val {cfg['val_start_week']}-{cfg['val_end_week']}: "
              f"users_with_rel={len(val_users)}/{len(sub_users)}, pool={len(candidates)}")

        t0 = time.perf_counter()
        # Популярность (fallback + тай-брейк): уникальные зрители за train
        pop_train = train.groupby("tv_show_id")["user_id"].nunique()
        pop_recent = train[train["week"] >= max_week - 3].groupby("tv_show_id")["user_id"].nunique()
        pop_recent = pop_recent.reindex(candidates).fillna(0.0).sort_values(ascending=False)
        pop_rank = {it: r for r, it in enumerate(pop_recent.index)}
        top5_pop = pop_recent.index.tolist()[:5]

        # Векторы программ (токены: категория + жанры + канал), train-расписание
        train_sched = load_train_schedule(cfg["train_end_week"])
        item_mat, vocab, item_tokens = build_item_tokens(train_sched)
        V = len(vocab)

        # Слот-популярность канала (час, день недели) по обучению
        slot_pop, channel_pop, global_mean = build_slot_popularity(train, train_sched, pop_train)
        slot_max = float(slot_pop.max()) if len(slot_pop) else 1.0

        # Новые (без истории просмотров в train) кандидаты: их эфиры в целевом окне -> слот
        val_sched_air = val_sched[val_sched["tv_show_id"] != 0].copy()
        val_sched_air["start_time"] = pd.to_datetime(val_sched_air["start_time"], dayfirst=True)
        val_sched_air["hour"] = val_sched_air["start_time"].dt.hour.astype(int)
        val_sched_air["day_of_week"] = val_sched_air["start_time"].dt.day_of_week.astype(int)
        val_sched_air["slot_raw"] = val_sched_air.apply(
            lambda r: lookup_slot_pop(r, slot_pop, channel_pop, global_mean), axis=1
        )
        new_slot_norm = {}
        if len(val_sched_air):
            for sid, g in val_sched_air.groupby("tv_show_id")["slot_raw"]:
                new_slot_norm[sid] = float(g.mean()) / slot_max if slot_max > 0 else 0.0

        known_cands = [c for c in candidates if c in pop_train.index and c in item_mat]
        # cold-item = программа, которую пользователь НЕ смотрел в train (стандартное определение
        # cold-start для content-based рекомендаций; глобально новая программа -- частный случай)
        # здесь список всех кандидатов с вектором (известных+новых глобально), отфильтруется per-user
        all_vec_cands = [c for c in candidates if c in item_mat]
        known_cands_set = set(known_cands)
        print(f"  кандидаты: с train-историей={len(known_cands)}, с вектором(для cold-ранжирования)={len(all_vec_cands)} "
              f"(без вектора, в отсечении)={len(candidates) - len(all_vec_cands)}; токенов в словаре={V}")

        rows_out.append({"stage": "CBFHW2_meta", "split": split_name, "variant": "n_candidates_known", "value": len(known_cands)})
        rows_out.append({"stage": "CBFHW2_meta", "split": split_name, "variant": "n_candidates_all_vec", "value": len(all_vec_cands)})
        rows_out.append({"stage": "CBFHW2_meta", "split": split_name, "variant": "n_tokens_vocab", "value": V})
        rows_out.append({"stage": "CBFHW2_meta", "split": split_name, "variant": "n_slots", "value": int(len(slot_pop))})

        # Матрицы кандидатов
        known_mat = np.stack([item_mat[c] for c in known_cands]) if known_cands else np.zeros((0, V))
        known_norm = np.linalg.norm(known_mat, axis=1)
        known_norm[known_norm == 0] = 1.0
        all_mat = np.stack([item_mat[c] for c in all_vec_cands]) if all_vec_cands else np.zeros((0, V))
        all_card = all_mat.sum(axis=1) if all_mat.size else np.array([])
        all_card[all_card == 0] = 1.0
        all_slot_vec = np.array([new_slot_norm.get(c, 0.0) for c in all_vec_cands])

        # Профили пользователей: взвешенные по секундам (log1p), только те, кто есть в train
        train_items = set(train["tv_show_id"].unique())
        warm_users = [u for u in val_users if u in set(train["user_id"].unique())]
        cold_users = [u for u in val_users if u not in set(train["user_id"].unique())]

        agg = train.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum().reset_index()
        agg["weight"] = np.log1p(agg["overlap_seconds"])
        agg = agg[agg["tv_show_id"].isin(item_mat)]
        user_idx = {u: i for i, u in enumerate(warm_users)}

        profile = np.zeros((len(warm_users), V))
        # накопление: profile[u] += weight * item_vec
        for (u, sid), w in zip(zip(agg["user_id"], agg["tv_show_id"]), agg["weight"]):
            ui = user_idx.get(u)
            if ui is None:
                continue
            profile[ui] += w * item_mat[sid]

        prof_norm = np.linalg.norm(profile, axis=1)
        prof_norm[prof_norm == 0] = 1.0
        pmax = profile.max(axis=1, keepdims=True)
        pmax[pmax == 0] = 1.0
        prof_unit = profile / pmax  # для взвешенного Jaccard: токены в [0,1]

        # --- диагностика масштабов двух групп скоров (на первом warm-пользователе и в среднем) ---
        probe = warm_users[0] if warm_users else None
        if probe is not None:
            b0 = 0
            cos_probe = (profile[b0] @ known_mat.T) / (prof_norm[b0] * known_norm) if known_cands else np.array([])
            jac_probe = np.array([])
            if all_vec_cands:
                jac_num = prof_unit[b0] @ all_mat.T
                jac_probe = (jac_num / all_card) * all_slot_vec
            print(f"  [диагностика] косинус (известные, n={len(cos_probe)}): min={cos_probe.min():.4f} "
                  f"max={cos_probe.max():.4f} mean={cos_probe.mean():.4f}; "
                  f"Jaccard*slotpop (кандидаты с вектором, n={len(jac_probe)}): min={jac_probe.min():.6f} "
                  f"max={jac_probe.max():.6f} mean={jac_probe.mean():.6f}")

        # Сколько строго-релевантных кандидатов без train-истории (могут ли новые вообще быть релевантны)
        rel_known = rel_new = 0
        train_item_set = train_items
        for u in val_users:
            for r in relevant.get(u, set()):
                if r in train_item_set:
                    rel_known += 1
                else:
                    rel_new += 1
        print(f"  [диагностика] релевантных кандидатов: с train-историей={rel_known}, без (новых)={rel_new}")
        rows_out.append({"stage": "CBFHW2_diag", "split": split_name, "variant": "relevant_known_slots", "value": rel_known})
        rows_out.append({"stage": "CBFHW2_diag", "split": split_name, "variant": "relevant_new_slots", "value": rel_new})

        preds = {}
        new_in_top5 = 0
        new_relevant_slots = 0
        new_slot_total = 0
        # Программы, которые пользователь смотрел в train -- для cold-ранжирования (без просмотров им)
        user_train_items = train.groupby("user_id")["tv_show_id"].apply(set).to_dict()

        for b, u in enumerate(warm_users):
            # 1) известные (глобально, с train-историей): косинус профиля с вектором программы
            if known_cands:
                cos = (profile[b] @ known_mat.T) / (prof_norm[b] * known_norm)
                known_scores = dict(zip(known_cands, cos))
            else:
                known_scores = {}
            # 2) cold-item: программа, которую ЭТОТ пользователь не смотрел в train (глобально новая -- тоже
            #    попадает сюда, т.к. её нет в user_train_items[u]); score = Jaccard x слот-популярность канала
            cold_cands_u = [c for c in all_vec_cands if c not in user_train_items.get(u, set())]
            new_scores = {}
            if cold_cands_u:
                cold_idx = np.array([all_vec_cands.index(c) for c in cold_cands_u])
                jac_num = prof_unit[b] @ all_mat[cold_idx].T
                jac = jac_num / all_card[cold_idx]
                new_scores = dict(zip(cold_cands_u, jac * all_slot_vec[cold_idx]))

            # ранжирование: сначала известные по косинусу, затем cold по Jaccard*slotpop (без смешивания
            # шкал -- два страта, как в классическом CBF pipeline: основной скор + fallback на cold)
            if known_scores:
                ids = list(known_scores.keys())
                scores = np.array([known_scores[i] for i in ids])
                order = sorted(range(len(ids)), key=lambda i: (-scores[i], pop_rank.get(ids[i], 10**9)))
                top = [ids[o] for o in order]
            else:
                top = []
            if len(top) < 5 and new_scores:
                ids2 = list(new_scores.keys())
                scores2 = np.array([new_scores[i] for i in ids2])
                order2 = sorted(range(len(ids2)), key=lambda i: (-scores2[i], pop_rank.get(ids2[i], 10**9)))
                top2 = [ids2[o] for o in order2]
                top = top + [c for c in top2 if c not in top]
            for p in top5_pop:
                if len(top) >= 5:
                    break
                if p not in top:
                    top.append(p)
            preds[u] = top[:5]

            # подсчёт: сколько cold-программ в топ-5 и сколько из них релевантны
            rel = relevant.get(u, set())
            seen_u = user_train_items.get(u, set())
            for c in preds[u][:5]:
                new_slot_total += 1
                if c not in seen_u:
                    new_in_top5 += 1
                    if c in rel:
                        new_relevant_slots += 1

        for u in cold_users:
            preds[u] = top5_pop

        m, n = evaluate(preds, relevant)
        t_run = time.perf_counter() - t0
        print(f"  CBF HW2 (cosine + Jaccard*slotpop для новых)  MAP@5={m:.4f} (n={n}), time={t_run:.2f}s")
        pct_new = (100.0 * new_in_top5 / new_slot_total) if new_slot_total else 0.0
        pct_rel = (100.0 * new_relevant_slots / new_in_top5) if new_in_top5 else 0.0
        print(f"  новых программ в топ-5: {new_in_top5} из {new_slot_total} слотов "
              f"({pct_new:.2f}%), из них релевантны: {new_relevant_slots} ({pct_rel:.2f}% от новых)")
        rows_out.append({"stage": "CBFHW2_model", "split": split_name, "variant": "cbf_hw2_cosine_jaccard_slotpop",
                          "MAP5": m, "n_eval": n})
        rows_out.append({"stage": "CBFHW2_novelty", "split": split_name, "variant": "new_items_in_top5", "value": new_in_top5})
        rows_out.append({"stage": "CBFHW2_novelty", "split": split_name, "variant": "top5_slots_total", "value": new_slot_total})
        rows_out.append({"stage": "CBFHW2_novelty", "split": split_name, "variant": "new_items_relevant", "value": new_relevant_slots})

        preds_pop = {u: top5_pop for u in val_users}
        m_pop, n_pop = evaluate(preds_pop, relevant)
        rows_out.append({"stage": "CBFHW2_reference", "split": split_name, "variant": "popularity_recent4w_unique",
                          "MAP5": m_pop, "n_eval": n_pop})
        print(f"  (справочно) Popularity recent4w  MAP@5={m_pop:.4f} (n={n_pop})")

    df_new = pd.DataFrame(rows_out)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(RESULTS / "exp_cbf_hw2.csv", index=False)
    print(f"\n[saved] {RESULTS/'exp_cbf_hw2.csv'}")

    full_path = RESULTS / "exp_stage1_full.csv"
    df_full_old = pd.read_csv(full_path) if full_path.exists() else pd.DataFrame()
    df_full_new = pd.concat([df_full_old, df_new], ignore_index=True)
    df_full_new.to_csv(full_path, index=False)
    print(f"[updated] {full_path} (append CBF-HW2 rows)")

    sorted_path = RESULTS / "exp_stage1_sorted.csv"
    df_map = df_full_new[df_full_new["MAP5"].notna()].sort_values("MAP5", ascending=False)
    df_map.to_csv(sorted_path, index=False)
    print(f"[updated] {sorted_path}")

    print("\n=== Таблица CBF-HW2 + справочные строки (отсортирована по MAP@5) ===")
    df_cbf = df_new[df_new["MAP5"].notna()].sort_values("MAP5", ascending=False)
    print(df_cbf[["stage", "split", "variant", "MAP5", "n_eval"]].to_string(index=False))


if __name__ == "__main__":
    main()
