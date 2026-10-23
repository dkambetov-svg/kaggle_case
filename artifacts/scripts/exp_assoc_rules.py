#!/usr/bin/env python3
"""Association rules (по образцу HW4: github.com/dryjins/RecSys-LLMs/tree/main/week4),
адаптированные под программы вместо товаров в корзине.

Транзакция = множество программ (tv_show_id, != 0), которые пользователь смотрел
(есть overlap_seconds > 0) за ОДНУ неделю обучения. N = число таких транзакций
(пользователь x неделя), не число пользователей.

Определения (как в HW4 §5), count(X) = число транзакций, содержащих все элементы X:
  support(A -> B)    = count(A u B) / N
  confidence(A -> B) = count(A u B) / count(A)
  lift(A -> B)       = confidence(A -> B) / (count(B) / N)

Частые itemsets ищутся Apriori-подобным способом (downward closure) только для пар
(2-itemsets), т.к. это и есть базовый уровень для правил A->B в этом задании.
Пороги: support >= 0.2% транзакций, lift > 1 (без порога confidence на этапе отбора
"сильных правил" для отчёта, но confidence используется как скор для кандидатов рекомендаций).

Скор кандидата c для пользователя u = сумма confidence(A -> c) по всем A из множества
программ пользователя за последние 4 недели обучения (каждая отдельная программа как
антецедент-синглтон; это стандартная упрощённая агрегация item-item association rules
в рекомендательных системах -- аналог "sum of confidences").

Валидация: строгая, оба сплита (main train<=20/val21-30, early train<=16/val17-20).
Не перезаписывает финал; добавляет строки в results/tables/exp_stage1_full.csv (append)
и сохраняет results/tables/exp_assoc_rules.csv, exp_assoc_rules_top5.csv.
"""
import time
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "tables"
MIN_SUPPORT = 0.002  # 0.2% транзакций


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


def build_transactions(train):
    """Транзакция = (user_id, week) -> множество просмотренных tv_show_id (overlap_seconds>0)."""
    g = train[train["overlap_seconds"] > 0].groupby(["user_id", "week"])["tv_show_id"].apply(
        lambda s: frozenset(s.unique())
    )
    txns = [s for s in g.values if len(s) >= 1]
    return txns


def count_itemsets(transactions):
    """Счётчики одиночных программ (count(A)) и пар (count(A,B)), count(X) = #txns содержащих все X."""
    item_count = defaultdict(int)
    pair_count = defaultdict(int)
    for txn in transactions:
        for a in txn:
            item_count[a] += 1
        # только пары внутри транзакции -- базовый Apriori-уровень 2 для association rules
        items = sorted(txn)
        for a, b in combinations(items, 2):
            pair_count[(a, b)] += 1
    return item_count, pair_count


def frequent_pairs(pair_count, n_txn, min_support):
    """Apriori downward-closure: пара частая, если support >= min_support.
    (Уровень-1 кандидаты для пар уже прошли implicit pruning: пара считалась только
    если оба элемента встретились хотя бы в одной транзакции вместе -- это и есть
    единственный способ появления пары в pair_count.)"""
    freq = {}
    for pair, cnt in pair_count.items():
        support = cnt / n_txn
        if support >= min_support:
            freq[pair] = (cnt, support)
    return freq


def generate_rules(freq_pairs, item_count, n_txn):
    """Генерирует правила в обе стороны (A->B и B->A), считает confidence и lift,
    оставляет только lift > 1."""
    rules = []
    for (a, b), (cnt_ab, sup_ab) in freq_pairs.items():
        cnt_a, cnt_b = item_count.get(a, 0), item_count.get(b, 0)
        if cnt_a == 0 or cnt_b == 0:
            continue
        conf_ab = cnt_ab / cnt_a
        lift_ab = conf_ab / (cnt_b / n_txn)
        if lift_ab > 1:
            rules.append({"antecedent": a, "consequent": b, "support": sup_ab,
                           "confidence": conf_ab, "lift": lift_ab, "count_ab": cnt_ab,
                           "count_a": cnt_a, "count_b": cnt_b})
        conf_ba = cnt_ab / cnt_b
        lift_ba = conf_ba / (cnt_a / n_txn)
        if lift_ba > 1:
            rules.append({"antecedent": b, "consequent": a, "support": sup_ab,
                           "confidence": conf_ba, "lift": lift_ba, "count_ab": cnt_ab,
                           "count_a": cnt_b, "count_b": cnt_a})
    return rules


def rank_from_scores(ids, scores, tie_breaker_rank):
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], tie_breaker_rank.get(ids[i], 10**9)))
    return [ids[o] for o in order]


def run_split(split_name, inter, sub_users, cfg, rows_out, title_lookup):
    train, val_sched, candidates, relevant, val_users = build_split(
        inter, sub_users, cfg["train_end_week"], cfg["val_start_week"], cfg["val_end_week"], cfg["sched_val_path"]
    )
    max_week = cfg["max_week"]
    print(f"\n[split {split_name}] train<= {cfg['train_end_week']}, val {cfg['val_start_week']}-{cfg['val_end_week']}: "
          f"users_with_rel={len(val_users)}/{len(sub_users)}, pool={len(candidates)}")

    pop_recent = train[train["week"] >= max_week - 3].groupby("tv_show_id")["user_id"].nunique()
    pop_recent = pop_recent.reindex(candidates).fillna(0.0).sort_values(ascending=False)
    pop_rank = {it: r for r, it in enumerate(pop_recent.index)}
    top5_pop = pop_recent.index.tolist()[:5]

    t0 = time.perf_counter()
    txns = build_transactions(train)
    n_txn = len(txns)
    item_count, pair_count = count_itemsets(txns)
    freq_pairs = frequent_pairs(pair_count, n_txn, MIN_SUPPORT)
    rules = generate_rules(freq_pairs, item_count, n_txn)
    t_mine = time.perf_counter() - t0
    print(f"  транзакций (user,week)={n_txn}, уникальных программ={len(item_count)}, "
          f"пар с support>=0.2%={len(freq_pairs)}, правил (lift>1)={len(rules)}, время={t_mine:.2f}s")
    rows_out.append({"stage": "AR_meta", "split": split_name, "variant": "n_transactions", "value": n_txn})
    rows_out.append({"stage": "AR_meta", "split": split_name, "variant": "n_items", "value": len(item_count)})
    rows_out.append({"stage": "AR_meta", "split": split_name, "variant": "n_frequent_pairs", "value": len(freq_pairs)})
    rows_out.append({"stage": "AR_meta", "split": split_name, "variant": "n_rules_lift_gt1", "value": len(rules)})
    rows_out.append({"stage": "AR_meta", "split": split_name, "variant": "mining_time_s", "value": round(t_mine, 2)})

    # rule index: antecedent -> list of (consequent, confidence)
    rules_by_ant = defaultdict(list)
    for r in rules:
        rules_by_ant[r["antecedent"]].append((r["consequent"], r["confidence"]))

    # --- скор кандидатов для каждого eval-пользователя: сумма confidence(A->c),
    #     A = программы пользователя за последние 4 обучающие недели ---
    recent4w = train[(train["week"] >= max_week - 3) & (train["overlap_seconds"] > 0)]
    user_recent_items = recent4w.groupby("user_id")["tv_show_id"].apply(lambda s: set(s.unique()))

    preds = {}
    for u in val_users:
        A_set = user_recent_items.get(u, set())
        if not A_set:
            preds[u] = top5_pop
            continue
        score = defaultdict(float)
        for a in A_set:
            for c, conf in rules_by_ant.get(a, []):
                if c in candidates:
                    score[c] += conf
        if not score:
            preds[u] = top5_pop
            continue
        ids = list(score.keys())
        scores = np.array([score[i] for i in ids])
        top = rank_from_scores(ids, scores, pop_rank)
        for p in top5_pop:
            if len(top) >= 5:
                break
            if p not in top:
                top.append(p)
        preds[u] = top[:5]

    m, n = evaluate(preds, relevant)
    print(f"  Association Rules (sum confidence over last-4w items)  MAP@5={m:.4f} (n={n})")
    rows_out.append({"stage": "AR_model", "split": split_name, "variant": "assoc_rules_sum_confidence_last4w",
                      "MAP5": m, "n_eval": n})

    preds_pop = {u: top5_pop for u in val_users}
    m_pop, n_pop = evaluate(preds_pop, relevant)
    rows_out.append({"stage": "AR_reference", "split": split_name, "variant": "popularity_recent4w_unique",
                      "MAP5": m_pop, "n_eval": n_pop})
    print(f"  (справочно) Popularity recent4w  MAP@5={m_pop:.4f} (n={n_pop})")

    return rules, item_count, pair_count, n_txn, txns


def main():
    rows_out = []
    inter = pd.read_parquet(DATA / "user_interactions.parquet")
    inter["user_id"] = inter["user_id"].astype(str)
    sub = pd.read_csv(DATA / "submission.csv", dtype={"user_id": str})
    sub_users = sub["user_id"].tolist()

    # Названия программ для отчёта
    sched_all = pd.concat([
        pd.read_csv(DATA / "export_arh_11-20-final.csv", low_memory=False),
        pd.read_csv(DATA / "export_arh_21-30-final.csv", low_memory=False),
    ])
    title_lookup = sched_all[sched_all["tv_show_id"] != 0].drop_duplicates("tv_show_id").set_index("tv_show_id")["tv_show_title"]

    splits = {
        "main (train<=20, val21-30)": dict(train_end_week=20, val_start_week=21, val_end_week=30,
                                            sched_val_path=DATA / "export_arh_21-30-final.csv", max_week=20),
        "early (train<=16, val17-20)": dict(train_end_week=16, val_start_week=17, val_end_week=20,
                                             sched_val_path=DATA / "export_arh_11-20-final.csv", max_week=16),
    }

    all_rules = {}
    for name, cfg in splits.items():
        rules, item_count, pair_count, n_txn, txns = run_split(name, inter, sub_users, cfg, rows_out, title_lookup)
        all_rules[name] = (rules, item_count, pair_count, n_txn, txns)

    # --- 5 сильных правил (main split): сортируем по lift, затем support, с названиями ---
    main_rules, main_item_count, main_pair_count, main_n_txn, main_txns = all_rules["main (train<=20, val21-30)"]
    strong = sorted(main_rules, key=lambda r: (-r["lift"], -r["support"]))[:5]
    print("\n=== 5 сильных правил (main split, support>=0.2%, lift>1, топ по lift) ===")
    strong_rows = []
    for r in strong:
        a_title = title_lookup.get(r["antecedent"], "?")
        c_title = title_lookup.get(r["consequent"], "?")
        print(f"  {r['antecedent']} ({a_title}) -> {r['consequent']} ({c_title}): "
              f"support={r['support']:.4f} confidence={r['confidence']:.4f} lift={r['lift']:.3f} "
              f"(count_ab={r['count_ab']}, count_a={r['count_a']}, count_b={r['count_b']})")
        strong_rows.append({
            "antecedent_id": r["antecedent"], "antecedent_title": a_title,
            "consequent_id": r["consequent"], "consequent_title": c_title,
            "support": r["support"], "confidence": r["confidence"], "lift": r["lift"],
            "count_ab": r["count_ab"], "count_a": r["count_a"], "count_b": r["count_b"],
        })

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(strong_rows).to_csv(RESULTS / "exp_assoc_rules_top5.csv", index=False)
    print(f"[saved] {RESULTS/'exp_assoc_rules_top5.csv'}")

    # --- Ручной пересчёт одного правила из сырых счётчиков (первое из топ-5) ---
    r0 = strong[0]
    a, b = r0["antecedent"], r0["consequent"]
    cnt_a = sum(1 for txn in main_txns if a in txn)
    cnt_b = sum(1 for txn in main_txns if b in txn)
    cnt_ab = sum(1 for txn in main_txns if a in txn and b in txn)
    support_manual = cnt_ab / main_n_txn
    confidence_manual = cnt_ab / cnt_a
    lift_manual = confidence_manual / (cnt_b / main_n_txn)
    print(f"\n=== Ручной пересчёт правила {a} -> {b} из сырых счётчиков (main split) ===")
    print(f"  N (транзакций) = {main_n_txn}")
    print(f"  count(A={a}) = {cnt_a}")
    print(f"  count(B={b}) = {cnt_b}")
    print(f"  count(A,B) = {cnt_ab}")
    print(f"  support = count(A,B)/N = {cnt_ab}/{main_n_txn} = {support_manual:.6f}")
    print(f"  confidence = count(A,B)/count(A) = {cnt_ab}/{cnt_a} = {confidence_manual:.6f}")
    print(f"  lift = confidence/(count(B)/N) = {confidence_manual:.6f}/({cnt_b}/{main_n_txn}) = {lift_manual:.6f}")
    print(f"  сверка с кодом: support={r0['support']:.6f}, confidence={r0['confidence']:.6f}, lift={r0['lift']:.6f}")
    assert abs(support_manual - r0["support"]) < 1e-9
    assert abs(confidence_manual - r0["confidence"]) < 1e-9
    assert abs(lift_manual - r0["lift"]) < 1e-9
    print("  [OK] ручной пересчёт совпал с кодом с точностью до 1e-9")

    rows_out.append({"stage": "AR_manual_check", "split": "main (train<=20, val21-30)",
                      "variant": f"rule {a}->{b} manual vs code match", "value": "OK"})

    # --- Сохранение и обновление общей таблицы ---
    df_new = pd.DataFrame(rows_out)
    df_new.to_csv(RESULTS / "exp_assoc_rules.csv", index=False)
    print(f"\n[saved] {RESULTS/'exp_assoc_rules.csv'}")

    full_path = RESULTS / "exp_stage1_full.csv"
    df_full_old = pd.read_csv(full_path) if full_path.exists() else pd.DataFrame()
    df_full_new = pd.concat([df_full_old, df_new], ignore_index=True)
    df_full_new.to_csv(full_path, index=False)
    print(f"[updated] {full_path} (append AR rows)")

    sorted_path = RESULTS / "exp_stage1_sorted.csv"
    df_map = df_full_new[df_full_new["MAP5"].notna()].sort_values("MAP5", ascending=False)
    df_map.to_csv(sorted_path, index=False)
    print(f"[updated] {sorted_path}")

    print("\n=== Таблица AR + справочные строки (отсортирована по MAP@5) ===")
    df_ar = df_new[df_new["MAP5"].notna()].sort_values("MAP5", ascending=False)
    print(df_ar[["stage", "split", "variant", "MAP5", "n_eval"]].to_string(index=False))


if __name__ == "__main__":
    main()
