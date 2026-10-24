# sweet.tv — TV program recommender (MAP@5)

Решение экзаменационного квиза (GSB, дедлайн 24.10.2026 19:00 МСК) по конкурсу
[sweet.tv — TV program recommender](https://www.kaggle.com/competitions/sweettv-tv-program-recommender)
(slug `sweettv-tv-program-recommender`, метрика MAP@{K}, K=5).

## Среда и воспроизведение

- Python 3.14.8 (`uv`, проект `sweettv-recsys`, lock-файл `uv.lock`), macOS arm64.
- Установка: `uv sync` (подтянет `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `implicit`, `scipy`).
- Seed везде фиксирован: `SEED = 42` (numpy, `AlternatingLeastSquares(random_state=42)`).
- Данные: `kaggle competitions download -c sweettv-tv-program-recommender -p data/`, затем распаковать в `data/` (5 CSV).

Порядок запуска (из корня репозитория):

```bash
uv sync
uv run python artifacts/scripts/eda.py              # шаг 2: EDA -> docs/facts.md (~27 с)
uv run python artifacts/scripts/join_interactions.py # шаг 3: джойн -> data/user_interactions.parquet (~9 с)
uv run python artifacts/scripts/final_models.py      # шаг 4: валидация всех методов -> results/tables/validation.csv (~165 с)
uv run python artifacts/scripts/examples.py          # §7: примеры 2 пользователей -> reports/examples.md
uv run python artifacts/scripts/make_submission.py   # генерация data/submission_final.csv
```

Сабмит на Kaggle:

```bash
kaggle competitions submit -c sweettv-tv-program-recommender \
  -f data/submission_final.csv -m "baseline popularity recent4w"
kaggle competitions submissions -c sweettv-tv-program-recommender
```

## Финальный сабмит (exp5 candidate-only)

- **Файл:** `submission.csv` (корень; копия `data/submission_exp5_candidate_map_tuned.csv`)
- **Метод:** candidate-only RRF (k=60) по 15 источникам: tripartite PPR (attr_weight=0.5), LightFM WARP (binary, epochs=15), ALS (f=64, reg=0.05, it=20), repeat history (top-150), recent pop (top-50), TV time-slot affinity (top-100), user×channel affinity (top-500), airing-popularity (top-500), schedule propensity (max/mean/sum/logsumexp), threshold-specific LightFM (0.5/0.8), RP3beta/EASE. Веса RRF подобраны **только на early temporal split** (history≤16→target17–20) по MAP@5 через coordinate search с grid {0,0.5,1,2,4,8,16}, 4 прохода. **LightGBM/LGBMRanker не обучался**.
- **Kaggle:** public MAP@5 **0.41874**, private MAP@5 **0.39477** (run-id 57593168).
  Это выше прежнего exp4_k10 (0.34586 / 0.32383, run-id 57554720) на обеих половинах.
- **Формат:** 1,260 пользователей, 5 уникальных show IDs на строку, все IDs из расписания 31–42.
- **Генератор кандидатов:** `uv run python artifacts/scripts/exp5_candidate_recall.py expanded9 final`;
  **Файл сабмита:** `uv run python artifacts/scripts/exp5_make_candidate_submission.py data/exp5_candidates_final_expanded9.parquet results/tables/exp5_candidate_map_weights_propensity_widegrid.csv data/submission_exp5_candidate_map_tuned.csv 0.8`

Корневой `submission.csv` **обновлён** и содержит этот сабмит. Правило смены лучшего результата — одновременно выше public и private — выполнено run-id 57593168.
Reranker (LightGBM binary / LGBMRanker) остаётся заблокирован: пользователь подтвердил требование main Recall@500≥0.80, а лучшая измеренная main Recall@500 candidate-only конфигурация 0.6362.

## Структура репозитория

```
submission.csv             — **ФИНАЛЬНЫЙ сабмит** (exp5 candidate-only MAP5-tuned RRF)
data/submission_exp5_candidate_map_tuned.csv — дубликат финального сабмита
data/submission_exp4_k10.csv — legacy exp4_k10 (0.34586 / 0.32383)
artifacts/scripts/
  eda.py                 — потоковый EDA (stdlib csv), пишет docs/facts.md
  join_interactions.py   — джойн лог×расписание (канал + overlap интервалов), -> user_interactions.parquet
  debug_join.py          — сверка типов user_id и повторный пересчёт пересечений (шаг 3, проверка)
  validate_baseline.py   — первые версии бейзлайна/CBF (сохранён для истории итераций)
  final_models.py        — валидация: Popularity, CBF, ALS (implicit), Hybrid rank fusion
  strict_validation.py   — строгая валидация (≥50% длительности / топ-20 по секундам), 8 методов
  exp_stage1.py          — этап 1 плана: окна/decay популярности, alpha, recall@K, ALS-пулы
  exp_ppr.py             — персонализированный + глобальный PageRank (по образцу HW6)
  exp_assoc_rules.py     — association rules (по образцу HW4), ручной пересчёт правила
  exp_cbf_hw2.py         — CBF (по образцу HW2): cosine + Jaccard×slotpop для cold-кандидатов
  exp_rrf_hybrid.py      — RRF k=60 + greedy ablation компонент (PPR/AR/ALS/CBF-new)
  exp_rrf_submit.py      — сабмиты RRF (best/ppr/ar)
  exp_grid_rp3_ease_ppr.py / exp_grid_rp3_ease_ppr_extra.py — grid: RP3beta, EASE, PPR restart
  exp_rrf2_submit.py     — сабмиты RRF с PPR restart=0.5 (вариант a — прежний финал)
  exp3_*.py              — раунд exp3: трёхдольный PPR (e1), decay (e2), сетка RRF (e3),
                           эфирный буст (e4), сабмиты (submit, submit_e3b)
  exp4_*.py              — раунд exp4: сетка ALS (e5), сетка PPR + сходимость (e6),
                           сетка fusion (e7), пул (e8), примеры §7 (examples_full),
                           генератор сабмитов (submit), финальная оценка (final_eval)
  examples.py            — §7: тяжёлый/лёгкий пользователь, Top-5 каждого метода
  exp5_*.py              — раунд exp5: rich interactions, candidate recall, fusion tuning, MAP tuning
  exp5_build_features.py — leakage-safe feature builder для early/main
  exp5_tune_candidate_fusion.py — early-only RRF weight tuning (Recall@K / MAP@5)
  exp5_tune_candidate_map.py — early-only MAP@5 tuning (base/widegrid/propensity)
  exp5_build_rich_interactions.py — threshold-free rich overlap table
  exp5_reference_check.py — reference notebook cross-check
  exp5_build_features.py — leakage-safe feature tables для early/main
  exp5_eval_rolling_fusion.py — rolling-window fusion eval
  exp5_candidate_diagnostics.py — per-source candidate recall diagnostics
  exp5_reference_check.py — reference notebook cross-check
  exp5_make_candidate_submission.py — генерация сабмита из candidate parquet
  exp5_eval_rolling_fusion.py — rolling-window fusion eval
  exp5_candidate_diagnostics.py — per-source candidate recall diagnostics
  exp5_reference_check.py — reference notebook cross-check
  exp5_make_candidate_submission.py — генерация сабмита из candidate parquet
  exp5_eval_rolling_fusion.py — rolling-window fusion eval
docs/facts.md            — факты EDA (шаг 2)
docs/improve_plan.md     — план улучшений (этапы §4A–E)
docs/research/claim-ledger.md — реестр утверждений (C1.x–C19) со статусами и источниками
reports/join_checks.txt  — проверки джойна
reports/examples.md      — §7 примеры рекомендаций (исходный объём, сохранён)
reports/examples_full.md — §7 примеры рекомендаций (все 7 методов)
reports/report.md        — отчёт для экзамена §1–§9 (+PDF см. §9)
experiments/             — ноутбуки экспериментов (артефакты раундов exp3/exp4/exp5)
results/tables/          — все таблицы экспериментов (validation, diagnostics*, exp_stage1*,
                           exp_rrf*, exp_grid_rp3_ease_ppr*, exp_ppr, exp_assoc_rules, exp_cbf_hw2)
data/submission_*.csv    — все отправленные варианты сабмитов (не перезаписываются)
data/exp5_features_early.parquet / data/exp5_features_main.parquet — leakage-safe feature tables
```

## Правила эксперимента (кратко; подробности — в отчёте)

- **Джойн:** канал (`vsetv_id` ≡ `channel_id`, подтверждено проверкой сдвигов часовых поясов —
  максимум совпадений ровно на сдвиге 0 ч) + пересечение интервалов `[start, stop)` просмотра
  и `[start_air, start_air+duration)` эфира. Порог «просмотрено»: пересечение ≥ 300 с **или**
  ≥ 50% длительности программы.
- **`tv_show_id = 0`** — неразмеченный эфир, из джойна исключён (не может стать item'ом).
- **Валидация:** train = недели 11–20, holdout = недели 21–30 (строго раньше теста 31–42,
  порядок сохранён). Пул кандидатов holdout = программы, идущие в эфир в 21–30 по расписанию
  (`export_arh_21-30-final.csv`, `tv_show_id != 0`). Для финального сабмита пул — `export_arh_31-42-final.csv`.
  Повторы ранее просмотренных программ разрешены — одинаково для всех методов.
- **MAP@5:** AP делится на `min(число релевантных, 5)` — как на Kaggle.
- **Cold-start fallback:** популярность (Recent 4w) — единый fallback для всех методов.
- **CBF ничьи:** при равенстве косинусной близости (одинаковый жанровый вектор) ранжирование
  по вторичному ключу — популярности программы. Это часть метода, а не случайность sort.
- **Ограничение:** 66.1% пула кандидатов holdout и 33.7% релевантных в holdout — программы,
  не встречавшиеся в train (11–20). ALS и CBF принципиально не могут их оценить по данным
  обучения; это основная причина, почему популярность остаётся сильнейшим бейзлайном на
  этом хорошо документированном разрезе.

## Статус

**Финальный:** `submission.csv` — RRF k=60 candidate-only MAP5-tuned, public 0.41874 / private 0.39477 (run-id 57593168). Он превосходит exp4_k10 на обеих половинах. `submission.csv` **перезаписан** и содержит этот сабмит. `data/submission_exp4_k10.csv` сохранён как legacy.
Валидация main (11–20→21–30) и rolling splits выполнены только с прошлыми просмотрами;
просмотры 31–42 не использованы. Main Recall@500 остаётся ниже подтверждённого порога
0.80, поэтому LightGBM baseline и LGBMRanker не обучались.

## Сводная таблица результатов

| Метод | Public MAP@5 | Private MAP@5 | Kaggle Run ID | Тип |
|-------|--------------|---------------|---------------|-----|
| exp5 candidate-only MAP5-tuned RRF | **0.41874** | **0.39477** | 57593168 | candidate-only RRF |
| exp5 LightFM threshold 0.8 | 0.40581 | 0.38178 | 57593473 | candidate-only |
| exp5 LightFM threshold 0.5 | 0.38093 | 0.36288 | 57593488 | candidate-only |
| exp4_k10 (prev final) | 0.34586 | 0.32383 | 57554720 | RRF k=10 (pop/PPR/ALS) |
| exp3_e3b | 0.34487 | 0.32361 | 57544938 | RRF k=20 |
| exp3_e3 | 0.34368 | 0.32311 | 57544807 | RRF k=20 |
| exp3_e2 | 0.32261 | 0.29323 | 57544803 | tripartite PPR + decay |
| exp3_e1 | 0.32306 | 0.29313 | 57544801 | tripartite PPR |
| exp3_disc | 0.33331 | 0.31547 | 57544809 | RRF k=60 |
| exp4_winner (full stack) | 0.34591 | 0.31580 | 57554681 | full stack |
| exp4 PPR r=0.7 | 0.34122 | 0.31624 | 57554703 | PPR restart=0.7 |
| exp4 PPR r=0.5 | 0.34320 | 0.31550 | 57554681? | PPR restart=0.5 |
| exp3 E2 | 0.32261 | 0.29323 | 57544803 | tripartite + decay |
| exp3 E3 | 0.34368 | 0.32311 | 57544807 | RRF k=20 |

## Команды для воспроизведения

**Полный пайплайн (до сабмита):**
```bash
uv sync
uv run python artifacts/scripts/eda.py
uv run python artifacts/scripts/join_interactions.py
uv run python artifacts/scripts/exp5_build_rich_interactions.py
uv run python artifacts/scripts/exp5_candidate_recall.py expanded9 final
uv run python artifacts/scripts/exp5_build_features.py main
uv run python artifacts/scripts/exp5_tune_candidate_map.py propensity_widegrid
uv run python artifacts/scripts/exp5_make_candidate_submission.py
# -> submission.csv в корне (финальный сабмит)
```

**Сабмит на Kaggle:**
```bash
kaggle competitions submit -c sweettv-tv-program-recommender \
  -f submission.csv -m "exp5 candidate-only MAP5-tuned RRF"
kaggle competitions submissions -c sweettv-tv-program-recommender
```

**Параметры модели (Seed=42 везде):**
- RRF: k=60, 15 источников, веса из early MAP@5 tuning (propensity_widegrid)
- Tripartite PPR: attr_weight=0.5, restart=0.5, n_iter=30, alpha=0.5
- LightFM: loss=warp, epochs=15, num_threads=4, binary interactions
- ALS: factors=64, regularization=0.05, iterations=20, num_threads=4
- PPR restarts: user items only (Vi=0), attr_weight=0.5
- Candidate pool: top-200 LightFM / top-500 others / union dedup → restrict to target schedule
- Target schedule filtering: weeks 31–42 only (`export_arh_31-42-final.csv`, ISO week filter)