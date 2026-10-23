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

## Финальный сабмит (зафиксирован 23.10.2026, раунд exp4)

- **Файл:** `submission.csv` (корень; копия `data/submission_exp4_k10.csv`)
- **Метод:** RRF (rank-score fusion, **k=10**) из трёх ранжирований на общем пуле топ-500
  (pop×log(1+число эфиров в 31–42)):
  1. `pop_airings` — популярность (уникальные зрители, недели 27–30) × log(1+эфиры);
  2. `PPR` — трёхдольный граф пользователь–программа–атрибут, **restart = 0.5**
     (α = 0.5), **attr_weight = 0.5**, без decay, 30 итераций, батч через
     `scipy.sparse`, `SEED=42`;
  3. `ALS` — implicit ALS, f=64 / reg=0.05 / 20 итераций, ранжирование по скору в пуле.
  Веса fusion: w = (1, 2, 1) (популярность, PPR, ALS).
- **Обучение:** недели 11–30 (весь лог). **Тестовые просмотры (31–42) не используются** —
  расписание 31–42 применяется только для эфиров/каналов/жанров и числа эфиров.
- **Kaggle: public MAP@5 = 0.34586, private MAP@5 = 0.32383** (run-id 57554720,
  лучший результат проекта; прежний финал 0.34487/0.32361 — run-id 57544938).
- Генерация: `uv run python artifacts/scripts/exp4_submit.py k10`.

Правило смены финала: новый сабмит становится финалом, только если он лучше и по public,
и по private (минимизация риска подгонки под половину теста). exp4_k10 выполнил правило
(+0.00099 / +0.00022 против exp3_e3b); полный стек exp4 (winner, 0.34591/0.31580) —
нет, поэтому отклонён.

## Структура репозитория

```
submission.csv             — ФИНАЛЬНЫЙ сабмит (exp4_k10: RRF k=10, w=(1,2,1))
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
docs/facts.md            — факты EDA (шаг 2)
docs/improve_plan.md     — план улучшений (этапы §4A–E)
docs/research/claim-ledger.md — реестр утверждений (C1.x, C2.x) со статусами и источниками
reports/join_checks.txt  — проверки джойна
reports/examples.md      — §7 примеры рекомендаций (исходный объём, сохранён)
reports/examples_full.md — §7 примеры рекомендаций (все 7 методов)
reports/report.md        — отчёт для экзамена §1–§9 (+PDF см. §9)
experiments/             — ноутбуки экспериментов (артефакты раундов exp3/exp4)
results/tables/          — все таблицы экспериментов (validation, diagnostics*, exp_stage1*,
                           exp_rrf*, exp_grid_rp3_ease_ppr*, exp_ppr, exp_assoc_rules, exp_cbf_hw2)
data/submission_*.csv    — все отправленные варианты сабмитов (не перезаписываются)
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

**Финал: `submission.csv` = `data/submission_exp4_k10.csv` — RRF k=10, w=(1,2,1) над
pop×log(1+airings) + PPR (трёхдольный, restart=0.5, attr_weight=0.5, без decay) + ALS
64/0.05/20; Kaggle public 0.34586 / private 0.32383 (run-id 57554720).** На Kaggle 22
сабмита (см. Приложение A отчёта); локальная строгая валидация (MAP@5 на неделях 21–30
и 17–20) использовалась только как отсев («не хуже референса на обоих сплитах»),
окончательные решения принимались по Kaggle public+private. Ключевые наблюдения:
(1) прирост финала над предыдущим финалом даёт один k: 20 → 10 (+0.00099/+0.00022);
(2) расхождение знаков локальной валидации и Kaggle задокументировано в 4 случаях
(C1.4): локально лучший PPR restart=0.7/aw=0.7 на Kaggle хуже (0.34122/0.31624);
(3) чистые повторы истории и CBF по контентным токенам провалились на Kaggle, вопреки
локальной валидации; (4) чистый PPR на сплите main чуть выше гибрида локально, но на
Kaggle гибрид сильнее на +0.0228 public.
