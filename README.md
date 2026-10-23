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

## Финальный сабмит (зафиксирован 23.10.2026)

- **Файл:** `submission.csv` (корень; копия `data/submission_rrf2_a_ppr05_als.csv`)
- **Метод:** RRF (rank-score fusion, k=60) из трёх ранжирований на общем пуле топ-500
  (pop×log(1+число эфиров в 31–42)):
  1. `pop_airings` — популярность (уникальные зрители, недели 27–30) × log(1+эфиры);
  2. `PPR` — персонализированный PageRank по двудольному графу пользователь–программа
     (веса рёбер log(1+секунды), row-normalized транзишены в обе стороны, restart-вектор =
     программы пользователя пропорционально весам, **restart = 0.5**, 30 итераций, батч через
     `scipy.sparse`, `SEED=42`, implicit ALS f=64/reg=0.05/iters=20);
  3. `ALS` — implicit ALS, ранжирование по скору в пуле.
- **Обучение:** недели 11–30 (весь лог). **Тестовые просмотры (31–42) не используются** —
  расписание 31–42 применяется только для связей программа–канал–жанр и числа эфиров.
- **Kaggle: public MAP@5 = 0.32691, private MAP@5 = 0.31441** (лучший результат проекта).
- Генерация: `uv run python artifacts/scripts/exp_rrf2_submit.py`
  (variant `a_ppr05_als` из `submission_rrf2_a_ppr05_als.csv`).

Правило смены финала: новый сабмит становится финалом, только если он лучше и по public,
и по private (минимизация риска подгонки под половину теста).

## Структура репозитория

```
submission.csv             — ФИНАЛЬНЫЙ сабмит (RRF k=60: pop×airings + PPR restart=0.5 + ALS)
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
  exp_rrf2_submit.py     — ФИНАЛЬНЫЕ сабмиты: RRF с PPR restart=0.5 (вариант a — финал)
  examples.py            — §7: тяжёлый/лёгкий пользователь, Top-5 каждого метода
docs/facts.md            — факты EDA (шаг 2)
docs/improve_plan.md     — план улучшений (этапы §4A–E)
reports/join_checks.txt  — проверки джойна
reports/examples.md      — §7 примеры рекомендаций
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

**Финал: `submission.csv` — RRF k=60 (pop×log(1+airings) + PPR restart=0.5 + ALS), Kaggle
public 0.32691 / private 0.31441.** Реализовано и оценено на Kaggle 13 сабмитов; локальная
строгая валидация (MAP@5 на неделях 21–30 и 17–20) использовалась для отсева, окончательные
решения принимались по Kaggle public+private. Ключевые наблюдения: (1) рост restart PPR
0.15→0.5 подтвердился на Kaggle (+0.019 public); (2) чистые повторы истории и CBF по
контентным токенам провалились на Kaggle, вопреки локальной валидации; (3) расхождение
локальной валидации и Kaggle — отдельно задокументированное ограничение (README, отчёт).
