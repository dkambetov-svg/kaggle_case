# Claim Ledger — sweettv-tv-program-recommender

## Эскалации: Tier 1: 0/6, Tier 2: 0/3, Tier 3: 0/2

## C1 (supported)
**Type**: empirical
**Status**: supported
**Claim**: Взвешенный RRF-фьюжн (k=20, w_pop=1, w_ppr=2, w_als=1) поверх трёхдольного атрибутного PPR (attr_weight=0.5, без decay) улучшает MAP@5 на Kaggle против прежнего финала (RRF k=60 + двудольный PPR restart=0.5 + ALS): public 0.34487 против 0.32691 и private 0.32361 против 0.31441.
**Scope and exclusions**: только конкурс sweettv-tv-program-recommender, тестовые недели 31–42; утверждение о сравнении двух конкретных конфигураций, не о всеобщем превосходстве фьюжн над чистыми моделями.
**Assumptions**: валидационный отбор (строгая релевантность, два сплита) корректен; пул топ-500 и формат сабмита идентичны для обоих вариантов; просмотры 31–42 нигде не использованы.
**Closest prior work and gap**: прежний финал (commit f326c13, public 0.32691/private 0.31441) — та же архитектура без атрибутов в графе и с другими весами RRF; гап: не было оценки атрибутного графа и подбора весов на Kaggle.
**Falsifier or counterexample**: сабмит этой же конфигурации на новом фиксированном тесте даёт private ниже 0.31441; либо воспроизведение скриптом exp3_submit_e3b.py даёт иной файл.
**Required support**: сабмит с public и private выше прежнего финала + воспроизводимый скрипт и таблицы.
**Current support**: Сабмит Kaggle run-id 57544938: public 0.34487, private 0.32361; прежний финал run-id 57543402: 0.32691/0.31441 — лучше по обеим метрикам. Генерация: artifacts/scripts/exp3_submit_e3b.py (проверка формата пройдена: 1260 строк, 5 уникальных id из расписания 31–42).
**Evidence sources**:
- kaggle-run:57544938
- kaggle-run:57543402
- artifacts/scripts/exp3_submit_e3b.py
- results/tables/exp3_e3_ppr_e1.csv:2
**Counterevidence and failures**: локально fusion хуже чистого PPR_best (results/tables/exp3_e3.csv:2 против :3) — знак расхождения локальной валидации и Kaggle задокументирован; чистые PPR-сабмиты exp3_e1/e2 на Kaggle хуже прежнего финала (0.32306/0.32261 < 0.32691).
**Permitted manuscript wording**: Итоговый метод достиг MAP@5 0.34487 public / 0.32361 private (run-id 57544938), превосходя прежний финал 0.32691/0.31441 (run-id 57543402) по обеим половинам теста.
**Next cheapest decisive action**: воспроизвести файл скриптом exp3_submit_e3b.py и свести md5 с отправленным data/submission_exp3_e3b.csv.
**Depends on**: C1.1, C1.3

## C1.1 (supported)
**Type**: empirical
**Status**: supported
**Claim**: Включение атрибутов (канал, категория, жанры) в граф PPR (трёхдольный граф, attr_weight=0.5) улучшает MAP@5 относительно двудольного PPR на локальной валидации: main 0.542712 против 0.533331, early 0.464273 против 0.457684.
**Scope and exclusions**: локальные сплиты main (train ≤20, val 21–30) и early (train ≤16, val 17–20); утверждение о знаке эффекта для этого графа и пула, magnitude не переносится на Kaggle автоматически.
**Assumptions**: контроль attr_weight=0 воспроизводит двудольный PPR бит-в-бит (подтверждено совпадением 0.533331/0.457684 с прежней таблицей); нормировка item→attr/item→user до смешивания корректна.
**Closest prior work and gap**: прежний двудольный PPR (restart=0.5) — без атрибутных рёбер; гап: влияние атрибутной структуры на MAP@5 не измерено.
**Falsifier or counterexample**: контрольный прогон attr_weight=0 на тех же сплитах даёт иной результат, чем 0.533331/0.457684; либо прирост исчезает после переупорядочивания атрибутов (пермютированный контроль).
**Required support**: таблица с контролем и тремя уровнями aw на обоих сплитах.
**Current support**: results/tables/exp3_e1.csv:2 (aw=0.5 → 0.542712/0.464273), :5 (aw=0 контроль → 0.533331/0.457684); монотонный рост по aw (строки 2–5).
**Evidence sources**:
- results/tables/exp3_e1.csv:2
- results/tables/exp3_e1.csv:5
- artifacts/scripts/exp3_e1_attr_ppr.py
**Counterevidence and failures**: на Kaggle чистый атрибутный PPR (run-id 57544801, 0.32306/0.29313) хуже прежнего финала 0.32691/0.31441 — прирост локально не переносится на Kaggle в чистом виде; пер-юзерный счётчик «не видел» даёт те же 59/22 слотов, релевантных 16/2 против 17/3 у контроля (0.533331), т.е. атрибуты не добавляют холодных попаданий.
**Permitted manuscript wording**: На локальной валидации атрибутный PPR (attr_weight=0.5) превосходит двудольный контроль на обоих сплитах (0.5427 против 0.5333 и 0.4643 против 0.4577); на Kaggle в чистом виде прирост не подтверждается.
**Next cheapest decisive action**: ничего — знак на Kaggle проверен сабмитами 57544801 (против 57543402) в конфигурации fusion (57544809/57544938).
**Depends on**: —

## C1.2 (weakened)
**Type**: empirical
**Status**: weakened
**Claim**: Временное затухание весов рёбер (exp(-Δ/τ), τ=8) улучшает PPR-компонент относительно взвешивания без decay.
**Scope and exclusions**: локальные сплиты, лучший E1-граф (aw=0.5); Δ — недели от просмотра до конца обучения.
**Assumptions**: decay применяется к агрегированному весу log1p(Σ overlap·exp(−Δ/τ)); restart берётся по сырым весам (per_user_raw).
**Closest prior work and gap**: E2-таблица (results/tables/exp3_e2.csv); гап: знак эффекта на Kaggle не был проверен до раунда exp3.
**Falsifier or counterexample**: на Kaggle fusion с decayed-PPR лучше, чем тот же fusion с PPR без decay.
**Required support**: лучший decayed-вариант не хуже baseline на обоих сплитах + положительный знак на Kaggle.
**Current support**: Локально tau=8 restart=raw: 0.543274/0.464525 против baseline 0.542712/0.464273 (results/tables/exp3_e2.csv:2 и :3), прирост +0.0006/+0.0003 — в пределах практического шума.
**Evidence sources**:
- results/tables/exp3_e2.csv:2
- results/tables/exp3_e2.csv:3
- kaggle-run:57544803
- kaggle-run:57544801
**Counterevidence and failures**: На Kaggle fusion с decay (exp3_e3, run-id 57544807: 0.34368/0.32311) хуже того же fusion без decay (exp3_e3b, run-id 57544938: 0.34487/0.32361) — знак отрицательный; чистые сабмиты: E2 0.32261/0.29323 против E1 0.32306/0.29313 — почти равны, private чуть выше у decay. Также найден и исправлен баг переполнения UInt32 (первый прогон E2 дал мусор 0.2893/0.0236, перезаписан исправленным).
**Permitted manuscript wording**: Локальный прирост от decay крайне мал (+0.0006 по среднему), а на Kaggle тот же fusion с decay хуже fusion без decay (0.34368 против 0.34487 по public); эффект свежести не подтверждается.
**Next cheapest decisive action**: зафиксировать отказ от decay в финальной конфигурации (уже выполнено: финал без decay).
**Depends on**: C1.1

## C1.3 (supported)
**Type**: empirical
**Status**: supported
**Claim**: Разумный вес PPR-компонента (w_ppr=2 при k=20) в взвешенном RRF даёт лучший fusion-результат; при этом fusion взвешивается лучше, чем равные веса k=60.
**Scope and exclusions**: сетка 54 комбинаций (w_pop ∈ {1,1.5,2}, w_ppr ∈ {1,1.5,2}, w_als ∈ {0,0.5,1}, k ∈ {20,60}); пул топ-500; компонент PPR = E1+E2 (aw=0.5, tau=8, raw).
**Assumptions**: локальный порядок fusion-вариантов частично переносится на Kaggle; PPR-компонент фиксирован при переборе весов.
**Closest prior work and gap**: прежний финал использовал равные веса 1/1/1 при k=60 без перебора; гап: влияние весов и k не измерено.
**Falsifier or counterexample**: на Kaggle fusion с другими весами/k стабильно превосходит конфигурацию k=20 w=(1,2,1).
**Required support**: таблица сетки с лучшей точкой + сабмит, превосходящий прежний финал по обеим метрикам.
**Current support**: Лучшая точка сетки: 0.535673/0.462592 (results/tables/exp3_e3.csv:3) против контроля прежнего финала локально 0.501455/0.431830 (exp3_e3.csv:44). Сабмит run-id 57544807 (E3 с decay): 0.34368/0.32311; run-id 57544938 (E3b без decay): 0.34487/0.32361 — оба лучше прежнего финала по обеим метрикам; абляция pop+PPR без ALS худшая из топа (0.528146/0.448850, exp3_e3.csv:8).
**Evidence sources**:
- results/tables/exp3_e3.csv:3
- results/tables/exp3_e3.csv:44
- results/tables/exp3_e3.csv:8
- kaggle-run:57544938
- kaggle-run:57544807
**Counterevidence and failures**: локально каждый fusion-вариант хуже чистого PPR_best (exp3_e3.csv:2: 0.543274/0.464525) — локальная валидация недооценивает fusion; на Kaggle чистый PPR при этом хуже fusion (0.32306 против 0.34487). Порядок fusion-вариантов внутри сетки на Kaggle не проверялся (отправлен только лучший по локальной оценке).
**Permitted manuscript wording**: Перебор весов RRF и k на 54 комбинациях подобрал конфигурацию k=20, w=(1,2,1), которая на Kaggle дала 0.34487/0.32361 против 0.32691/0.31441 у прежнего финала с равными весами; локально fusion при этом хуже чистого PPR, знак переносится на Kaggle инвертированно.
**Next cheapest decisive action**: отправка нескольких топовых точек сетки (w_ppr=1.5/2 при k=20) для проверки внутри-сеточного порядка — исчерпан лимит сабмитов раунда.
**Depends on**: C1.1, C1.2

## C1.4 (weakened)
**Type**: empirical
**Status**: weakened
**Claim**: Локальная строгая валидация предсказывает относительный порядок методов на Kaggle (использовалась для отсева вариантов перед сабмитом).
**Scope and exclusions**: сплиты main/early против public/private Kaggle для вариантов раундов exp/exp3.
**Assumptions**: метрика и пул локально эквивалентны Kaggle-разметке (те же правила join, строгая релевантность, MAP@5).
**Closest prior work and gap**: прежние расхождения (repeat, window=2w, fusion, CBF); гап: систематическая оценка корреляции знаков.
**Falsifier or counterexample**: вариант, локально худший контроля на обоих сплитах, на Kaggle окажется лучше — либо локально лучший, на Kaggle хуже.
**Required support**: ≥5 пар вариантов с согласованным знаком локально и на Kaggle.
**Current support**: Расхождения задокументированы: чистый PPR локально 0.5427/0.4643 > контроль финала 0.5015/0.4318, на Kaggle 0.32306/0.29313 < 0.32691/0.31441; fusion локально хуже чистого PPR (exp3_e3.csv:3 против :2), на Kaggle fusion лучше (0.34487 против 0.32306); decay локально +0.0006, на Kaggle −0.00119 (public, fusion-пара 57544807 против 57544938). Согласование по знаку подтверждено для атрибутов (C1.1 локально, fusion-сабмиты на Kaggle).
**Evidence sources**:
- results/tables/exp3_e1.csv:2
- results/tables/exp3_e3.csv:2
- results/tables/exp3_e3.csv:3
- kaggle-run:57544801
- kaggle-run:57544807
- kaggle-run:57544938
- kaggle-run:57543402
**Counterevidence and failures**: см. Current support — три расхождения знака в раунде exp3 (чистый PPR, fusion-знак, decay); прежние: repeat, window=2w, CBF-инверсия токенов.
**Permitted manuscript wording**: Локальная валидация использовалась только как отсев (фильтр «не хуже финала на обоих сплитах»), но знаки эффектов расходятся с Kaggle не менее чем в трёх случаях раунда exp3, поэтому окончательные решения принимались по public+private.
**Next cheapest decisive action**: на будущее — собирать таблицу знаков «локальный эффект vs Kaggle-эффект» по каждому сабмиту (13+5 = 18 пар).
**Depends on**: —

## C1.5 (rejected)
**Type**: empirical
**Status**: rejected
**Claim**: Буст по числу эфиров поверх PPR (score = PPR × (1+airings)^a, a ∈ {0.25, 0.5}) как самостоятельный компонент улучшает качество относительно отдельных компонентов pop×airings и PPR.
**Scope and exclusions**: альтернативная замена пары компонентов pop_airings+PPR одним комбинированным; локальные сплиты, оба значения a.
**Assumptions**: airings берутся из расписания val-окна (эквивалент 31–42 в финале); PPR-компонент = E1+E2.
**Closest prior work and gap**: прежний финал держал pop×airings и PPR раздельно в RRF; гап: комбинированный компонент не проверялся.
**Falsifier or counterexample**: вариант с boost локально не хуже контроля на обоих сплитах.
**Required support**: оба варианта a проходят локальный фильтр «не хуже 0.5015/0.4318».
**Current support**: —
**Evidence sources**:
- results/tables/exp3_e4.csv:3
- results/tables/exp3_e4.csv:5
- results/tables/exp3_e3.csv:44
**Counterevidence and failures**: Лучший вариант a=0.25 отдельно: 0.449241/0.385668 (exp3_e4.csv:3), fusion boost+ALS a=0.5: 0.397553/0.318189 (exp3_e4.csv:5) — оба значительно ниже контроля финала 0.501455/0.431830 (exp3_e3.csv:44); провал локального фильтра → сабмит не отправлялся. Референс без буста: 0.512463/0.437065 (exp3_e4.csv:2).
**Permitted manuscript wording**: Идея буста эфиров поверх PPR отвергнута по локальной валидации: даже лучший вариант a=0.25 заметно хуже контроля на обоих сплитах (0.4492/0.3857 против 0.5015/0.4318), на Kaggle не проверялась.
**Next cheapest decisive action**: не предпринимать; при возврате к идее — сначала параметр a < 0.25 на тех же сплитах.
**Depends on**: —

## C1.6 (supported)
**Type**: mathematical
**Status**: supported
**Claim**: Глобально «нетренированных» программ в топ-5 не может быть структурно: пул топ-500 по pop×log(1+airings) целиком состоит из программ, виденных в train (во всех трёх окнах pool500 pos_score=500, «в train не встречено»=0).
**Scope and exclusions**: только текущее правило пула (top-500 по pop×airings с положительным score); иные правила пула могут включать необученные программы.
**Assumptions**: pop считается по train-окну, эфиры по расписанию val/тест-окна; число кандидатов с положительным score ≥ 500 (фактически 1495/1559/1655).
**Closest prior work and gap**: README утверждал 66.1% необученных кандидатов в holdout-пуле (иной разрез — весь пул, не top-500); гап: не было прямой проверки top-500.
**Falsifier or counterexample**: контрольный пересчёт пула покажет ≥1 программу с pop=0 внутри топ-500.
**Required support**: пересчёт состава пула на трёх окнах.
**Current support**: Пересчёт 2026-10-23: main — candidates 6363, pos_score 1495, pool500 pos_score 500, не встречено в train 0; early — 5847/1559/500/0; final — 6636/1655/500/0. Счётчики new_top5_* во всех строках exp3_e1.csv равны 0.
**Evidence sources**:
- results/tables/exp3_e1.csv:2
- results/tables/exp3_e1.csv:5
- experiments/2026-10-23-0525-exp3-round.ipynb
**Counterevidence and failures**: нет; численный вывод пересчёта совпадает с нулевыми столбцами диагностики в exp3_e1.csv.
**Permitted manuscript wording**: По правилу пула топ-500 все кандидаты имеют положительную pop×airings-оценку и поэтому видены в обучении: глобальный счётчик холодных программ в топ-5 всегда 0; осмысленной является только пер-юзерная диагностика (59 слотов / 16 релевантных на main).
**Next cheapest decisive action**: указать структурное ограничение в §5 отчёта (пул и fallback).
**Depends on**: —
