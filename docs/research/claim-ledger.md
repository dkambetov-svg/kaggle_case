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
**Current support**: Расхождения задокументированы: чистый PPR локально 0.5427/0.4643 > контроль финала 0.5015/0.4318, на Kaggle 0.32306/0.29313 < 0.32691/0.31441; fusion локально хуже чистого PPR (exp3_e3.csv:3 против :2), на Kaggle fusion лучше (0.34487 против 0.32306); decay локально +0.0006, на Kaggle −0.00119 (public, fusion-пара 57544807 против 57544938); **раунд exp4**: PPR restart=0.7/aw=0.7 локально +0.0059 mean (exp4_ppr.csv:2 против :8), на Kaggle −0.00365/−0.00737 (57554703 против 57544938) — 4-е расхождение; при этом k=10 изолированно знак согласовал: локально все верхние строки E7 при k=10, на Kaggle +0.00099/+0.00022 (57554720 против 57544938). Согласование по знаку подтверждено для атрибутов (C1.1 локально, fusion-сабмиты на Kaggle) и для k=10 (exp4).
**Evidence sources**:
- results/tables/exp3_e1.csv:2
- results/tables/exp3_e3.csv:2
- results/tables/exp3_e3.csv:3
- results/tables/exp4_ppr.csv:2
- results/tables/exp4_ppr.csv:8
- kaggle-run:57544801
- kaggle-run:57544807
- kaggle-run:57544938
- kaggle-run:57543402
- kaggle-run:57554703
- kaggle-run:57554720
**Counterevidence and failures**: см. Current support — четыре расхождения знака (три раунда exp3 + PPR-этап exp4); прежние: repeat, window=2w, CBF-инверсия токенов.
**Permitted manuscript wording**: Локальная валидация использовалась только как отсев (фильтр «не хуже финала на обоих сплитах»), но знаки эффектов расходятся с Kaggle не менее чем в четырёх случаях (раунды exp3 и exp4), поэтому окончательные решения принимались по public+private.
**Next cheapest decisive action**: на будущее — собирать таблицу знаков «локальный эффект vs Kaggle-эффект» по каждому сабмиту (19+3 пары).
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

## C2 (rejected)
**Type**: empirical
**Status**: rejected
**Claim**: Конфигурация exp4 (ALS f=128/it=20/reg=0.05 + PPR restart=0.7/aw=0.7 alpha=0.3 + RRF k=10 w=(1,2,1), пул-500) превосходит победителя exp3 (exp3_e3b: public 0.34487 / private 0.32361, run-id 57544938) по обеим половинам Kaggle-теста.
**Scope and exclusions**: только этот конкурс и эта конфигурация против exp3_e3b; утверждение о Kaggle public+private, не о локальной валидации (она — отдельное C2.1).
**Assumptions**: формат сабмита, пул топ-500, train 11–30 идентичны exp3_e3b; просмотры 31–42 не использованы; локальный фильтр (C2.1) корректно отбирает варианты.
**Closest prior work and gap**: exp3_e3b (run-id 57544938) — та же архитектура с PPR restart=0.5/aw=0.5, ALS 64/0.05/20, k=20; гап: новые параметры E5–E8 на Kaggle не проверены.
**Falsifier or counterexample**: сабмит exp4-конфигурации даёт public или private ниже 0.34487/0.32361.
**Required support**: сабмит с public и private выше exp3_e3b + воспроизводимый скрипт exp4_submit.py variant=winner.
**Current support**: — (отвергнуто: сабмит не прошло правило обеих половин)
**Evidence sources**:
- kaggle-run:57554681
- kaggle-run:57544938
- artifacts/scripts/exp4_submit.py
**Counterevidence and failures**: Сабмит run-id 57554681: public 0.34591 (выше 0.34487 ✓), но private 0.31580 ниже 0.32361 (−0.00781 ✗) — правило «лучше на обеих половинах» нарушено. Разложение (57554703 ppr-only: 0.34122/0.31624) показывает: провал private в основном из-за PPR restart=0.7/aw=0.7, который локально давал +0.0059 mean (C2.2), на Kaggle — −0.00365/−0.00737 (расхождение знака, пополняет C1.4). Локальное обоснование C2.1 остаётся верным для локальных сплитов, но на Kaggle не подтвердилось.
**Permitted manuscript wording**: Полная конфигурация exp4 на Kaggle улучшила public до 0.34591, но ухудшила private до 0.31580 (run-id 57554681) и поэтому не принята как финальная; изолированный PPR-инкремент также хуже на обеих половинах (0.34122/0.31624, run-id 57554703).
**Next cheapest decisive action**: не предпринимать; финальной принята точка C2.6 (k=10 изолированно).
**Depends on**: C2.1

## C2.1 (supported)
**Type**: empirical
**Status**: supported
**Claim**: Локально конфигурация exp4 (победитель E5–E8: ALS f=128/it=20, PPR restart=0.7/aw=0.7, RRF k=10 w=(1,2,1), пул-500) превосходит референс exp3_e3b (0.536525/0.463176) на обоих сплитах: main 0.547871 (+0.0113), early 0.471534 (+0.0084).
**Scope and exclusions**: сплиты main (train ≤20, val 21–30) и early (train ≤16, val 17–20), строгая релевантность; magnitude не переносится на Kaggle автоматически (C1.4).
**Assumptions**: обе точки (референс и победитель) посчитаны одним кодом пула/метрики; RESTART/SEED=42; референс воспроизведён бит-в-бит в E6 (0.5365245321873032/0.46317616959064295 совпадает с exp3_e3_ppr_e1.csv:2).
**Closest prior work and gap**: референс exp3_e3b; гап: пошаговые улучшения E5–E8 не были сложены в одну проверку против референса.
**Falsifier or counterexample**: пересчёт победителя на тех же сплитах даёт значение ниже 0.536525 или 0.463176 хотя бы на одном сплите.
**Required support**: таблицы E5–E8 с точкой победителя и точкой референса на обоих сплитах.
**Current support**: exp4_fusion.csv:2 (0.547870668815/0.471534, mean 0.509703) против референса exp4_als.csv:6 (0.5365245321873032/0.46317616959064295, mean 0.499850); воспроизведение пула: exp4_pool.csv:2 (pool=500 даёт ровно 0.547871/0.471534 — идентично E7).
**Evidence sources**:
- results/tables/exp4_fusion.csv:2
- results/tables/exp4_als.csv:6
- results/tables/exp4_pool.csv:2
- results/tables/exp3_e3_ppr_e1.csv:2
**Counterevidence and failures**: вклад ALS f=128 — на уровне шума (E5: mean 0.500014 против 0.499850, но early 0.461828 ниже референса 0.463176, exp4_als.csv:2) — почти весь прирост E6+E7; E5 не проходил фильтр «на обоих сплитах» отдельно от остальных стадий.
**Permitted manuscript wording**: Сложенная конфигурация exp4 локально превосходит победителя exp3 на обоих сплитах (+0.0113 main, +0.0084 early, строгая MAP@5); вклад этапа ALS отдельно мал и знак его эффекта нестабилен между сплитами.
**Next cheapest decisive action**: выполнен — победитель отправлен (57554681), правило обеих половин не прошёл (C2); финальной принята точка C2.6.
**Depends on**: C2.2, C2.3, C2.5

## C2.2 (supported)
**Type**: empirical
**Status**: supported
**Claim**: Сильный рестарт PPR (restart=0.7, alpha=0.3) с высоким весом атрибутов (attr_weight=0.7) даёт лучший fusion-результат в сетке restart × attr_weight: mean 0.505766 против 0.499850 у референтной точки (restart=0.5/aw=0.5), на обоих сплитах лучше.
**Scope and exclusions**: сетка 12 точек (restart ∈ {0.3,0.5,0.7} × aw ∈ {0.4,0.5,0.6,0.7}), fusion k=20 w=(1,2,1) с ALS 64/0.05/20, пул-500; чистый PPR отдельно оптимален при других параметрах (restart=0.5) — выбор по fusion.
**Assumptions**: нулевая decay-функция (E1-конфигурация); 30 итераций PPR достаточно (C2.4); референтная точка воспроизводит exp3_e3b.
**Closest prior work and gap**: E1-сетка только по aw при restart=0.5; гап: рестарт не перебирался.
**Falsifier or counterexample**: точка (0.7, 0.7) в пересчёте ниже референта на обоих сплитах; либо референтная точка не воспроизводит 0.536525/0.463176.
**Required support**: 12-точечная сетка на обоих сплитах с референтной точкой.
**Current support**: exp4_ppr.csv:2 (0.5444587698092933/0.46707259685672387, mean 0.5057656833330086) против exp4_ppr.csv:8 (референт 0.5365245321873032/0.46317616959064295 — совпадает с exp3_e3_ppr_e1.csv:2 бит-в-бит); на ранге PPR отдельно точка (0.7,0.7) не лучшая (standalone mean 0.481822 против 0.503493 у референта, те же строки).
**Evidence sources**:
- results/tables/exp4_ppr.csv:2
- results/tables/exp4_ppr.csv:8
- results/tables/exp3_e3_ppr_e1.csv:2
**Counterevidence and failures**: чистый PPR-компонент при restart=0.7/aw=0.7 заметно хуже референтного (0.4818 против 0.5035 mean) — улучшение достигается только внутри fusion; **на Kaggle не переносится**: изолированный сабмит (run-id 57554703) 0.34122/0.31624 против 0.34487/0.32361 у exp3_e3b — хуже по обеим половинам, при том что локально точка лучше референта на обоих сплитах (расхождение знака, C1.4).
**Permitted manuscript wording**: В сетке 12×2 точка restart=0.7/aw=0.7 даёт лучший fusion (mean 0.5058 против 0.4999 у референта на обоих сплитах), хотя как самостоятельный метод PPR при этих параметрах хуже restart=0.5 — отбор вёлся по fusion-метрике.
**Next cheapest decisive action**: выполнен — вариант отправлен (run-id 57554703) и на Kaggle отвергнут (см. Counterevidence).
**Depends on**: C2.4

## C2.3 (supported)
**Type**: empirical
**Status**: supported
**Claim**: Меньший масштаб RRF (k=10 вместо 20) при весах (1,2,1) поверх лучших ALS/PPR улучшает fusion: mean 0.509703 против 0.505553 у контрольной точки k=20 (оба сплита лучше); топ-4 сетки — все при k=10.
**Scope and exclusions**: сетка 18 комбинаций (k ∈ {10,20,40}, w_ppr ∈ {1.5,2,3}, w_als ∈ {0.5,1}) при PPR E6 и ALS E5; оба сплита.
**Assumptions**: PPR/ALS-компоненты зафиксированы на победителях E6/E5; tie-break — порядок пула.
**Closest prior work and gap**: exp3 E3 сетка содержала только k ∈ {20,60}; гап: k < 20 не проверялся.
**Falsifier or counterexample**: контрольная точка k=20 с теми же компонентами не хуже k=10 на обоих сплитах.
**Required support**: сетка с контролем k=20 и лучшей точкой на обоих сплитах.
**Current support**: exp4_fusion.csv:2 (k=10, mean 0.509703, 0.547870668815/0.471534) против exp4_fusion.csv:6 (k=20 контроль, mean 0.505553); четыре верхних строки сетки — k=10 (exp4_fusion.csv:2-5).
**Evidence sources**:
- results/tables/exp4_fusion.csv:2
- results/tables/exp4_fusion.csv:6
**Counterevidence and failures**: прирост k=10 против k=20 меньше, чем прирост PPR-этапа E6 локально (+0.0041 против +0.0059 mean); **на Kaggle знак подтверждён**: изолированный сабмит k=10 (run-id 57554720) 0.34586/0.32383 против 0.34487/0.32361 — лучше по обеим половинам (C2.6), в то время как составной победитель E7 того же k=10 с новыми компонентами private ухудшил (C2) — эффекты компонентов не аддитивны.
**Permitted manuscript wording**: Уменьшение масштаба RRF до k=10 изолированно дало +0.0041 по среднему локальной строгой MAP@5 против контрольной точки k=20 и подтвердилось на Kaggle: 0.34586/0.32383 против 0.34487/0.32361 (run-id 57554720 против 57544938).
**Next cheapest decisive action**: выполнен — вариант отправлен (57554720) и принят как финальный (C2.6).
**Depends on**: C2.2

## C2.4 (rejected)
**Type**: empirical
**Status**: rejected
**Claim**: Гипотеза E5: увеличение числа эпох (итераций) ALS выше 20 (40–160) улучшает качество ALS-компонента и fusion-результат.
**Scope and exclusions**: итерации {20,40,80,160} × факторы {64,128} × reg {0.01,0.05,0.2}, fusion k=20 w=(1,2,1) с PPR E1, оба сплита.
**Assumptions**: implicit ALS при SEED=42; оценка отдельно (ALS standalone) и внутри fusion.
**Closest prior work and gap**: прежний финал — 20 итераций без перебора; гап: «больше эпох» не проверялся.
**Falsifier or counterexample**: прирост standalone или fusion с итерациями > 20 заметен и знак положителен на обоих сплитах.
**Required support**: монотонный или устойчивый рост MAP@5 с числом итераций.
**Current support**: — (гипотеза отвергнута)
**Evidence sources**:
- results/tables/exp4_als.csv:5
- results/tables/exp4_als.csv:9
- results/tables/exp4_als.csv:10
- results/tables/exp4_als.csv:11
- results/tables/exp4_als.csv:2
- results/tables/exp4_als.csv:6
**Counterevidence and failures**: ALS standalone при f=64 монотонно падает с ростом итераций: it=20 → 0.139857, it=80 → 0.136261, it=160 → 0.132168 (main; exp4_als.csv:5/:9/:10); при f=128 хуже: it=20 → 0.122306, it=160 → 0.104070 (:2/:11). Fusion mean всех точек с it≥80 ниже референса 0.499850 (it=80 f=64: 0.499594; f=128 it=160: 0.498202). Единственные точки выше референса — it=40 (+0.0001…+0.0002, шум) и f=128/it=20 (+0.00016 mean, но early ниже референса). Итог: эпохи больше 20 не помогают; выбранный f=128/it=20 — не «больше эпох», а больше факторов с прежним числом эпох.
**Permitted manuscript wording**: Гипотеза «больше эпох ALS» не подтвердилась: качество компонента монотонно снижается с ростом итераций (0.1399 → 0.1322 при 160), fusion-приросты на уровне шума и знак нестабилен между сплитами; в конфигурацию exp4 взят f=128 при прежних 20 итерациях (лучший fusion-mean, вклад в итоговый прирост мал).
**Next cheapest decisive action**: не предпринимать; при возврате — регуляризация под reg при большем числе эпох.
**Depends on**: —

## C2.5 (supported)
**Type**: empirical
**Status**: supported
**Claim**: Увеличение пула кандидатов сверх топ-500 (1000, 2000) не улучшает fusion-результат при конфигурации победителя exp4: mean 0.509703 (500) против 0.509272 (1000) и 0.508843 (2000).
**Scope and exclusions**: pool ∈ {500,1000,2000} по правилу pop(нед. max−3)×log(1+эфиров), конфигурация E7-победителя; три значения на обоих сплитах.
**Assumptions**: для пула >500 в граф попадают новые программы (main: 3321 узел при 2000 против 2900 при 500) — эффект расширенного пула учтён честно.
**Closest prior work and gap**: прежние раунды — только топ-500; гап: расширение пула не проверялось.
**Falsifier or counterexample**: pool=1000 или 2000 даёт mean выше 0.509703 на обоих сплитах.
**Required support**: три значения пула, оба сплита, та же конфигурация.
**Current support**: exp4_pool.csv:2 (500 → 0.547871/0.471534, mean 0.509703, идентично exp4_fusion.csv:2), :3 (1000 → mean 0.509272), :4 (2000 → mean 0.508843).
**Evidence sources**:
- results/tables/exp4_pool.csv:2
- results/tables/exp4_pool.csv:3
- results/tables/exp4_pool.csv:4
- results/tables/exp4_fusion.csv:2
**Counterevidence and failures**: различия малы (≤0.0009 mean) — эффект близок к шуму; знак на Kaggle не проверен.
**Permitted manuscript wording**: Расширение пула до 1000/2000 не дало роста локальной MAP@5 (0.5097 при 500 против 0.5093/0.5088), поэтому оставлен топ-500 — тот же пул, что и в прежнем финале.
**Next cheapest decisive action**: не предпринимать; пул-500 зафиксирован в exp4_submit.py.
**Depends on**: C2.3

## C2.6 (supported)
**Type**: empirical
**Status**: supported
**Claim**: Изолированное изменение RRF k=20 → k=10 (веса w=(1,2,1), PPR E1 restart=0.5/aw=0.5, ALS 64/0.05/20 — всё остальное как в exp3_e3b) улучшает Kaggle MAP@5 по обеим половинам: public 0.34586 против 0.34487 и private 0.32383 против 0.32361. Принято как финальная конфигурация.
**Scope and exclusions**: только этот конкурс, только пара конфигураций (exp4_k10 против exp3_e3b); приросты малы (+0.00099/+0.00022) и не утверждаются для других тестов.
**Assumptions**: оба файла сгенерированы при одинаковом train 11–30, пуле-500 и формате (отличие — только k в fusion); просмотры 31–42 не использованы; сабмиты в одном дне, леденец/квота не влияют на возвращённые скоры.
**Closest prior work and gap**: exp3_e3b (k=20, run-id 57544938); гап: k<20 в exp3 не проверялся, в exp4 E7 локальный выигрыш k=10 был +0.0041 mean.
**Falsifier or counterexample**: воспроизведение файла скриптом exp4_submit.py k10 даёт иной файл; либо сабмит той же конфигурации даёт ниже 0.34487/0.32361.
**Required support**: сабмит с public и private выше exp3_e3b + воспроизводимый скрипт.
**Current support**: Сабмит run-id 57554720: public 0.34586, private 0.32383; референс exp3_e3b run-id 57544938: 0.34487/0.32361 — выше по обеим половинам. Локально выбор k=10 обоснован сеткой E7 (exp4_fusion.csv:2-5 — все четыре верхние строки при k=10, против k=20-контроля exp4_fusion.csv:6), но точка «k=10 + старые компоненты exp3_e3b» локально не считалась — k=10 выбран по Kaggle-проверке изолированного инкремента. Генерация: artifacts/scripts/exp4_submit.py variant=k10 (format OK: 1260 строк, 5 id из расписания 31–42); файл назначен финалом: submission.csv в корне == data/submission_exp4_k10.csv.
**Evidence sources**:
- kaggle-run:57554720
- kaggle-run:57544938
- artifacts/scripts/exp4_submit.py
- results/tables/exp4_fusion.csv:6
**Counterevidence and failures**: прирост частный и мал (private +0.00022 — на уровне близкого к шуму); знак k=10 изолированно согласовался локально и на Kaggle (редкий случай, C1.4), но k=10 в составе полной стек-конфигурации (variant=winner) private не улучшил — эффекты компонентов не аддитивны.
**Permitted manuscript wording**: Финальная конфигурация — RRF k=10 при весах (1,2,1) поверх pop×log(1+airings), PPR (restart=0.5, attr_weight=0.5, без decay) и ALS 64/0.05/20; она превосходит прежний финал (k=20) по обеим половинам теста: 0.34586/0.32383 против 0.34487/0.32361 (run-id 57554720 против 57544938).
**Next cheapest decisive action**: воспроизвести data/submission_exp4_k10.csv скриптом exp4_submit.py variant=k10 и свести md5; скриншот возвращённого public для §9 отчёта.
**Depends on**: C2.3
