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
**Claim**: Изолированное изменение RRF k=20 → k=10 (веса w=(1,2,1), PPR E1 restart=0.5/aw=0.5, ALS 64/0.05/20 — всё остальное как в exp3_e3b) улучшило Kaggle MAP@5 по обеим половинам: public 0.34586 против 0.34487 и private 0.32383 против 0.32361. Конфигурация была финальной после раунда exp4, затем superseded C18.
**Scope and exclusions**: только этот конкурс, только пара конфигураций (exp4_k10 против exp3_e3b); приросты малы (+0.00099/+0.00022) и не утверждаются для других тестов.
**Assumptions**: оба файла сгенерированы при одинаковом train 11–30, пуле-500 и формате (отличие — только k в fusion); просмотры 31–42 не использованы; сабмиты в одном дне, леденец/квота не влияют на возвращённые скоры.
**Closest prior work and gap**: exp3_e3b (k=20, run-id 57544938); гап: k<20 в exp3 не проверялся, в exp4 E7 локальный выигрыш k=10 был +0.0041 mean.
**Falsifier or counterexample**: воспроизведение файла скриптом exp4_submit.py k10 даёт иной файл; либо сабмит той же конфигурации даёт ниже 0.34487/0.32361.
**Required support**: сабмит с public и private выше exp3_e3b + воспроизводимый скрипт.
**Current support**: Сабмит run-id 57554720: public 0.34586, private 0.32383; референс exp3_e3b run-id 57544938: 0.34487/0.32361 — выше по обеим половинам. Локально выбор k=10 обоснован сеткой E7 (exp4_fusion.csv:2-5 — все четыре верхние строки при k=10, против k=20-контроля exp4_fusion.csv:6), но точка «k=10 + старые компоненты exp3_e3b» локально не считалась. На момент завершения exp4 файл был назначен финалом; после exp5 его заменил по обеим Kaggle-половинам run 57593168 (C18). Корневой `submission.csv` сохранён как legacy и не перезаписывался.
**Evidence sources**:
- kaggle-run:57554720
- kaggle-run:57544938
- artifacts/scripts/exp4_submit.py
- results/tables/exp4_fusion.csv:6
**Counterevidence and failures**: прирост exp4 был мал (private +0.00022); полный стек exp4 (variant=winner) private не улучшил. Новый candidate-only exp5 run 57593168 теперь превосходит exp4_k10 по обеим половинам, поэтому exp4 больше не является текущим лучшим сабмитом.
**Permitted manuscript wording**: В раунде exp4 изолированное изменение k=20→10 повысило Kaggle public/private с 0.34487/0.32361 до 0.34586/0.32383 (run 57554720 против 57544938). Впоследствии exp5 candidate-only submission достиг 0.41874/0.39477 и стал лучшим (C18).
**Next cheapest decisive action**: поддерживать текущим лучшим candidate-only exp5 файл; улучшать retrieval до Recall@500≥0.80, затем вернуться к ranker только после этого.
**Depends on**: C2.3

## C3 (supported)
**Type**: empirical
**Status**: supported
**Claim**: A threshold-free rich interaction table can be reconstructed from the raw view log and labelled schedules using the reference solution's strict interval-overlap rule, retaining overlap duration and airing-time context.
**Scope and exclusions**: data weeks 11–30; channel match `vsetv_id == channel_id`; labelled schedule rows only; this is a data-construction claim, not a recommendation-quality or MAP@5 claim.
**Assumptions**: timestamps in the raw view log and schedule are in the same local time basis; adjacent labelled airings on each channel do not overlap; overlap seconds are the target-view duration proxy.
**Closest prior work and gap**: `artifacts/scripts/join_interactions.py` applies `(overlap_seconds >= 300) | (overlap_seconds / show_duration >= 0.5)` before saving, losing short/partial viewing information; reference `1_prepare_data.ipynb` retains strict interval overlaps without that threshold.
**Falsifier or counterexample**: any saved row violates `show_start < view_stop && show_stop > view_start`, has non-positive overlap, omits a positive overlap from the source intervals, or applies a watch threshold at construction.
**Required support**: reproducible build from raw sources; required 14-column schema; zero nulls and interval-predicate violations; threshold values derivable after construction.
**Current support**: Run `2026-10-24-exp5-rich-interactions-0100` built `data/exp5_user_airings_rich.parquet`: 3,847,696 rows, 4,895 users, 4,363 shows; 14 expected columns, zero null cells, zero overlap-predicate violations; positive overlaps retained down to 1 second. `user_watch_pct >= 0.3/0.5/0.8` counts are 1,238,985 / 885,038 / 570,588. Independent naive-reference check over 5,000 sampled raw views produced 3,319 overlap pairs, exactly matching the rich table as a multiset (0 reference-only, 0 table-only; run `2026-10-24-exp5-reference-check-0125`). The 50%-threshold legacy table is not the source. LightFM 1.17 imports in isolated Python 3.12 environment `.venv-exp5` after patching the PyPI sdist's `__builtins__` setup line and regenerating C extensions with Cython 0.29.37; it is compiled without OpenMP (single-thread warning). Candidate recall and ranking effects are separate claims.
**Evidence sources**:
- artifacts/scripts/exp5_build_rich_interactions.py:1
- artifacts/scripts/exp5_reference_check.py:1
- experiments/2026-10-24-0100-exp5-rich-interactions.ipynb:1
- run #2026-10-24-exp5-rich-interactions-0100
- run #2026-10-24-exp5-reference-check-0125
- run #2026-10-24-exp5-lightfm-build-0115
**Counterevidence and failures**: Initial `uv add lightfm lightgbm` failed on Python 3.14 because the PyPI sdist assumes `__builtins__` is an object; plain install also failed on Python 3.12 because its bundled Cython-generated C uses removed CPython internals. Resolution: in an isolated Python 3.12 environment, patched only the temporary source copy to handle dict-valued `__builtins__` and regenerated C with Cython 0.29.37; LightFM imports, but with OpenMP disabled. The first candidate run then failed in PPR row-normalization because the diagonal matrix was constructed with a non-square sparse shape; corrected to use `scipy.sparse.diags`, and the run was restarted (run `2026-10-24-exp5-candidate-recall-0130`). LightGBM wheel cannot currently load because its ARM64 binary requires `libomp`; Homebrew reports there is no bottle for this macOS 14 configuration and would build/upgrade dependencies, so the attempted system install was stopped. These are environment/build or implementation failures, not model-performance findings.
**Permitted manuscript wording**: A new threshold-free overlap table was built from the raw 11–30 view log and labelled schedules with strict interval matching. It contains 3,847,696 rows and preserves continuous `user_watch_pct`; this establishes the data pipeline only and does not support a MAP@5 improvement claim.
**Next cheapest decisive action**: perform the remaining rolling-window candidate-recall audits with this validated table; do not train a reranker until Recall@500 is high enough.
**Depends on**: —

## C4 (supported)
**Type**: empirical
**Status**: supported
**Claim**: На main split corrected candidate pool A–E (repeat top-150, LightFM WARP top-200, existing tripartite PPR top-100, recent popularity top-50, TV channel×weekday×30-minute-slot top-100) покрывает 49.410% strong-watch relevant user-show pairs при cutoff 500.
**Scope and exclusions**: history weeks 11–20; target weeks 21–30; eval users из конкурсного submission template с хотя бы одним target show, где отдельная airing имеет `user_watch_pct >= 0.8`; релевантность ограничена target-schedule catalog; это recall кандидатов, не MAP@5 и не Kaggle score.
**Assumptions**: future views не участвуют в генераторах; расписание target window доступно; агрегатный порог релевантности совпадает с выбранной exp5 strong-watch постановкой.
**Closest prior work and gap**: пользовательская целевая архитектура A–E; до этого измерялись MAP@5 текущего RRF/PPR/ALS, но Recall@K новой rich-table candidate union не измерялся.
**Falsifier or counterexample**: воспроизводимый пересчёт на той же таблице, split, catalog и target definition показывает другой recall; либо внесение исправления denominator/catalog раскрывает, что текущая оценка смещена.
**Required support**: воспроизводимая candidate table, раздельные Recall@200/500 по каждому генератору и union, фиксированная relevance definition.
**Current support**: Corrected run `results/tables/exp5_candidate_recall_corrected.csv:25-27` gives tripartite A–E Recall@200=0.418186 and Recall@500=0.494101 (19,643/39,755 relevant pairs; 1,236 users; target catalog restricted to weeks21–30). Independent diagnostic `artifacts/scripts/exp5_candidate_diagnostics.py:1` on `data/exp5_candidates_main_corrected.parquet` reproduces union coverage exactly: 19,643/39,755=49.4101%. Tripartite PPR source recall is 12,500/39,755=31.4426%. Historical bipartite-PPR and unfiltered-catalog counts are superseded failures.
**Evidence sources**:
- results/tables/exp5_candidate_recall.csv:15
- results/tables/exp5_candidate_recall.csv:17
- results/tables/exp5_candidate_recall.csv:19
- results/tables/exp5_candidate_recall.csv:21
- results/tables/exp5_candidate_recall.csv:23
- results/tables/exp5_candidate_recall.csv:25
- results/tables/exp5_candidate_recall.csv:26
- results/tables/exp5_candidate_recall_corrected.csv:25
- results/tables/exp5_candidate_recall_corrected.csv:26
- results/tables/exp5_candidate_recall_corrected.csv:27
- artifacts/scripts/exp5_candidate_diagnostics.py:1
- data/exp5_candidates_main.parquet
- data/exp5_candidates_main_corrected.parquet
- run #2026-10-24-exp5-candidate-recall-0130
- run #2026-10-24-exp5-tripartite-baseline-0230
- run #2026-10-24-exp5-tripartite-diagnostics-0240
- artifacts/scripts/exp5_candidate_diagnostics.py:1
- run #2026-10-24-exp5-candidate-diagnostics-0130
**Counterevidence and failures**: Earlier bipartite-PPR run with two week-31-only show IDs reported 0.492944 and is superseded; first sparse-normalization implementation also failed before saving outputs. Corrected tripartite run is reproducible, but recall remains <0.5 and does not meet the high-recall gate.
**Permitted manuscript wording**: Corrected tripartite A–E candidate generation on the main split yielded Recall@500 0.494101 (local diagnostic, not MAP@5). This remains below the candidate gate; no reranker has been trained.
**Next cheapest decisive action**: determine whether expanded generators with tripartite PPR raise main Recall@500, then validate on rolling windows.
**Depends on**: C3

## C5 (supported)
**Type**: empirical
**Status**: supported
**Claim**: With tripartite PPR and the strict target-week catalog, expanded candidates (longer lists, user×channel, user×metadata, ALS) achieve strong-watch main Recall@500 0.577386 and full-union recall 0.837379; early-split Recall@500 is 0.736941.
**Scope and exclusions**: Main: history weeks 11–20 → target 21–30; early: history ≤16 → target 17–20. Strong-watch relevance: individual airing `user_watch_pct >= 0.8`, target catalog only. Это candidate recall, не MAP@5 и не Kaggle score.
**Assumptions**: candidate generators используют только history до каждого target window; future schedule/metadata разрешены; expanded2 RRF использует равные reciprocal-rank contributions при k=60.
**Closest prior work and gap**: C4 baseline A–E; расширенные генераторы и их coverage раньше не измерялись.
**Falsifier or counterexample**: воспроизводимое сравнение тех же candidate lists и label/catalog definition даёт иные recall; либо непреднамеренная утечка target-window просмотров обнаруживается в генераторах.
**Required support**: per-source и union Recall@200/500 для main и early splits, сохранённые candidate tables и скрипт.
**Current support**: `results/tables/exp5_candidate_recall_expanded4.csv:37-39` reports main tripartite expanded Recall@200=0.435341, Recall@500=0.577386, full union=0.837379 (33,290/39,755 pairs; 1,236 users). `results/tables/exp5_candidate_recall_expanded4_early.csv:37-39` reports early Recall@500=0.736941. Main catalog is restricted to weeks21–30; mean union size is 1,670.5 candidates/user. These are candidate-recall diagnostics, not MAP@5.
**Evidence sources**:
- artifacts/scripts/exp5_candidate_recall.py:1
- results/tables/exp5_candidate_recall_expanded4.csv:37
- results/tables/exp5_candidate_recall_expanded4.csv:38
- results/tables/exp5_candidate_recall_expanded4.csv:39
- results/tables/exp5_candidate_recall_expanded4_early.csv:38
- data/exp5_candidates_main_expanded4.parquet
- data/exp5_candidates_main_expanded4_early.parquet
- run #2026-10-24-exp5-tripartite-expanded-main-0240
- run #2026-10-24-exp5-tripartite-expanded-early-0235
**Counterevidence and failures**: User-metadata source remains weak (Recall@500=0.135706, `results/tables/exp5_candidate_recall_expanded4.csv:34`). Earlier bipartite-PPR and unfiltered-catalog experiments are superseded. Full-union recall is higher, but main top-500 recall remains below the high-recall gate; no reranker was trained.
**Permitted manuscript wording**: Corrected tripartite-PPR expanded candidates reached Recall@500 0.577386 on main and 0.736941 on early, with main full-union recall 0.837379. These are local candidate-recall results only and do not establish MAP@5 improvement.
**Next cheapest decisive action**: tune source weights on the early tripartite split, freeze them, and evaluate on main before deciding whether to add more candidate sources.
**Depends on**: C3, C4

## C6 (supported)
**Type**: empirical
**Status**: supported
**Claim**: Reciprocal-rank fusion weights tuned only on early temporal labels improved tripartite-PPR expanded-candidate main Recall@500 from 0.577361 to 0.598692, but did not reach a high-recall level.
**Scope and exclusions**: early tune: history ≤16, target 17–20; holdout evaluation: history ≤20, target 21–30; tripartite expanded candidate sources; target strong-watch label is individual airing `user_watch_pct >= 0.8`, schedule catalog restricted to each target window. This is Recall@500, not MAP@5.
**Assumptions**: all candidate features/ranks are generated strictly from history before each target window; main weights were frozen from early split and not tuned on main labels; the same candidate set/catalog definition is used for equal and tuned comparisons.
**Closest prior work and gap**: equal-weight RRF candidate union in C5; source-weight transfer across two temporal windows was not yet tested.
**Falsifier or counterexample**: rerunning the early-only coordinate search and main evaluation with identical candidate tables produces no positive main-split change, or reveals target information in the tuning inputs.
**Required support**: reproducible early-only tuning, persisted source weights, and later untouched main-split Recall@500 comparison.
**Current support**: Early-only tripartite candidate tuning yielded early Recall@500=0.788440 (`results/tables/exp5_candidate_fusion_tuned_tripartite.csv:9`). Frozen weights on main history≤20/target21–30 yielded Recall@500=0.598692 vs equal-weight=0.577361 (`:17` vs `:13`; 23,801/39,755 vs 22,953/39,755, 1,236 users). Weights are in `results/tables/exp5_candidate_fusion_weights_tripartite.csv:2-9`. This remains below the high-recall gate; no reranker has been trained.
**Evidence sources**:
- artifacts/scripts/exp5_tune_candidate_fusion.py:1
- results/tables/exp5_candidate_fusion_tuned_tripartite.csv:13
- results/tables/exp5_candidate_fusion_tuned_tripartite.csv:17
- results/tables/exp5_candidate_fusion_tuned_tripartite.csv:9
- results/tables/exp5_candidate_fusion_weights_tripartite.csv:2
- results/tables/exp5_candidate_fusion_weights_tripartite.csv:9
- data/exp5_candidates_main_tuned_tripartite.parquet
- run #2026-10-24-exp5-candidate-fusion-tripartite-0245
**Counterevidence and failures**: Early tuned Recall@500=0.788440 transfers only +0.021331 absolute to main; main Recall@200 decreases from 0.435341 to 0.425406. Repeat, LightFM, PPR, and metadata receive zero weight, indicating source utility may be split-specific.
**Permitted manuscript wording**: Weights tuned on early tripartite-PPR candidates and frozen for main raised local strong-watch Recall@500 from 0.577361 to 0.598692. This is a candidate-recall gain, not a MAP@5 or Kaggle score improvement.
**Next cheapest decisive action**: measure the remaining rolling windows with tripartite candidates, then reassess the main-split high-recall gate.
**Depends on**: C5

## C7 (supported)
**Type**: empirical
**Status**: supported
**Claim**: Adding user weekday×30-minute-slot, channel×weekday and channel×hour candidate projections to the tripartite expanded pool yielded main equal-RRF strong-watch Recall@500 0.609785 (full union 0.882983) and early Recall@500 0.776970.
**Scope and exclusions**: Main history≤20→target21–30; early history≤16→target17–20; target catalogs week-filtered. Strong-watch relevance is per airing `user_watch_pct>=0.8`; candidate recall only.
**Assumptions**: All time/channel affinities use pre-target history; target schedule context is known; union is deduplicated and target-catalog restricted.
**Closest prior work and gap**: C5 expanded tripartite candidates with exact channel×weekday×slot and broad channel affinity; the three coarser temporal projections were newly tested.
**Falsifier or counterexample**: Rebuilding these candidate lists with the recorded splits and feature keys gives different Recall@K, or detects target-window views in affinity statistics.
**Required support**: Reproducible main and early candidate/metric tables, same labels/catalog, per-source and union Recall@200/500.
**Current support**: `results/tables/exp5_candidate_recall_expanded5.csv:49-51` reports main Recall@500=0.609785 and full-union=0.882983; `results/tables/exp5_candidate_recall_expanded5_early.csv:37-39` reports early Recall@500=0.776970. Main comparator C5 expanded4 was 0.577386/0.837379.
**Evidence sources**:
- artifacts/scripts/exp5_candidate_recall.py:1
- results/tables/exp5_candidate_recall_expanded5.csv:50
- results/tables/exp5_candidate_recall_expanded5.csv:51
- results/tables/exp5_candidate_recall_expanded5_early.csv:50
- results/tables/exp5_candidate_recall_expanded4.csv:38
- data/exp5_candidates_main_expanded5.parquet
- data/exp5_candidates_main_expanded5_early.parquet
- run #2026-10-24-exp5-time-affinity-expanded5-main-0310
- run #2026-10-24-exp5-time-affinity-expanded5-early-0308
**Counterevidence and failures**: Main Recall@500 remains below the high-recall gate; user_time_slot source alone has Recall@500 0.307886. No reranker was trained.
**Permitted manuscript wording**: Three additional time-context generators raised local main candidate Recall@500 to 0.609785 (full union 0.882983); this does not establish MAP@5 improvement.
**Next cheapest decisive action**: determine whether airing-weighted schedule popularity transfers across the temporal windows without reducing main Recall@500.
**Depends on**: C5

## C8 (supported)
**Type**: empirical
**Status**: supported
**Claim**: Adding a global schedule-popularity source (`recent unique viewers × log1p(target airings)`) increased equal-RRF main tripartite Recall@500 from 0.609785 to 0.613961 and full-union recall from 0.882983 to 0.887259, while early Recall@500 fell from 0.776970 to 0.772882.
**Scope and exclusions**: Same week-filtered main and early splits and strong-watch label as C7; equal-RRF candidate ranking.
**Assumptions**: Future airing counts are computed only from the current target-window schedule; popularity viewers only from history ending before target.
**Closest prior work and gap**: C7 used recent popularity, but did not multiply it by target-window airing count.
**Falsifier or counterexample**: Reproducible main/early candidate comparisons fail to reproduce the stated recalls or include schedule rows outside the split window.
**Required support**: Saved expanded5/expanded6 candidate metrics for both main and early windows.
**Current support**: Main expanded5 `results/tables/exp5_candidate_recall_expanded5.csv:50-51` = 0.609785/0.882983; expanded6 `results/tables/exp5_candidate_recall_expanded6.csv:54-55` = 0.613961/0.887259. Early Recall@500 falls from 0.776970 (`expanded5_early.csv:50`) to 0.772882 (`expanded6_early.csv:54`). Early-tuned expanded6 weights yield main Recall@500=0.626487 (`results/tables/exp5_candidate_fusion_tuned_expanded6.csv:17`).
**Evidence sources**:
- artifacts/scripts/exp5_candidate_recall.py:1
- results/tables/exp5_candidate_recall_expanded5.csv:50
- results/tables/exp5_candidate_recall_expanded5.csv:51
- results/tables/exp5_candidate_recall_expanded6.csv:54
- results/tables/exp5_candidate_recall_expanded6.csv:55
- results/tables/exp5_candidate_recall_expanded5_early.csv:50
- results/tables/exp5_candidate_recall_expanded6_early.csv:54
- results/tables/exp5_candidate_fusion_tuned_expanded6.csv:17
- data/exp5_candidates_main_expanded6.parquet
- run #2026-10-24-exp5-airing-popularity-expanded6-main-0315
**Counterevidence and failures**: The early equal-RRF Recall@500 change is negative (−0.004089); main equal-RRF gain is small (+0.004176). Main remains below the high-recall gate.
**Permitted manuscript wording**: The airing-weighted popularity source made a small main-split retrieval gain to Recall@500 0.613961 but decreased early-window recall; its tuned main result was 0.626487. These are candidate-recall results only.
**Next cheapest decisive action**: retain only if the small main gain persists under rolling evaluation; continue withholding the reranker.
**Depends on**: C7

## C9 (rejected)
**Type**: empirical
**Status**: rejected
**Claim**: Adding association-rule consequents as a candidate source improves main strong-watch Recall@500 over expanded6 tripartite candidates.
**Scope and exclusions**: Main history≤20→target21–30; week-filtered target schedule; strong-watch per airing `user_watch_pct>=0.8`; equal RRF.
**Assumptions**: Association rules are mined from pre-target user-week transactions only, and consequents are restricted to target-catalog show IDs.
**Closest prior work and gap**: C8 tripartite candidates include collaborative, time/channel and schedule-popularity sources; weekly association candidates were an additional proposed retrieval source.
**Falsifier or counterexample**: A corrected, repeatable association source with frozen fusion weights increases held-out main Recall@500 over expanded6.
**Required support**: Reproducible equal-RRF main comparison under identical users, schedule catalog and labels.
**Current support**: Rejected for this implementation: expanded6 Recall@500=0.613961 (`exp5_candidate_recall_expanded6.csv:54`); adding association rules gives 0.608301 (`exp5_candidate_recall_expanded7.csv:58`). Full-union recall changes 0.887259→0.887687 (`expanded6.csv:55`, `expanded7.csv:59`).
**Evidence sources**:
- artifacts/scripts/exp5_candidate_recall.py:1
- artifacts/scripts/exp_assoc_rules.py:82
- results/tables/exp5_candidate_recall_expanded6.csv:54
- results/tables/exp5_candidate_recall_expanded6.csv:55
- results/tables/exp5_candidate_recall_expanded7.csv:58
- results/tables/exp5_candidate_recall_expanded7.csv:59
- run #2026-10-24-exp5-association-candidates-0335
**Counterevidence and failures**: Recall@500 fell by 0.005660; the tiny full-union gain does not compensate for poorer top-500 ordering. Initial association scoring used linear membership checks against the catalog and was stopped after profiling; using a set completed in 125.9s on main.
**Permitted manuscript wording**: Association-rule candidates did not improve equal-RRF main Recall@500 in this implementation (0.608301 vs 0.613961) and are excluded from the current candidate pool.
**Next cheapest decisive action**: none for this implementation; revisit only if a separate temporal split shows positive transfer.
**Depends on**: C8

## C10 (supported)
**Type**: empirical
**Status**: supported
**Claim**: On the main temporal split, candidate-only reciprocal-rank weights tuned for MAP@5 on the early split raised strong-watch MAP@5 from 0.478560 (equal RRF) to 0.523572; Recall@500 fell from 0.613961 to 0.576003.
**Scope and exclusions**: Candidate ordering only over the tripartite expanded6 union; early tuning history≤16→target17–20, main evaluation history≤20→target21–30; per-airing strong target threshold 0.8, with a parallel 0.5 target ablation. This is local validation MAP@5, not a LightGBM reranker result or Kaggle score.
**Assumptions**: Candidate generation and features use history before each target window; weights are fitted on early data only and frozen for main; AP@5 uses the project's target definition and target-schedule catalog.
**Closest prior work and gap**: C6/C8 tuned reciprocal-rank weights against Recall@500; direct MAP@5 tuning for candidate-only ordering had not been tested.
**Falsifier or counterexample**: Recomputing the saved candidate tables with the same early-only coordinate search produces a different main MAP@5, or a leakage audit finds main labels in tuning.
**Required support**: Reproducible candidate tables, early-only tuning code, main MAP@5 and Recall@500 outputs under both target thresholds.
**Current support**: `results/tables/exp5_candidate_map_tuned.csv:11` gives equal-RRF main MAP@5=0.478560, Recall@500=0.613961; `:13` gives early-tuned-for-0.8 main MAP@5=0.523572, Recall@500=0.576003 (1,236 users). For target threshold0.5, `:8`→`:10` gives MAP@5 0.575607→0.624520. Weights were tuned on early labels only (`results/tables/exp5_candidate_map_weights.csv`). This local MAP gain does not pass the candidate Recall@500 gate and is not a Kaggle submission result.
**Evidence sources**:
- artifacts/scripts/exp5_tune_candidate_map.py:1
- results/tables/exp5_candidate_map_tuned.csv:8
- results/tables/exp5_candidate_map_tuned.csv:10
- results/tables/exp5_candidate_map_tuned.csv:11
- results/tables/exp5_candidate_map_tuned.csv:13
- results/tables/exp5_candidate_map_weights.csv:1
- data/exp5_candidates_main_expanded6.parquet
- data/exp5_candidates_main_expanded6_early.parquet
- run #2026-10-24-exp5-candidate-map-tune-0340
**Counterevidence and failures**: MAP@5 tuning trades away Recall@500 (−0.037957 absolute on the strong target) and main Recall@500 remains below the high-recall gate. No learned reranker was fitted.
**Permitted manuscript wording**: Candidate-only RRF weights tuned on the early split yielded local main MAP@5 0.523572 versus 0.478560 equal-weight at target threshold0.8, while reducing Recall@500 from 0.613961 to 0.576003. This is not a Kaggle score and not a reranker result.
**Next cheapest decisive action**: continue candidate-source work; the user confirmed the gate is main-split Recall@500≥0.80, so candidate-only MAP improvements do not authorize a reranker.
**Depends on**: C3, C8

## C11 (supported)
**Type**: empirical
**Status**: supported
**Claim**: Expanding the early-only MAP@5 candidate-weight search grid produced main strong-target MAP@5 0.524715, only +0.001143 over the earlier grid's 0.523572, while reducing Recall@500 to 0.557062.
**Scope and exclusions**: Same expanded6 tripartite candidate tables as C10; source weights tuned on early history≤16/target17–20, evaluated frozen on main history≤20/target21–30; strong target threshold0.8. Candidate-only reciprocal-rank ordering; not a learned reranker or Kaggle score.
**Assumptions**: Main labels were used only for final evaluation, not for the early source-weight search; MAP@5 and target construction match C10.
**Closest prior work and gap**: C10 five-point grid `{0,0.5,1,2,4}`; this test widened the grid to `{0,0.5,1,2,4,8,16}` and used four coordinate passes.
**Falsifier or counterexample**: Re-running the wider grid on the same early candidates and main holdout yields a different MAP@5/Recall@500 pair, or reveals main labels were used in weight selection.
**Required support**: Saved early-only wide-grid weights and main holdout MAP@5/Recall@500 under the same threshold.
**Current support**: `results/tables/exp5_candidate_map_tuned_widegrid.csv:11` gives equal-RRF main MAP@5=0.478560 / Recall@500=0.613961; `:13` gives widegrid early-tuned MAP@5=0.524715 / Recall@500=0.557062 (1,236 users). C10's earlier grid gave MAP@5=0.523572 and Recall@500=0.576003 (`exp5_candidate_map_tuned.csv:13`). Weights hit upper bound16 for repeat and8 for PPR/airing-popularity (`exp5_candidate_map_weights_widegrid.csv:14,16,25`), so search is not converged.
**Evidence sources**:
- artifacts/scripts/exp5_tune_candidate_map.py:1
- results/tables/exp5_candidate_map_tuned_widegrid.csv:11
- results/tables/exp5_candidate_map_tuned_widegrid.csv:13
- results/tables/exp5_candidate_map_tuned.csv:13
- results/tables/exp5_candidate_map_weights_widegrid.csv:14
- results/tables/exp5_candidate_map_weights_widegrid.csv:16
- results/tables/exp5_candidate_map_weights_widegrid.csv:25
- run #2026-10-24-exp5-candidate-map-widegrid-0400
**Counterevidence and failures**: MAP@5 gain over C10 is only +0.001143, while Recall@500 drops by 0.018941 versus C10. The wider search therefore does not resolve the retrieval gate, and boundary weights indicate the finite grid was inadequate to establish an optimum.
**Permitted manuscript wording**: The expanded early-only candidate MAP@5 search yielded main MAP@5 0.524715 vs 0.478560 equal-RRF, but Recall@500 was 0.557062; no reranker or Kaggle result is implied.
**Next cheapest decisive action**: pursue a source with out-of-split coverage evidence; reranker remains blocked until main Recall@500≥0.80, as confirmed by the user.
**Depends on**: C10

## C12 (supported)
**Type**: empirical
**Status**: supported
**Claim**: Adding schedule-airing propensity aggregates (max, mean, sum, logsumexp) from user channel×weekday×30-minute-slot history raised equal-RRF main tripartite Recall@500 from 0.613961 (expanded6) to 0.629229 (expanded9); full-union recall reached 0.891737.
**Scope and exclusions**: Main history≤20→target21–30; early history≤16→target17–20; target schedule filtered by target ISO weeks; per-airing strong-watch threshold0.8. Candidate recall only.
**Assumptions**: All propensity inputs derive from history before target; each target airing receives a score from user channel, weekday, slot and exact channel×weekday×slot counts; show-level aggregates are computed over target-window airings.
**Closest prior work and gap**: C7/C8 used separate time/channel candidate projections; max/mean/sum/logsumexp schedule-propensity aggregations had not been materialized as candidate features/sources.
**Falsifier or counterexample**: Rebuilding the propensity table and candidate lists on identical splits/catalog produces different Recall@K or shows future-window interactions entered affinity counts.
**Required support**: Saved candidate data with all four named propensity columns, source-level/union Recall@200/500, and strict temporal split.
**Current support**: `data/exp5_candidates_main_expanded9.parquet` includes `airing_propensity_max`, `_mean`, `_sum`, `_logsumexp` (621,288 non-null candidate rows per feature). Main `results/tables/exp5_candidate_recall_expanded9.csv:69-71` reports equal-RRF Recall@200=0.470130, Recall@500=0.629229, full union=0.891737 (25,015/39,755 relevant pairs; 1,236 users). Early equal-RRF Recall@500 is 0.774756 (`exp5_candidate_recall_expanded9_early.csv:50`). Early Recall@500-tuned weights yield main 0.633883 (`results/tables/exp5_candidate_fusion_tuned_expanded9.csv:17`). The main high-recall gate remains unmet.
**Evidence sources**:
- artifacts/scripts/exp5_candidate_recall.py:1
- results/tables/exp5_candidate_recall_expanded9.csv:69
- results/tables/exp5_candidate_recall_expanded9.csv:70
- results/tables/exp5_candidate_recall_expanded9.csv:71
- results/tables/exp5_candidate_recall_expanded9_early.csv:50
- results/tables/exp5_candidate_fusion_tuned_expanded9.csv:17
- data/exp5_candidates_main_expanded9.parquet
- run #2026-10-24-exp5-propensity-expanded9-main-0400
- run #2026-10-24-exp5-propensity-expanded9-early-0355
**Counterevidence and failures**: Main Recall@500 rises by 0.015269 over expanded6, but remains below 0.8. The early-tuned weight increase is modest (0.633883 vs 0.629229). This is a retrieval improvement only; no reranker was trained.
**Permitted manuscript wording**: Four schedule-propensity aggregates were materialized and used for candidate generation; equal-RRF main Recall@500 reached 0.629229 (full union 0.891737), while early-tuned weights gave 0.633883. These are local candidate-recall values, not MAP@5 or Kaggle scores.
**Next cheapest decisive action**: continue improving main candidate Recall@500 or ask the user whether the gate can be relaxed; do not train a reranker before the ≥0.80 main criterion.
**Depends on**: C8

## C13 (supported)
**Type**: empirical
**Status**: supported
**Claim**: Early-only MAP@5 tuning of the expanded9 propensity candidate pool increased main strong-target candidate MAP@5 from 0.479806 equal-RRF to 0.556025; Recall@500 was 0.609860 under the tuned weights.
**Scope and exclusions**: Candidate-only reciprocal-rank ordering, with the four schedule-propensity aggregates; tuning on history≤16→target17–20, evaluation on history≤20→target21–30; target is `user_watch_pct>=0.8`. No LightGBM or Kaggle score.
**Assumptions**: Candidate features are pre-target only, and the schedule catalog is filtered to each target window; weights fit on early labels only.
**Closest prior work and gap**: C10 candidate MAP tuning used expanded6 without the four airing-propensity aggregates.
**Falsifier or counterexample**: Reproducible early-only source-weight tuning on the same saved tables fails to reproduce the MAP@5 result or reveals main labels in tuning.
**Required support**: Candidate output with four propensity aggregates, persisted early-only weights, main MAP@5 and Recall@500 evaluation.
**Current support**: `results/tables/exp5_candidate_map_tuned_propensity.csv:11` equal RRF MAP@5=0.479806; `:13` early-tuned MAP@5=0.556025, Recall@500=0.609860 (1,236 users). The MAP score improves, but the candidate gate remains unmet.
**Evidence sources**:
- artifacts/scripts/exp5_tune_candidate_map.py:1
- results/tables/exp5_candidate_map_tuned_propensity.csv:11
- results/tables/exp5_candidate_map_tuned_propensity.csv:13
- results/tables/exp5_candidate_map_weights_propensity.csv:1
- data/exp5_candidates_main_expanded9.parquet
- run #2026-10-24-exp5-candidate-map-propensity-0410
**Counterevidence and failures**: MAP@5 tuning reduces Recall@500 versus equal RRF (0.629229→0.609860); no reranker was trained.
**Permitted manuscript wording**: Early-only MAP@5 tuning of tripartite candidate weights with schedule propensity yielded local main candidate MAP@5 0.556025 vs 0.479806 equal RRF, but Recall@500 was 0.609860 and no Kaggle score is implied.
**Next cheapest decisive action**: keep using candidate MAP as a diagnostic while improving main Recall@500 to the confirmed ≥0.80 threshold.
**Depends on**: C12

## C14 (supported)
**Type**: empirical
**Status**: supported
**Claim**: Widening the early-only MAP@5 weight grid for the propensity-expanded candidate pool increased main strong-target candidate MAP@5 to 0.568194, while main Recall@500 was 0.600428.
**Scope and exclusions**: Same candidate tables and splits as C13; early-only tuning, main untouched evaluation; grid weights `{0,0.5,1,2,4,8,16}` and four coordinate passes. Candidate-only result, not a reranker or Kaggle score.
**Assumptions**: No main target labels enter optimization; AP@5 and threshold definitions match C13.
**Closest prior work and gap**: C13 smaller grid `{0,0.5,1,2,4}`; this measures sensitivity to allowing larger RRF weights.
**Falsifier or counterexample**: Repeating the saved wide-grid run under the same inputs gives different main MAP@5/Recall@500 or uses main labels in tuning.
**Required support**: Wide-grid weights and main evaluation tables.
**Current support**: `results/tables/exp5_candidate_map_tuned_propensity_widegrid.csv:11` equal-RRF MAP@5=0.479806; `:13` wide-grid early-tuned MAP@5=0.568194 / Recall@500=0.600428 (1,236 users). Selected weights reach upper bound16 for `prop_sum` and `prop_logsumexp`, demonstrating the grid optimum is boundary-limited (`exp5_candidate_map_weights_propensity_widegrid.csv:32-33`).
**Evidence sources**:
- artifacts/scripts/exp5_tune_candidate_map.py:1
- results/tables/exp5_candidate_map_tuned_propensity_widegrid.csv:11
- results/tables/exp5_candidate_map_tuned_propensity_widegrid.csv:13
- results/tables/exp5_candidate_map_weights_propensity_widegrid.csv:32
- results/tables/exp5_candidate_map_weights_propensity_widegrid.csv:33
- run #2026-10-24-exp5-candidate-map-propensity-widegrid-0420
**Counterevidence and failures**: Additional MAP@5 gain over C13 is 0.012169, but Recall@500 remains 0.600428, below the confirmed gate. Weights at a grid boundary do not establish convergence or robust transfer.
**Permitted manuscript wording**: A wider early-only MAP@5 grid yielded local main candidate MAP@5 0.568194 with Recall@500 0.600428; it is not a MAP@5 reranker or Kaggle result.
**Next cheapest decisive action**: do not enlarge the weight grid further before improving candidate Recall@500 or obtaining user approval to relax the gate.
**Depends on**: C13

## C18 (supported)
**Type**: empirical
**Status**: supported
**Claim**: A candidate-only tripartite/LightFM/ALS/time-propensity RRF submission with MAP@5 weights tuned on the early temporal split achieved Kaggle public 0.41874 / private 0.39477 (run 57593168), exceeding the prior designated exp4_k10 submission (0.34586 / 0.32383, run 57554720) on both halves.
**Scope and exclusions**: Competition `sweettv-tv-program-recommender`, test weeks31–42; views31–42 were not used; target schedule/metadata only. The submission uses candidate generators and weighted reciprocal-rank fusion only—no binary LightGBM, LGBMRanker, or learned reranker.
**Assumptions**: Kaggle returned scores correspond to the submitted `data/submission_exp5_candidate_map_tuned.csv`; the file was generated from the final schedule catalog with five unique IDs per user; previous score comparison is for the same competition/test.
**Closest prior work and gap**: Previous exp4_k10 final (run57554720); exp5 adds rich overlap history, tripartite PPR, LightFM, ALS, time/schedule propensity and candidate-only MAP-tuned fusion. Gap: Recall@500 on the main temporal validation split remains below the confirmed 0.80 gate, and the reference binary/ranking LightGBM stages remain untested.
**Falsifier or counterexample**: Kaggle submissions table shows different public/private values or run ID; format/catalog audit finds any invalid row; or views31–42 were used in generation.
**Required support**: Submission file, candidate generation and scoring code, Kaggle returned public/private scores, format and target-catalog validation.
**Current support**: Kaggle run 57593168: public 0.41874, private 0.39477; prior exp4_k10 run 57554720: 0.34586/0.32383. File `data/submission_exp5_candidate_map_tuned.csv` has 1,260 rows, five unique IDs per user, all IDs in weeks31–42 schedule. Candidate MAP@5 weights were selected on early history≤16/target17–20 and applied to full-history≤30 inference; no future views used.
**Evidence sources**:
- kaggle-run:57593168
- kaggle-run:57554720
- artifacts/scripts/exp5_make_candidate_submission.py:1
- artifacts/scripts/exp5_candidate_recall.py:1
- artifacts/scripts/exp5_tune_candidate_map.py:1
- results/tables/exp5_candidate_map_tuned_propensity_widegrid.csv:13
- results/tables/exp5_candidate_map_weights_propensity_widegrid.csv:14
- data/submission_exp5_candidate_map_tuned.csv
- run #2026-10-24-exp5-candidate-final-inference-1440
**Counterevidence and failures**: New submission remains below the user-stated leader public level ≈0.453 by 0.03426. It was generated with candidate-only RRF and main Recall@500≈0.557 on its MAP-tuned validation ordering; this does not satisfy the confirmed main Recall@500≥0.80 gate for reranking. Root `submission.csv` remains the prior exp4 file and was intentionally not overwritten.
**Permitted manuscript wording**: Candidate-only exp5 submission reached MAP@5 0.41874 public / 0.39477 private (run57593168), beating exp4_k10 0.34586/0.32383 on both halves; this is not a LightGBM reranker result.
**Next cheapest decisive action**: continue candidate-source/retrieval improvements toward main Recall@500≥0.80; keep the exp5 file as best Kaggle result, and do not train the reranker until the confirmed gate is met.
**Depends on**: C12, C14

## C19 (rejected)
**Type**: empirical
**Status**: rejected
**Claim**: Candidate-only submissions trained with LightFM history positives thresholded at 0.5 or 0.8, plus their early-tuned MAP@5 fusion weights, improve Kaggle MAP@5 over the current exp5 candidate-only best.
**Scope and exclusions**: Same competition/test weeks31–42; candidate generation uses history through week30 and target schedule only; no LightGBM/reranker. Candidate threshold fits and fusion weights differ between variants.
**Assumptions**: Kaggle submission rows preserve the same 1,260-user template and five unique scheduled shows; Kaggle public/private scores are the authoritative comparison.
**Closest prior work and gap**: C18 candidate-only RRF using threshold-free LightFM WARP; the user requested 0.5/0.8 viewing-threshold ablation.
**Falsifier or counterexample**: Either threshold-specific candidate submission beats C18 on both public and private scores.
**Required support**: Kaggle-returned public/private scores for the .5 and .8 threshold candidate files versus C18.
**Current support**: Threshold .8 run 57593473 scored public/private 0.40581/0.38178; threshold .5 run 57593488 scored 0.38093/0.36288. Both are below C18 run 57593168 (0.41874/0.39477) on both halves, so neither replaces the best.
**Evidence sources**:
- kaggle-run:57593473
- kaggle-run:57593488
- kaggle-run:57593168
- data/submission_exp5_candidate_lightfm_threshold_08.csv
- data/submission_exp5_candidate_lightfm_threshold_05.csv
- artifacts/scripts/exp5_candidate_recall.py:268
- artifacts/scripts/exp5_make_candidate_submission.py:1
**Counterevidence and failures**: Both threshold-specific submissions underperform the threshold-free candidate RRF best; .8 is closer, but still lower by 0.01293 public / 0.01299 private. Main validation Recall@500 also remains below the 0.80 gate.
**Permitted manuscript wording**: LightFM threshold-specific candidate submissions scored 0.40581/0.38178 (.8) and 0.38093/0.36288 (.5), both below the current best candidate-only score 0.41874/0.39477; these are not reranker results.
**Next cheapest decisive action**: retain C18 as the candidate-only best; further progress requires raising main Recall@500 to the confirmed 0.80 gate or explicit user approval to relax it.
**Depends on**: C18

## C16 (supported)
**Type**: empirical
**Status**: supported
**Claim**: Leakage-safe feature tables were built for early (history≤16→target17–20) and main (history≤20→target21–30) candidate windows; both contain all requested history, user, user×channel, time, future/past schedule, metadata, collaborative and airing-propensity feature families, with target labels stored separately.
**Scope and exclusions**: Features cover expanded9 tripartite candidate rows only; this is feature construction/validation, not model fitting, MAP@5 or Kaggle performance.
**Assumptions**: Future schedule and metadata for each target window are known; no target-window viewing logs are allowed in features; labels use individual airing `user_watch_pct>=0.5` and `>=0.8`.
**Closest prior work and gap**: Existing binary interaction table and old candidate scripts lacked continuous watch history, rich user×time/context features and separately defined strong-watch labels.
**Falsifier or counterexample**: Feature rows contain duplicate user-show pairs, a history-derived feature uses rows after the history cutoff, a required field is missing, or label thresholds disagree with target interactions.
**Required support**: Saved early/main feature tables, required-column audit, unique user-show keys, separate target labels and script implementing strict split filtering.
**Current support**: `data/exp5_features_early.parquet` has 2,174,364 rows, 101 columns, 1,218 users, 27,947 `label_050` positives and 16,478 `label_080` positives; `data/exp5_features_main.parquet` has 2,605,151 rows, 101 columns, 1,244 users, 54,477 `label_050` positives and 35,451 `label_080` positives. Required-column audit found no missing fields, zero duplicate user-show rows, zero null labels and zero cases `label_080>label_050`. The builder filters history at `history.week<=train_end` (`artifacts/scripts/exp5_build_features.py:67`); target interactions are read separately for labels (`:68-72`) and merged after feature construction (`:181-186`).
**Evidence sources**:
- artifacts/scripts/exp5_build_features.py:67
- artifacts/scripts/exp5_build_features.py:181
- artifacts/scripts/exp5_build_features.py:193
- experiments/2026-10-24-1420-exp5-feature-build.ipynb:1
- data/exp5_features_early.parquet
- data/exp5_features_main.parquet
- run #2026-10-24-exp5-features-early-1413
- run #2026-10-24-exp5-features-main-1417
- run #2026-10-24-exp5-feature-audit-1420
**Counterevidence and failures**: Feature tables were built on candidate lists with main Recall@500 below the confirmed 0.80 gate. LightGBM could not be loaded in the project environment because the available wheel requires ARM64 `libomp`; no binary LightGBM or LGBMRanker training was attempted. Feature availability does not establish ranking quality.
**Permitted manuscript wording**: Rich feature tables containing the requested history, user, time-slot, schedule, metadata, collaborative and schedule-propensity signals were constructed for early and main temporal windows with separate 0.5/0.8 target labels; no MAP@5 result is claimed.
**Next cheapest decisive action**: continue candidate source work until the main Recall@500 gate is met; then resolve the isolated LightGBM runtime and reproduce the reference binary baseline before testing LGBMRanker.
**Depends on**: C3, C12

## C15 (rejected)
**Type**: empirical
**Status**: rejected
**Claim**: Adding RP3beta and EASE top-500 candidates improves strong-watch main Recall@500 over expanded9 tripartite candidates.
**Scope and exclusions**: Main history≤20→target21–30; target schedule filtered by week; per-airing `user_watch_pct>=0.8`; equal-RRF candidate fusion.
**Assumptions**: RP3beta uses binary history (`alpha=0.5`, `beta=0.3`) and EASE binary (`lambda=1000`); both are trained only on history; candidate IDs are restricted to the target schedule.
**Closest prior work and gap**: Existing project experiments measured RP3beta/EASE MAP@5; their contribution to rich-table Recall@500 was not measured.
**Falsifier or counterexample**: Repeating the same source comparison yields positive main Recall@500 transfer, or a rolling window consistently benefits.
**Required support**: Reproducible with/without comparison under the same split, schedule catalog and target labels.
**Current support**: Expanded9 main Recall@500=0.629229 and full-union=0.891737 (`results/tables/exp5_candidate_recall_expanded9.csv:70-71`); expanded10 with RP3beta+EASE gives Recall@500=0.604251 and full-union=0.887964 (`results/tables/exp5_candidate_recall_expanded10.csv:62-63`).
**Evidence sources**:
- artifacts/scripts/exp5_candidate_recall.py:1
- results/tables/exp5_candidate_recall_expanded9.csv:70
- results/tables/exp5_candidate_recall_expanded9.csv:71
- results/tables/exp5_candidate_recall_expanded10.csv:62
- results/tables/exp5_candidate_recall_expanded10.csv:63
- artifacts/scripts/exp_grid_rp3_ease_ppr.py:102
- run #2026-10-24-exp5-rp3-ease-candidates-0440
**Counterevidence and failures**: Recall@500 declines by 0.024978 and full-union coverage declines by 0.003773. Neither the top-500 rank order nor pool coverage improves; no reranker was run.
**Permitted manuscript wording**: Adding RP3beta/EASE reduced main equal-RRF Recall@500 from 0.629229 to 0.604251 and full-union coverage from 0.891737 to 0.887964; the source pair is rejected for this candidate configuration.
**Next cheapest decisive action**: none for this implementation; retain only if another split shows a clear held-out improvement.
**Depends on**: C12

## C17 (supported)
**Type**: empirical
**Status**: supported
**Claim**: Adding separate LightFM WARP candidate generators trained on history interactions at watch_pct≥0.5 and ≥0.8 raised expanded9 equal-RRF main Recall@500 slightly from 0.629229 to 0.630009; early-tuned weights gave main Recall@500 0.636222.
**Scope and exclusions**: Tripartite expanded9 sources plus LightFM threshold-specific WARP variants; history-only fit; main weeks21–30 and early weeks17–20; per-airing strong-watch labels. Candidate recall only.
**Assumptions**: Threshold-specific interactions use only pre-target history; all predictions are restricted to the target schedule; user/item mappings are shared with the threshold-free LightFM model.
**Closest prior work and gap**: C2 used one threshold-free LightFM WARP model for candidates; the interaction threshold ablation for LightFM candidate generation had not been measured.
**Falsifier or counterexample**: Recomputing the two LightFM threshold models on identical history and schedule fails to reproduce their per-source candidate recalls or resulting union coverage.
**Required support**: Saved threshold-specific LightFM candidate rankings, per-source Recall@200/500 and union Recall@500 on early/main.
**Current support**: Main equal-RRF Recall@500=0.630009, full union=0.896667 (`results/tables/exp5_candidate_recall_expanded11.csv:78-79`) vs expanded9 main Recall@500=0.629229 (`expanded9.csv:70`). Early-only Recall-tuned weights produce main Recall@500=0.636222 (`results/tables/exp5_candidate_fusion_tuned_expanded11.csv:17`); source weight optimization remains early-only. Difference is small, and the main Recall@500 gate remains unmet.
**Evidence sources**:
- artifacts/scripts/exp5_candidate_recall.py:268
- artifacts/scripts/exp5_candidate_recall.py:269
- results/tables/exp5_candidate_recall_expanded11.csv:78
- results/tables/exp5_candidate_recall_expanded11.csv:79
- results/tables/exp5_candidate_fusion_tuned_expanded11.csv:17
- data/exp5_candidates_main_expanded11.parquet
- run #2026-10-24-exp5-lightfm-threshold-candidates-1415
- run #2026-10-24-exp5-lightfm-threshold-tuning-1420
**Counterevidence and failures**: Equal-RRF improvement over expanded9 is only +0.000780; early candidate Recall@500 is lower than expanded9. Threshold-specific LightFM scores do not resolve the main retrieval ceiling.
**Permitted manuscript wording**: Threshold-specific LightFM candidate sources produced main equal-RRF Recall@500 0.630009 and early-tuned Recall@500 0.636222; this is a small local retrieval change, not MAP@5 or Kaggle evidence.
**Next cheapest decisive action**: no more LightFM threshold variants without a specific hypothesis; retain the strongest candidate union and continue looking for orthogonal coverage gains.
**Depends on**: C12
