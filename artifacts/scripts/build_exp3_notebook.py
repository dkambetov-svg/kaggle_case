#!/usr/bin/env python3
"""Сборка ноутбука experiments/*-exp3-round.ipynb из реального выполнения ячеек.

Каждый code-cell выполняется через exec с перенаправлением stdout; вывод
встраивается в ноутбук как outputs (capture-at-creation). Никаких выдуманных
чисел: только то, что реально напечатано скриптом.
"""
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CELLS = [
    ("md", """# exp3 — раунд улучшений: атрибутный PPR, свежесть, взвешенный RRF, буст эфиров

**Дата:** 2026-10-23 (сабмиты 01:39–01:45 MSK). **Seed:** 42 везде.

## Гипотезы / намерение
- **E1**: трёхдольный граф (пользователь–программа–атрибут: канал/категория/жанры)
  улучшит PPR против двудольного; `attr_weight ∈ {0.1, 0.3, 0.5}`.
- **E2**: временной decay весов `exp(-Δ/τ)`, `τ ∈ {4, 8}`, два варианта restart.
- **E3**: взвешенный RRF `score = Σ w_c/(k+rank_c)`, сетка 54 комбинации.
- **E4**: буст по числу эфиров поверх PPR `PPR × (1+airings)^a`, `a ∈ {0.25, 0.5}`.

## Жёсткие правила раунда
Тестовые просмотры 31–42 не используются (только расписание: связи и эфиры).
Ни одна существующая таблица/версия не перезаписывается, новые — с префиксом `exp3_`.
Локальный фильтр перед сабмитом: не хуже текущего финала (0.5015/0.4318) на обоих
сплитах; решения по Kaggle. Максимум 5 сабмитов.

## Метод (кратко)
Оба сплита: main (train ≤20, val 21–30) и early (train ≤16, val 17–20); строгая
релевантность (пересечение ≥50% длительности); пул топ-500 по pop×log(1+эфиру).
PPR: батч `scipy.sparse`, 30 итераций, restart 0.5. Диагностика холодных программ:
глобальный счётчик (видна ли программа в train) и пер-юзерный (не видел этот юзер)."""),
    ("code", """import pandas as pd
e1 = pd.read_csv("results/tables/exp3_e1.csv")
print(e1[["variant","MAP5_main","MAP5_early","MAP5_mean","time_s","seed"]].to_string(index=False))"""),
    ("md", """**Результат E1.** Атрибуты помогают: рост монотонный по `attr_weight`, лучший aw=0.5
(0.5427/0.4643). Контроль aw=0 точно воспроизводит прежний двудольный PPR
(0.533331/0.457684) — реализация проверена. Глобально «нетренированных» программ в
топ-5 всегда 0 — структурно: пул топ-500 целиком состоит из программ, виденных в
train (pop×airings > 0 у всех 500). Пер-юзерно не видел: 59 слотов (16 релевантных)
на main и 22 (2) на early.

**Найденные и исправленные баги E1/E2:**
1. Нормировка: эффективная доля атрибутов была ~0.0007 (сырые Σw ≈ 622 против aw ≤ 0.5).
   Исправлено нормированием внутри групп ДО смешивания: item→user = (1−aw)·w/Σw,
   item→attr = aw/|attrs|.
2. `week` имеет dtype UInt32: унарный минус `-(max_week − week)` переполнял unsigned
   (−9 → 4294967287) → `exp` → inf → inf/inf = NaN (514200 битых весов), τ=4 и τ=8
   давали идентичный мусор. Исправлено приведением к int64 + `assert np.isfinite`."""),
    ("code", """e2 = pd.read_csv("results/tables/exp3_e2.csv")
print(e2[["variant","MAP5_main","MAP5_early","MAP5_mean","time_s","seed"]].to_string(index=False))"""),
    ("md", """**Результат E2.** Свежесть даёт пренебрежимый прирост: лучший tau=8 restart=raw
(0.503899 против baseline 0.503493, +0.0004). Затухание в restart-векторе вредит
(tau=4 decayed: 0.498279)."""),
    ("code", """e3 = pd.read_csv("results/tables/exp3_e3.csv")
grid = e3[e3["variant"].str.startswith("E3")]
abl = e3[e3["ablation_pop_ppr_no_als"] == True]
print("=== лучшие 5 из сетки 54 ===")
print(grid.head(5)[["variant","MAP5_main","MAP5_early","MAP5_mean"]].to_string(index=False))
print("\\n=== контроль и справочные строки ===")
print(e3[~e3["variant"].str.startswith("E3")][["variant","MAP5_main","MAP5_early","MAP5_mean"]].to_string(index=False))
print(f"\\nвсего сеточных вариантов: {len(grid)}; лучший из абляции pop+PPR без ALS:")
print(abl.head(1)[["variant","MAP5_main","MAP5_early","MAP5_mean"]].to_string(index=False))"""),
    ("md", """**Результат E3.** Лучший fusion-точка: w=(1,2,1), k=20 (0.5357/0.4626), но **все
fusion-комбинации локально хуже чистого PPR_best** (0.5433/0.4645) — знак известного
расхождения локальной валидации и Kaggle (там fusion был лучше чистого PPR)."""),
    ("code", """e4 = pd.read_csv("results/tables/exp3_e4.csv")
print(e4[["variant","MAP5_main","MAP5_early","MAP5_mean"]].to_string(index=False))"""),
    ("md", """**Результат E4 — провал.** Буст по эфирам поверх PPR локально вредит: лучший
a=0.25 отдельно 0.4492/0.3857 против референса 0.5125/0.4371. Не прошёл локальный
фильтр «не хуже финала (0.5015/0.4318) на обоих сплитах» → **не отправлен** (правило
раунда)."""),
    ("code", """e3b = pd.read_csv("results/tables/exp3_e3_ppr_e1.csv")
print(e3b[["variant","MAP5_main","MAP5_early","MAP5_mean","seed"]].to_string(index=False))"""),
    ("md", """**5-й кандидат (на усмотрение).** Гипотеза по Kaggle: decay слегка вредит
(E1 0.32306 > E2 0.32261 на public). Fusion-конфигурация лучшего E3, но с PPR E1
без decay: локально 0.5365/0.4632 — не хуже decayed-версии на обоих сплитах, проходит
фильтр → отправлен."""),
    ("code", """# Итоговые сабмиты Kaggle (вывод `kaggle competitions submissions`, раунд exp3)
subs = [
    ("submission_exp3_e3b.csv", "RRF k=20 w=(1,2,1), PPR=E1 без decay", 0.34487, 0.32361),
    ("submission_exp3_e3.csv",  "RRF k=20 w=(1,2,1), PPR=E1+E2",        0.34368, 0.32311),
    ("submission_exp3_disc.csv", "RRF k=60 1/1/1 (структура финала), PPR=E1+E2", 0.33331, 0.31547),
    ("submission_exp3_e1.csv",  "чистый трёхдольный PPR aw=0.5",        0.32306, 0.29313),
    ("submission_exp3_e2.csv",  "PPR aw=0.5 + decay tau=8",             0.32261, 0.29323),
]
prev_final = ("submission_rrf2_a_ppr05_als.csv", "RRF k=60, бипартитный PPR restart=0.5", 0.32691, 0.31441)
df = pd.DataFrame(subs, columns=["file","components","public","private"])
prev_public, prev_private = prev_final[2], prev_final[3]
df["лучше финала (оба)"] = (df["public"] > prev_public) & (df["private"] > prev_private)
new_row = pd.DataFrame([{"file": prev_final[0], "components": prev_final[1],
                         "public": prev_public, "private": prev_private}])
df = pd.concat([df, new_row], ignore_index=True)
df["лучше финала (оба)"] = df["лучше финала (оба)"].map(
    {True: "ДА", False: "нет"})
df.loc[df["file"] == prev_final[0], "лучше финала (оба)"] = "— это финал —"
print(f"Прежний финал {prev_final[0]}: public={prev_public} / private={prev_private}")
print(df.to_string(index=False))"""),
    ("md", """## Итог раунда
5 из 5 сабмитов использованы. Победитель — **exp3_e3b: 0.34487 / 0.32361**, лучше
прежнего финала и по public (0.32691), и по private (0.31441) — квалифицируется по
правилу «лучше обоих метрик». Рост: +0.01796 public, +0.00920 private.

## Выводы
1. **Сработало:** атрибуты (канал/категория/жанры) в графе PPR внутри взвешенного
   fusion — единственная комбинация, прошедшая Kaggle-фильтр с большим запасом.
2. **Не сработало:** буст эфиров поверх PPR (E4, отвергнут локально); decay в
   restart-векторе; чистый PPR (E1/E2 без fusion) на Kaggle хуже финала — fusion
   остаётся нужным.
3. **Локальная валидация согласилась с Kaggle** по знаку: атрибуты помогают; fusion
   компоненты уместны.
4. **Локальная валидация разошлась с Kaggle:** decay локально +0.0004, на Kaggle
   E2 < E1; чистый PPR локально сильно лучше контроля финала, на Kaggle — хуже.
   Окончательные решения принимались по Kaggle, как и в прошлых раундах."""),
]

SHARED_GLOBALS = {"__name__": "__main__", "pd": __import__("pandas")}


def main():
    cells = []
    for kind, src in CELLS:
        cell = {"cell_type": "markdown" if kind == "md" else "code",
                "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "md":
            cells.append(cell)
            continue
        buf = io.StringIO()
        exec_globals = SHARED_GLOBALS  # общий namespace: ячейки используют pd из первой
        try:
            with redirect_stdout(buf):
                exec(compile(src, "<cell>", "exec"), exec_globals)
            status = "ok"
        except Exception as exc:  # noqa: BLE001 — фиксируем в ноутбуке
            buf.write(f"\n[ERROR] {type(exc).__name__}: {exc}")
            status = "error"
        text = buf.getvalue()
        cell["execution_count"] = None
        cell["outputs"] = [{"output_type": "stream", "name": "stdout",
                            "text": text.splitlines(keepends=True)}]
        cell["metadata"] = {"execution_status": status}
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": sys.version.split()[0]},
            "exp3": {"seed": 42, "date": "2026-10-23",
                     "winner": "submission_exp3_e3b.csv 0.34487/0.32361"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = ROOT / "experiments" / f"{datetime.now():%Y-%m-%d-%H%M}-exp3-round.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] {out} ({out.stat().st_size} bytes, {len(cells)} cells)")


if __name__ == "__main__":
    main()
