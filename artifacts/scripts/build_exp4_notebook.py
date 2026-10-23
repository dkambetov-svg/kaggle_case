#!/usr/bin/env python3
"""Сборка ноутбука experiments/*-exp4-round.ipynb из реального выполнения ячеек.

Как build_exp3_notebook.py: каждый code-cell выполняется через exec с перенаправлением
stdout, вывод встраивается в ноутбук (capture-at-creation). Числа берутся из
таблиц results/tables/exp4_*.csv и логов Kaggle, никаких выдуманных значений.
"""
import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CELLS = [
    ("md", """# exp4 — раунд увеличения скора: больше эпох ALS, PPR-рефинемент, fusion-доводка, размер пула

**Дата:** 2026-10-23. **Seed:** 42 везде. **Контекст:** победитель exp3 —
`exp3_e3b` (RRF k=20 w=(1,2,1) + PPR E1 aw=0.5 без decay + ALS 64/0.05/20),
Kaggle 0.34487/0.32361. Цель раунда — найти конфигурацию лучше.

## Гипотезы
- **E5 («больше эпох»)**: ALS — самый слабый компонент (отдельно 0.140/0.092);
  больше итераций/факторов усилит его вклад в fusion.
- **E6**: оптимум `attr_weight` может быть выше 0.5 (в E1 ряд был монотонный);
  `restart ∈ {0.3,0.5,0.7}`. Плюс численная проверка: при alpha=1−restart
  сжатие alpha^30, итераций 30 достаточно?
- **E7**: в окрестности победителя (k, w_ppr, w_als) есть близкая точка лучше.
- **E8**: пул топ-500 урезает кандидатов; 1000/2000 дадут рост recall.

## Метод
Оба сплита (main: train≤20/val21–30; early: train≤16/val17–20), строгая
релевантность, join = канал + overlap ≥300s или ≥50%. Локальный фильтр:
не хуже победителя exp3 (0.5365/0.4632) на обоих сплитах. Решения по Kaggle."""),
    ("code", """import pandas as pd
e5 = pd.read_csv("results/tables/exp4_als.csv")
print(e5.to_string(index=False))"""),
    ("md", """**E5 результат.** Таблица `exp4_als.csv`: ALS-гиперпараметры, fusion MAP@5
(к k=20 w=(1,2,1), PPR E1 фиксирован) и ALS отдельно на обоих сплитах."""),
    ("code", """e6 = pd.read_csv("results/tables/exp4_ppr.csv")
print(e6.to_string(index=False))"""),
    ("md", """**E6 результат.** Сетка restart × attr_weight + строка референса
(0.32306/0.29313 из exp3_e1.csv). Численная проверка сходимости —
`exp4_ppr_convergence.csv`: max|score(30 iter) − score(100 iter)|."""),
    ("code", """conv = pd.read_csv("results/tables/exp4_ppr_convergence.csv")
print(conv.groupby(["restart", "alpha"])["max_abs_score_diff"]
        .agg(["max", "mean"]).to_string())
print("\\nalpha^30 (теоретическое сжатие):")
print(conv.groupby("restart")["contraction_alpha_pow30"].first().to_string())"""),
    ("md", """**Численная проверка сходимости.** Если max|Δ| на уровне 1e-9 при alpha=0.5
(теория: 0.5^30 ≈ 9.3e-10), то 30 итераций достаточно и увеличивать эпохи PPR
бессмысленно. Для restart=0.3 (alpha=0.7) сжатие 0.7^30 ≈ 2.3e-5 — разность
должна быть больше, но по-прежнему мала для порядка рангов."""),
    ("code", """e7 = pd.read_csv("results/tables/exp4_fusion.csv")
print(e7.head(10).to_string(index=False))"""),
    ("md", """**E7 результат.** Fusion-доводка из 18 комбинаций (k ∈ {10,20,40},
w_ppr ∈ {1.5,2,3}, w_als ∈ {0.5,1}) поверх лучших ALS/PPR из E5/E6."""),
    ("code", """e8 = pd.read_csv("results/tables/exp4_pool.csv")
print(e8.to_string(index=False))"""),
    ("md", """**E8 результат.** Размер пула 500/1000/2000 с конфигурацией-победителем."""),
]


def main():
    cells = []
    shared = {"__name__": "__main__", "pd": __import__("pandas")}
    for kind, src in CELLS:
        cell = {"cell_type": "markdown" if kind == "md" else "code",
                "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "md":
            cells.append(cell)
            continue
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(compile(src, "<cell>", "exec"), shared)
            status = "ok"
        except Exception as exc:  # noqa: BLE001 — фиксируем в ноутбуке
            buf.write(f"\n[ERROR] {type(exc).__name__}: {exc}")
            status = "error"
        cell["execution_count"] = None
        cell["outputs"] = [{"output_type": "stream", "name": "stdout",
                            "text": buf.getvalue().splitlines(keepends=True)}]
        cell["metadata"] = {"execution_status": status}
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": sys.version.split()[0]},
            "exp4": {"seed": 42, "date": "2026-10-23",
                     "context": "победитель exp3 exp3_e3b 0.34487/0.32361"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = ROOT / "experiments" / f"{datetime.now():%Y-%m-%d-%H%M}-exp4-round.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = sum(1 for c in cells
              if c["cell_type"] == "code"
              and c["metadata"].get("execution_status") != "ok")
    print(f"[saved] {out} ({out.stat().st_size} bytes, {len(cells)} cells, "
          f"errors={bad})")


if __name__ == "__main__":
    main()
