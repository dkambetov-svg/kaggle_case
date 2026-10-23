#!/usr/bin/env python3
"""EDA для sweettv-tv-program-recommender.

Только stdlib (csv, bisect, collections, datetime), потоковое чтение, одна детерминированная
стратегия сэмплирования (фиксированные шаги строк, без random). Печатает факт-лист в stdout
и записывает его же в docs/facts.md. Результаты шага 2 плана: покрывают период ~весь dataset.

Запуск: uv run python artifacts/scripts/eda.py
"""
from __future__ import annotations

import csv
import platform
import sys
import time
from array import array
from bisect import bisect_right
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT_MD = ROOT / "docs" / "facts.md"

VIEW_LOG = DATA / "dataset11-30.csv"
TRAIN_SCHED = [DATA / "export_arh_11-20-final.csv", DATA / "export_arh_21-30-final.csv"]
TEST_SCHED = DATA / "export_arh_31-42-final.csv"
ALL_SCHED = TRAIN_SCHED + [TEST_SCHED]
SUBMISSION = DATA / "submission.csv"

PROBE_STRIDE = 291  # детерминированная выборка строк view-лога для джойн-пробы
CONSIST_STRIDE = 9973  # шаг проверки duraton == stop-start

OUT: list[str] = []


def emit(line: str = "") -> None:
    OUT.append(line)
    print(line)


def iso_ts(s: str) -> int:
    """'YYYY-MM-DD HH:MM:SS' -> минуты-безопасная эпоха (ordinal*86400 + сек дня)."""
    day = date(int(s[0:4]), int(s[5:7]), int(s[8:10])).toordinal()
    return day * 86400 + int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])


_DAY_CACHE: dict[str, int] = {}


def sched_ts(s: str) -> int:
    """'DD.MM.YYYY HH:MM:SS' -> эпоха (ordinal*86400 + сек дня), день кешируется."""
    key = s[:10]
    day = _DAY_CACHE.get(key)
    if day is None:
        day = date(int(key[6:10]), int(key[3:5]), int(key[0:2])).toordinal()
        _DAY_CACHE[key] = day
    return day * 86400 + int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])


_WEEK_CACHE: dict[str, int] = {}


def iso_week(s: str) -> int:
    """ISO-неделя даты из 'YYYY-MM-DD HH:MM:SS' (первые 10 символов)."""
    key = s[:10]
    w = _WEEK_CACHE.get(key)
    if w is None:
        w = date(int(key[0:4]), int(key[5:7]), int(key[8:10])).isocalendar()[1]
        _WEEK_CACHE[key] = w
    return w


def pct(xs, q: float) -> float:
    """Ближайший по рангу перцентиль (не интерполяция) — детерминированно."""
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(q / 100 * (len(xs) - 1) + 0.5)))
    return xs[i]


def pcts(xs) -> str:
    return ", ".join(f"p{q}={pct(xs, q):,.0f}" for q in (1, 5, 25, 50, 75, 95, 99, 100))


def open_csv(path: Path):
    """UTF-8 (BOM-тоерантный), newline=''. Строгая декодировка — падай громко."""
    return path.open(newline="", encoding="utf-8-sig")


# ---------------------------------------------------------------- 1. Расписания
# Результат: [файл, статистика, каналы, id-множество, (по каналу массивы airing'ов)]
SchedStat = dict


def scan_schedule(path: Path) -> SchedStat:
    t0 = time.perf_counter()
    n = lab = title_ne = year_f = dir_f = act_f = dur_bad = id_bad = 0
    ids: set[str] = set()
    raw_ids: set[str] = set()
    raw_not_int: list[str] = []
    chans: set[str] = set()
    cat: Counter[str] = Counter()
    g1: Counter[str] = Counter()
    durs = array("l")
    titles: dict[str, set[str]] = {}
    weeks: Counter[int] = Counter()
    day_min = day_max = None
    with open_csv(path) as f:
        for r in csv.reader(f):
            if not r or r[0] == "channel_id":
                continue
            n += 1
            chans.add(r[0])
            st = sched_ts(r[2])
            if day_min is None or st < day_min:
                day_min = st
            if day_max is None or st > day_max:
                day_max = st
            d = date.fromordinal(st // 86400)
            weeks[d.isocalendar()[1]] += 1
            try:
                dur = int(r[3])
            except ValueError:
                dur = -1
                dur_bad += 1
            title = r[4].strip()
            if title:
                title_ne += 1
            raw = r[5].strip()
            raw_ids.add(raw)
            try:
                sid = int(raw)
            except ValueError:
                sid = 0
                id_bad += 1
                if len(raw_not_int) < 10 and raw:
                    raw_not_int.append(raw)
            if sid == 0:
                continue
            lab += 1
            skey = str(sid)
            ids.add(skey)
            if 0 < dur < 24 * 7 * 3600:
                durs.append(dur)
            cat[r[6].strip()] += 1
            if r[7].strip():
                g1[r[7].strip()] += 1
            if r[10].strip():
                year_f += 1
            if r[11].strip():
                dir_f += 1
            if r[12].strip():
                act_f += 1
            tt = titles.setdefault(skey, set())
            if title:
                tt.add(title)
    return {
        "path": path,
        "n": n, "lab": lab, "title_ne": title_ne,
        "ids": ids, "raw_ids": raw_ids, "raw_not_int": raw_not_int,
        "chans": chans, "cat": cat, "g1": g1, "durs": durs,
        "titles": titles, "weeks": weeks, "year_f": year_f,
        "dir_f": dir_f, "act_f": act_f,
        "dur_bad": dur_bad, "id_bad": id_bad,
        "day_min": day_min, "day_max": day_max,
        "sec": time.perf_counter() - t0,
    }


# ------------------------------------------------------- 2. Джойн-проба: airing'ы
# Индекс по каналам: ch -> (starts[], [(start, stop, dur, sid), ...]) только размеченные
# строки TRAIN-файлов (11-20, 21-30), отсортированные по start.
AirIndex = dict[str, tuple[list[int], list[tuple[int, int, int, str]]]]


def build_air_index(stats: list[SchedStat]) -> AirIndex:
    per_ch: dict[str, list[tuple[int, int, int, str]]] = {}
    for st in stats:
        cur = st["path"]
        if cur not in TRAIN_SCHED:
            continue
        with open_csv(cur) as f:
            for i, r in enumerate(csv.reader(f)):
                if i == 0:
                    continue
                try:
                    sid = int(r[5])
                except ValueError:
                    continue
                if sid == 0:
                    continue
                try:
                    dur = int(r[3])
                except ValueError:
                    continue
                s = sched_ts(r[2])
                per_ch.setdefault(r[0], []).append((s, s + dur, dur, str(sid)))
    idx: AirIndex = {}
    for ch, rows in per_ch.items():
        rows.sort(key=lambda t: t[0])
        idx[ch] = ([t[0] for t in rows], rows)
    return idx


def probe_overlap(idx: AirIndex, ch: str, start: int, stop: int):
    """Есть ли в индексе airing того же канала с пересечением [start, stop)?

    Возвращает (best или None, число всех пересечений, секунд перекрытия с best).
    best — airing с максимальным перекрытием. Смотрим максимум назад до 2 суток эфира.
    """
    arr = idx.get(ch)
    if arr is None:
        return None, 0, 0
    starts, data = arr
    j = bisect_right(starts, stop) - 1
    lo = bisect_right(starts, stop - 2 * 86400)
    best = None
    best_ov = 0
    n_matches = 0
    for k in range(j, lo - 1, -1):
        s, e, dur, sid = data[k]
        if s >= stop:
            continue
        if e <= start:
            break
        n_matches += 1
        ov = (e if e < stop else stop) - (s if s > start else start)
        if best is None or ov > best_ov:
            best = (s, e, dur, sid)
            best_ov = ov
    return (best if best else None), n_matches, best_ov


# ---------------------------------------------------------------- 3. View-лог
def scan_view_log(idx: AirIndex, channel_set: set[str]):
    t0 = time.perf_counter()
    t_start = t_end = None
    n = dur_eq0 = dur_lt0 = dur_bad = 0
    users: dict[str, list] = {}
    week_rows: Counter[int] = Counter()
    vsetv: Counter[str] = Counter()
    durs = array("l")
    cons = [0, 0, 0]  # проверено, совпало, max|diff|
    lab_rows = 0
    prob = {"n": 0, "miss_channel": 0, "no_air": 0, "match": 0, "multi": 0}
    frac_air = array("l")  # % программы (overlap/duration), накрывшаяся взглядом
    frac_view = array("l")  # % взгляда (overlap/duraton), лёгшего на программу

    with open_csv(VIEW_LOG) as f:
        for i, r in enumerate(csv.reader(f)):
            if i == 0:
                continue
            n += 1
            uid, ch, start, stop, dr = r
            try:
                d = int(dr)
            except ValueError:
                d = 0
                dur_bad += 1
            if d == 0:
                dur_eq0 += 1
            elif d < 0:
                dur_lt0 += 1
            elif i % 13 == 0:
                durs.append(d)
            s_t, e_t = iso_ts(start), iso_ts(stop)
            if t_start is None or start < t_start:
                t_start = start
            if t_end is None or start > t_end:
                t_end = start
            u = users.get(uid)
            if u is None:
                u = users[uid] = [0, 0, 0, start, start]
            u[0] += 1
            if d > 0:
                u[1] += d
            if ch in channel_set:
                u[2] += 1
            if ch in idx:
                lab_rows += 1
            week_rows[iso_week(start)] += 1
            vsetv[ch] += 1
            if i % CONSIST_STRIDE == 0:
                cons[0] += 1
                diff = e_t - s_t - d
                if diff == 0:
                    cons[1] += 1
                cons[2] = max(cons[2], abs(diff))
            if i % PROBE_STRIDE == 0:
                prob["n"] += 1
                if ch not in idx:
                    prob["miss_channel"] += 1
                else:
                    air, n_m, ov = probe_overlap(idx, ch, s_t, e_t)
                    if air is None or ov <= 0:
                        prob["no_air"] += 1
                    else:
                        prob["match"] += 1
                        if n_m > 1:
                            prob["multi"] += 1
                        air_dur = air[2]
                        if air_dur > 0:
                            frac_air.append((ov * 100) // air_dur)  # % программы, попавшая во взгляд
                        if d > 0:
                            frac_view.append((ov * 100) // d)  # % взгляда, легшего на программу (cap 1000)
    return {
        "n": n, "dur_eq0": dur_eq0, "dur_lt0": dur_lt0, "dur_bad": dur_bad,
        "users": users, "week_rows": week_rows, "vsetv": vsetv, "lab_rows": lab_rows,
        "durs": durs, "cons": cons, "prob": prob,
        "frac_air": frac_air, "frac_view": frac_view,
        "t_start": t_start, "t_end": t_end,
        "sec": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------- 4. Submission
def scan_submission():
    users: list[str] = []
    zero_rows = 0
    other = Counter()
    with open_csv(SUBMISSION) as f:
        for i, r in enumerate(csv.reader(f)):
            if i == 0:
                continue
            users.append(r[0])
            v = r[1].strip() if len(r) > 1 else ""
            if v == "0 0 0 0 0":
                zero_rows += 1
            else:
                other[v] += 1
    return users, zero_rows, other


# ---------------------------------------------------------------- main
def human(dur: float) -> str:
    if dur < 60:
        return f"{dur:.1f} s"
    return f"{dur/60:.1f} min"


def ordinal_to_iso(day_sec: int) -> str:
    return date.fromordinal(day_sec // 86400).isoformat()


def main() -> None:  # noqa: D103
    t_all = time.perf_counter()

    emit("# Sweet.tv recommender — факты EDA (шаг 2)")
    emit()
    emit(f"Сгенерировано: {date.today().isoformat()}, скрипт `artifacts/scripts/eda.py` (stdlib, streaming).")
    emit()
    emit("## 0. Среда")
    emit()
    emit(f"- Python: `{platform.python_version()}`, платформа: `{platform.platform()}`")
    emit(f"- Правило чтения: stdlib `csv`, `utf-8-sig`, потоково; сэмплинг детерминированный (шаги строк).")

    emit()
    emit("## 1. Сканирование расписаний")
    sched_stats = [scan_schedule(p) for p in ALL_SCHED]
    for st in sched_stats:
        p: Path = st["path"]
        emit()
        emit(f"### {p.name}  ({human(st['sec'])})")
        emit(f"- строк всего: **{st['n']:,}**; с `tv_show_id != 0`: **{st['lab']:,} ({st['lab']/st['n']*100:.1f}%)**;"
             f" с непустым заголовком: {st['title_ne']:,}")
        emit(f"- канал-ов уникальных: {len(st['chans']):,}; программ (id) размеченных: **{len(st['ids']):,}**")
        emit(f"- длительности размеченных (`duration`): {pcts(st['durs'])}")
        emit(f"- категорий уникальных: {len(st['cat'])}; топ-9: {st['cat'].most_common(9)}")
        emit(f"- genre_1 топ-5: {st['g1'].most_common(5)}")
        emit(f"- заполненность у размеченных: year={st['year_f']:,} dir={st['dir_f']:,} actors={st['act_f']:,}")
        emit(f"- id→titles: один заголовок у {sum(1 for v in st['titles'].values() if len(v)==1):,};"
             f" >1 заголовка у {sum(1 for v in st['titles'].values() if len(v)>1):,};"
             f" максимум заголовков на id: {max((len(v) for v in st['titles'].values()), default=0)}")
        emit(f"- span start_time: {ordinal_to_iso(st['day_min'])} … {ordinal_to_iso(st['day_max'])}")
        wk = f"недели ISO в файле: {dict(sorted(st['weeks'].items()))}"
        emit(f"- {wk}")
        emit(f"- [сверка id] уникальных raw-строк tv_show_id: {len(st['raw_ids']):,};"
             f" не парсится int: {st['id_bad']:,}")

    ids = [st["ids"] for st in sched_stats]
    chans_all: set[str] = set()
    for st in sched_stats:
        chans_all |= st["chans"]
    emit()
    emit("### Перекрытие программ между файлами")
    emit(f"- |ids(11-20)|={len(ids[0]):,}, |ids(21-30)|={len(ids[1]):,}, |ids(31-42)|={len(ids[2]):,}")
    emit(f"- |11-20 ∩ 21-30| = {len(ids[0] & ids[1]):,}; |11-20 ∩ 21-30 ∩ 31-42| = {len(ids[0] & ids[1] & ids[2]):,}")
    emit(f"- новых программ в тесте (31-42), отсутствующих в train: **{len(ids[2] - ids[0] - ids[1]):,}**")
    emit(f"- канал-ов во всех файлах (union): {len(chans_all):,}")

    emit()
    emit("## 2. Джойн-индекс = train-расписания (11-20, 21-30), размеченный эфир")
    t_idx = time.perf_counter()
    idx = build_air_index(sched_stats)
    n_air = sum(len(v[1]) for v in idx.values())
    emit(f"- каналов с размеченным эфиром: {len(idx):,}; airing'ов (labelled): {n_air:,}  ({human(time.perf_counter() - t_idx)} на сборку)")

    emit()
    emit("## 3. View-лог dataset11-30.csv")
    t_view = time.perf_counter()
    view = scan_view_log(idx, chans_all)
    known = sum(u[2] for u in view["users"].values())
    emit(f"- строк: **{view['n']:,}**; span start_time: **{view['t_start']} … {view['t_end']}**  ({human(view['sec'])})")
    emit(f"- duraton: {pcts(view['durs'])}  (сэмпл 1/13, детерминированный)")
    emit(f"- строк с vsetv_id ∈ каналов расписаний: {known:,}/{view['n']:,} = {known/view['n']*100:.1f}%")
    emit(f"- из них на каналах, где ЕСТЬ размеченный эфир (лабель-каналы): **{view['lab_rows']:,}** ({view['lab_rows']/view['n']*100:.1f}%)")
    emit(f"- duraton == 0: {view['dur_eq0']:,}; < 0: {view['dur_lt0']:,}; не парсится: {view['dur_bad']:,}")
    emit(f"- из них на каналах, где ЕСТЬ размеченный эфир (лабель-каналы): **{view['lab_rows']:,}** ({view['lab_rows']/view['n']*100:.1f}%)")
    emit(f"- duraton == 0: {view['dur_eq0']:,}; < 0: {view['dur_lt0']:,}; не парсится: {view['dur_bad']:,}")
    c = view["cons"]
    emit(f"- проверка `duraton == stop-start` (сэмпл 1/{CONSIST_STRIDE}): совпало {c[1]}/{c[0]}"
         f" ({c[1]/max(1,c[0])*100:.1f}%); max |diff| = {c[2]:,} s")
    emit(f"- пользователей: **{len(view['users']):,}**")
    nv = array("l", [u[0] for u in view["users"].values()])
    sv = array("l", [u[1] for u in view["users"].values()])
    emit(f"- просмотров на пользователя: {pcts(nv)}")
    emit(f"- сумм-просмотр (сек) на пользователя: {pcts(sv)}")
    weak = sum(1 for u in view["users"].values() if u[0] < 5)
    emit(f"- «слабых» пользователей (<5 просмотров): {weak:,} ({weak/len(view['users'])*100:.1f}%)")
    emit(f"- просмотров по неделям ISO: {dict(sorted(view['week_rows'].items()))}")
    emit(f"- vsetv_id уникальных: {len(view['vsetv']):,}; топ-16: {view['vsetv'].most_common(16)}")
    known = sum(u[2] for u in view["users"].values())
    emit(f"- строк с vsetv_id ∈ каналов расписаний: {known:,}/{view['n']:,} = {known/view['n']*100:.1f}%")

    emit()
    emit("## 4. Джойн-проба (сэмпл 1/{}, детерминированный)".format(PROBE_STRIDE))
    p = view["prob"]
    if p["n"]:
        emit(f"- проб: {p['n']:,}")
        emit(f"- канал НЕ из расписаний (union всех файлов): {p['miss_channel']:,};"
             f" пересечение с размеченным эфиром не найдено: {p['no_air']:,};"
             f" найдено: **{p['match']:,}** ({p['match']/p['n']*100:.1f}%)")
        emit(f"- из найденных с >1 airing-кандидатом: {p['multi']:,} ({p['multi']/max(1,p['match'])*100:.1f}%)")
        emit(f"- % программы, накрывшаяся взглядом (overlap/duration): {pcts(view['frac_air'])}")
        emit(f"- % взгляда, легшего на программу (overlap/duraton): {pcts(view['frac_view'])}")

    emit()
    emit("## 5. Submission и cold-start")
    users, zero_rows, other = scan_submission()
    vu = view["users"]
    hist = [u for u in users if u in vu]
    emit(f"- пользователей: {len(users):,}; строк-шаблонов `0 0 0 0 0`: {zero_rows:,}; иных значений: {sum(other.values()):,}")
    emit(f"- есть история в view-логе: {len(hist):,} ({len(hist)/len(users)*100:.1f}%);"
         f" **cold-start (нет истории): {len(users)-len(hist):,} ({(len(users)-len(hist))/len(users)*100:.1f}%)**")
    emit(f"- (history, но только на неразмеченном эфире — считать на шаге 3 после полного джойна)")

    total = time.perf_counter() - t_all
    emit()
    emit(f"Время всего скрипта: {human(total)}.")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(OUT) + "\n", encoding="utf-8")
    print(f"\n[saved] {OUT_MD}", file=sys.stderr)


if __name__ == "__main__":
    main()
