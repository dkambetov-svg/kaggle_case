#!/usr/bin/env python3
"""Финальные сабмиты для 3 лучших по строгой валидации методов.
Обучение: недели 11-30 (весь train). Кандидаты: export_arh_31-42 (tv_show_id != 0)."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def main():
    inter = pd.read_parquet(DATA / "user_interactions.parquet")
    inter["user_id"] = inter["user_id"].astype(str)
    train = inter[inter["week"] <= 30].copy()

    sub = pd.read_csv(DATA / "submission.csv", dtype={"user_id": str})

    sched42 = pd.read_csv(DATA / "export_arh_31-42-final.csv")
    candidates = sched42[sched42["tv_show_id"] != 0]["tv_show_id"].unique()
    candidates_set = set(candidates)

    pop_recent4w = (
        train[train["week"] >= 27].groupby("tv_show_id")["user_id"].nunique().sort_values(ascending=False)
    )
    pop_recent4w = pop_recent4w[pop_recent4w.index.isin(candidates_set)]
    top5_pop = pop_recent4w.index.tolist()[:5]

    # (b) популярность x log(1+airings в целевом окне 31-42)
    airing_count_42 = sched42[sched42["tv_show_id"] != 0]["tv_show_id"].value_counts()
    score_b = {}
    for show in candidates_set:
        pr = pop_recent4w.get(show, 0)
        ac = airing_count_42.get(show, 0)
        score_b[show] = pr * np.log1p(ac)
    top5_b = [it for it, _ in sorted(score_b.items(), key=lambda kv: -kv[1])[:5]]

    own_sec = train.groupby(["user_id", "tv_show_id"])["overlap_seconds"].sum()
    own_idx0 = set(own_sec.index.get_level_values(0))

    def repeat_fill(fallback_top5):
        preds = {}
        for u in sub["user_id"]:
            if u in own_idx0:
                own_u = own_sec.loc[u]
                own_u = own_u[own_u.index.isin(candidates_set)].sort_values(ascending=False)
                top = own_u.index.tolist()
                for p in fallback_top5:
                    if p not in top:
                        top.append(p)
                preds[u] = top[:5]
            else:
                preds[u] = fallback_top5
        return preds

    preds_repeat = repeat_fill(top5_pop)
    preds_repeat_b = repeat_fill(top5_b)
    preds_b = {u: top5_b for u in sub["user_id"]}

    def write_submission(preds, path):
        out = sub.copy()
        out["tv_show_id"] = out["user_id"].map(lambda u: " ".join(str(int(x)) for x in preds[u][:5]))
        assert len(out) == len(sub)
        assert (out["user_id"].values == sub["user_id"].values).all()
        for ids in out["tv_show_id"]:
            parts = ids.split()
            assert len(parts) == 5
            for p in parts:
                assert int(p) in candidates_set
        out.to_csv(path, index=False)
        print(f"[saved] {path}  rows={len(out)}")

    write_submission(preds_repeat, DATA / "submission_repeat_full.csv")
    write_submission(preds_repeat_b, DATA / "submission_repeat_plus_b.csv")
    write_submission(preds_b, DATA / "submission_pop_x_log_airings.csv")


if __name__ == "__main__":
    main()
