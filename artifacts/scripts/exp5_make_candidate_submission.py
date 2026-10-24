#!/usr/bin/env python3
"""Create an exp5 candidate-only submission from the MAP@5-tuned RRF blend.

This is a submission generator only; it does not train a reranker or overwrite
the designated root submission.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CANDIDATES = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "exp5_candidates_final_expanded9.parquet"
WEIGHTS = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "tables" / "exp5_candidate_map_weights_propensity_widegrid.csv"
OUTPUT = Path(sys.argv[3]) if len(sys.argv) > 3 else DATA / "submission_exp5_candidate_map_tuned.csv"
TARGET_THRESHOLD = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8
if not CANDIDATES.is_absolute(): CANDIDATES = ROOT / CANDIDATES
if not WEIGHTS.is_absolute(): WEIGHTS = ROOT / WEIGHTS
if not OUTPUT.is_absolute(): OUTPUT = ROOT / OUTPUT
SOURCES = ["repeat", "lightfm", "ppr", "popularity", "tv_slot", "user_channel",
           "user_metadata", "als", "user_time_slot", "user_channel_dow",
           "user_channel_hour", "airing_popularity", "prop_max", "prop_mean",
           "prop_sum", "prop_logsumexp"]
RRF_K = 60.0


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    template = pd.read_csv(DATA / "submission.csv", dtype={"user_id": str})
    candidates = pd.read_parquet(CANDIDATES)
    candidates["user_id"] = candidates.user_id.astype(str)
    candidates["tv_show_id"] = candidates.tv_show_id.astype(np.int64)
    weights_df = pd.read_csv(WEIGHTS)
    weights_df = weights_df[weights_df.target_threshold.eq(TARGET_THRESHOLD)].set_index("source")
    weights = weights_df.weight.reindex(SOURCES).fillna(0.0).to_numpy(dtype=float)
    rank_matrix = candidates[[f"rank_{source}" for source in SOURCES]].to_numpy(dtype=float)
    components = np.divide(1.0, RRF_K + 1.0 + rank_matrix,
                           out=np.zeros_like(rank_matrix), where=np.isfinite(rank_matrix))
    candidates["score"] = components @ weights

    output_rows = []
    missing_users = []
    for user in template.user_id:
        group = candidates[candidates.user_id.eq(user)]
        if group.empty:
            missing_users.append(user)
            continue
        top = group.sort_values(["score", "tv_show_id"], ascending=[False, True], kind="stable").tv_show_id.tolist()
        unique = list(dict.fromkeys(top))
        if len(unique) < 5:
            raise ValueError(f"User {user} has only {len(unique)} unique candidates")
        output_rows.append({"user_id": user, "tv_show_id": " ".join(map(str, unique[:5]))})
    if missing_users:
        raise ValueError(f"No candidates for {len(missing_users)} template users; first={missing_users[:5]}")
    out = pd.DataFrame(output_rows)
    if len(out) != len(template) or out.user_id.tolist() != template.user_id.tolist():
        raise AssertionError("Submission rows/order differ from template")
    schedule = pd.read_csv(DATA / "export_arh_31-42-final.csv", usecols=["tv_show_id", "start_time"], low_memory=False)
    schedule["start_time"] = pd.to_datetime(schedule.start_time, dayfirst=True)
    weeks = schedule.start_time.dt.isocalendar().week
    catalog = set(schedule.loc[weeks.between(31, 42) & schedule.tv_show_id.ne(0), "tv_show_id"].astype(int))
    for ids in out.tv_show_id:
        parsed = list(map(int, ids.split()))
        if len(parsed) != 5 or len(set(parsed)) != 5 or not set(parsed).issubset(catalog):
            raise AssertionError("Invalid prediction row or show outside weeks31-42 schedule")
    out.to_csv(OUTPUT, index=False)
    print(f"[exp5] output={OUTPUT}; rows={len(out)}; predictions_per_user=5; all IDs in target schedule")
    print(f"[exp5] unique scheduled shows in output={len(set(int(x) for row in out.tv_show_id for x in row.split()))}")
    print(f"[exp5] target threshold={TARGET_THRESHOLD}; source weights:", dict(zip(SOURCES, weights.tolist())))


if __name__ == "__main__":
    main()
