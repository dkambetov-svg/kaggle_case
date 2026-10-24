#!/usr/bin/env python3
"""One fixed Phase A experiment; no tuning against weeks 21–30."""
import contextlib
import gc
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from e9_data import build_features, load_split
from e9_models import FEATURES, LGBM_CONFIG, LIGHTFM_CONFIG, fit_predict, generate_candidates

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TABLES = ROOT / "results" / "tables"


def evaluate(predictions, relevant, users, k=5, recall=False):
    """Macro average over fixed users with nonempty truth; missing predictions score zero."""
    values = []
    for user in users:
        actual = relevant[user]
        predicted = list(dict.fromkeys(predictions.get(user, [])))[:k]
        hits, ap = 0, 0.0
        for rank, item in enumerate(predicted, 1):
            if item in actual:
                hits += 1
                ap += hits / rank
        values.append(hits / len(actual) if recall else ap / min(k, len(actual)))
    if not values:
        raise ValueError("No evaluable users")
    return float(np.mean(values))


def prepare(split, users):
    history, past, future, truth = load_split(*split)
    print(f"split={split}: history_rows={len(history):,}, future_shows={future.tv_show_id.nunique():,}", flush=True)
    candidates = generate_candidates(history, future, users)
    baselines = {name: candidates.attrs[name] for name in ("repeat_top5", "lightfm_top5")}
    candidates.attrs = {}
    assert not candidates.duplicated(["user_id", "tv_show_id"]).any()
    assert candidates.tv_show_id.isin(future.tv_show_id).all()
    assert set(candidates.user_id) == set(users)
    assert candidates.groupby("user_id").size().max() <= 500
    features = build_features(history, past, future, candidates)
    truth = truth[truth.user_id.isin(users)].copy()
    features = features.merge(truth.assign(label=1), on=["user_id", "tv_show_id"], how="left", validate="one_to_one")
    features["label"] = features.label.fillna(0).astype("int8")
    print(f"split={split}: candidate_rows={len(features):,}, positive_rows={int(features.label.sum()):,}", flush=True)
    return features, truth, baselines


def main():
    start = time.monotonic()
    toy_truth = {"a": {1, 3}, "b": {8}}
    toy_predictions = {"a": [1, 1, 2, 3]}
    assert np.isclose(evaluate(toy_predictions, toy_truth, ["a", "b"]), 5 / 12)
    assert np.isclose(evaluate(toy_predictions, toy_truth, ["a", "b"], 2, True), 0.25)
    print("Metric checks passed: duplicates, missing predictions, AP denominator, macro recall.")
    users = pd.read_csv(DATA / "submission.csv", dtype={"user_id": str}).user_id.tolist()
    assert len(users) == len(set(users))
    # Supervised examples are wholly inside 11–20; final candidate models use 11–20.
    train, _, _ = prepare((16, 17, 20), users)
    gc.collect()
    validation, truth, baselines = prepare((20, 21, 30), users)
    gc.collect()
    assert train.label.nunique() == 2
    scores, model = fit_predict(train, validation)
    assert len(scores) == len(validation) and np.isfinite(scores).all()
    validation["reranker_score"] = scores
    relevant = truth.groupby("user_id").tv_show_id.agg(set).to_dict()
    eval_users = [u for u in users if relevant.get(u)]
    candidates = validation.sort_values(["user_id", "candidate_rank"]).groupby("user_id").tv_show_id.agg(list).to_dict()
    reranked = validation.sort_values(["user_id", "reranker_score", "tv_show_id"], ascending=[True, False, True])
    top5 = reranked.groupby("user_id", sort=False).head(5).groupby("user_id").tv_show_id.agg(list).to_dict()
    metrics = {f"candidate Recall@{k}": evaluate(candidates, relevant, eval_users, k, True) for k in (100, 200, 500)}
    for label, predictions in (("repeat baseline", baselines["repeat_top5"]), ("LightFM", baselines["lightfm_top5"]), ("LightGBM reranker", top5)):
        metrics[f"MAP@5 {label}"] = evaluate(predictions, relevant, eval_users)
    TABLES.mkdir(parents=True, exist_ok=True)
    details = {
        "metrics": metrics, "users_requested": len(users), "users_evaluated": len(eval_users),
        "users_without_truth_excluded": len(users) - len(eval_users),
        "relevance": "unique user-show with at least one target overlap watch_pct >= 0.5",
        "recall_aggregation": "macro average over the same nonempty-truth users as MAP@5",
        "history_weeks": [11, 20], "validation_weeks": [21, 30],
        "reranker_training_history_weeks": [11, 16], "reranker_training_label_weeks": [17, 20],
        "lightfm_config": LIGHTFM_CONFIG, "lightgbm_config": LGBM_CONFIG,
        "features": FEATURES, "train_rows": len(train), "validation_rows": len(validation),
        "elapsed_seconds": time.monotonic() - start,
    }
    (TABLES / "e9_phase_a_metrics.json").write_text(json.dumps(details, indent=2, default=str) + "\n")
    pd.DataFrame(list(metrics.items()), columns=["metric", "value"]).to_csv(TABLES / "e9_phase_a_metrics.csv", index=False)
    baselines["reranker_top5"] = top5
    (DATA / "e9_validation_top5.json").write_text(json.dumps(baselines, default=int))
    validation[["user_id", "tv_show_id", "candidate_rank", "reranker_score", "label"]].to_parquet(DATA / "e9_validation_candidates.parquet", index=False)
    truth.to_parquet(DATA / "e9_validation_truth.parquet", index=False)
    model.booster_.save_model(str(ROOT / "checkpoints" / "e9_phase_a_lightgbm.txt"))
    print(f"Evaluated users: {len(eval_users)}/{len(users)}; relevance: watch_pct >= 0.5")
    for label, value in metrics.items():
        print(f"{label}: {value:.8f}")
    print(f"Elapsed seconds: {details['elapsed_seconds']:.1f}", flush=True)
    return details


class Tee(io.StringIO):
    def write(self, text):
        sys.__stdout__.write(text)
        sys.__stdout__.flush()
        return super().write(text)


if __name__ == "__main__":
    output = Tee()
    runtime_env = {key: os.environ[key] for key in ("DYLD_LIBRARY_PATH", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS") if key in os.environ}
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    result = None
    try:
        with contextlib.redirect_stdout(output):
            result = main()
    finally:
        interpretation = ("Completed one fixed-config temporal experiment.\n\n" + json.dumps(result["metrics"], indent=2)) if result else "Run interrupted or failed; inspect captured output and process stderr."
        notebook = {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [
                {"cell_type": "markdown", "id": "intent", "metadata": {}, "source": [f"# e9 Phase A — {stamp}\n", "Intent: test repeat + LightFM candidates and a binary LightGBM reranker.\n", "Method: train features 11–16 / labels 17–20; evaluate history 11–20 / labels 21–30. One config per model. Relevance: watch_pct >= 0.5. Macro metrics exclude users without positives.\n"]},
                {"cell_type": "code", "id": "run", "metadata": {"interpreter": sys.executable, "environment": runtime_env, "last_known_output": output.getvalue()}, "execution_count": 1,
                 "source": ["import os, subprocess\n", "from pathlib import Path\n", "root = Path.cwd() if (Path.cwd() / 'artifacts').exists() else Path.cwd().parent\n", f"subprocess.run([{sys.executable!r}, 'artifacts/scripts/e9_phase_a.py'], cwd=root, env=os.environ | {runtime_env!r}, check=True)\n"],
                 "outputs": [{"output_type": "stream", "name": "stdout", "text": output.getvalue().splitlines(True)}]},
                {"cell_type": "markdown", "id": "result", "metadata": {}, "source": [interpretation]},
            ],
        }
        (ROOT / "experiments" / f"{stamp}-e9-phase-a.ipynb").write_text(json.dumps(notebook, indent=2) + "\n")
