#!/usr/bin/env python3
"""Train the accepted E9 Phase A protocol and create one weeks 31–42 submission."""
import contextlib
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
from e9_models import fit_predict, generate_candidates

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUTPUT = DATA / "submission_e9_phase_a.csv"


def prepare_training(users):
    history, past, future, truth = load_split(16, 17, 20)
    candidates = generate_candidates(history, future, users)
    candidates.attrs = {}
    features = build_features(history, past, future, candidates)
    features = features.merge(truth.assign(label=1), on=["user_id", "tv_show_id"],
                              how="left", validate="one_to_one")
    features["label"] = features.label.fillna(0).astype("int8")
    return features


def prepare_test(users):
    history, past, future, _ = load_split(30, 31, 42)
    candidates = generate_candidates(history, future, users)
    candidates.attrs = {}
    features = build_features(history, past, future, candidates)
    return features, future


def main():
    started = time.monotonic()
    template = pd.read_csv(DATA / "submission.csv", dtype={"user_id": str})
    users = template.user_id.tolist()
    if len(users) != 1260 or len(set(users)) != 1260:
        raise AssertionError("Expected exactly 1260 unique submission users")
    train = prepare_training(users)
    print(f"training rows={len(train):,}; positives={int(train.label.sum()):,}", flush=True)
    test, future = prepare_test(users)
    print(f"test candidates={len(test):,}; users={test.user_id.nunique():,}; "
          f"scheduled shows={future.tv_show_id.nunique():,}", flush=True)
    scores, _ = fit_predict(train, test)
    if len(scores) != len(test) or not np.isfinite(scores).all():
        raise AssertionError("Reranker produced invalid scores")
    test["score"] = scores
    ranked = test.sort_values(["user_id", "score", "tv_show_id"],
                              ascending=[True, False, True], kind="stable")
    top5 = ranked.groupby("user_id", sort=False).head(5)
    predictions = top5.groupby("user_id", sort=False).tv_show_id.agg(list).to_dict()
    submission = template[["user_id"]].copy()
    submission["tv_show_id"] = submission.user_id.map(
        lambda user: " ".join(map(str, predictions.get(user, []))))

    catalog = set(future.tv_show_id.astype(int))
    parsed = submission.tv_show_id.str.split().map(lambda values: [int(v) for v in values])
    checks = {
        "users": len(submission),
        "unique_users": submission.user_id.nunique(),
        "five_per_user": bool(parsed.map(len).eq(5).all()),
        "unique_within_top5": bool(parsed.map(lambda x: len(set(x)) == 5).all()),
        "all_in_schedule_31_42": bool(parsed.map(lambda x: set(x).issubset(catalog)).all()),
        "no_zeros": bool(parsed.map(lambda x: 0 not in x).all()),
        "candidate_rows": len(test),
        "candidate_users": test.user_id.nunique(),
        "min_candidates_per_user": int(test.groupby("user_id").size().min()),
        "max_candidates_per_user": int(test.groupby("user_id").size().max()),
        "target_catalog_size": len(catalog),
    }
    if checks != checks | {"users": 1260, "unique_users": 1260}:
        raise AssertionError(checks)
    required = ("five_per_user", "unique_within_top5", "all_in_schedule_31_42", "no_zeros")
    if not all(checks[key] for key in required):
        raise AssertionError(checks)
    submission.to_csv(OUTPUT, index=False)
    checks["elapsed_seconds"] = time.monotonic() - started
    print(json.dumps(checks, indent=2), flush=True)
    print(f"wrote {OUTPUT}", flush=True)
    return checks


class Tee(io.StringIO):
    def write(self, text):
        sys.__stdout__.write(text)
        sys.__stdout__.flush()
        return super().write(text)


if __name__ == "__main__":
    capture = Tee()
    result = None
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    runtime_env = {key: os.environ[key] for key in
                   ("DYLD_LIBRARY_PATH", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
                   if key in os.environ}
    try:
        with contextlib.redirect_stdout(capture):
            result = main()
    finally:
        summary = json.dumps(result, indent=2) if result else "Run failed; inspect output."
        notebook = {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [
                {"cell_type": "markdown", "id": "intent", "metadata": {}, "source": [
                    f"# E9 Phase A final submission — {stamp}\n",
                    "Intent: apply the accepted fixed Phase A architecture to history weeks 11–30 and schedule weeks 31–42. The reranker keeps the accepted supervised fold: history 11–16, labels 17–20.\n"]},
                {"cell_type": "code", "id": "run", "execution_count": 1,
                 "metadata": {"interpreter": sys.executable, "environment": runtime_env,
                              "last_known_output": capture.getvalue()},
                 "source": ["import os, subprocess\n", "from pathlib import Path\n",
                            "root = Path.cwd() if (Path.cwd() / 'artifacts').exists() else Path.cwd().parent\n",
                            f"subprocess.run([{sys.executable!r}, 'artifacts/scripts/e9_submit.py'], cwd=root, env=os.environ | {runtime_env!r}, check=True)\n"],
                 "outputs": [{"output_type": "stream", "name": "stdout",
                              "text": capture.getvalue().splitlines(True)}]},
                {"cell_type": "markdown", "id": "result", "metadata": {},
                 "source": ["Submission validation result:\n\n```json\n", summary, "\n```\n"]},
            ],
        }
        (ROOT / "experiments" / f"{stamp}-e9-submission.ipynb").write_text(
            json.dumps(notebook, indent=2) + "\n")
