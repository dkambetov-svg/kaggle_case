# Task: Improve Kaggle sweet.tv TV Program Recommender MAP@5

You are a senior recommender-systems engineer and ML researcher. Your task is to improve the existing solution for the Kaggle competition **sweet.tv — TV program recommender**.

Do not start from scratch. First inspect the existing repository, implementation, experiment results, validation code, feature engineering, and submission generation pipeline. Preserve working components and build on the strongest verified results.

## 1. Current results and baseline

The current best submission is:

- `submission_hybrid_pop_als_w01.csv`
- Candidate generation: top-50 programs ranked by popularity × log(1 + airings)
- Re-ranking: implicit ALS with weight `w=0.1`
- Public MAP@5: 0.30215
- Private MAP@5: 0.29635

The strongest pure popularity baseline is:

- `submission_pop_x_log_airings.csv`
- Score: Public 0.30050, Private 0.29538

Other observed results:

- Recent-four-week popularity: Public 0.28234, Private 0.27809
- Popularity + content-based re-ranking: Public 0.28914, Private 0.28692
- Repeat-own-history approaches: Public 0.27339, Private 0.24269

Strict temporal holdout results from the existing experiments:

- Pure popularity: MAP@5 = 0.4565
- ALS hybrid: MAP@5 = 0.4544
- Channel-based content model: MAP@5 = 0.4182

The Kaggle improvement of the ALS hybrid over the strongest pure baseline is small. The local validation ranking disagrees with Kaggle. Treat the improvement as uncertain rather than assuming ALS is universally superior.

## 2. Primary objective

Find a reproducible recommendation strategy that improves MAP@5 on a valid chronological holdout and produces a valid Kaggle submission.

Optimize MAP@5, not accuracy, RMSE, AUC, or an unrelated proxy metric.

Use the existing Kaggle evaluation results as historical evidence, not as a substitute for local validation. Do not claim an improvement unless the relevant experiment has actually been run.

## 3. First inspect the repository

Before changing code:

1. Inspect the repository tree, README, data schemas, preprocessing, joins, training scripts, validation code, diagnostics, and submission generation.
2. Read `results/tables/hybrid_validation.csv` and `results/tables/diagnostics_strict.csv`.
3. Locate the exact code that generates the best submission and reproduce its predictions if feasible.
4. Verify how viewing intervals are joined to scheduled programs, how viewing duration is thresholded, how `tv_show_id = 0` is handled, and how candidate IDs are mapped to submission IDs.
5. Check for data leakage, train-validation contamination, duplicate interactions, incorrect temporal boundaries, and inconsistencies between local validation and test inference.
6. Identify which code paths and settings generated each recorded score.

Do not overwrite the existing best submission or experiment outputs. Use versioned experiment files.

## 4. Highest-priority experiments

Run experiments incrementally, measuring the marginal contribution of each change.

### A. Improve the popularity baseline

Test the following alternatives on exactly the same chronological validation splits:

- Different recent-history windows, such as 2, 4, 6, 8, and 10 weeks.
- Exponential time decay instead of a hard cutoff.
- Unique-user popularity versus interaction-count popularity.
- Duration-weighted and binary interactions.
- Popularity normalized by channel, category, or airing frequency.
- Smoothed popularity estimates for rare programs.
- Robust combinations of recent popularity and long-term popularity.
- Alternative exponents for the airing-frequency adjustment:
  score = popularity × (1 + airings)^alpha.
- Compare raw counts, logarithmic transforms, and normalized scores.

Avoid optimizing dozens of parameters blindly. Use a structured search, record all results, and retain only changes supported by validation.

### B. Improve candidate generation

The current ALS hybrid considers only the top-50 popularity candidates. This may exclude relevant programs before re-ranking.

Test candidate pools of 50, 100, 200, and 500 programs where computationally feasible.

Measure candidate recall@5 and recall@K on validation. Determine whether the best ranking model is constrained by missing relevant candidates.

Test unions of candidate generators:

- Popularity candidates.
- ALS recommendations.
- Item-item collaborative filtering.
- User-program co-view similarity.
- Content-based candidates from genres, categories, actors, and directors.
- Channel or time-slot affinity, only where supported by the data and valid at recommendation time.

Deduplicate candidates and ensure all methods use the same eligible program pool and repeat policy.

### C. Improve collaborative filtering

Inspect the existing ALS implementation before replacing it.

Test a small, controlled grid of:

- Latent factors.
- Regularization.
- Confidence scaling for implicit feedback.
- Binary versus count-weighted interactions.
- Duration weighting with robust clipping.
- Time-decayed user-program interactions.
- Random seeds and initialization stability where applicable.

Compare ALS with item-item cosine similarity and co-view graph recommendations. Consider BPR only if the available interaction construction and runtime justify it.

Do not assume that a more complex model will perform better.

### D. Improve the hybrid ranking

Compare these approaches:

1. Pure popularity ranking.
2. Popularity candidates plus ALS re-ranking.
3. Popularity candidates plus item-item scoring.
4. Candidate union plus score fusion.
5. Candidate union plus rank fusion.
6. A lightweight LightGBM ranker, if reliable out-of-time training examples can be constructed.

Tune fusion weights using validation only. Consider user-history length and confidence-dependent blending, but keep such rules only if they improve the same validation metric.

For LightGBM, prevent target leakage when creating training features. Use temporally valid historical examples rather than features calculated from future interactions.

### E. Improve interaction construction

Audit the view-to-schedule join and run controlled comparisons of defensible viewing thresholds.

Check interval overlap, duration normalization, channel-ID mapping, schedule boundaries, timezone consistency, duplicate rows, and programme-ID coverage.

Do not silently change preprocessing while comparing models. Track each preprocessing change as a separate experiment.

## 5. Validation protocol

Use a chronological validation design that mimics the real task: predict future viewing behavior from earlier data.

- Fit all model parameters and interaction statistics on training history only.
- Use validation labels only for evaluation and model selection.
- Do not use validation or test viewing behavior as input features.
- Use the same candidate eligibility and repeat policy across methods.
- Define cold-start behavior explicitly.
- Report MAP@5 on identical validation users and time windows.
- Where feasible, evaluate multiple forward-chaining temporal windows.
- Report the number of users, relevant items, and eligible candidates for every split.

Investigate why the current strict holdout ranks popularity above ALS while Kaggle ranks the ALS hybrid slightly higher. Quantify temporal variability before drawing conclusions.

## 6. Metric and submission requirements

Implement or verify MAP@5 against the competition's actual evaluation convention.

Check whether users with no relevant validation items are excluded or included according to the correct metric implementation.

Every prediction must contain exactly five valid program IDs in the required format, without accidental duplicates or invalid IDs.

Ensure that all test users are covered, including cold-start users.

Do not use test-period viewing behavior. Use the future schedule only for permitted candidate and availability information.

## 7. Experiment management

For every experiment, save:

- Experiment ID and code/configuration.
- Data split and interaction-construction settings.
- Model parameters and random seed.
- Validation MAP@5.
- Candidate recall@K, when applicable.
- Training and inference time.
- Memory usage if measurable.
- A short explanation of what changed.
- Whether the experiment should be retained.

Create a comparison table sorted by validation MAP@5. Preserve the current best-known baseline and final submission.

Prefer simple, reproducible improvements over fragile gains from extensive tuning.

## 8. Final deliverables

Produce:

1. A ranked table of all experiments.
2. The best reproducible model and its configuration.
3. A final submission CSV with the exact required format.
4. Reproduction instructions.
5. A concise report explaining which changes improved MAP@5, which failed, and why.
6. A clear comparison against the existing baseline and ALS hybrid.
7. An explicit statement of any unresolved discrepancy between validation and Kaggle scores.

Do not fabricate results, Kaggle scores, citations, experiments, or screenshots.

Do not claim a new Kaggle score unless the generated file has actually been submitted and the returned score verified.

## 9. Execution strategy

Start by inspecting the repository and presenting a concise audit with the top three likely opportunities for improvement.

Then implement the highest-value, lowest-cost experiment first. Run it, compare it against the baseline on the same validation split, and continue iteratively.

Do not stop after proposing ideas. Execute experiments wherever the available environment and data permit.

The goal is a real, measured improvement in MAP@5, not merely a more complicated model.