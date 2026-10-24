"""Fixed E9 candidates and binary reranker; callers supply history-only inputs."""
import numpy as np
import pandas as pd
from lightfm import LightFM
from lightgbm import LGBMClassifier
from scipy.sparse import coo_matrix

LIGHTFM_CONFIG = dict(loss="warp", no_components=32, learning_rate=0.05,
                      random_state=42, epochs=15, num_threads=4)
LGBM_CONFIG = dict(objective="binary", n_estimators=250, num_leaves=31,
                   learning_rate=0.05, min_child_samples=50,
                   colsample_bytree=0.9, subsample=1, n_jobs=4,
                   random_state=42, verbosity=-1)
FEATURES = ["user_show_freq_03", "user_show_freq_05", "user_show_freq_08",
            "recent_freq_1w", "recent_freq_4w", "user_show_rank",
            "lightfm_score", "lightfm_rank", "lightfm_user_bias",
            "lightfm_item_bias", "user_channel_count", "user_channel_share",
            "category", "genre_1", "genre_2", "genre_3", "past_airtime",
            "future_airtime", "future_past_airtime_ratio"]
CATEGORICAL = ["category", "genre_1", "genre_2", "genre_3"]
COLUMNS = ["user_id", "tv_show_id", "lightfm_score", "lightfm_rank",
           "lightfm_user_bias", "lightfm_item_bias", "candidate_rank"]


def generate_candidates(history: pd.DataFrame, future_schedule: pd.DataFrame,
                        users: list[str]) -> pd.DataFrame:
    """Interleave repeat150/LF500. Cold users: item bias; cold items: zero dot.

    Cold means no >=0.5 positive. Item/user bias features remain actual fitted
    LightFM biases, including negative-sampled item biases. Ranks cover every
    distinct future show; ties use ascending show ID. No seen-item exclusion.
    """
    users = list(dict.fromkeys(map(str, users)))
    target = np.sort(future_schedule.tv_show_id.dropna().astype("int64").unique())
    result = pd.DataFrame({c: pd.Series(dtype="object" if c == "user_id" else
                          "int64" if c.endswith("rank") or c == "tv_show_id"
                          else "float64") for c in COLUMNS})
    repeat_top5, lightfm_top5 = {u: [] for u in users}, {u: [] for u in users}
    result.attrs.update(repeat_top5=repeat_top5, lightfm_top5=lightfm_top5)
    if not users or not len(target):
        return result
    pct = "watch_pct" if "watch_pct" in history else "user_watch_pct"
    hist = history[["user_id", "tv_show_id", pct]].copy()
    hist["user_id"] = hist.user_id.astype(str)
    hist["tv_show_id"] = hist.tv_show_id.astype("int64")
    hist["positive"] = hist[pct].ge(0.5)
    all_users = sorted(set(hist.user_id) | set(users))
    catalog = sorted(set(hist.tv_show_id) | set(target))
    umap, imap = {u: i for i, u in enumerate(all_users)}, {s: i for i, s in enumerate(catalog)}
    positives = hist.loc[hist.positive, ["user_id", "tv_show_id"]].drop_duplicates()
    matrix = coo_matrix((np.ones(len(positives), dtype=np.float32),
                         (positives.user_id.map(umap).to_numpy(dtype=int),
                          positives.tv_show_id.map(imap).to_numpy(dtype=int))),
                        shape=(len(all_users), len(catalog)))
    config = LIGHTFM_CONFIG.copy()
    epochs, threads = config.pop("epochs"), config.pop("num_threads")
    model = LightFM(**config).fit(matrix, epochs=epochs if matrix.nnz else 0,
                                 num_threads=threads)
    target_indices = np.array([imap[s] for s in target])
    item_bias = model.item_biases[target_indices]
    item_vectors = model.item_embeddings[target_indices].copy()
    warm_users, warm_items = set(positives.user_id), set(positives.tv_show_id)
    item_vectors[[s not in warm_items for s in target]] = 0
    counts = hist[hist.tv_show_id.isin(target)].groupby(["user_id", "tv_show_id"]).agg(
        freq=("positive", "sum"), overlap=("positive", "size")).reset_index()
    counts = counts.sort_values(["user_id", "freq", "overlap", "tv_show_id"],
                                ascending=[True, False, False, True])
    repeats = {u: g.tv_show_id.to_numpy()[:150] for u, g in counts.groupby("user_id", sort=False)}
    blocks = []
    target_position = {int(s): i for i, s in enumerate(target)}
    for user in users:
        uid = umap[user]
        user_bias = float(model.user_biases[uid])
        scores = item_bias.copy()
        if user in warm_users:
            scores += item_vectors @ model.user_embeddings[uid] + user_bias
        order = np.lexsort((target, -scores))
        ranks = np.empty(len(target), dtype=np.int64)
        ranks[order] = np.arange(1, len(target) + 1)
        repeat = list(map(int, repeats.get(user, [])))
        repeat_top5[user] = repeat[:5]
        lightfm_top5[user] = target[order[:5]].astype(int).tolist()
        # Each source nominates a (rank, source-priority, show) key.
        keys = {int(target[i]): (r, 1, int(target[i])) for r, i in enumerate(order[:500], 1)}
        for rank, show in enumerate(repeat, 1):
            keys[show] = min(keys.get(show, (rank, 0, show)), (rank, 0, show))
        selected = sorted(keys, key=keys.get)[:500]
        positions = np.array([target_position[s] for s in selected])
        blocks.append(pd.DataFrame(dict(user_id=user, tv_show_id=selected,
            lightfm_score=scores[positions], lightfm_rank=ranks[positions],
            lightfm_user_bias=user_bias, lightfm_item_bias=item_bias[positions],
            candidate_rank=np.arange(1, len(selected) + 1))))
    result = pd.concat(blocks, ignore_index=True)[COLUMNS]
    result.attrs.update(repeat_top5=repeat_top5, lightfm_top5=lightfm_top5)
    return result


def fit_predict(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[np.ndarray, LGBMClassifier]:
    """Fit the fixed allowlist; validation labels are never read or passed to fit."""
    xtrain, xvalid = train[FEATURES].copy(), validation[FEATURES].copy()
    vocab = {}
    for column in CATEGORICAL:
        values = xtrain[column].astype("string")
        vocab[column] = sorted(values.dropna().unique().tolist())
        xtrain[column] = pd.Categorical(values, categories=vocab[column])
        valid_values = xvalid[column].astype("string")
        xvalid[column] = pd.Categorical(valid_values.where(valid_values.isin(vocab[column])),
                                        categories=vocab[column])
    for column in set(FEATURES) - set(CATEGORICAL):
        xtrain[column] = pd.to_numeric(xtrain[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        xvalid[column] = pd.to_numeric(xvalid[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    labels = train["label"]
    if labels.isna().any() or not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise ValueError("Training label must contain both binary classes and no missing values")
    model = LGBMClassifier(**LGBM_CONFIG)
    model.fit(xtrain, labels.astype(int), categorical_feature=CATEGORICAL)
    model.e9_category_vocab_ = vocab
    probabilities = model.predict_proba(xvalid)[:, 1] if len(xvalid) else np.empty(0)
    return probabilities, model
