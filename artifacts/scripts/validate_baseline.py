#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "tables"

def evaluate(predictions, ground_truth, k=5):
    ap_sum = 0
    gt_grouped = ground_truth.groupby('user_id')['tv_show_id'].apply(set)
    users = set(predictions.keys()) & set(gt_grouped.index)
    if not users: return 0
    for u in users:
        preds = predictions[u]
        actuals = gt_grouped[u]
        hits = 0
        ap = 0
        for i, p in enumerate(preds[:k]):
            if p in actuals:
                hits += 1
                ap += hits / (i + 1)
        ap_sum += ap / min(len(actuals), k)
    return ap_sum / len(users)

def run():
    inter = pd.read_parquet(DATA / "user_interactions.parquet")
    inter['user_id'] = inter['user_id'].astype(str)
    
    train = inter[inter['week'] <= 20]
    val = inter[(inter['week'] > 20) & (inter['week'] <= 30)]
    val_users = val['user_id'].unique()
    
    val_sched = pd.read_csv(DATA / "export_arh_21-30-final.csv")
    candidates_ids = val_sched[val_sched['tv_show_id'] != 0]['tv_show_id'].unique()
    
    gt_grouped = val.groupby('user_id')['tv_show_id'].apply(set)
    n_rel = [len(s) for s in gt_grouped.values]
    print(f"Stats: avg_rel={np.mean(n_rel):.2f}, med_rel={np.median(n_rel):.2f}, pool={len(candidates_ids)}")

    # 1. Бейзлайн
    pop_recent = train[train['week'] >= 17].groupby('tv_show_id')['user_id'].nunique().sort_values(ascending=False)
    preds_recent = {u: pop_recent.index.tolist()[:5] for u in val_users}
    m_r = evaluate(preds_recent, val)
    
    # 2. CBF (Manual Cosine)
    sched_all = pd.concat([pd.read_csv(DATA / f) for f in ["export_arh_11-20-final.csv", "export_arh_21-30-final.csv"]])
    feat = sched_all[sched_all['tv_show_id'] != 0].drop_duplicates('tv_show_id').copy()
    
    # One-hot категорий
    cats = pd.get_dummies(feat['tv_show_category']).values
    # year binning
    feat['year'] = pd.to_numeric(feat['year_of_production'], errors='coerce').fillna(2000)
    years = (feat['year'] // 10).values.reshape(-1, 1)
    
    feats = np.hstack([cats, years])
    # Нормализация
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    
    id_to_idx = {id_: i for i, id_ in enumerate(feat['tv_show_id'].values)}
    
    train['weight'] = np.log1p(train['overlap_seconds'])
    user_profiles = {}
    for uid, group in train.groupby('user_id'):
        idxs = [id_to_idx[sid] for sid in group['tv_show_id'] if sid in id_to_idx]
        if not idxs: continue
        weights = group.loc[[sid in id_to_idx for sid in group['tv_show_id']], 'weight'].values
        profile = (feats[idxs].T @ weights) / weights.sum()
        user_profiles[uid] = profile / (np.linalg.norm(profile) + 1e-9)
        
    preds_cbf = {}
    cand_mask = np.array([sid in id_to_idx for sid in candidates_ids])
    cand_idxs = [id_to_idx[sid] for sid in candidates_ids[cand_mask]]
    cand_feats = feats[cand_idxs]
    
    for u in val_users:
        if u not in user_profiles:
            preds_cbf[u] = pop_recent.index.tolist()[:5]
            continue
        scores = (cand_feats @ user_profiles[u])
        
        df_scores = pd.DataFrame({'id': candidates_ids[cand_mask], 'score': scores})
        df_scores = df_scores.merge(pop_recent.reset_index(name='pop').rename(columns={'tv_show_id': 'id'}), on='id', how='left').fillna(0)
        preds_cbf[u] = df_scores.sort_values(['score', 'pop'], ascending=False).head(5)['id'].tolist()
        
    m_cbf = evaluate(preds_cbf, val)
    
    results = pd.DataFrame({
        'method': ['Popularity (Recent 4w)', 'CBF (Cat+Year)'],
        'MAP@5': [m_r, m_cbf]
    })
    results.to_csv(RESULTS / "validation.csv", index=False)
    print(results)

run()
