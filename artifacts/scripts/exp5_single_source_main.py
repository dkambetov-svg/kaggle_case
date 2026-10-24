"""Script used to compute exp5_single_source_main.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path('/Users/davidkk/Kaggle_Case')
DATA = ROOT / 'data'
TABLES = ROOT / 'results' / 'tables'

# Load data
inter = pd.read_parquet(DATA / 'exp5_user_airings_rich.parquet')
inter['user_id'] = inter['user_id'].astype(str)
inter['tv_show_id'] = inter['tv_show_id'].astype(int)

# Main split: history ≤20, target 21-30
history = inter[inter.week.le(20)].copy()
target = inter[inter.week.between(21, 30)].copy()

users_df = pd.read_csv(DATA / 'submission.csv', dtype={'user_id': str})
sub_users = set(users_df['user_id'].tolist())

target = target[target.user_id.isin(sub_users)]

# Target schedule for 21-30
sched = pd.read_csv(DATA / 'export_arh_21-30-final.csv', low_memory=False)
sched['start_time'] = pd.to_datetime(sched['start_time'], dayfirst=True)
sched['duration'] = pd.to_numeric(sched['duration'], errors='coerce')
sched = sched[sched['tv_show_id'].ne(0)].copy()
sched['tv_show_id'] = sched['tv_show_id'].astype(int)
catalog = set(sched['tv_show_id'].unique())

# Filter target to catalog
target = target[target.tv_show_id.isin(catalog)]

# Labels: watch_pct >= 0.8
relevant = target[target.user_watch_pct.ge(0.8)].groupby('user_id')['tv_show_id'].apply(set).to_dict()
eval_users = [u for u in sub_users if relevant.get(u)]
print(f'eval users: {len(eval_users)}')

# Load candidate parquet
cand = pd.read_parquet('data/exp5_candidates_main_expanded9.parquet')
cand['user_id'] = cand['user_id'].astype(str)
cand['tv_show_id'] = cand['tv_show_id'].astype(int)
cand = cand[cand.user_id.isin(eval_users)]

# Sources to evaluate
sources = ['repeat', 'ppr', 'als', 'lightfm', 'popularity', 
           'airing_popularity', 'tv_slot', 'user_channel', 'user_metadata',
           'prop_max', 'prop_mean', 'prop_sum', 'prop_logsumexp']

# MAP@5 function
def map5_from_ranked(preds, relevant):
    ap_sum = 0.0
    n = 0
    for u in preds:
        if u not in relevant or not relevant[u]:
            continue
        actuals = relevant[u]
        hits = 0
        ap = 0.0
        for i, p in enumerate(preds[u][:5]):
            if p in actuals:
                hits += 1
                ap += hits / (i + 1)
        ap_sum += ap / min(len(actuals), 5)
        n += 1
    return ap_sum / n if n else 0.0, n

# Single source MAP@5
results = []
for src in sources:
    rank_col = f'rank_{src}'
    if rank_col not in cand.columns:
        print(f'Warning: {rank_col} not in candidates')
        continue
    preds = {}
    for u in eval_users:
        grp = cand[cand.user_id.eq(u)]
        if grp.empty:
            preds[u] = []
            continue
        grp = grp.sort_values(rank_col, kind='stable')
        preds[u] = grp['tv_show_id'].tolist()[:5]
    map5, n = map5_from_ranked(preds, relevant)
    results.append({'source': src, 'MAP5': map5, 'n_users': n})
    print(f'{src}: MAP5={map5:.6f} (n={n})')

df_res = pd.DataFrame(results)
df_res.to_csv('results/tables/exp5_single_source_main.csv', index=False)
print('\nSaved to results/tables/exp5_single_source_main.csv')
