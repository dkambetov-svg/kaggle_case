import pandas as pd
import numpy as np
from pathlib import Path
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

def run_submit():
    # 1. Данные
    inter = pd.read_parquet(DATA / "user_interactions.parquet")
    inter['user_id'] = inter['user_id'].astype(str)
    
    # Трейн для сабмита: недели 11-30 (вся доступная история)
    train = inter[inter['week'] <= 30]
    sub = pd.read_csv(DATA / "submission.csv", dtype={'user_id': str})
    
    # 2. Популярность (fallback)
    pop = train.groupby('tv_show_id')['user_id'].nunique().sort_values(ascending=False).index.tolist()
    
    # 3. Сабмит
    preds = []
    for uid in sub['user_id']:
        # Если пользователь есть в трейне - можно было бы ALS, но для Baseline-сабмита возьмем Pop
        preds.append(" ".join(map(str, pop[:5])))
        
    sub['tv_show_id'] = preds
    sub.to_csv(DATA / "submission_final.csv", index=False)
    print("Submission saved to data/submission_final.csv")

run_submit()
