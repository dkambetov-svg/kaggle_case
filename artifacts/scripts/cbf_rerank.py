import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Установим sklearn через uv в текущей сессии (если возможно) или используем существующий код
# Поскольку sklearn недоступен, реализуем упрощенный TF-IDF на numpy/pandas

def get_cbf_rerank(train, val_sched, candidates_ids, pop_recent, user_profiles, feat_df, id_to_idx, tfidf_matrix):
    preds = {}
    val_users = train['user_id'].unique() # Упрощенно для примера
    for u in val_users:
        if u not in user_profiles:
            preds[u] = pop_recent.index.tolist()[:5]
            continue
        # Берем топ-50 популярных кандидатов
        top_50 = pop_recent.index.tolist()[:50]
        cand_idxs = [id_to_idx[sid] for sid in top_50 if sid in id_to_idx]
        if not cand_idxs:
            preds[u] = top_50[:5]
            continue
            
        scores = (tfidf_matrix[cand_idxs] @ user_profiles[u])
        df_scores = pd.DataFrame({'id': np.array(top_50)[np.array([sid in id_to_idx for sid in top_50])], 'score': scores})
        preds[u] = df_scores.sort_values('score', ascending=False).head(5)['id'].tolist()
    return preds
