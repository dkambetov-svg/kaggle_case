#!/usr/bin/env python3
"""Джойн лога и расписания: векторизованный поиск пересечений."""
import pandas as pd
import numpy as np
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

def perform_vectorized_join():
    # 1. Загрузка
    log = pd.read_csv(DATA / "dataset11-30.csv")
    log['start_time'] = pd.to_datetime(log['start_time'])
    log['stop_time'] = pd.to_datetime(log['stop_time'])
    
    sched_files = ["export_arh_11-20-final.csv", "export_arh_21-30-final.csv", "export_arh_31-42-final.csv"]
    scheds = []
    for f in sched_files:
        df = pd.read_csv(DATA / f)
        df['start_time'] = pd.to_datetime(df['start_time'], dayfirst=True)
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
        df['end_time'] = df['start_time'] + pd.to_timedelta(df['duration'], unit='s')
        scheds.append(df[df['tv_show_id'] != 0])
    
    sched = pd.concat(scheds)
    
    # 2. Векторизованный джойн по каналам
    interactions = []
    t0 = time.time()
    
    # Предполагаем vsetv_id == channel_id
    common_channels = set(log['vsetv_id'].unique()) & set(sched['channel_id'].unique())
    
    for ch in common_channels:
        l_ch = log[log['vsetv_id'] == ch].sort_values('start_time').reset_index(drop=True)
        s_ch = sched[sched['channel_id'] == ch].sort_values('start_time').reset_index(drop=True)
        
        # Исправление перекрытий (жадная обрезка предыдущего по началу текущего)
        for i in range(1, len(s_ch)):
            if s_ch.loc[i, 'start_time'] < s_ch.loc[i-1, 'end_time']:
                s_ch.loc[i-1, 'end_time'] = s_ch.loc[i, 'start_time']
                s_ch.loc[i-1, 'duration'] = (s_ch.loc[i-1, 'end_time'] - s_ch.loc[i-1, 'start_time']).total_seconds()
        
        s_starts = s_ch['start_time'].values
        s_ends = s_ch['end_time'].values
        s_ids = s_ch['tv_show_id'].values
        s_durs = s_ch['duration'].values
        
        v_starts = l_ch['start_time'].values
        v_stops = l_ch['stop_time'].values
        
        lo = np.searchsorted(s_ends, v_starts, side='right')
        hi = np.searchsorted(s_starts, v_stops, side='left')
        
        # Кол-во пересечений для каждого просмотра
        counts = hi - lo
        mask = counts > 0
        
        if mask.any():
            v_idx = np.where(mask)[0]
            v_repeats = counts[mask]
            
            # Разворачиваем индексы
            v_ids_rep = np.repeat(v_idx, v_repeats)
            
            # Индексы расписания
            s_idx_rep = np.concatenate([np.arange(lo[i], hi[i]) for i in v_idx])
            
            # Оверлап
            ov = np.minimum(v_stops[v_ids_rep], s_ends[s_idx_rep]) - np.maximum(v_starts[v_ids_rep], s_starts[s_idx_rep])
            ov_sec = ov.astype('timedelta64[s]').astype(int)
            
            # Порог
            pass_mask = (ov_sec >= 300) | (ov_sec / s_durs[s_idx_rep] >= 0.5)
            
            res = pd.DataFrame({
                'user_id': l_ch.loc[v_ids_rep[pass_mask], 'user_id'].values,
                'tv_show_id': s_ids[s_idx_rep[pass_mask]],
                'week': l_ch.loc[v_ids_rep[pass_mask], 'start_time'].dt.isocalendar().week.values,
                'overlap_seconds': ov_sec[pass_mask]
            })
            interactions.append(res)
            
    df_res = pd.concat(interactions)
    print(f"Time: {time.time() - t0:.2f}s")
    print(f"Interactions: {len(df_res):,}")
    print(f"Unique users: {df_res['user_id'].nunique():,}")
    print(f"Unique programmes: {df_res['tv_show_id'].nunique():,}")
    
    sub_users = pd.read_csv(DATA / "submission.csv")['user_id']
    test_users_in_inter = df_res[df_res['user_id'].isin(sub_users)]['user_id'].nunique()
    print(f"Test users with history: {test_users_in_inter:,}")
    
    df_res.to_parquet(DATA / "user_interactions.parquet")

perform_vectorized_join()
