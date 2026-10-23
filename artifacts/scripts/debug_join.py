#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

def check_consistency():
    # 1. Загрузка
    log = pd.read_csv(DATA / "dataset11-30.csv", dtype={'user_id': str})
    sub = pd.read_csv(DATA / "submission.csv", dtype={'user_id': str})
    inter = pd.read_parquet(DATA / "user_interactions.parquet")
    inter['user_id'] = inter['user_id'].astype(str)
    
    # 2. Сверка пользователей (Submission vs Log vs Inter)
    log_users = set(log['user_id'].unique())
    sub_users = set(sub['user_id'].unique())
    inter_users = set(inter['user_id'].unique())
    
    print(f"Users in submission: {len(sub_users)}")
    print(f"Users in submission found in log: {len(sub_users & log_users)}")
    print(f"Users in submission found in interactions: {len(sub_users & inter_users)}")
    
    # 3. Статистика взаимодействий
    print(f"Total interactions rows: {len(inter)}")
    inter_unique = inter.groupby(['user_id', 'tv_show_id', 'week']).size().reset_index(name='count')
    print(f"Unique (user, tv_show, week) pairs: {len(inter_unique)}")
    print(f"Min tv_show_id: {inter['tv_show_id'].min()}")
    
    # 4. Взгляд на просмотры
    log_chans = log['vsetv_id'].unique()
    # Нужен импорт sched для точного покрытия (сделаем быстро)
    print("Done")

check_consistency()
