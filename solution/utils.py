# -*- coding: utf-8 -*-

import sys
import json
import logging
import inspect
import pandas as pd

from datetime import datetime, timedelta
from collections import Counter, defaultdict

simple_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(simple_formatter)

logger = logging.getLogger('retailhero')
logger.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)


TOP_PRODUCTS = [
    '4009f09b04', '4dcf79043e', '15ccaa8685', 'bf07df54e1', '3e038662c0',
    '113e3ace79', '31dcf71bbd', 'f4599ca21a','ea27d5dc75', '5cb93c9bc5',
    '0a46068efc', '439498bce2', '080ace8748', '5645789fdf', '1c257c1a1b',
    '53fc95e177', '76ae00433f', 'dc2001d036', '5186e12ff4', 'ad865591c6',
    'f95785964a', '4d3ab3e72c', '343e841aaa', 'e6f8ac5174', 'e29cab0243',
    '6d0f84a0ac', 'e6a5597d19', 'fc5b0d84e8', '719b704cb6', 'cf1a5be7fb'
]

def read_json(path):
    with open(path, 'r') as f:
        js = json.load(f)
        
    return js

def write_json(path, dict_obj):
    with open(path, 'w') as f:
        json.dump(dict_obj, f)

def get_top_prods(data, days_offset):
    query_datetime = datetime.fromisoformat(data['query_time'])
    first_date_offset = query_datetime - timedelta(days=days_offset)
    transactions = data['transaction_history']

    recs = []
    if len(transactions) == 0:
        return TOP_PRODUCTS

    counter = defaultdict(int)

    for transaction in transactions:
        d = datetime.fromisoformat(transaction['datetime'])
        if d >= first_date_offset:
            for product in transaction['products']:
                counter[product['product_id']] += 1

    if len(counter) > 0:
        ls = sorted(counter.items(), key=lambda x: -x[1])
        recs.extend([pid for (pid, hits) in ls])

    if len(recs) < 30:
        recs.extend(TOP_PRODUCTS)

    return recs[:30]

def extract_data_from_json_dict(json_dict):
    n = 0
    counter = Counter()
    records = {'age': json_dict['age'], 'gender': json_dict['gender']}
    history = sorted(json_dict['transaction_history'],
        key=lambda x: pd.to_datetime(x['datetime']), reverse=True)

    for h in history:
        for p in h['products']:
            n += 1
            if p['product_id'] not in counter:
                records[p['product_id']] = dict(
                        store_id=h['store_id'],
                        regular_points_received=h['regular_points_received'],
                        express_points_received=h['express_points_received'],
                    )

            counter.update({p['product_id']: 1.0})

    counter = {k: v / n for (k,v) in counter.items()}

    return counter, records

def get_json_rows_from_purchases(data, cid):
    res_json = dict(client_id=cid, age=data.age.iloc[0], gender=data.gender.iloc[0])
    hist = []
    for dt, ds in data.groupby('transaction_datetime'):
        cur_records = {}
        cur_records['datetime'] = str(dt)
        cur_records['store_id'] = ds.store_id.iloc[0]
        cur_records['regular_points_received'] = ds.regular_points_received.iloc[0]
        cur_records['express_points_received'] = ds.express_points_received.iloc[0]
        cur_records['products'] = [{'product_id': pid} for pid in ds.product_id]
        hist.append(cur_records)
        
    res_json['transaction_history'] = hist
    return res_json

def average_precision(actual, recommended, k=30):
    ap_sum = 0
    hits = 0
    for i in range(k):
        product_id = recommended[i] if i < len(recommended) else None
        if product_id is not None and product_id in actual:
            hits += 1
            ap_sum += hits / (i + 1)
    return ap_sum / k


def normalized_average_precision(actual, recommended, k=30):
    actual = set(actual)
    if len(actual) == 0:
        return 0.0
    
    ap = average_precision(actual, recommended, k=k)
    ap_ideal = average_precision(actual, list(actual)[:k], k=k)
    return ap / ap_ideal