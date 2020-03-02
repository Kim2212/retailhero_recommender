# -*- coding: utf-8 -*-

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from utils import (
    extract_data_from_json_dict, logger,
    normalized_average_precision
)

DIR = os.path.abspath(os.path.dirname(__file__))
TOP_PRODUCTS = [
    '4009f09b04', '4dcf79043e', '15ccaa8685', 'bf07df54e1', '3e038662c0',
    '113e3ace79', '31dcf71bbd', 'f4599ca21a','ea27d5dc75', '5cb93c9bc5',
    '0a46068efc', '439498bce2', '080ace8748', '5645789fdf', '1c257c1a1b',
    '53fc95e177', '76ae00433f', 'dc2001d036', '5186e12ff4', 'ad865591c6',
    'f95785964a', '4d3ab3e72c', '343e841aaa', 'e6f8ac5174', 'e29cab0243',
    '6d0f84a0ac', 'e6a5597d19', 'fc5b0d84e8', '719b704cb6', 'cf1a5be7fb'
]

sample_queries = pd.read_csv(os.path.join(DIR,'check_queries.tsv'),
    sep='\t', header=None, names=['query', 'next_trans'])

metadata_products = pd.read_csv(os.path.join(DIR, 'products.csv'))


with open(os.path.join(DIR, 'model.pkl'), 'rb') as f:
    model = joblib.load(f)

time_bechmark = []
scores = []
if __name__ == '__main__':
    for i in range(len(sample_queries)):
        query = json.loads(sample_queries.at[i, 'query'])
        actual = json.loads(sample_queries.at[i, 'next_trans'])['product_ids']
        if len(query['transaction_history']) == 0:
            continue

        start_time = time.time()
        products_hist_counter, histdata_products = extract_data_from_json_dict(query)
        recs = model.recommend(products_hist_counter, metadata_products, histdata_products)
        # recs = model.recommend(products_hist_counter)
        finish_time = time.time()

        score = normalized_average_precision(actual, recs)
        scores.append(score)
        # logger.debug(f'Query: {i}, processed in {(finish_time - start_time):.5f} seconds')
        logger.debug(f'Query {i}, NAP@30: {score}, Num transactions: {len(query["transaction_history"])}')
        time_bechmark.append(finish_time - start_time)

    logger.debug(f'Ran {len(sample_queries)} queries, Average elapsed time: {np.mean(time_bechmark)} ± {np.std(time_bechmark)}')
    logger.debug(f'MNAP@30: {np.mean(scores)}')
