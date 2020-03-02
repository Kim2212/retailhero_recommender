# -*- coding: utf-8 -*-

import os
import json
import joblib
import numpy as np
import pandas as pd

from utils import (
    read_json, write_json, logger, normalized_average_precision,
    extract_data_from_json_dict, get_json_rows_from_purchases
)
from tqdm import tqdm
from datetime import datetime
from models import RetailHeroRecommender, CosineRecommenderModel

np.random.seed(4242)

DIR = os.path.abspath(os.path.dirname(__file__))
SPLIT_DATE = '2019-03-02 10:05:38'

params_rec = dict(K=1, num_threads=12)
params_catboost = dict(
    random_seed=34,
    iterations=1000,
    loss_function='PairLogit',
    custom_metric='MAP:top=30',
)

if __name__ == '__main__':
    logger.debug('Reading Data...')
    df_clients = pd.read_csv(os.path.join(DIR, 'clients.csv'))
    df_products = pd.read_csv(os.path.join(DIR, 'products.csv'))
    df_purchases = pd.read_csv(os.path.join(DIR, 'purchases.csv'),
                                    parse_dates=['transaction_datetime'])

    df_purchases['client_len_products'] = df_purchases.groupby('client_id').product_id.transform('size')
    df_purchases['relevance'] = df_purchases.eval('1 / client_len_products')

    df_purchases = pd.merge(df_purchases, df_products, on='product_id', how='left')
    df_purchases = pd.merge(df_purchases, df_clients[['client_id', 'age', 'gender']],
                            on='client_id', how='left')

    logger.debug('Splitting train / valid data...')
    df_train = df_purchases.query(f'transaction_datetime < "{SPLIT_DATE}"')
    df_val = df_purchases.query(f'transaction_datetime >= "{SPLIT_DATE}"')

    eval_clients = tuple(set(df_train.client_id.unique()) & set(df_val.client_id.unique()))
    val_clients = np.random.choice(eval_clients, size=20000, replace=False)
    train_clients = list(set(eval_clients) - set(val_clients))

    logger.debug(f'First 3, Train clients: {train_clients[:3]}, Valid clients: {val_clients[:3]}')
    df_train_rec = df_train[df_train.client_id.isin(train_clients)]

    val_dict = pd.merge(
        df_val.groupby('client_id').transaction_datetime.min().reset_index(),
        df_val[['client_id', 'transaction_datetime', 'product_id']].drop_duplicates(),
        how='inner', on=['client_id', 'transaction_datetime']
    )
    val_dict = val_dict.groupby('client_id').product_id.apply(set).to_dict()

    df_train_ranker = df_train[df_train.client_id.isin(val_clients[:10000])]
    df_test_clients = df_train[df_train.client_id.isin(val_clients[10000:])]

    logger.debug('Training Model...')
    cosine_model = CosineRecommenderModel(df_products.product_id.unique(), params_rec)
    cosine_model.fit_recommender(df_train_rec)

    retailHeroModel = RetailHeroRecommender(df_products, params_rec, params_catboost)
    retailHeroModel.train_model(df_train_rec, df_train_ranker, val_dict)

    logger.debug('Succsessfully trained Model')
    logger.debug('Saving...')

    with open(os.path.join(DIR, 'model.pkl'), 'wb') as f:
        joblib.dump(retailHeroModel, f)

    logger.debug('Validation...')
    scores = []
    cosine_scores = []
    for (cid, ds) in tqdm(df_test_clients.groupby('client_id')):
        query = get_json_rows_from_purchases(ds, cid)
        products_hist_counter, histdata_products = extract_data_from_json_dict(query)
        recs = retailHeroModel.recommend(products_hist_counter, histdata_products)
        cosine_recs = cosine_model.recommend(products_hist_counter)

        scores.append(normalized_average_precision(val_dict[cid], recs))
        cosine_scores.append(normalized_average_precision(val_dict[cid], cosine_recs))

    logger.debug(f'RetailHeroRecommender MNAP@30: {np.mean(scores)}')
    logger.debug(f'CosineRecommenderModel MNAP@30: {np.mean(cosine_scores)}')
