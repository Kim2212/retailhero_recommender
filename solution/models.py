# -*- coding: utf-8 -*-

import os
import joblib
import catboost
import numpy as np
import pandas as pd
import scipy.sparse as sp

from implicit.nearest_neighbours import (
    CosineRecommender,
    TFIDFRecommender,
    BM25Recommender,
)
from utils import logger, TOP_PRODUCTS


CB_FEATURES = [
    'age',
    'score',
    'netto',
    'gender',
    'level_1',
    'level_2',
    'level_3',
    'level_4',
    'brand_id',
    'store_id',
    'vendor_id',
    'product_id',
    'is_alcohol',
    'segment_id',
    'is_own_trademark',
    # 'implicit_score',
    # 'days_of_last_purchase',
    'regular_points_received',
    # 'express_points_received',
]

class BaseItemItemRecommenderModel:
    def __init__(self, products: np.ndarray, params: dict):
        self.recommender = CosineRecommender(**params)
        self._product_idx = dict(zip(products, range(len(products))))
        self._idx_product = products.tolist()

    def _make_user_item_csr_row(self, values, item_idx):
        row = sp.coo_matrix(
            (values,
                (
                    np.zeros(len(values)),
                    [self._product_idx[p] for p in item_idx]
                )
            ), shape=(1, len(self._product_idx))
        )

        return row.tocsr()

    def _create_user_item_matrix_from_purchases(self, purchases):
        clients = purchases.client_id.unique()
        clients_mapper = dict(zip(clients, range(len(clients))))

        user_item_matrix = sp.coo_matrix(
            (purchases.relevance.values,
                (
                    purchases.client_id.map(clients_mapper).values,
                    purchases.product_id.map(self._product_idx).values
                )

            )
        )

        user_item_matrix = user_item_matrix.tocsr()
        user_item_matrix.eliminate_zeros()
        return user_item_matrix

    def _fit_recommender(self, purchases):
        user_item_matrix = self._create_user_item_matrix_from_purchases(purchases)
        logger.debug('Training Recommender Model...')
        self.recommender.fit(user_item_matrix.T)


class CosineRecommenderModel(BaseItemItemRecommenderModel):
    def __init__(self, products: np.ndarray, params: dict):
        self.cosine_recommender = CosineRecommender(**params)
        self._product_idx = dict(zip(products, range(len(products))))
        self._idx_product = products.tolist()

    def fit_recommender(self, purchases):
        user_item_matrix = self._create_user_item_matrix_from_purchases(purchases)
        logger.debug('Training CosineRecommender ...')
        self.cosine_recommender.fit(user_item_matrix.T)
        return self

    def recommend(self, products_counter):
        recs = []
        user_item_csr_row = self._make_user_item_csr_row(
            values=list(products_counter.values()),
            item_idx=products_counter.keys()
        )

        cosine_preds = self.cosine_recommender.recommend(
            0, user_item_csr_row, N=30, 
            recalculate_user=True,
            filter_already_liked_items=False
        )

        cosine_preds = [self._idx_product[idx] for (idx, score) in cosine_preds]

        return cosine_preds


class RetailHeroRecommender(BaseItemItemRecommenderModel):
    def __init__(
        self,
        products: pd.DataFrame,
        params_rec: dict,
        params_catboost: dict,
        catboost_features=CB_FEATURES
    ):
        self.ranker = catboost.CatBoost(params_catboost)
        self._catboost_features = catboost_features
        self._nan_fill_dict = dict()

        self.recommender = CosineRecommender(**params_rec)
        self._product_idx = dict(zip(products.product_id, range(len(products))))
        self._idx_product = products.product_id.tolist()
        self._product_features = {
            row['product_id']: dict(row.drop(index='product_id')) \
                        for (i , row) in products.iterrows()
        }

    def _cat_features(self):
        return (
            'gender', 'level_1', 'level_2', 'level_3', 'level_4',
            'product_id', 'is_alcohol', 'brand_id', 'store_id',
            'vendor_id', 'segment_id', 'is_own_trademark',
        )

    def _fillna(self, df):
        for feature, fill_value in self._nan_fill_dict.items():
            df.loc[:, feature] = df.loc[:, feature].fillna(fill_value)

        return df

    def _fit_ranker(self, train, valid=None):
        features = self._catboost_features
        cat_features = self._cat_features()
        cat_inds = [i for i, col in enumerate(features) if col in cat_features]

        for feature in features:
            if feature in cat_features:
                self._nan_fill_dict[feature] = 'unknown'
            else:
                self._nan_fill_dict[feature] = np.nanmedian(train[feature])

        train = self._fillna(train)

        logger.debug(f'Train shape: {train.shape}')
        logger.debug(f'Train target mean: {train.target.mean()}')
        for feature, nuniques in train[features].nunique().to_dict().items():
            logger.debug(f'{feature} has {nuniques} values')

        train_pool = catboost.Pool(
            data=train[features],
            label=train['target'],
            weight=train.weight,
            cat_features=cat_inds,
            group_id=train['client_id']
        )

        if valid is not None:
            valid = self._fillna(valid)
            val_pool = catboost.Pool(
                data=valid[features],
                label=valid['target'],
                weight=valid.weight,
                cat_features=cat_inds,
                group_id=valid['client_id']
            )

        else:
            val_pool = None

        logger.debug('Training Ranker Model...')
        self.ranker.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)


    def train_model(self, train_rec: pd.DataFrame, train_ranker: pd.DataFrame, ranker_labels_dict: dict):
        self._fit_recommender(train_rec)

        cb_feats_df = train_ranker.sort_values(by=['client_id', 'transaction_datetime']) \
                                .drop_duplicates(subset=['client_id', 'product_id'], keep='last')

        logger.debug('Preparing Train data for Ranker Model...')
        implicit_preds = []
        for cid, df in train_ranker.groupby('client_id'):
            csr_row = self._make_user_item_csr_row(
                values=df.relevance.values,
                item_idx=df.product_id.values
            )

            pred = self.recommender.recommend(
                0, csr_row, N=30, 
                recalculate_user=True,
                filter_already_liked_items=False
            )

            for (i , (idx , score)) in enumerate(pred):
                implicit_preds.append(
                    {
                        'client_id': cid,
                        'score': score,
                        'product_id': self._idx_product[idx],
                        'weight': len(pred) - i,
                        'target': int(self._idx_product[idx] in ranker_labels_dict[cid]), 
                    }
                )

        logger.debug('Finished preparing Train data')

        len_before = len(implicit_preds)
        implicit_preds = pd.DataFrame(implicit_preds)
        cb_feats_df = pd.merge(
            implicit_preds, cb_feats_df,
            on=['client_id', 'product_id'], how='left'
        )

        assert len(cb_feats_df) == len_before, 'Shape after merge is different'
        uniq_clients = cb_feats_df.client_id.unique()
        train = cb_feats_df[cb_feats_df.client_id.isin(uniq_clients[:8000])]
        valid = cb_feats_df[cb_feats_df.client_id.isin(uniq_clients[8000:])]

        self._fit_ranker(train, valid=valid)

    def recommend(self, products_counter: dict, histdata_products: dict):
        user_item_csr_row = self._make_user_item_csr_row(
            values=list(products_counter.values()),
            item_idx=products_counter.keys()
        )
        rec_preds = self.recommender.recommend(
            0, user_item_csr_row, N=30, 
            recalculate_user=True,
            filter_already_liked_items=False
        )

        data_list = []
        nan_hist_data = dict(
            store_id=np.nan,
            regular_points_received=np.nan,
            express_points_received=np.nan,
        )

        for i, score in rec_preds:
            pid = self._idx_product[i]
            row_dic = dict(
                product_id=pid, score=score,
                age=histdata_products['age'],
                gender=histdata_products['gender'],
            )

            row_dic.update(self._product_features[pid])
            row_dic.update(histdata_products.get(pid, nan_hist_data))
            data_list.append(row_dic)

        preds_df = pd.DataFrame(data_list)
        preds_df = self._fillna(preds_df)

        preds_df.loc[:, 'catb_score'] = self.ranker.predict(preds_df[self._catboost_features])
        result = preds_df.sort_values(by='catb_score', ascending=False).product_id.tolist()
        if len(result) < 30:
            for t_prod in TOP_PRODUCTS:
                if t_prod not in result and len(result) < 30:
                    result.append(t_prod)

        return result[:30]

