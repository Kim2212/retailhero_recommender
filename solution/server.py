from flask import Flask, jsonify, request
from collections import Counter, defaultdict
from utils import extract_data_from_json_dict
import pandas as pd
import joblib
import json
import os

cur_dir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)

with open(os.path.join(cur_dir, 'model.pkl'), 'rb') as f:
    model = joblib.load(f)

TOP_PRODUCTS = [
    '4009f09b04', '4dcf79043e', '15ccaa8685', 'bf07df54e1', '3e038662c0',
    '113e3ace79', '31dcf71bbd', 'f4599ca21a','ea27d5dc75', '5cb93c9bc5',
    '0a46068efc', '439498bce2', '080ace8748', '5645789fdf', '1c257c1a1b',
    '53fc95e177', '76ae00433f', 'dc2001d036', '5186e12ff4', 'ad865591c6',
    'f95785964a', '4d3ab3e72c', '343e841aaa', 'e6f8ac5174', 'e29cab0243',
    '6d0f84a0ac', 'e6a5597d19', 'fc5b0d84e8', '719b704cb6', 'cf1a5be7fb'
]


@app.route('/ready')
def ready():
    return "OK"

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    if len(data['transaction_history']) == 0:
        recs = TOP_PRODUCTS

    else:
        products_hist_counter, histdata_products = extract_data_from_json_dict(data)
        recs = model.recommend(products_hist_counter, histdata_products)

    return jsonify({
        'recommended_products': recs
    })

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=8000)
