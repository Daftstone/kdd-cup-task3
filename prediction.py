import pickle
import numpy as np
import pandas as pd
from functools import lru_cache
import time
import os
from utils import *

train_data_dir = '.'
test_data_dir = '.'
task = 'task3'


@lru_cache(maxsize=1)
def read_product_data():
    return pd.read_csv(os.path.join(train_data_dir, 'data/products_train.csv'))


@lru_cache(maxsize=1)
def read_train_data():
    return pd.read_csv(os.path.join(train_data_dir, 'data/sessions_train.csv'))


@lru_cache(maxsize=3)
def read_test_data(task):
    return pd.read_csv(os.path.join(test_data_dir, f'data/sessions_test_{task}_phase2.csv'))


products = read_product_data()  # load products
hist_data = read_train_data()  # load training data
test_sessions = read_test_data(task)  # load test data
test_locale_names = test_sessions['locale'].unique()
with open("data/products_map.pkl", "rb") as tf:  # load id:title dictionary
    products_map = pickle.load(tf)

# construct co-visiting graph
all_id = products['id'].unique()
id_to_idx = {id: i for i, id in enumerate(all_id)}
idx_to_id = {i: id for i, id in enumerate(all_id)}
graph = {id: {} for i, id in enumerate(all_id)}
phase1 = pd.read_csv('data/sessions_test_task3_phase1.csv')
phase2 = pd.read_csv('data/append_data.csv')
for _, row in hist_data.iterrows():  # use training data to construct co-visiting graph
    items = ([s.strip("'\n") for s in row['prev_items'][1:-1].split(" ")] + [row['next_item'].strip()])[::-1][:5]

    for i in range(0, len(items)):
        for j in range(i + 1, len(items)):
            if (items[j] != items[i]):
                try:
                    graph[items[i]][items[j]] += 1.1 - abs(j - i) * 0.2
                except:
                    graph[items[i]][items[j]] = 1.1 - abs(j - i) * 0.2
                try:
                    graph[items[j]][items[i]] += 1.1 - abs(j - i) * 0.2
                except:
                    graph[items[j]][items[i]] = 1.1 - abs(j - i) * 0.2
for _, row in phase1.iterrows():  # use task3 phase1 data to construct co-visiting graph
    items = ([s.strip("'\n") for s in row['prev_items'][1:-1].split(" ")])[::-1][:5]
    for i in range(0, len(items)):
        for j in range(i + 1, len(items)):
            if (items[j] != items[i]):
                try:
                    graph[items[i]][items[j]] += 1.1 - abs(j - i) * 0.2
                except:
                    graph[items[i]][items[j]] = 1.1 - abs(j - i) * 0.2
                try:
                    graph[items[j]][items[i]] += 1.1 - abs(j - i) * 0.2
                except:
                    graph[items[j]][items[i]] = 1.1 - abs(j - i) * 0.2
with open("data/graph_session_last_copy.pkl", "wb") as tf:
    pickle.dump(graph, tf)
for _, row in phase2.iterrows():  # use task1& task2 data to construct co-visiting graph
    items = ([s.strip("'\n") for s in row['prev_items'][1:-1].split(" ")])[::-1][:5]
    for i in range(0, len(items)):
        for j in range(i + 1, len(items)):
            if (items[j] != items[i]):
                try:
                    graph[items[i]][items[j]] += 1.1 - abs(j - i) * 0.2
                except:
                    graph[items[i]][items[j]] = 1.1 - abs(j - i) * 0.2
                try:
                    graph[items[j]][items[i]] += 1.1 - abs(j - i) * 0.2
                except:
                    graph[items[j]][items[i]] = 1.1 - abs(j - i) * 0.2
with open("data/graph_session_last.pkl", "wb") as tf:
    pickle.dump(graph, tf)
with open("data/graph_session_last_copy.pkl", "rb") as tf:
    graph = pickle.load(tf)
with open("data/graph_session_last.pkl", "rb") as tf:
    graph1 = pickle.load(tf)

languages = ["ES", "DE", "FR", "IT", "UK", "JP"]
predictions = []
for locale in languages:
    sess_test_locale = test_sessions.query(f'locale == "{locale}"').copy()
    cur_pred = title_predictions(locale, sess_test_locale, products_map, graph, graph1)
    predictions.append(cur_pred)

predictions = pd.concat(predictions)
if ("Unnamed: 0" in predictions.columns):
    predictions.drop("Unnamed: 0", inplace=True, axis=1)
predictions.sort_index(inplace=True)


def check_predictions(predictions):
    """
    These tests need to pass as they will also be applied on the evaluator
    """
    test_locale_names = test_sessions['locale'].unique()
    for locale in test_locale_names:
        print(locale)
        sess_test = test_sessions.query(f'locale == "{locale}"')
        preds_locale = predictions[predictions['locale'] == sess_test['locale'].iloc[0]]
        assert sorted(preds_locale.index.values) == sorted(
            sess_test.index.values), f"Session ids of {locale} doesn't match"
        assert predictions['next_item_prediction'].apply(
            lambda x: isinstance(x, str)).all(), "Predictions should all be strings"


check_predictions(predictions)
from datetime import datetime

file_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.makedirs(f"result/{file_name}")
predictions.to_parquet(f'result/{file_name}/submission_{task}.parquet', engine='pyarrow')
