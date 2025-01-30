import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import argparse
import json
import subprocess
import os
import itertools
import random
from tabpfn import TabPFNClassifier
from tqdm import tqdm
from .model.utils import (
    get_deep_args,show_results,tune_hyper_parameters,
    get_method,set_seeds
)
from .model.lib.data import (
    get_dataset
)
import shutil
from sklearn.model_selection import GridSearchCV
import pickle

models = [
    'danets',
    'mlp',
    'node',
    'resnet',
    'switchtab',
    'tabcaps',
    'tabnet',
    'tangos'
]

indices_models = [
    'autoint',
    'dcn2',
    'ftt',
    'grownet',
    'saint',
    'snn',
    'tabtransformer'
]

tabr_ohe_models = [
    'tabr',
    'modernNCA'
]


def test_model(dataset, dataset_task, model, train_set, test_sets):
    metric1_by_model = []
    metric2_by_model = []

    if model =="TabPFN":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(current_dir, "../../configs/default/tabpfn.json")
        with open(file, 'r') as f:
            param_grid = json.load(f)
        model = TabPFNClassifier(device='cuda', N_ensemble_configurations=32)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        length = min(1024, len(train_set))
        grid_search.fit(train_set.iloc[:length, :-1], train_set.iloc[:length, -1])
        best_params = grid_search.best_params_
        downstream = TabPFNClassifier(**best_params)
        downstream.fit(train_set.iloc[:length, :-1], train_set.iloc[:length, -1])
        for test_set in test_sets:
            X_test = test_set.iloc[:, :-1]
            y_test = test_set.iloc[:, -1]
            y_pred = downstream.predict(X_test)
            if dataset_task=="binary":
                y_pred_proba = downstream.predict_proba(X_test)[:, 1]
                accuracy = accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            else:
                y_pred_proba = downstream.predict_proba(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            metric1_by_model.append(accuracy)
            metric2_by_model.append(roc_auc)
        with open('tabpfn.pkl', 'wb') as f:
             pickle.dump(downstream, f)
        return metric1_by_model, metric2_by_model

    else:
        preprocessing(dataset, train_set, type="train")
        preprocessing(dataset, test_sets[0], type="test")
        loss_list, results_list, time_list = [], [], []
        args, default_para, opt_space = get_deep_args(dataset, model)
        train_val_data, test_data, info = get_dataset(args.dataset, args.dataset_path)
        if args.tune:
            args = tune_hyper_parameters(args, opt_space, train_val_data, info)
        method = get_method(args.model_type)(args, info['task'] == 'regression')
        time_cost = method.fit(train_val_data, info)

        for test_set in test_sets:
            preprocessing(dataset, test_set, type="test")
            train_val_data, test_data, info = get_dataset(args.dataset, args.dataset_path)
            vl, vres, metric_name, predict_logits = method.predict(test_data, info, model_name=args.evaluate_option)
            loss_list.append(vl)
            results_list.append(vres)
            time_list.append(time_cost)
            metric1, metric2 = show_results(args, info, metric_name, loss_list, results_list, time_list)
            metric1_by_model.append(metric1)
            metric2_by_model.append(metric2)

    return metric1_by_model, metric2_by_model

def preprocessing(dataset, data, type):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    number = X.select_dtypes(exclude=['object'])
    category = X.select_dtypes(include=['object'])
    newfile = './model/dlmodel/model/dataset/' + dataset + '/'

    if type == "train":
        if number.shape[1] != 0:
            np.save(newfile + 'N_train.npy', number)
            np.save(newfile + 'N_val.npy', number)
        else:
            if os.path.exists(newfile + 'N_train.npy'):
                os.remove(newfile + 'N_train.npy')
            if os.path.exists(newfile + 'N_val.npy'):
                os.remove(newfile + 'N_val.npy')
        if category.shape[1] != 0:
            np.save(newfile + 'C_train.npy', category)
            np.save(newfile + 'C_val.npy', category)
        else:
            if os.path.exists(newfile + 'C_train.npy'):
                os.remove(newfile + 'C_train.npy')
            if os.path.exists(newfile + 'C_val.npy'):
                os.remove(newfile + 'C_val.npy')
        np.save(newfile + 'y_train.npy', y)
        np.save(newfile + 'y_val.npy', y)
    elif type == "test":
        if number.shape[1] != 0:
            np.save(newfile + 'N_test.npy', number)
        else:
            if os.path.exists(newfile + 'N_test.npy'):
                os.remove(newfile + 'N_test.npy')
        if category.shape[1] != 0:
            np.save(newfile + 'C_test.npy', category)
        else:
            if os.path.exists(newfile + 'C_test.npy'):
                os.remove(newfile + 'C_test.npy')
        np.save(newfile + 'y_test.npy', y)

    json_file = './model/dlmodel/model/dataset/' + dataset + '/info.json'
    with open(json_file, 'r') as file:
        info = json.load(file)
    info['n_num_features'] = number.shape[1]
    info['n_cat_features'] = category.shape[1]
    with open(json_file, 'w') as file:
        json.dump(info, file)
