import numpy as np
import pandas as pd
from astropy.modeling.tabular import tabular_model
from sklearn.model_selection import train_test_split
import argparse
import json
import os
from model.utils import *

# model name list
tree_based_models = [
    'LightGBM',
    'XGBoost',
    'CatBoost'
]

deep_learning_models = [
    'danets',
    'mlp',
    'node',
    'resnet',
    'switchtab',
    'tabcaps',
    'tabnet',
    'tangos',
    'autoint',
    'dcn2',
    'ftt',
    'grownet',
    'saint',
    'snn',
    'tabtransformer',
    'tabr',
    'modernNCA',
    'TabPFN'
]

llms = [
    'LLaMA3-8B'
]

tabular_llms = [
    'TabLLM',
    'UniPredict'
]

def pearson(df, ascending):
    correlation = df.corr(method='pearson')
    last_column_correlation = correlation.iloc[:, -1]
    last_column_correlation = last_column_correlation.drop(last_column_correlation.index[last_column_correlation == 1])
    sorted_correlation = last_column_correlation.abs().sort_values(ascending=ascending)
    sorted_columns = sorted_correlation.index
    return sorted_columns

def split_dataset(dataset, task, degree):
    filename = './datasets/' + dataset + '/' + dataset + '.csv'
    df = pd.read_csv(filename)
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    all_test_sets = []

    all_test_sets.append(test_set) # In Distribution

    if task == 'single':
        sorted_columns = pearson(df, ascending=True)
        test = test_set.copy()
        for num in range(1, len(test.columns)):
            if test[sorted_columns[num-1]].dtype == 'object':
                mode_value = train_set[sorted_columns[num-1]].mode()[0]
                test[sorted_columns[num-1]] = mode_value
            else:
                column_means = train_set[sorted_columns[num-1]].mean()
                test[sorted_columns[num-1]] = column_means
            all_test_sets.append(test)
    elif task == 'multi-removeleast':
        sorted_columns = pearson(df, ascending=True)
        test = test_set.copy()
        for num in range(1, len(test.columns)):
            for i in range(0, num):
                if test[sorted_columns[i]].dtype == 'object':
                    mode_value = train_set[sorted_columns[i]].mode()[0]
                    test[sorted_columns[i]] = mode_value
                else:
                    column_means = train_set[sorted_columns[i]].mean()
                    test[sorted_columns[i]] = column_means
            all_test_sets.append(test)
    elif task == 'multi-removemost':
        sorted_columns = pearson(df, ascending=False)
        test = test_set.copy()
        for num in range(1, len(test.columns)):
            for i in range(0, num):
                if test[sorted_columns[i]].dtype == 'object':
                    mode_value = train_set[sorted_columns[i]].mode()[0]  # mode() 返回一个序列，取第一个值
                    test[sorted_columns[i]] = mode_value
                else:
                    column_means = train_set[sorted_columns[i]].mean()
                    test[sorted_columns[i]] = column_means
            all_test_sets.append(test)
    elif task == 'random':
        part_test_sets = []
        if degree != 'all':
            degree = int(degree)
            num = degree * len(df.columns)
            combinations = list(itertools.combinations(data.columns[:-1], num))
            for combination in combinations:
                column_list = train.columns.tolist()
                for i in combination:
                    index = column_list.index(i)
                    if test[sorted_columns[index]].dtype == 'object':
                        mode_value = train_set[sorted_columns[index]].mode()[0]
                        test[sorted_columns[index]] = mode_value
                    else:
                        column_means = train_set[sorted_columns[index]].mean()
                        test[sorted_columns[index]] = column_means
                test = test[df.columns]
                part_test_sets.append(test)
            part_test_sets = pd.concat(part_test_sets, ignore_index=True)
            all_test_sets.append(part_test_sets)
        else:
            for num in range(1, len(data.columns)):
                combinations = list(itertools.combinations(data.columns[:-1], num))
                for combination in combinations:
                    column_list = train.columns.tolist()
                    for i in combination:
                        index = column_list.index(i)
                        if test[sorted_columns[index]].dtype == 'object':
                            mode_value = train_set[sorted_columns[index]].mode()[0]
                            test[sorted_columns[index]] = mode_value
                        else:
                            column_means = train_set[sorted_columns[index]].mean()
                            test[sorted_columns[index]] = column_means
                    test = test[df.columns]
                    part_test_sets.append(test)
                part_test_sets = pd.concat(part_test_sets, ignore_index=True)
                all_test_sets.append(part_test_sets)
    return train_set, all_test_sets

def evaluate_model(dataset, model, train_set, test_sets):
    if model in tabular_llm:
        tabular_llm(dataset, model, train_set, test_sets)
    elif model in llm:
        llm(dataset, model, train_set, test_sets)
    elif model in deep_learning_model:
        deep_learning(dataset, model, train_set, test_sets)
    else:
        tree_model(dataset, model, train_set, test_sets)

def main(dataset, model, task, degree, export_dataset):
    # 1. get train and test set
    train_set, test_sets = split_dataset(dataset, task, degree)

    # 2. whether to export the dataset or not
    if export_dataset:
        train_set.to_csv('train.csv', index=False)
        for index, test_set in enumerate(test_sets):
            test_set.to_csv('test_' + index + '.csv', index=False)

    # 3. evaluate model
    evaluate_model(dataset, model, train_set, test_sets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset Name")
    parser.add_argument('--model', type=str, required=True,
                        help="Model Name")
    parser.add_argument('--task', type=str, required=True,
                        help="Task Name")
    parser.add_argument('--degree', type=str,
                        help="Feature Shift Degree")
    parser.add_argument('--export_dataset', type=bool,
                        help="whether to export the dataset or not")
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    task = args.task
    degree = args.degree
    export_dataset = args.export_dataset
    if task == 'random' and degree == None:
        print("Please specify a degree for random")
    else:
        main(dataset, model, task, degree, export_dataset)
