import torch
import json
import numpy as np

from pandas.api.types import is_numeric_dtype, is_integer_dtype
from sklearn.metrics import roc_auc_score

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_json(path, data):
    with open(path, 'w+') as f:
        json.dump(data, f, indent=4)


def data_to_prompt(data, column_names):
    listed_data = data.values.tolist()
    prompts = []
    for row in listed_data:
        prompt = []
        for i in range(len(column_names)):
            column = column_names[i].replace('_', ' ')
            value = row[i]
            prompt_segment = f'{column} is {value}; '
            prompt.append(prompt_segment)
        prompt = ''.join(prompt)[:-2] + '.\n'
        prompts.append(prompt)
    return prompts


def label_to_prompt(label, length):
    classification = []
    prompt = []
    for key in label.keys():
        classification.append(f'class {label[key]}: xxx; ')
        prompt_segment = f'class {label[key]} stands for "{key}"; '
        prompt.append(prompt_segment)
    # Class 0 is xxx; class 1 is xxx; where class 0 stands for yyy; class 1 stands for yyy.
    full_prompt = ''.join(classification) + 'where ' + ''.join(prompt)[:-2] + '.\n'
    return [full_prompt] * length


def numericalize(samples, labels, column_names):
    samples = samples.reset_index(drop=True)
    for i in range(len(column_names)):
        col = column_names[i]
        if is_numeric_dtype(samples[col]) or is_integer_dtype(samples[col]):
            continue
        categories = [cat for cat in set(samples[col].to_list())]
        cat_dict = {categories[i]: i for i in range(len(categories))}
        samples[col] = samples[col].map(cat_dict).astype(int)
    categories = [cat for cat in set(labels.to_list())]
    cat_dict = {categories[i]: i for i in range(len(categories))}
    labels = labels.map(cat_dict).astype(int)
    return samples.to_numpy(), labels.to_numpy().reshape(-1, 1)


def calculate_auc(labels, preds):
    # ground truth
    if len(labels.shape) > 1:
        y_gt = labels.squeeze(1)
    else: 
        y_gt = labels
    onehot = np.zeros((y_gt.size, y_gt.max().astype(int) + 1))
    onehot[np.arange(y_gt.size), y_gt] = 1
    y_gt = onehot
    # pred
    y_pred = preds
    return roc_auc_score(y_gt, y_pred, average=None)

def serialize_output(preds):
    outputs = []
    for i in preds:
        out_strs = []
        for j, k in enumerate(i):
            out_str = f'class {j}: {np.round(k, 2)}; '
            out_strs.append(out_str)
        out_strs = ''.join(out_strs)
        out_strs = out_strs[:-2] + '.'
        outputs.append(out_strs)
    return outputs