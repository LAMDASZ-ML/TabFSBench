import os
import json
import pandas as pd
import numpy as np
from .preprocess_kaggle_dataset import transform_data
from .xgb import *

DEFAULT_DATASET_INDEXING_PATH = 'files/unified/dataset_list/'
DEFAULT_DATASET_SAVING_PATH = 'dataset/'

def preprocess_data(data, dataset_name, set_type):
    """
    Preprocess the given DataFrame and save it as a .pt file for training or testing.

    Args:
        data (pd.DataFrame): The input dataset as a DataFrame.
        dataset_name (str): Name of the dataset (used for defining file paths).
        set_type (str): Type of the data, either 'train' or 'test'.

    Raises:
        AssertionError: If necessary metadata files are missing in the dataset directory.
        ValueError: If `data_type` is neither 'train' nor 'test'.
    """
    file_path = DEFAULT_DATASET_SAVING_PATH + dataset_name
    dataset = data

    # filter out the unqualified datasets
    files_lst = []
    for root, dirs, files in os.walk(file_path):
        files_lst.extend(files)
    assert 'metadata.json' in files_lst, f'Preprocessed metadata not included in this folder: {files_lst}, {file_path}'

    # define paths to datasets and metadata files
    files_lst.remove('dataset-metadata.json')
    files_lst.remove('metadata.json')
    metadata_path = os.path.join(file_path, 'metadata.json')
    # open files and extract values
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    target_name = metadata['target']
    bin_lst = metadata['bins']
    label_lst = metadata['labels']
    annotations = metadata['metadata']
    print(target_name)

    # prepare samples (x), column values, labels (y) and annotations (dataset description/metadata)
    samples = dataset.drop(target_name, axis=1).round(4)
    col = samples.columns.to_list()
    if metadata['bins'] != 'N/A' and dataset[target_name].nunique() > 10:
        segs = [0] + bin_lst + [np.inf]
        labels = pd.cut(dataset[target_name], bins=segs, labels=label_lst)
    else:
        labels = dataset[target_name]
    annotations = [annotations] * len(samples)

    # preprocessing
    xgb_baseline, prompts, output = transform_data(samples, col, labels, annotations)
    augmentor = DataAugmentor(xgb_baseline[0], xgb_baseline[1])
    o_train, auc = augmentor.generate_label_prompt()
    train = ((xgb_baseline[0], xgb_baseline[1]), (prompts[0], prompts[1], prompts[2]), o_train)

    # Save preprocessed data
    if set_type == 'train':
        save_path = os.path.join(file_path, 'train_set.pt')
        torch.save(train, save_path)
    elif set_type == 'test':
        save_path = os.path.join(file_path, 'test_set.pt')
        torch.save(train, save_path)
    else:
        raise ValueError(f"Invalid data_type '{set_type}'. Expected 'train' or 'test'.")