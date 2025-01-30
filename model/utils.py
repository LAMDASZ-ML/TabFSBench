import os
import numpy as np
import json
import pandas as pd
from .tabularLLM.evaluating.light import *
from .tabularLLM.evaluating.tabllm import *
from .treemodel.LGBM import *
from .treemodel.CatB import *
from .treemodel.XGB import *
from .dlmodel.utils import *
from .download_data import *


THIS_PATH = os.path.dirname(__file__)

def get_dataset(dataset):
    filename = './dataset/' + dataset + '/info.json'
    with open(filename, 'r') as file:
        data = json.load(file)

    task = data['task']
    link = data['link']
    return task, link

def llm(dataset, model, train_set, test_sets):

    # prompt
    prompt = '''
           "instruction": "{question}",
           "input": "",
           "output": "{answer}"
       '''

    # dataset information
    background = " Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease. People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help."
    features_information = "Age: Refers to the age of the patient. (Numerical value of {28} ~ {77})\\n Sex: Indicates the sex of the patient. (Categorical value of M (which means male) and F (which means female)\\n ChestPainType: Describes the type of chest pain experienced. (Categorical value of Asymptomatic, Atypical Angina, Non-Anginal Pain and Typical Angina)\\n RestingBP: Measures the blood pressure at rest. (Numerical value of {0} ~ {200})\\n Cholesterol: Represents the level of serum cholesterol. (Numerical value of {0} ~ {603})\\n FastingBS: Indicates whether the fasting blood sugar is above 120 mg/dl. (Boolean, 0 means False and 1 means True)\\n RestingECG: Records the results of the resting ECG. (Categorical value of LVH, Normal, ST)\\n MaxHR: The highest heart rate achieved. (Numerical value of {60} ~ {202})\\n ExerciseAngina: Presence of angina during exercise. (Boolean, N means No and Y means Yes)\\n Oldpeak: A measure of ST depression. (Numerical value of {-2.6} ~ {6.2})\\n ST_Slope: Describes the slope of the peak exercise ST segment. (Categorical value of Down, Flat, Up)\\n"
    declaration = "\\n *** \\n Here's the specifics about one patient :  "
    question = "\\n In this case, this patient's heart disease is likely to be (present/absent) : "

    # List Template
    text = ""
    head = np.array(train_set.columns)
    for j in range(0, len(train_set.columns) - 1):
        target = "- " + head[j] + " : {} . "
        text = text + target

    # train.json
    file = dataset + "_train.json"
    with open(file, 'a') as write_f:
        write_f.write('[\n')
    for i in range(0, len(train_set)):
        sentence = text.format(*train_set.iloc[i, :-1])
        sentence = background + "Features and their explanations: " + features_information + declaration + sentence + question
        answer = train_set.iloc[i, -1]
        result = prompt.format(question=sentence, answer=answer)
        if i == len(train_set) - 1:
            result = "{" + result + "}\n"
        else:
            result = "{" + result + "},\n"
        with open(file, 'a', encoding='utf-8') as write_f:
            write_f.write(result)
    with open(file, 'a') as write_f:
        write_f.write('\n]')
    write_f.close()

    print("train set is saved in: " + file)

    # test.json
    for i, test_set in enumerate(test_sets):
        file = str(dataset) + "_test_" + str(i) + ".json"
        with open(file, 'a') as write_f:
            write_f.write('[\n')
        for i in range(0, len(test_set)):
            sentence = text.format(*test_set.iloc[i, :-1])
            sentence = background + "Features and their explanations: " + features_information + declaration + sentence + question
            answer = test_set.iloc[i, -1]
            result = prompt.format(question=sentence, answer=answer)
            if i == len(test_set) - 1:
                result = "{" + result + "}\n"
            else:
                result = "{" + result + "},\n"
            with open(file, 'a', encoding='utf-8') as write_f:
                write_f.write(result)
        with open(file, 'a') as write_f:
            write_f.write('\n]')
        write_f.close()

        print("test set is saved in: " + file)

def tabular_llm(dataset, model, train_set, test_sets):

    if model == 'TabLLM':

        dataset_metadata_path = "../dataset/"+ dataset +"/dataset-metadata.json"
        metadata_path = "../dataset/"+ dataset + "/metadata.json"

        if not (os.path.isfile(dataset_metadata_path) and os.path.isfile(metadata_path)):
            print("Files do not exist, downloading the dataset...")
            download_metadata(dataset)

        tabllm = TabLLM()

        # train
        tabllm.train(dataset_name=dataset, train_set=train_set)

        # test
        for i, test_set in enumerate(test_sets):
            auc_tab, acc_tab = tabllm.test(dataset_name=dataset, test_set=test_set)
            print(f"Tabllm: test_{i}: acc: {acc_tab}, auc: {auc_tab}\n")

    elif model == 'UniPredict':

        light = Light()

        # train
        light.train(dataset_name=dataset, train_set=train_set)

        # test
        for i, test_set in enumerate(test_sets):
            auc_light, acc_light = light.test(dataset_name=dataset, test_set=test_set)
            print(f"UniPredict: test_{i}: acc: {acc_light}, auc: {auc_light}\n")

def tree_model(dataset, model, train_set, test_sets):

    dataset_task, _ = get_dataset(dataset) # Get whether the dataset is a binary, multiclassification or regression task

    if model == 'LightGBM':
        metric1, metric2 = LGBM(dataset_task, train_set, test_sets)
    elif model == 'XGBoost':
        metric1, metric2 = XGB(dataset_task, train_set, test_sets)
    elif model == 'CatBoost':
        metric1, metric2 = CatB(dataset_task, train_set, test_sets)

    if dataset_task == 'binary' or dataset_task == 'multiclass':
        for i in range(0,len(metric1)):
            print(f"{model}: test_{i}: acc: {metric1[i]}, auc: {metric2[i]}\n")
    elif dataset_task == 'regression':
        for i in range(0,len(metric1)):
            print(f"{model}: test_{i}: rmse: {metric1[i]}\n")

def deep_learning(dataset, model, train_set, test_sets):

    dataset_task, _ = get_dataset(dataset)  # Get whether the dataset is a binary, multiclassification or regression task

    metric1, metric2 = test_model(dataset, dataset_task, model, train_set, test_sets)

    if dataset_task == 'binary' or dataset_task == 'multiclass':
        for i in range(0,len(metric1)):
            print(f"{model}: test_{i}: acc: {metric1[i]}, auc: {metric2[i]}\n")
    elif dataset_task == 'regression':
        for i in range(0,len(metric1)):
            print(f"{model}: test_{i}: rmse: {metric1[i]}\n")