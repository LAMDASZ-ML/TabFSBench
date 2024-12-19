import pandas as pd
from sklearn.model_selection import train_test_split
from tabularLLM.evaluating.light import *
from tabularLLM.evaluating.tabllm import *

import itertools
import random

import tabularLLM.evaluating
import tabularLLM.evaluating.light




filename = '/data0/jiazy/my_code/dataset/credit/dataset_31_credit-g.csv'
df = pd.read_csv(filename)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category').cat.codes


correlation = df.corr(method='pearson')
last_column_correlation = correlation.iloc[:, -1]
last_column_correlation = last_column_correlation.drop(
    last_column_correlation.index[last_column_correlation == 1])
sorted_correlation = last_column_correlation.abs().sort_values(ascending=True)
sorted_columns = sorted_correlation.index
data = df

num = 1

data = df.copy(deep=True)
train, test = train_test_split(data, test_size=0.2, random_state=42)
X_train = train.iloc[:, :-1]
X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

# 训练
tabllm = TabLLM()
light = Light()

tabllm.train(dataset_name='credit',train_set=train)
light.train(dataset_name='credit',train_set=train)


# 测试
column_means = X_train[sorted_columns[num - 1]].mean()
X_test[sorted_columns[num - 1]] = column_means

test = pd.concat([X_test, y_test], axis=1)

auc_tab,acc_tab = tabllm.test(dataset_name='credit',test_set=test)
auc_light,acc_light = light.test(dataset_name='credit',test_set=data)
print(f"Tabllm:auc:{auc_tab} acc:{acc_tab}\n")
print(f"light:auc:{auc_light} acc:{acc_light}\n")