from sklearn.model_selection import GridSearchCV
import json
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import os
import pandas as pd
import pickle


def LGBM(task, train_set, test_sets):
        metric1_by_model = []
        metric2_by_model = []
        train_set[train_set.columns[train_set.dtypes == 'object']] = train_set.select_dtypes(['object']).apply(
                lambda x: pd.Categorical(x).codes)
        test_sets = [df.assign(**{col: pd.Categorical(df[col]).codes for col in df.columns[df.dtypes == 'object']}) for
                     df in test_sets]
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(current_dir, "../../configs/lightgbm.json")
        with open(file, 'r') as f:
                param_grid = json.load(f)

        if task == 'binary':
                model = LGBMClassifier()
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
                grid_search.fit(train_set.iloc[:, :-1], train_set.iloc[:, -1])
                best_params = grid_search.best_params_
                downstream = LGBMClassifier(**best_params)
                downstream.fit(train_set.iloc[:, :-1], train_set.iloc[:, -1])
                for test_set in test_sets:
                        X_test = test_set.iloc[:, :-1]
                        y_test = test_set.iloc[:, -1]
                        y_pred = downstream.predict(X_test)
                        y_pred_proba = downstream.predict_proba(X_test)[:, 1]  # 获取正类概率
                        accuracy = accuracy_score(y_test, y_pred)
                        roc_auc = roc_auc_score(y_test, y_pred_proba)
                        metric1_by_model.append(accuracy)
                        metric2_by_model.append(roc_auc)

        elif task == 'multiclass':
                model = LGBMClassifier()
                param_grid['num_classes'] = 3
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
                grid_search.fit(train_set.iloc[:, :-1], train_set.iloc[:, -1])
                best_params = grid_search.best_params_
                downstream = LGBMClassifier(**best_params)
                downstream.fit(train_set.iloc[:, :-1], train_set.iloc[:, -1])
                for test_set in test_sets:
                        X_test = test_set.iloc[:, :-1]
                        y_test = test_set.iloc[:, -1]
                        y_pred = downstream.predict(X_test)
                        y_pred_proba = downstream.predict_proba(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                        metric1_by_model.append(accuracy)
                        metric2_by_model.append(roc_auc)

        elif task == 'regression':
                model = LGBMRegressor()
                param_grid.pop("num_classes", None)
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
                grid_search.fit(train_set.iloc[:, :-1], train_set.iloc[:, -1])
                best_params = grid_search.best_params_
                downstream = LGBMRegressor(**best_params)
                downstream.fit(train_set.iloc[:, :-1], train_set.iloc[:, -1])
                for test_set in test_sets:
                        X_test = test_set.iloc[:, :-1]
                        y_test = test_set.iloc[:, -1]
                        y_pred = downstream.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        metric1_by_model.append(rmse)
        with open('lightgbm.pkl', 'wb') as f:
             pickle.dump(downstream, f)
        return metric1_by_model, metric2_by_model



