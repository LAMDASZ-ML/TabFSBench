import xgboost as xgb
import sklearn
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from .utils import *

class DataAugmentor():
    """
    Example usage:
        >>> augmentor = DataAugmentor(samples, labels, column_names)
        >>> outputs, auc = augmentor.generate_label_prompt()
    """
    def __init__(self, samples, labels, column_names=None):
        self.samples = samples
        self.labels = labels
        if column_names:
            self.column_names = column_names
            self.numericalize_samples()
        # if no column names are provided, assume that samples are numericalized
        self.preds = None

    def numericalize_samples(self):
        self.samples, self.labels = numericalize(self.samples, self.labels, self.column_names)

    def augment_label(self):
        clf = xgb.XGBClassifier(n_estimators = 100)
        calibrated_clf = CalibratedClassifierCV(estimator=clf, method='isotonic')
        calibrated_clf.fit(self.samples, self.labels)
        self.preds = calibrated_clf.predict_proba(self.samples)
        auc = calculate_auc(self.labels, self.preds)
        return auc

    def generate_label_prompt(self, print_auc=False):
        auc = self.augment_label()
        if print_auc:
            print(auc)
        outputs = serialize_output(self.preds)
        return outputs, auc


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