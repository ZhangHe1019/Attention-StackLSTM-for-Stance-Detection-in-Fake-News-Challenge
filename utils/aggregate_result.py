# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
from collections import OrderedDict


class aggregate_result():
    def __init__(self, path, params, k_fold):
        self.accuracy = 0
        self.F1_macro = 0
        self.F1_agree = 0
        self.F1_disagree = 0
        self.F1_discuss = 0
        self.F1_unrelated = 0
        self.FNC_score = 0
        self.path = path
        self.params = params
        self.k_fold = k_fold
        self.data = []
        self.results = OrderedDict()
    def run(self, results):
        self.accuracy += results['Accuracy']/self.k_fold
        self.FNC_score += results["FNC_score"]/self.k_fold
        self.F1_agree += results['Agree F1']/self.k_fold
        self.F1_disagree += results['Disagree F1']/self.k_fold
        self.F1_discuss += results['Discuss F1']/self.k_fold
        self.F1_unrelated += results['Unrelated F1']/self.k_fold
        self.F1_macro += results['Macro F1']/self.k_fold
    def output(self, mode):
        for k, v in self.params._asdict().items():
            self.results[k] = v
        self.results["Accuracy"] = self.accuracy
        self.results["FNC_score"] = self.FNC_score
        self.results["F1_agree"] = self.F1_agree
        self.results["F1_disagree"] = self.F1_disagree
        self.results["F1_discuss"] = self.F1_discuss
        self.results["F1_unrelated"] = self.F1_unrelated
        self.results["Macro_F1"] = self.F1_macro
        self.data.append(self.results)
        pd.DataFrame.from_dict(self.data, orient='columns').to_csv(f'{self.path}/aggregate_{mode}.csv')





