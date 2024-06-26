# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from sklearn import metrics
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from collections import namedtuple, OrderedDict
from itertools import product
import utils.fnc_scorer as fnc_scorer
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle


class Runbuilder():
    @staticmethod
    def get_runs(param):
        Run = namedtuple('Run', param.keys())
        runs = []
        for run in product(*param.values()):
            runs.append(Run(*run))
        return runs

class RunManager():
    def __init__(self):
        # code smell

        self.train_epoch_count = 0
        self.val_epoch_count = 0
        self.epoch_loss = 0
        self.epoch_start_time = None
        self.run_params = None
        self.run_start_time = None
        self.run_count = 0
        self.run_data_train = []
        self.run_data_val = []
        self.run_data_test = []
        self.true_label = []
        self.pred_label = []
        self.pred_score = []
        self.train_results = OrderedDict()
        self.val_results = OrderedDict()
        self.test_results = OrderedDict()

        self.network = None
        self.tb = None
        self.device = None

    def begin_run(self, path, params):
        self.run_start_time = time.time()
        self.run_params = params
        self.save_path = path
        self.run_count += 1
        self.train_epoch_count = 0
        self.val_epoch_count = 0
        self.tb = SummaryWriter(log_dir=self.save_path)
        self.device = params.device
        self.run_data_train = []
        self.run_data_val = []
        self.run_data_test = []
        self.train_results = OrderedDict()
        self.val_results = OrderedDict()
        self.test_results = OrderedDict()

    def begin_train_epoch(self):
        self.epoch_start_time = time.time()
        self.train_epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_instance = 0
        self.true_label = []
        self.pred_label = []
        self.pred_score = []

    def end_train_epoch(self):
        self.epoch_duration = time.time() - self.epoch_start_time
        self.run_duration = time.time() - self.run_start_time
        loss = self.epoch_loss / self.epoch_num_instance
        # print(f'the training loss for the {self.train_epoch_count} epoch is :', loss)
        self.tb.add_scalar('Train_Loss', loss, self.train_epoch_count)

        results = OrderedDict()
        for k, v in self.run_params._asdict().items():
            results[k] = v
        results['run'] = self.run_count
        results['train epoch'] = self.train_epoch_count
        results['train loss'] = loss
        results['train epoch duration'] = self.epoch_duration
        results['train run duration'] = self.run_duration

        if len(set(self.true_label)) == 2:
            p_class, r_class, f_class, support_micro = metrics.precision_recall_fscore_support(y_true=self.true_label,
                                                                                               y_pred=self.pred_label,
                                                                                               labels=[0, 1],
                                                                                               average=None)
            f1_agree = f_class[0]
            f1_disagree = f_class[1]
            F1_macro = metrics.f1_score(self.true_label, self.pred_label, labels=[0, 1], average='macro')
            accuracy = metrics.accuracy_score(self.true_label, self.pred_label)
            cm = metrics.confusion_matrix(self.true_label, self.pred_label)
            roc_auc_dict = self.roc_auc(self.pred_score, self.true_label, "Train", "False")
            # print("Training Process ROC Area Under The Curve:", roc_auc_dict)
            # print('Train Macro F1:', F1_macro)

            results['Accuracy'] = accuracy
            results['Agree F1'] = f1_agree
            results['Disagree F1'] = f1_disagree
            results['Macro F1'] = F1_macro
            results['Confusion matrix'] = cm
            results['Agree AUC'] = roc_auc_dict[0]
            results['Disagree AUC'] = roc_auc_dict[1]
            results['Macro AUC'] = roc_auc_dict["macro"]

        if len(set(self.true_label)) == 4:
            p_class, r_class, f_class, support_micro = metrics.precision_recall_fscore_support(y_true=self.true_label,
                                                                                               y_pred=self.pred_label,
                                                                                               labels=[0, 1, 2, 3],
                                                                                               average=None)
            f1_agree = f_class[0]
            f1_disagree = f_class[1]
            f1_discuss = f_class[2]
            f1_unrelated = f_class[3]
            F1_macro = metrics.f1_score(self.true_label, self.pred_label, labels=[0, 1, 2, 3], average='macro')
            accuracy = metrics.accuracy_score(self.true_label, self.pred_label)
            cm = metrics.confusion_matrix(self.true_label, self.pred_label)
            roc_auc_dict = self.roc_auc(self.pred_score, self.true_label, "Train", "False")
            # print("Training Process ROC Area Under The Curve:", roc_auc_dict)
            # print('Train Macro F1:', F1_macro)
            score, _ = fnc_scorer.score_submission(self.true_label, self.pred_label)
            _, max_score = fnc_scorer.score_defaults(self.true_label)
            fnc_score = score / max_score

            results['Accuracy'] = accuracy
            results["FNC_score"] = fnc_score
            results['Agree F1'] = f1_agree
            results['Disagree F1'] = f1_disagree
            results['Discuss F1'] = f1_discuss
            results['Unrelated F1'] = f1_unrelated
            results['Macro F1'] = F1_macro
            results['Confusion matrix'] = cm
            results['Agree AUC'] = roc_auc_dict[0]
            results['Disagree AUC'] = roc_auc_dict[1]
            results['Discuss AUC'] = roc_auc_dict[2]
            results['Unrelated AUC'] = roc_auc_dict[3]
            results['Macro AUC'] = roc_auc_dict["macro"]
        self.train_results = results
        self.run_data_train.append(results)
        df = pd.DataFrame.from_dict(self.run_data_train, orient='columns')

    def begin_val_epoch(self):
        self.epoch_start_time = time.time()
        self.val_epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_instance = 0
        self.true_label = []
        self.pred_label = []
        self.pred_score = []

    def end_val_epoch(self):
        self.epoch_duration = time.time() - self.epoch_start_time
        self.run_duration = time.time() - self.run_start_time
        loss = self.epoch_loss / self.epoch_num_instance
        # print(f'the validation loss for the {self.val_epoch_count} epoch is :', loss)
        self.tb.add_scalar('Val_Loss', loss, self.val_epoch_count)

        results = OrderedDict()
        for k, v in self.run_params._asdict().items():
            results[k] = v
        results['run'] = self.run_count
        results['val epoch'] = self.val_epoch_count
        results['val loss'] = loss
        results['val epoch duration'] = self.epoch_duration
        results['run duration'] = self.run_duration
        if len(set(self.true_label)) == 2:
            p_class, r_class, f_class, support_micro = metrics.precision_recall_fscore_support(y_true=self.true_label,
                                                                                               y_pred=self.pred_label,
                                                                                               labels=[0, 1],
                                                                                               average=None)
            f1_agree = f_class[0]
            f1_disagree = f_class[1]
            F1_macro = metrics.f1_score(self.true_label, self.pred_label, labels=[0, 1], average='macro')
            accuracy = metrics.accuracy_score(self.true_label, self.pred_label)
            cm = metrics.confusion_matrix(self.true_label, self.pred_label)
            roc_auc_dict = self.roc_auc(self.pred_score, self.true_label, mode="Val", plot_mode="False")
            # print("Validation Process ROC Area Under The Curve:", roc_auc_dict)
            # print('Val Macro F1:', F1_macro)

            results['Accuracy'] = accuracy
            results['Agree F1'] = f1_agree
            results['Disagree F1'] = f1_disagree
            results['Macro F1'] = F1_macro
            results['Confusion matrix'] = cm
            results['Agree AUC'] = roc_auc_dict[0]
            results['Disagree AUC'] = roc_auc_dict[1]
            results['Macro AUC'] = roc_auc_dict["macro"]

        if len(set(self.true_label)) == 4:
            p_class, r_class, f_class, support_micro = metrics.precision_recall_fscore_support(y_true=self.true_label,
                                                                                               y_pred=self.pred_label,
                                                                                               labels=[0, 1, 2, 3],
                                                                                               average=None)
            f1_agree = f_class[0]
            f1_disagree = f_class[1]
            f1_discuss = f_class[2]
            f1_unrelated = f_class[3]
            F1_macro = metrics.f1_score(self.true_label, self.pred_label, labels=[0, 1, 2, 3], average='macro')
            accuracy = metrics.accuracy_score(self.true_label, self.pred_label)
            cm = metrics.confusion_matrix(self.true_label, self.pred_label)
            roc_auc_dict = self.roc_auc(self.pred_score, self.true_label, mode='Val', plot_mode="False")
            # print("Validation Process ROC Area Under The Curve:", roc_auc_dict)
            # print('Val Macro F1:', F1_macro)
            score, _ = fnc_scorer.score_submission(self.true_label, self.pred_label)
            _, max_score = fnc_scorer.score_defaults(self.true_label)
            fnc_score = score / max_score

            results['Accuracy'] = accuracy
            results["FNC_score"] = fnc_score
            results['Agree F1'] = f1_agree
            results['Disagree F1'] = f1_disagree
            results['Discuss F1'] = f1_discuss
            results['Unrelated F1'] = f1_unrelated
            results['Macro F1'] = F1_macro
            results['Confusion matrix'] = cm
            results['Agree AUC'] = roc_auc_dict[0]
            results['Disagree AUC'] = roc_auc_dict[1]
            results['Discuss AUC'] = roc_auc_dict[2]
            results['Unrelated AUC'] = roc_auc_dict[3]
            results['Macro AUC'] = roc_auc_dict["macro"]
        self.val_results = results
        self.run_data_val.append(results)
        val_df = pd.DataFrame.from_dict(self.run_data_val, orient='columns')

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        # early_stopping(loss,self.network)

    def begin_test_epoch(self):
        self.true_label = []
        self.pred_label = []
        self.pred_score = []

    def end_test_epoch(self):
        results = OrderedDict()
        for k, v in self.run_params._asdict().items():
            results[k] = v
        if len(set(self.true_label)) == 2:
            p_class, r_class, f_class, support_micro = metrics.precision_recall_fscore_support(y_true=self.true_label,
                                                                                               y_pred=self.pred_label,
                                                                                               labels=[0, 1],
                                                                                               average=None)
            f1_agree = f_class[0]
            f1_disagree = f_class[1]
            F1_macro = metrics.f1_score(self.true_label, self.pred_label, labels=[0, 1], average='macro')
            accuracy = metrics.accuracy_score(self.true_label, self.pred_label)
            cm = metrics.confusion_matrix(self.true_label, self.pred_label)
            roc_auc_dict = self.roc_auc(self.pred_score, self.true_label, "Test", plot_mode="True")
            # print("Test Process ROC Area Under The Curve:", roc_auc_dict)
            # print('Test Macro F1:', F1_macro)

            results['Accuracy'] = accuracy
            results['Agree F1'] = f1_agree
            results['Disagree F1'] = f1_disagree
            results['Macro F1'] = F1_macro
            results['Confusion matrix'] = cm
            results['Agree AUC'] = roc_auc_dict[0]
            results['Disagree AUC'] = roc_auc_dict[1]
            results['Macro AUC'] = roc_auc_dict["macro"]
        if len(set(self.true_label)) == 4:
            p_class, r_class, f_class, support_micro = metrics.precision_recall_fscore_support(y_true=self.true_label,
                                                                                               y_pred=self.pred_label,
                                                                                               labels=[0, 1, 2, 3],
                                                                                               average=None)
            f1_agree = f_class[0]
            f1_disagree = f_class[1]
            f1_discuss = f_class[2]
            f1_unrelated = f_class[3]
            F1_macro = metrics.f1_score(self.true_label, self.pred_label, labels=[0, 1, 2, 3], average='macro')
            accuracy = metrics.accuracy_score(self.true_label, self.pred_label)
            cm = metrics.confusion_matrix(self.true_label, self.pred_label)
            roc_auc_dict = self.roc_auc(self.pred_score, self.true_label, mode="Test", plot_mode="True")
            # print("Test Process ROC Area Under The Curve:", roc_auc_dict)
            # print('Test Macro F1:', F1_macro)
            score, _ = fnc_scorer.score_submission(self.true_label, self.pred_label)
            _, max_score = fnc_scorer.score_defaults(self.true_label)
            fnc_score = score / max_score

            results['Accuracy'] = accuracy
            results["FNC_score"] = fnc_score
            results['Agree F1'] = f1_agree
            results['Disagree F1'] = f1_disagree
            results['Discuss F1'] = f1_discuss
            results['Unrelated F1'] = f1_unrelated
            results['Macro F1'] = F1_macro
            results['Confusion matrix'] = cm
            results['Agree AUC'] = roc_auc_dict[0]
            results['Disagree AUC'] = roc_auc_dict[1]
            results['Discuss AUC'] = roc_auc_dict[2]
            results['Unrelated AUC'] = roc_auc_dict[3]
            results['Macro AUC'] = roc_auc_dict["macro"]
        self.test_results = results
        self.run_data_test.append(results)
        test_df = pd.DataFrame.from_dict(self.run_data_test, orient='columns')

    def end_run(self):
        self.tb.close()

    def track_loss(self, loss, batch):
        self.epoch_loss += loss.item() * batch.stance.size(0)

    def track_num_instance(self, preds, labels):
        self.epoch_num_instance += len(labels)

    def collect_data(self, preds, labels):
        if self.device == 'cuda':
            prediction = preds.argmax(dim=1).cuda().cpu().numpy().tolist()
            self.pred_label.extend(prediction)
            labels = labels.cuda().cpu().numpy().tolist()
            self.true_label.extend(labels)
            self.pred_score.extend(preds.detach().cpu().numpy())
        else:
            prediction = preds.argmax(dim=1).cpu().numpy().tolist()
            self.pred_label.extend(prediction)
            labels = labels.numpy().tolist()
            self.true_label.extend(labels)
            self.pred_score.extend(preds.detach().numpy())

    def roc_auc(self, pred_score, true_label, mode, plot_mode):
        Stance = list()
        num_class = len(set(true_label))
        if num_class == 2:
            Stance = ["Agree", "Disagree"]
        if num_class == 4:
            Stance = ["Agree", "Disagree", "Discuss", "Unrelated"]
        score_array = np.array(pred_score)
        # transfrom the label to the onehot representation
        label_tensor = torch.tensor(true_label)
        label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
        label_onehot = torch.zeros(label_tensor.shape[0], num_class)
        label_onehot.scatter_(dim=1, index=label_tensor, value=1)
        label_onehot = np.array(label_onehot)

        fpr_dict = dict()
        tpr_dict = dict()
        roc_auc_dict = dict()
        for i in range(num_class):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

        # macro
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_class):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        # Finally average it and compute AUC
        mean_tpr /= num_class
        fpr_dict["macro"] = all_fpr
        tpr_dict["macro"] = mean_tpr
        roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

        if plot_mode == "True":
            # plot the average roc curve of all class
            plt.figure()
            lw = 2
            plt.plot(fpr_dict["macro"], tpr_dict["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc_dict["macro"]),
                     color='navy', linestyle=':', linewidth=4)

            colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            for i, color in zip(range(num_class), colors):
                plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                               ''.format(Stance[i], roc_auc_dict[i]))
            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Some extension of Receiver operating characteristic to multi-class')
            plt.legend(loc="lower right")
            plt.savefig(f'{self.save_path}/{mode}_roc.jpg')
            plt.show()
        return roc_auc_dict

    def save(self, filename):
        # print("Saving the train, validation, and test result")
        pd.DataFrame.from_dict(self.run_data_train, orient='columns').to_csv(f'{self.save_path}/train_{filename}.csv')
        pd.DataFrame.from_dict(self.run_data_val, orient='columns').to_csv(f'{self.save_path}/val_{filename}.csv')
        pd.DataFrame.from_dict(self.run_data_test, orient='columns').to_csv(f'{self.save_path}/test_{filename}.csv')

    def final_result(self):
        return self.train_results,self.val_results,self.test_results


