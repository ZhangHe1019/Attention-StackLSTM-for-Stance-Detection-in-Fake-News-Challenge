# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import init
import os
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from torchtext.vocab import Vectors
from collections import namedtuple
from collections import OrderedDict
from itertools import product
import FNC_scorer
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
import torchtext.data as data
from sklearn.model_selection import StratifiedKFold


class Runbuilder():
    @staticmethod
    def get_runs(param):
        Run = namedtuple('Run', param.keys())
        runs = []
        for run in product(*param.values()):
            runs.append(Run(*run))
        return runs

class aggregate_result():
    def __init__(self,path,params,k_fold):
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
    def run(self,results):
        self.accuracy += results['Accuracy']/self.k_fold
        self.FNC_score += results["FNC_score"]/self.k_fold
        self.F1_agree += results['Agree F1']/self.k_fold
        self.F1_disagree += results['Disagree F1']/self.k_fold
        self.F1_discuss += results['Discuss F1']/self.k_fold
        self.F1_unrelated += results['Unrelated F1']/self.k_fold
        self.F1_macro += results['Macro F1']/self.k_fold
    def output(self,mode):
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
        self.device = run.device
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
        print(f'the training loss for the {self.train_epoch_count} epoch is :', loss)
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
            print("Training Process ROC Area Under The Curve:", roc_auc_dict)
            print('Train Macro F1:', F1_macro)

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
            print("Training Process ROC Area Under The Curve:", roc_auc_dict)
            print('Train Macro F1:', F1_macro)
            score, _ = FNC_scorer.score_submission(self.true_label, self.pred_label)
            _, max_score = FNC_scorer.score_defaults(self.true_label)
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
        print(f'the validation loss for the {self.val_epoch_count} epoch is :', loss)
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
            print("Validation Process ROC Area Under The Curve:", roc_auc_dict)
            print('Val Macro F1:', F1_macro)

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
            print("Validation Process ROC Area Under The Curve:", roc_auc_dict)
            print('Val Macro F1:', F1_macro)
            score, _ = FNC_scorer.score_submission(self.true_label, self.pred_label)
            _, max_score = FNC_scorer.score_defaults(self.true_label)
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
            print("Test Process ROC Area Under The Curve:", roc_auc_dict)
            print('Test Macro F1:', F1_macro)

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
            print("Test Process ROC Area Under The Curve:", roc_auc_dict)
            print('Test Macro F1:', F1_macro)
            score, _ = FNC_scorer.score_submission(self.true_label, self.pred_label)
            _, max_score = FNC_scorer.score_defaults(self.true_label)
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
        print("Saving the train, validation, and test result")
        pd.DataFrame.from_dict(self.run_data_train, orient='columns').to_csv(f'{self.save_path}/train_{filename}.csv')
        pd.DataFrame.from_dict(self.run_data_val, orient='columns').to_csv(f'{self.save_path}/val_{filename}.csv')
        pd.DataFrame.from_dict(self.run_data_test, orient='columns').to_csv(f'{self.save_path}/test_{filename}.csv')

    def final_result(self):
        return self.train_results,self.val_results,self.test_results

class LSTM_With_Attention_Network(nn.Module):
    def __init__(self, num_layers, dropout_rate, concat_embeddings, hidden_size, embedding_dim, attention_window,
                 num_class,trainable_embeddings):
        super(LSTM_With_Attention_Network, self).__init__()  # 调用父类的构造方法
        self.num_layers = num_layers
        self.attention_window = attention_window
        self.concat_embedding = nn.Embedding(len(concat_embeddings),embedding_dim)  # vocab_size词汇表大小， embedding_dim词嵌入维度
        self.concat_embedding.weight.data.copy_(concat_embeddings)  # 第一句就是导入词向量
        self.concat_embedding.weight.requires_grad = trainable_embeddings

        self.BiLstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                              bidirectional=False, dropout=dropout_rate)
        self.LinearLayer1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.LinearLayer2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.LinearLayer3 = nn.Linear(hidden_size, 1, bias=False)
        self.LinearLayer4 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.LinearLayer5 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.LinearLayer6 = nn.Linear(hidden_size, num_class)

    def forward(self, text):
        concat_embedding = self.concat_embedding(text)  # torch.Size([sentence_lens,batch_size,embedding_size]),[75, 32, 50]
        output, (hidden, cell) = self.BiLstm(concat_embedding)
        # output shape: torch.Size([sentence_lens,batch_size,hidden_size]) torch.Size([75,32,100])
        # hidden shape: torch.Size([layer_num,batch_size,hidden_size])   torch.Size([2,32,100])
        output_state = output[:self.attention_window]  # torch.Size([15,32,100])
        final_state = hidden[self.num_layers - 1]  # torch.Size([1,32,100])
        final_state_ = final_state.expand(output_state.shape[0], hidden.shape[1],
                                          hidden.shape[2])  # torch.Size([15,32,100])
        M = F.tanh(self.LinearLayer1(output_state) + self.LinearLayer2(final_state_))  # torch.Size([15,32,100])
        alpha = F.softmax(self.LinearLayer3(M))  # torch.Size([15,32,1])
        a = output_state.permute(1, 0, 2)  # torch.Size([32,1,15])
        b = alpha.permute(1, 2, 0)  # torch.Size([32,15,100])
        r = torch.bmm(b, a).permute(1, 0, 2)  # torch.Size([1, 32, 100])
        h = F.tanh(self.LinearLayer4(final_state) + self.LinearLayer5(r))  # torch.Size([1, 32, 100])
        output = self.LinearLayer6(h.squeeze())
        return output


# define the initial function to init the layer's parameters for the network
def weigth_init(m):
    if isinstance(m, nn.LSTM):
        init.xavier_uniform_(m.weight_ih_l0.data, gain=1)
        init.xavier_uniform_(m.weight_hh_l0.data, gain=1)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight, gain=1)

embedding_dim = {"glove.twitter.27B.50d":50,
                 "glove.6B.300d":300,
                 "glove.6B.50d":50,
                 "glove.twitter.27B.25d":25,
                 "glove.6B.100d":100,
                 "glove.6B.200d":200,
                 "glove.twitter.27B.100d":100,
                 "glove.twitter.27B.200d":200,
                 "wiki_news_300d_sub":300,
                 "wiki_news_300d":300,
                 "crawl.300d.2M":300,
                 "GoogleNews.vector300":300}
filenames = {"wiki_news_300d_sub":"wiki-news-300d-1M-subword.vec",
           "glove.twitter.27B.50d":"glove.twitter.27B.50d.txt",
           "glove.6B.300d":"glove.6B.300d.txt",
           "glove.6B.50d":"glove.6B.50d.txt",
           "glove.twitter.27B.25d":"glove.twitter.27B.25d.txt" ,
           "glove.6B.100d": "glove.6B.100d.txt",
           "glove.6B.200d": "glove.6B.200d.txt",
           "glove.twitter.27B.100d": "glove.twitter.27B.100d.txt",
           "glove.twitter.27B.200d":"glove.twitter.27B.200d.txt",
           "crawl.300d.2M":"crawl-300d-2M.vec",
           "wiki_news_300d":"wiki-news-300d-1M.vec",
           "GoogleNews.vector300":"GoogleNews-vectors-negative300.txt"}
params = OrderedDict(epoch_num=[40],
                lr=[0.001],
                batch_size=[128],
                num_layer=[2],
                dropout_rate=[0.8],
                max_length=[75],
                device=['cuda'],
                weight_decay=[0],
                trainset=['FNC-1'],
                valset=['FNC-1'],
                testset=['FNC-1'],
                word_embedding=["glove.6B.50d",
                                "glove.6B.100d",
                                "glove.6B.300d",
                                "glove.6B.200d",
                                "glove.twitter.27B.25d",
                                "glove.twitter.27B.50d",
                                "glove.twitter.27B.100d",
                                "wiki_news_300d",
                                "crawl.300d.2M",
                                "GoogleNews.vector300",
                                ],
                trainable_embeddings=[True,False])

runs = Runbuilder().get_runs(params)
m = RunManager()
Train = pd.read_csv(r".\fnc-1\dataset\train_concat_dataset.csv", header=None)
Train.columns = ["Text", "Stance", "Index"]
X = Train["Text"].values
y = Train["Stance"].values
ind = Train["Index"].values
kf = StratifiedKFold(n_splits=5, shuffle=False)
for run in runs:

    fold_count = 0
    device = torch.device(run.device)
    path = f"./LSTM_With_Attention/Full fnc-1 corpus/5 Fold Train/Word Embedding Selection/{run.word_embedding}_{run.trainable_embeddings}/output"
    if not os.path.exists(path):
        os.makedirs(path)
    train_output = aggregate_result(path,run,5)
    val_output = aggregate_result(path,run,5)
    test_output = aggregate_result(path,run,5)
    for train_index, test_index in kf.split(X, y):
        print(len(train_index))
        print(len(test_index))
        fold_count += 1
        save_path = f"./LSTM_With_Attention/Full fnc-1 corpus/5 Fold Train/Word Embedding Selection/{run.word_embedding}_{run.trainable_embeddings}/fold{fold_count}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print('train_index:%s , test_index: %s ' % (train_index, test_index))
        Train = pd.DataFrame({"Text": X[train_index], "Stance": y[train_index], "Index": ind[train_index]})
        Val = pd.DataFrame({"Text": X[test_index], "Stance": y[test_index], "Index": ind[test_index]})
        Train.to_csv("Train_fold.csv", header=None,index=None)
        Val.to_csv("Val_fold.csv", header=None,index=None)


        LABEL = data.Field(sequential=False,use_vocab=False)
        CONCAT = data.Field(lower=True,fix_length=75)
        INDEX = data.Field(sequential=False,use_vocab=False)
        SEQLEN = data.Field(sequential=False,use_vocab=False)

        TrainDataset = data.TabularDataset(
        path = f'./Train_fold.csv',
        format = 'csv',
        fields = [('text',CONCAT),('stance',LABEL),("index",INDEX)],
        skip_header = False)

        ValDataset = data.TabularDataset(
        path=f'./Val_fold.csv',
        format='csv',
        fields=[('text', CONCAT), ('stance', LABEL), ("index", INDEX)],
        skip_header=False)

        TestDataset = data.TabularDataset(
        path = r".\fnc-1\dataset\test_concat_dataset.csv",
        format = 'csv',
        fields = [('text',CONCAT),('stance',LABEL),("index",INDEX)],
        skip_header = False)

        if not os.path.exists("./.vector_cache"):
            os.mkdir("./.vector_cache")
        vectors = Vectors(name=f'./.vector_cache/{filenames[run.word_embedding]}')
        CONCAT.build_vocab(TrainDataset, vectors=vectors)
        CONCAT.vocab.vectors.unk_init = init.xavier_uniform
        print(CONCAT.vocab.vectors.shape)
        Attention_LSTM = LSTM_With_Attention_Network(num_layers=run.num_layer,
                                                     dropout_rate=run.dropout_rate,
                                                     concat_embeddings=CONCAT.vocab.vectors,
                                                     hidden_size=100,
                                                     embedding_dim=embedding_dim[run.word_embedding],
                                                     attention_window=15,
                                                     num_class=4,
                                                     trainable_embeddings=run.trainable_embeddings).to(device)
        Attention_LSTM.apply(weigth_init)
        print(Attention_LSTM)
        train_iter = data.BucketIterator(TrainDataset,
                                    batch_size = run.batch_size,
                                    device = device,
                                    sort_within_batch = True,
                                    sort_key = lambda x : len(x.text))
        val_iter = data.BucketIterator(ValDataset, batch_size=run.batch_size,
                                    device = device,
                                    sort_within_batch = True,
                                    sort_key = lambda x : len(x.text))
        test_iter = data.BucketIterator(TestDataset, batch_size=run.batch_size,
                                  device = device,
                                  sort_within_batch = False,
                                  sort_key = lambda x : len(x.text))
        optimizer = optim.Adam(Attention_LSTM.parameters(),lr=run.lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=run.weight_decay)

        m.begin_run(save_path,run)
        for epoch in range(run.epoch_num):
            #-------Training process---#
            Attention_LSTM.train()
            m.begin_train_epoch()
            for i,batch in enumerate(train_iter):
                text = batch.text.to(device)
                # print(text.shape) [75,128]
                labels = batch.stance.to(device)
                preds = Attention_LSTM(text)
                loss = F.cross_entropy(preds,labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                m.track_loss(loss,batch)
                m.collect_data(preds, labels)
                m.track_num_instance(preds,labels)
            m.end_train_epoch()
            #-------Validation process-----#
            Attention_LSTM.eval()
            m.begin_val_epoch()
            for i,batch in enumerate(val_iter):
                text = batch.text.to(device)
                labels = batch.stance.to(device)
                preds = Attention_LSTM(text)
                loss = F.cross_entropy(preds,labels)

                m.track_loss(loss,batch)
                m.collect_data(preds, labels)
                m.track_num_instance(preds,labels)
            m.end_val_epoch()
        #-------Test process-----#
        Attention_LSTM.eval()
        with torch.no_grad():
            m.begin_test_epoch()
            for i,batch in enumerate(test_iter):
                text = batch.text.to(device)
                # print(text.shape) [75,128]
                labels = batch.stance.to(device)
                preds = Attention_LSTM(text)

                m.collect_data(preds,labels)
                m.track_num_instance(preds,labels)
            m.end_test_epoch()
        m.end_run()
        m.save('results')
        train,val,test = m.final_result()
        train_output.run(train)
        val_output.run(val)
        test_output.run(test)
    train_output.output("train")
    val_output.output("val")
    test_output.output("test")


