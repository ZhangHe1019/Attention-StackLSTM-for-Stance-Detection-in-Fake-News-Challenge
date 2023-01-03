import warnings
warnings.filterwarnings('ignore')
import os
from collections import Counter
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import os
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
import pickle as cPickle


class Runbuilder():
    @staticmethod
    def get_runs(param):
        Run = namedtuple('Run', param.keys())
        runs = []
        for run in product(*param.values()):
            runs.append(Run(*run))
        return runs

def load_checkpoint(model,checkpoint_PATH):
    model_CKPT = torch.load(checkpoint_PATH)
    model.load_state_dict(model_CKPT['state_dict'],strict=False)
    print('loading checkpoint!')
    return model

class Prediction_Process():
    def __init__(self,path,device):
        self.total_pred = list()
        self.device = device
        print(self.device)
        self.data = []
        self.path = path
        self.voting_label = None

    def begin_test(self):
        self.true_label = []
        self.pred_label = []

    def end_test(self):
        self.total_pred += [self.pred_label]

    def collect_data(self, preds, labels):
        if self.device == 'cuda':
            prediction = preds.argmax(dim=1).cuda().cpu().numpy().tolist()
            self.pred_label.extend(prediction)
            labels = labels.cuda().cpu().numpy().tolist()
            self.true_label.extend(labels)
        else:
            prediction = preds.argmax(dim=1).cpu().numpy().tolist()
            self.pred_label.extend(prediction)
            labels = labels.cpu().numpy().tolist()
            self.true_label.extend(labels)

    def voting_process(self):
        self.total_pred = np.array(self.total_pred)
        print(self.total_pred)
        self.total_pred = self.total_pred.T
        self.voting_label = np.zeros(len(self.total_pred))
        for i in range(len(self.total_pred)):
            predictions = self.total_pred[i].tolist()
            dict1 = Counter(predictions)
            self.voting_label[i] = dict1.most_common(1)[0][0]

    def metrics_statistic(self):
        results = OrderedDict()
        p_class, r_class, f_class, support_micro = metrics.precision_recall_fscore_support(y_true=self.true_label,
                                                                                           y_pred=self.voting_label,
                                                                                           labels=[0, 1],
                                                                                           average=None)
        f1_agree = f_class[0]
        f1_disagree = f_class[1]
        F1_macro = metrics.f1_score(self.true_label, self.voting_label, labels=[0, 1], average='macro')
        accuracy = metrics.accuracy_score(self.true_label, self.voting_label)
        cm = metrics.confusion_matrix(self.true_label, self.voting_label)

        results['Accuracy'] = accuracy
        results['Agree F1'] = f1_agree
        results['Disagree F1'] = f1_disagree
        results['Macro F1'] = F1_macro
        results['Confusion matrix'] = cm
        self.data.append(results)
        pd.DataFrame.from_dict(self.data, orient='columns').to_csv(f'{self.path}/voting_result.csv')

class LSTM_With_Attention_Network(nn.Module):
    def __init__(self, num_layers, dropout_rate, concat_embeddings, hidden_size, input_unit_num, unit_num,
                 embedding_dim, num_class, trainable_embeddings,bidirectional):
        super(LSTM_With_Attention_Network, self).__init__()  # 调用父类的构造方法
        self.num_layers = num_layers
        self.attention_window = 15
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

class StackLSTM(nn.Module):
    def __init__(self, num_layers, dropout_rate, concat_embeddings, hidden_size, input_unit_num, unit_num,
                 embedding_dim, num_class, trainable_embeddings,bidirectional):
        super(StackLSTM, self).__init__()  # 调用父类的构造方法
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.concat_embedding = nn.Embedding(len(concat_embeddings),embedding_dim)  # vocab_size词汇表大小,embedding_dim词嵌入维度
        self.concat_embedding.weight.data.copy_(concat_embeddings)  # 第一句就是导入词向量
        self.concat_embedding.weight.requires_grad = trainable_embeddings

        if self.bidirectional == True:
            self.BiLstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                                  bidirectional=True, dropout=dropout_rate)
            self.LinearLayer1 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
            self.LinearLayer2 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
            self.LinearLayer3 = nn.Linear(2 * hidden_size, 1, bias=False)
            self.LinearLayer4 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
            self.LinearLayer5 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
            self.fc1 = nn.Linear(input_unit_num + 2*hidden_size, unit_num)
            self.fc2 = nn.Linear(unit_num, unit_num)
            self.fc3 = nn.Linear(unit_num, unit_num)
            self.fc4 = nn.Linear(unit_num, num_class)


        else:
            self.BiLstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                                  bidirectional=False, dropout=dropout_rate)
            self.LinearLayer1 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.LinearLayer2 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.LinearLayer3 = nn.Linear(hidden_size, 1, bias=False)
            self.LinearLayer4 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.LinearLayer5 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.fc1 = nn.Linear(input_unit_num + hidden_size, unit_num)
            self.fc2 = nn.Linear(unit_num, unit_num)
            self.fc3 = nn.Linear(unit_num, unit_num)
            self.fc4 = nn.Linear(unit_num, num_class)


    def forward(self,text,features):
        concat_embedding = self.concat_embedding(text)
        # torch.Size([sentence_lens,batch_size,embedding_size])
        # torch.Size([75, 32, 50])
        output, (hidden, cell) = self.BiLstm(concat_embedding)
        # output shape: torch.Size([sentence_lens,batch_size,hidden_size*2])
        # torch.Size([75,32,200])
        # hidden shape: torch.Size([layer_num*2,batch_size,hidden_size])
        # torch.Size([4,32,100])
        # Forward output
        # Bidirectional:torch.Size([15,32,100])
        if self.bidirectional == True:
            final_state = torch.cat((hidden[-2], hidden[-1]), 1)
        else:
            final_state = hidden[-1]
        concat_features = torch.cat([final_state, features], 1)
        o1 = F.tanh(self.fc1(concat_features))
        o2 = F.tanh(self.fc2(o1))
        o3 = F.tanh(self.fc3(o2))
        output = self.fc4(o3)
        return output

##########################Loading Features############################
corpus_name = "fnc-1"
path = f'./{corpus_name}/features'

with open(f"{path}/test_topic_features.pkl", "rb") as test_feature1:
    test_topic_features = cPickle.load(test_feature1,encoding='latin1')
    test_topic_features = torch.tensor(test_topic_features).float()

with open(f"{path}/test_bow_features.pkl", "rb") as test_feature2:
    test_bow_feature = cPickle.load(test_feature2,encoding='latin1')
    test_bow_feature = torch.tensor(test_bow_feature).float()

with open(f"{path}/test_baseline_features.pkl", "rb") as test_feature3:
    test_baseline_features = cPickle.load(test_feature3,encoding='latin1')
    test_baseline_features = torch.tensor(test_baseline_features).float()

features={"4 Topic model features": {"test": test_topic_features},
          "1 BOW feature":{"test": test_bow_feature},
          "4 Baseline features":{"test": test_baseline_features},
          "Baseline and BOW features":{"test": torch.cat([test_baseline_features,test_bow_feature],1)},
          "Baseline and Topic model features":{"test": torch.cat([test_baseline_features,test_topic_features],1)},
          "BOW and Topic model features":{"test": torch.cat([test_bow_feature,test_topic_features],1)},
          "All":{"test": torch.cat([test_baseline_features,test_bow_feature,test_topic_features],1)}
          }
embedding_dim = {"glove.twitter.27B.50d":50,
                 "glove.6B.300d":300,
                 "glove.6B.50d":50,
                 "glove.twitter.27B.25d":25,
                 "charngram.100d":100,
                 "fasttext.en.300d":300,
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
           "charngram.100d":"charNgram.txt",
           "fasttext.en.300d": "wiki.en.vec",
           "glove.6B.100d": "glove.6B.100d.txt",
           "glove.6B.200d": "glove.6B.200d.txt",
           "glove.twitter.27B.100d": "glove.twitter.27B.100d.txt",
           "glove.twitter.27B.200d":"glove.twitter.27B.200d.txt",
           "crawl.300d.2M":"crawl-300d-2M.vec",
           "wiki_news_300d":"wiki-news-300d-1M.vec",
           "GoogleNews.vector300":"GoogleNews-vectors-negative300.txt"}
params = OrderedDict(
            epoch_num=[40],#50
            lr=[0.0005],
            batch_size=[128],
            num_layer=[2],
            dropout_rate=[0.8],#0.2,0.4,0.6
            max_length=[75],
            device=['cuda'],
            weight_decay=[0],
            trainset=['FNC-1'],
            valset=['FNC-1'],
            testset=['FNC-1'],
            features=[#"4 Topic model features",
                      #"1 BOW feature",
                      #"4 Baseline features",
                      #"Baseline and BOW features",
                      "Baseline and Topic model features",
                      #"BOW and Topic model features",
                      #"All"
                      ],
            bidirectional=[True],
            word_embedding=[#"glove.6B.50d",
                            #"glove.6B.100d",
                            #"glove.6B.300d",
                            "glove.6B.200d",
                            #"glove.twitter.27B.25d",
                            #"glove.twitter.27B.50d",
                            #"glove.twitter.27B.100d",
                            #"fasttext.en.300d",
                            #"wiki.news.300d",
                            #"crawl.300d.2M",
                            #"GoogleNews.vector300"
                            ],
            trainable_embeddings=[True])

runs = Runbuilder().get_runs(params)
Train = pd.read_csv(r".\fnc-1\dataset\train_dataset_agree_and_disagree_concat_representation.csv", header=None)
Train.columns = ["Text", "Stance", "Index"]
X = Train["Text"].values
y = Train["Stance"].values
ind = Train["Index"].values
kf = StratifiedKFold(n_splits=5, shuffle=False)
mode_name = "StackLSTM"
for run in runs:
    fold_count = 0
    device = torch.device(run.device)
    path = f"./{mode_name}/Partial FNC corpus/Model Training/parameters"
    if not os.path.exists(path):
        os.makedirs(path)
    m = Prediction_Process(path=f"{path}",device=device)
    for train_index, test_index in kf.split(X, y):
        fold_count += 1
        checkpoint_path = f"./{mode_name}/Partial FNC corpus/Model Training/parameters/fold{fold_count}/checkpoint/fold{fold_count}.pth.tar"

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
        path=r".\fnc-1\dataset\test_dataset_agree_and_disagree_concat_representation.csv",
        format='csv',
        fields=[('text',CONCAT),('stance',LABEL),("index",INDEX)],
        skip_header=False)

        if not os.path.exists("./.vector_cache"):
            os.mkdir("./.vector_cache")
        vectors = Vectors(name=f'./.vector_cache/{filenames[run.word_embedding]}')
        CONCAT.build_vocab(TrainDataset, vectors=vectors)
        CONCAT.vocab.vectors.unk_init = init.xavier_uniform

        if mode_name == "StackLSTM":
            Attention_LSTM = StackLSTM(num_layers=run.num_layer,
                                                     dropout_rate=run.dropout_rate,
                                                     concat_embeddings=CONCAT.vocab.vectors,
                                                     hidden_size=100,
                                                     input_unit_num=features[run.features]['test'].shape[1],
                                                     unit_num=600,
                                                     embedding_dim=embedding_dim[run.word_embedding],
                                                     num_class=2,
                                                     trainable_embeddings=run.trainable_embeddings,
                                                     bidirectional=run.bidirectional).to(device)

            Attention_LSTM = load_checkpoint(Attention_LSTM, checkpoint_path)
            print(Attention_LSTM)
            test_iter = data.BucketIterator(TestDataset, batch_size=run.batch_size,
                                      device = device,
                                      sort_within_batch = False,
                                      sort_key = lambda x : len(x.text))
            test_features_set = features[run.features]["test"]
            m.begin_test()
            for i, batch in enumerate(test_iter):
                text = batch.text.to(device)
                labels = batch.stance.to(device)
                if run.device == "cuda":
                    index = batch.index.cpu().numpy().tolist()
                else:
                    index = batch.index.numpy().tolist()
                input_features = test_features_set[index].to(device)
                preds = Attention_LSTM(text, input_features)
                m.collect_data(preds, labels)
            m.end_test()
        else:
            Attention_LSTM = LSTM_With_Attention_Network(num_layers=run.num_layer,
                                                         dropout_rate=run.dropout_rate,
                                                         concat_embeddings=CONCAT.vocab.vectors,
                                                         hidden_size=100,
                                                         input_unit_num=features[run.features]['test'].shape[1],
                                                         unit_num=600,
                                                         embedding_dim=embedding_dim[run.word_embedding],
                                                         num_class=2,
                                                         trainable_embeddings=run.trainable_embeddings,
                                                         bidirectional=run.bidirectional).to(device)

            Attention_LSTM = load_checkpoint(Attention_LSTM, checkpoint_path)
            print(Attention_LSTM)
            test_iter = data.BucketIterator(TestDataset, batch_size=run.batch_size,
                                      device = device,
                                      sort_within_batch = False,
                                      sort_key = lambda x : len(x.text))
            m.begin_test()
            for i, batch in enumerate(test_iter):
                text = batch.text.to(device)
                labels = batch.stance.to(device)
                if run.device == "cuda":
                    index = batch.index.cpu().numpy().tolist()
                else:
                    index = batch.index.numpy().tolist()
                preds = Attention_LSTM(text)
                m.collect_data(preds, labels)
            m.end_test()
    m.voting_process()
    m.metrics_statistic()