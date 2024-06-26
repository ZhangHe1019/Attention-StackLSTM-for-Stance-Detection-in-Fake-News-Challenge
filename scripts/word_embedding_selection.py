# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import init
import os
import pandas as pd
from torchtext.vocab import Vectors
from collections import OrderedDict
from torchtext.legacy import data
from sklearn.model_selection import StratifiedKFold
from models.lstm_attention import LSTM_With_Attention_Network, weigth_init
from utils.aggregate_result import aggregate_result
from utils.run_manager import Runbuilder, RunManager


if __name__ == "__main__":

    # Get the directory where this script is located
    script_dir = os.path.dirname(__file__)
    # Construct the path relative to the script's directory
    dataset_path = os.path.join(script_dir, '..', 'fnc-1', 'dataset')
    result_path = os.path.join(script_dir, '..', 'result')


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
               "glove.twitter.27B.25d":"glove.twitter.27B.25d.txt",
               "glove.6B.100d": "glove.6B.100d.txt",
               "glove.6B.200d": "glove.6B.200d.txt",
               "glove.twitter.27B.100d": "glove.twitter.27B.100d.txt",
               "glove.twitter.27B.200d":"glove.twitter.27B.200d.txt",
               "crawl.300d.2M":"crawl-300d-2M.vec",
               "wiki_news_300d":"wiki-news-300d-1M.vec",
               "GoogleNews.vector300":"GoogleNews-vectors-negative300.txt"}

    params = OrderedDict(epoch_num=[1],
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
                word_embedding=["glove.6B.200d"],
                trainable_embeddings=[True, False])

    runs = Runbuilder().get_runs(params)
    m = RunManager()
    Train = pd.read_csv(f"{dataset_path}/train_concat_dataset.csv", header=None)
    Train.columns = ["Text", "Stance", "Index"]
    X = Train["Text"].values
    y = Train["Stance"].values
    ind = Train["Index"].values
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for run in runs:
        fold_count = 0
        device = torch.device(run.device)
        path = f"{result_path}/word_embedding_selection/{run.word_embedding}_{run.trainable_embeddings}/output"
        if not os.path.exists(path):
            os.makedirs(path)
        train_output = aggregate_result(path,run,5)
        val_output = aggregate_result(path,run,5)
        test_output = aggregate_result(path,run,5)
        for train_index, test_index in kf.split(X, y):
            fold_count += 1
            save_path = f"{result_path}/word embedding selection/{run.word_embedding}_{run.trainable_embeddings}/fold{fold_count}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            Train = pd.DataFrame({"Text": X[train_index], "Stance": y[train_index], "Index": ind[train_index]})
            Val = pd.DataFrame({"Text": X[test_index], "Stance": y[test_index], "Index": ind[test_index]})
            Train.to_csv(f"{dataset_path}/Train_fold.csv", header=None,index=None)
            Val.to_csv(f"{dataset_path}/Val_fold.csv", header=None,index=None)


            LABEL = data.Field(sequential=False,use_vocab=False)
            CONCAT = data.Field(lower=True,fix_length=75)
            INDEX = data.Field(sequential=False,use_vocab=False)
            SEQLEN = data.Field(sequential=False,use_vocab=False)

            TrainDataset = data.TabularDataset(
            path=f'{dataset_path}/Train_fold.csv',
            format='csv',
            fields=[('text',CONCAT),('stance',LABEL),("index",INDEX)],
            skip_header = False)

            ValDataset = data.TabularDataset(
            path=f'{dataset_path}/Val_fold.csv',
            format='csv',
            fields=[('text', CONCAT), ('stance', LABEL), ("index", INDEX)],
            skip_header=False)

            TestDataset = data.TabularDataset(
            path=f"{dataset_path}/test_concat_dataset.csv",
            format='csv',
            fields=[('text',CONCAT),('stance',LABEL),("index",INDEX)],
            skip_header=False)

            if not os.path.exists(f"{script_dir}/../.vector_cache"):
                os.mkdir(f"{script_dir}/../.vector_cache")
            vectors = Vectors(name=f'{script_dir}/../.vector_cache/{filenames[run.word_embedding]}')
            CONCAT.build_vocab(TrainDataset, vectors=vectors)
            CONCAT.vocab.vectors.unk_init = init.xavier_uniform
            Attention_LSTM = LSTM_With_Attention_Network(num_layers=run.num_layer,
                                                     dropout_rate=run.dropout_rate,
                                                     concat_embeddings=CONCAT.vocab.vectors,
                                                     hidden_size=100,
                                                     embedding_dim=embedding_dim[run.word_embedding],
                                                     attention_window=15,
                                                     num_class=4,
                                                     trainable_embeddings=run.trainable_embeddings).to(device)
            Attention_LSTM.apply(weigth_init)
            train_iter = data.BucketIterator(TrainDataset,
                                    batch_size=run.batch_size,
                                    device=device,
                                    sort_within_batch=True,
                                    sort_key=lambda x: len(x.text))
            val_iter = data.BucketIterator(ValDataset, batch_size=run.batch_size,
                                    device=device,
                                    sort_within_batch=True,
                                    sort_key=lambda x: len(x.text))
            test_iter = data.BucketIterator(TestDataset, batch_size=run.batch_size,
                                  device=device,
                                  sort_within_batch=False,
                                  sort_key=lambda x: len(x.text))
            optimizer = optim.Adam(Attention_LSTM.parameters(),
                                   lr=run.lr,
                                   betas=(0.9, 0.999),
                                   eps=1e-08,
                                   weight_decay=run.weight_decay)

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
                for i, batch in enumerate(test_iter):
                    text = batch.text.to(device)
                    # print(text.shape) [75,128]
                    labels = batch.stance.to(device)
                    preds = Attention_LSTM(text)

                    m.collect_data(preds,labels)
                    m.track_num_instance(preds,labels)
                m.end_test_epoch()
            m.end_run()
            m.save('results')
            train, val, test = m.final_result()
            train_output.run(train)
            val_output.run(val)
            test_output.run(test)
        train_output.output("train")
        val_output.output("val")
        test_output.output("test")

