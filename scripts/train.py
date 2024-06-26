# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import os
import pandas as pd
from collections import OrderedDict
from torchtext.legacy import data
from sklearn.model_selection import StratifiedKFold
from models.lstm_attention import Attention_StackLSTM_With_Multi_FC, weigth_init
from utils.aggregate_result import aggregate_result
from utils.run_manager import Runbuilder, RunManager
from utils.dataset_setup import load_features, save_folds, setup_datasets, setup_embedding


def main(params, output_file='train partial',
         train_file='train_dataset_agree_and_disagree_concat_representation',
         test_file='test_dataset_agree_and_disagree_concat_representation',
         num_class=2):
    script_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(script_dir, '..', 'fnc-1', 'dataset')
    feature_path = os.path.join(script_dir, '..', 'fnc-1', 'features')
    result_path = os.path.join(script_dir, '..', 'result')
    vector_path = os.path.join(script_dir, '..', '.vector_cache')

    # Load features
    features = load_features(feature_path)
    # Define embedding dimensions and filenames
    embedding_dim = {
        "glove.twitter.27B.50d": 50, "glove.6B.300d": 300, "glove.6B.50d": 50,
        "glove.twitter.27B.25d": 25, "glove.6B.100d": 100, "glove.6B.200d": 200,
        "glove.twitter.27B.100d": 100, "glove.twitter.27B.200d": 200,
        "wiki_news_300d_sub": 300, "wiki_news_300d": 300, "crawl.300d.2M": 300,
        "GoogleNews.vector300": 300
    }

    filenames = {
        "wiki_news_300d_sub": "wiki-news-300d-1M-subword.vec",
        "glove.twitter.27B.50d": "glove.twitter.27B.50d.txt",
        "glove.6B.300d": "glove.6B.300d.txt", "glove.6B.50d": "glove.6B.50d.txt",
        "glove.twitter.27B.25d": "glove.twitter.27B.25d.txt",
        "glove.6B.100d": "glove.6B.100d.txt", "glove.6B.200d": "glove.6B.200d.txt",
        "glove.twitter.27B.100d": "glove.twitter.27B.100d.txt",
        "glove.twitter.27B.200d": "glove.twitter.27B.200d.txt",
        "crawl.300d.2M": "crawl-300d-2M.vec", "wiki_news_300d": "wiki-news-300d-1M.vec",
        "GoogleNews.vector300": "GoogleNews-vectors-negative300.txt"
    }

    # Generate hyperparameter combinations
    runs = Runbuilder().get_runs(params)
    # Initialize RunManager
    m = RunManager()
    # Load and prepare training data
    Train = pd.read_csv(f"{dataset_path}/{train_file}.csv", header=None)
    Train.columns = ["Text", "Stance", "Index"]
    X = Train["Text"].values
    y = Train["Stance"].values
    ind = Train["Index"].values
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for run in runs:
        fold_count = 0
        device = torch.device(run.device)
        path = f"{result_path}/{output_file}/output"
        if not os.path.exists(path):
            os.makedirs(path)
        train_output = aggregate_result(path, run, 5)
        val_output = aggregate_result(path, run, 5)
        test_output = aggregate_result(path, run, 5)
        for train_index, test_index in kf.split(X, y):
            fold_count += 1
            save_path = f"{result_path}/{output_file}/fold{fold_count}"
            checkpoint_path = f"{result_path}/{output_file}/fold{fold_count}/checkpoint"

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            Train = pd.DataFrame({"Text": X[train_index], "Stance": y[train_index], "Index": ind[train_index]})
            Val = pd.DataFrame({"Text": X[test_index], "Stance": y[test_index], "Index": ind[test_index]})
            save_folds(Train, Val, dataset_path, fold_count)

            # Prepare training and validation datasets
            # Setup CONCAT field and word embeddings
            LABEL = data.Field(sequential=False, use_vocab=False)
            CONCAT = data.Field(lower=True, fix_length=75)
            INDEX = data.Field(sequential=False, use_vocab=False)
            TrainDataset, ValDataset, TestDataset = setup_datasets(dataset_path, CONCAT, LABEL, INDEX, fold_count, testset_name=test_file)
            CONCAT = setup_embedding(CONCAT, run, TrainDataset, vector_path, filenames)

            Attention_StackLSTM = Attention_StackLSTM_With_Multi_FC(
                                                                 num_layers=run.num_layer,
                                                                 num_fc=run.num_fc,
                                                                 dropout_rate=run.dropout_rate,
                                                                 concat_embeddings=CONCAT.vocab.vectors,
                                                                 hidden_size=100,
                                                                 input_unit_num=features[run.features]['train'].shape[1],
                                                                 unit_num=600,
                                                                 embedding_dim=embedding_dim[run.word_embedding],
                                                                 attention_window=15,
                                                                 num_class=num_class,
                                                                 trainable_embeddings=run.trainable_embeddings,
                                                                 bidirectional=run.bidirectional).to(device)
            Attention_StackLSTM.apply(weigth_init)
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

            train_features_set = features[run.features]['train']
            test_features_set = features[run.features]["test"]
            optimizer2 = optim.Adam(Attention_StackLSTM.parameters(), lr=run.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=run.weight_decay)
            m.begin_run(save_path, run)
            for epoch in range(run.epoch_num):
                # -------Training process---#
                Attention_StackLSTM.train()
                m.begin_train_epoch()
                for i, batch in enumerate(train_iter):
                    text = batch.text.to(device)
                    # print(text.shape) [75,128]
                    labels = batch.stance.to(device)
                    if run.device == "cuda":
                        index = batch.index.cpu().numpy().tolist()
                    else:
                        index = batch.index.numpy().tolist()
                    input_features = train_features_set[index].to(device)
                    preds = Attention_StackLSTM(text, input_features)
                    loss = F.cross_entropy(preds, labels)

                    optimizer2.zero_grad()
                    loss.backward()
                    optimizer2.step()

                    m.track_loss(loss, batch)
                    m.collect_data(preds, labels)
                    m.track_num_instance(preds, labels)
                m.end_train_epoch()
                # -------Validation process-----#
                Attention_StackLSTM.eval()
                m.begin_val_epoch()
                for i, batch in enumerate(val_iter):
                    text = batch.text.to(device)
                    # print(text.shape) [75,128]
                    labels = batch.stance.to(device)
                    if run.device == "cuda":
                        index = batch.index.cpu().numpy().tolist()
                    else:
                        index = batch.index.numpy().tolist()
                    input_features = train_features_set[index].to(device)
                    preds = Attention_StackLSTM(text, input_features)
                    loss = F.cross_entropy(preds, labels)

                    m.track_loss(loss, batch)
                    m.collect_data(preds, labels)
                    m.track_num_instance(preds, labels)
                m.end_val_epoch()
            # -------Test process-----#
            Attention_StackLSTM.eval()
            with torch.no_grad():
                m.begin_test_epoch()
                for i, batch in enumerate(test_iter):
                    text = batch.text.to(device)
                    # print(text.shape) [75,128]
                    labels = batch.stance.to(device)
                    if run.device == "cuda":
                        index = batch.index.cpu().numpy().tolist()
                    else:
                        index = batch.index.numpy().tolist()
                    input_features = test_features_set[index].to(device)
                    preds = Attention_StackLSTM(text, input_features)

                    m.collect_data(preds, labels)
                    m.track_num_instance(preds, labels)
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

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Specify the dataset size for the operation.")
    # Add the --subset argument
    parser.add_argument('--subset', type=str, required=True, choices=['full', 'partial'],
                        help='Mode to run: full or partial. "full" processes the entire dataset, while "partial" processes a subset.')
    args = parser.parse_args()
    params = OrderedDict(
        epoch_num=[40],
        lr=[0.0005],
        batch_size=[128],
        num_layer=[3],
        dropout_rate=[0.8],
        max_length=[75],
        device=['cuda'],
        weight_decay=[0],
        trainset=['FNC-1'],
        valset=['FNC-1'],
        testset=['FNC-1'],
        features=["Baseline and Topic model features"],
        bidirectional=[True],
        word_embedding=["glove.6B.200d"],
        trainable_embeddings=[True],
        num_fc=[4])

    if args.subset == "partial":
        main(params=params)
    elif args.subset == 'full':
        main(params=params, output_file='train full',
             train_file='train_concat_dataset',
             test_file='test_concat_dataset',
             num_class=4)
