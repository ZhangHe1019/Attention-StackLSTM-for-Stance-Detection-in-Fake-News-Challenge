# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import os
import pandas as pd
from collections import OrderedDict
from torchtext.legacy import data
from sklearn.model_selection import StratifiedKFold
from models.lstm_attention import Attention_StackLSTM_With_Multi_FC, weigth_init
from utils.run_manager import Runbuilder, RunManager
from utils.dataset_setup import load_features, save_folds, setup_datasets, setup_embedding
import argparse

def load_checkpoint(model,checkpoint_PATH):
    model_CKPT = torch.load(checkpoint_PATH)
    model.load_state_dict(model_CKPT['state_dict'])
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
            if dict1.most_common(1)[0][1] == 1:
                self.voting_label[i] = self.total_pred[i][2]
            else:
                self.voting_label[i] = dict1.most_common(1)[0][0]

    def metrics_statistic(self):
        results = OrderedDict()
        p_class, r_class, f_class, support_micro = metrics.precision_recall_fscore_support(y_true=self.true_label,
                                                                                           y_pred=self.voting_label,
                                                                                           labels=[0, 1, 2, 3],
                                                                                           average=None)
        f1_agree = f_class[0]
        f1_disagree = f_class[1]
        f1_discuss = f_class[2]
        f1_unrelated = f_class[3]
        F1_macro = metrics.f1_score(self.true_label, self.voting_label, labels=[0, 1, 2, 3], average='macro')
        accuracy = metrics.accuracy_score(self.true_label, self.voting_label)
        cm = metrics.confusion_matrix(self.true_label, self.voting_label)
        score, _ = FNC_scorer.score_submission(self.true_label, self.voting_label)
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
        self.data.append(results)
        pd.DataFrame.from_dict(self.data, orient='columns').to_csv(f'{self.path}/voting_result.csv')


def main(params, output_file='test partial',
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
        m = Prediction_Process(path=f"{path}", device=device)
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
            TrainDataset, ValDataset, TestDataset = setup_datasets(dataset_path, CONCAT, LABEL, INDEX,
                                                                   fold_count, testset_name=test_file)
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
            Attention_StackLSTM = load_checkpoint(Attention_StackLSTM, checkpoint_path)
            test_iter = data.BucketIterator(TestDataset, batch_size=run.batch_size,
                                      device=device,
                                      sort_within_batch=False,
                                      sort_key=lambda x: len(x.text))
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
                preds = Attention_StackLSTM(text, input_features)
                m.collect_data(preds, labels)
            m.end_test()
        m.voting_process()
        m.metrics_statistic()


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
        main(params=params, output_file='test full',
             train_file='train_concat_dataset',
             test_file='test_concat_dataset',
             num_class=4)