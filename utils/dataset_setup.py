# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from torch.nn import init
import os
import pandas as pd
from torchtext.vocab import Vectors
from torchtext.legacy import data
import pickle as cPickle
import torch

def load_features(feature_path):
    """
    Load precomputed features from pickle files.
    Args:
    - feature_path (str): Path to the directory containing feature files.
    Returns:
    - dict: Dictionary containing loaded features.
    """
    train_topic_features = torch.tensor(
        cPickle.load(open(f"{feature_path}/train_topic_features.pkl", "rb"), encoding='latin1')).float()
    train_bow_feature = torch.tensor(
        cPickle.load(open(f"{feature_path}/train_bow_features.pkl", "rb"), encoding='latin1')).float()
    train_baseline_features = torch.tensor(
        cPickle.load(open(f"{feature_path}/train_baseline_features.pkl", "rb"), encoding='latin1')).float()
    test_topic_features = torch.tensor(
        cPickle.load(open(f"{feature_path}/test_topic_features.pkl", "rb"), encoding='latin1')).float()
    test_bow_feature = torch.tensor(
        cPickle.load(open(f"{feature_path}/test_bow_features.pkl", "rb"), encoding='latin1')).float()
    test_baseline_features = torch.tensor(
        cPickle.load(open(f"{feature_path}/test_baseline_features.pkl", "rb"), encoding='latin1')).float()
    features = {
        "4 Topic model features": {"train": train_topic_features, "test": test_topic_features},
        "1 BOW feature": {"train": train_bow_feature, "test": test_bow_feature},
        "4 Baseline features": {"train": train_baseline_features, "test": test_baseline_features},
        "Baseline and BOW features": {"train": torch.cat([train_baseline_features, train_bow_feature], 1),
                                      "test": torch.cat([test_baseline_features, test_bow_feature], 1)},
        "Baseline and Topic model features": {"train": torch.cat([train_baseline_features, train_topic_features], 1),
                                              "test": torch.cat([test_baseline_features, test_topic_features], 1)},
        "BOW and Topic model features": {"train": torch.cat([train_bow_feature, train_topic_features], 1),
                                         "test": torch.cat([test_bow_feature, test_topic_features], 1)},
        "All": {"train": torch.cat([train_baseline_features, train_bow_feature, train_topic_features], 1),
                "test": torch.cat([test_baseline_features, test_bow_feature, test_topic_features], 1)}
    }
    return features


def load_datasets(dataset_path, fold_count):
    """
    Load train and validation datasets from CSV files.

    Args:
    - dataset_path (str): Path to the dataset directory.
    - fold_count (int): Fold index for the dataset split.

    Returns:
    - pd.DataFrame: DataFrame containing the loaded training data.
    - pd.DataFrame: DataFrame containing the loaded validation data.
    """
    Train = pd.read_csv(f"{dataset_path}/Train_fold_{fold_count}.csv", header=None)
    Train.columns = ["Text", "Stance", "Index"]
    Val = pd.read_csv(f"{dataset_path}/Val_fold_{fold_count}.csv", header=None)
    Val.columns = ["Text", "Stance", "Index"]

    X_train, y_train, ind_train = Train["Text"].values, Train["Stance"].values, Train["Index"].values
    X_val, y_val, ind_val = Val["Text"].values, Val["Stance"].values, Val["Index"].values

    return X_train, y_train, ind_train, X_val, y_val, ind_val


def save_folds(Train, Val, dataset_path, fold_count):
    Train.to_csv(f"{dataset_path}/Train_fold_{fold_count}.csv", header=None,index=None)
    Val.to_csv(f"{dataset_path}/Val_fold_{fold_count}.csv", header=None,index=None)

def setup_datasets(dataset_path, CONCAT, LABEL, INDEX, fold_count, testset_name='test_concat_dataset'):
    """
    Setup training, validation, and test datasets using torchtext TabularDataset.

    Args:
    - dataset_path (str): Path to the dataset directory.
    - feature_path (str): Path to the directory containing feature files.
    - fold_count (int): Fold index for the dataset split.

    Returns:
    - data.TabularDataset: Training dataset.
    - data.TabularDataset: Validation dataset.
    - data.TabularDataset: Test dataset.
    """
    TrainDataset = data.TabularDataset(
        path=f'{dataset_path}/Train_fold_{fold_count}.csv',
        format='csv',
        fields=[('text', CONCAT),
                ('stance', LABEL),
                ("index", INDEX)],
        skip_header=False
    )

    ValDataset = data.TabularDataset(
        path=f'{dataset_path}/Val_fold_{fold_count}.csv',
        format='csv',
        fields=[('text', CONCAT),
                ('stance', LABEL),
                ("index", INDEX)],
        skip_header=False
    )

    TestDataset = data.TabularDataset(
        path=f'{dataset_path}/{testset_name}.csv',
        format='csv',
        fields=[('text', CONCAT),
                ('stance', LABEL),
                ("index", INDEX)],
        skip_header=False
    )

    return TrainDataset, ValDataset, TestDataset


def setup_embedding(CONCAT, run, TrainDataset, vector_path, embedding_filenames):
    """
    Setup word embeddings for torchtext field.
    Args:
    - CONCAT (data.Field): Field for concatenating word embeddings.
    - run (object): Current run configuration object.
    - vector_path (str): Path to the directory containing vector cache.
    - filenames (dict): Dictionary mapping word embeddings to their corresponding filenames.
    """
    if not os.path.exists(vector_path):
        os.mkdir(vector_path)

    vectors = Vectors(name=f'{vector_path}/{embedding_filenames[run.word_embedding]}')
    CONCAT.build_vocab(TrainDataset, vectors=vectors)
    CONCAT.vocab.vectors.unk_init = init.xavier_uniform
    return CONCAT