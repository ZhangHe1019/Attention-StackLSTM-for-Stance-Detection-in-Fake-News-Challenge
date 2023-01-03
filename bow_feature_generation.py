"""
This part of code is responsible for generating fnc training bag-of-word features and test bag-of-word features.
The content from line 13 to line 52 is cited from the following website:
https://github.com/UKPLab/coling2018_fake-news-challenge/blob/master/fnc/refs/feature_engineering.py
"""


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
import torch

def word_ngrams_concat_tf5000_l2_w_holdout_and_test(headlines,bodies):
    """
    Simple bag of words feature extraction
    """
    def combine_head_and_body(headlines, bodies):
        head_and_body = [headline + " " + body for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]
        return head_and_body

    def get_features(vocab):
        vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=True,
                                          norm="l2", stop_words='english')
        X_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=True,
                                          norm="l2", stop_words='english')
        X_body = vectorizer_body.fit_transform(bodies)

        X = np.concatenate([X_head.toarray(), X_body.toarray()], axis=1)

        return X

    h, b = get_head_body_tuples("fnc-1")
    h_test, b_test = get_head_body_tuples_test("fnc-1")

    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of heads: " + str(len(h)))
    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of bodies: " + str(len(b)))
    h.extend(h_test)
    b.extend(b_test)
    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of heads after ext: " + str(len(h)))
    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of bodies after ext: " + str(len(b)))

    tfidf = TfidfVectorizer(ngram_range=(1,1), stop_words='english', max_features=5000, use_idf=True,
                            norm='l2')
    tfidf.fit_transform(combine_head_and_body(h,b))
    vocab = tfidf.vocabulary_

    X = get_features(vocab)

    return X

def get_head_body_tuples_test(corpus_name):
    # file paths
    if "snli" in corpus_name:
      test_data_path = f"./{corpus_name}/dataset/test_dataset.csv"
      test = pd.read_csv(test_data_path,header=None)
      test.columns = ["Premise","Hypothesis",'Stance']
      p = test["Premise"].values.tolist()
      h = test["Hypothesis"].values.tolist()
      return p,h

    if "fnc" in corpus_name:
      test_data_path = f"./{corpus_name}/dataset/test_dataset.csv"
      test = pd.read_csv(test_data_path,header=None)
      test.columns = ["Body","Headline",'Stance']
      h = test["Headline"].values.tolist()
      b = test["Body"].values.tolist()
      return h,b

def get_head_body_tuples(corpus_name):
    # file paths
    if "snli" in corpus_name:
      train_data_path = f"./{corpus_name}/dataset/train_dataset.csv"
      val_data_path = f"./{corpus_name}/dataset/val_dataset.csv"
      train = pd.read_csv(train_data_path,header=None)
      val = pd.read_csv(val_data_path,header=None)
      train.columns = ["Premise","Hypothesis",'Stance']
      val.columns = ["Premise","Hypothesis",'Stance']
      p = val["Premise"].values.tolist()+train["Premise"].values.tolist()
      h = val["Hypothesis"].values.tolist()+train["Hypothesis"].values.tolist()
      return p,h
    if "fnc" in corpus_name:
      train_data_path = f"./{corpus_name}/dataset/train_dataset.csv"
      train = pd.read_csv(train_data_path,header=None)
      train.columns = ["Body","Headline",'Stance']
      h = train["Headline"].values.tolist()
      b = train["Body"].values.tolist()
      return h,b

import pandas as pd
def read_data(corpus_name,file_name):
    # Extracting data
    if corpus_name =="fnc-1":
        data = pd.read_csv(f"./{corpus_name}/dataset/{file_name}.csv",header=None)
        data.columns = ["Body","Headline","Stance"]
        stance = data["Stance"].values.tolist()
        headline = data["Headline"].values.tolist()
        body = data["Body"].values.tolist()
        stance = np.asarray(stance, dtype=np.int64)
        return headline,body,stance
    if corpus_name =="snli":
        data = pd.read_csv(f"./{corpus_name}/dataset/{file_name}.csv", header=None)
        data.columns = ["Premise", "Hypothesis", "Class"]
        stance = data["Class"].values.tolist()
        headline = data["Premise"].values.tolist()
        body = data["Hypothesis"].values.tolist()
        stance = np.asarray(stance, dtype=np.int64)
        return headline,body,stance

if __name__ == '__main__':
    mode = "train" #test
    corpus = "fnc-1"
    headlines,bodies,stance = read_data(corpus,f"{mode}_dataset")
    filepath = f"./{corpus}/features"
    bow1 = word_ngrams_concat_tf5000_l2_w_holdout_and_test(headlines, bodies)
    bow1 = torch.tensor(bow1)
    with open(f"{filepath}/{mode}_bow_features.pkl", "wb") as handle:
        pickle.dump(bow1, handle)
        handle.close()