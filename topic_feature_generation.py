import pandas as pd
import numpy as np
import joblib
from nltk.corpus import stopwords
from sklearn import feature_extraction
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
from sklearn.decomposition import LatentDirichletAllocation, NMF
import string
import nltk
import time
import pickle
path = r"F:/Individual_Project"

def get_head_body_tuples_test(corpus_name):
    # file paths
    if "snli" in corpus_name:
      test_data_path = f"{path}/{corpus_name}/dataset/test_dataset.csv"
      test = pd.read_csv(test_data_path,header=None)
      test.columns = ["Premise","Hypothesis",'Stance']
      p = test["Premise"].values.tolist()
      h = test["Hypothesis"].values.tolist()
      return p,h

    if "fnc" in corpus_name:
      test_data_path = f"{path}/{corpus_name}/dataset/test_dataset.csv"
      test = pd.read_csv(test_data_path,header=None)
      test.columns = ["Body","Headline",'Stance']
      h = test["Headline"].values.tolist()
      b = test["Body"].values.tolist()
      return h,b

def get_head_body_tuples(corpus_name):
    # file paths
    if "snli" in corpus_name:
      train_data_path = f"{path}/{corpus_name}/dataset/train_dataset.csv"
      val_data_path = f"{path}/{corpus_name}/dataset/val_dataset.csv"
      train = pd.read_csv(train_data_path,header=None)
      val = pd.read_csv(val_data_path,header=None)
      train.columns = ["Premise","Hypothesis",'Stance']
      val.columns = ["Premise","Hypothesis",'Stance']
      p = val["Premise"].values.tolist()+train["Premise"].values.tolist()
      h = val["Hypothesis"].values.tolist()+train["Hypothesis"].values.tolist()
      return p,h
    if "fnc" in corpus_name:
      train_data_path = f"{path}/{corpus_name}/dataset/train_dataset.csv"
      train = pd.read_csv(train_data_path,header=None)
      train.columns = ["Body","Headline",'Stance']
      h = train["Headline"].values.tolist()
      b = train["Body"].values.tolist()
      return h,b

def NMF_fit_all_incl_holdout_and_test(headlines,bodies,corpus_name):
    #http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-topics-extraction-with-nmf-lda-py
    # https://pypi.python.org/pypi/lda on bottom see suggestions like MALLET, hca
    # https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
    # https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text

    print("WARNING: IF SIZE OF HEAD AND BODY DO NOT MATCH, "
          "RUN THIS FEATURE EXTRACTION METHOD SEPERATELY (WITHOUT ANY OTHER FE METHODS) TO CREATE THE FEATURES ONCE!")

    def combine_head_and_body(headlines, bodies):
        head_and_body = [str(headline) + " " + str(body) for i, (headline, body) in enumerate(zip(headlines, bodies))]
        return head_and_body

    def get_all_data(head_and_body):
        features_dir = f"{path}/{corpus_name}"
        filename = "NMF_fit_all_incl_holdout_and_test"
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1,1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            vocab = vectorizer_all.vocabulary_
            print("NMF_fit_all_incl_holdout_and_test: complete vocabulary length=" + str(len(list(vocab.keys()))))
            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return X_all, vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                vocab = pickle.load(handle)
            vectorizer_all = TfidfVectorizer(vocabulary=vocab, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            return X_all, vectorizer_all.vocabulary_

    def get_vocab(head_and_body):
        features_dir = f"{path}/{corpus_name}"
        filename = "NMF_fit_all_incl_holdout_and_test"
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            vocab = vectorizer_all.vocabulary_
            print("NMF_fit_all_incl_holdout_and_test: complete vocabulary length=" + str(len(X_all[0])))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                return pickle.load(handle)


    def get_features(head_and_body):
        features_dir = f"{path}/{corpus_name}"
        filename = "NMF_fit_all_incl_holdout_and_test"
        if not (os.path.exists(features_dir + "/" + filename + ".pkl")):
            X_all, vocab = get_all_data(head_and_body)

            # calculates n most important topics of the bodies. Each topic contains all words but ordered by importance. The
            # more important topic words a body contains of a certain topic, the higher its value for this topic
            nfm = NMF(n_components=300, random_state=1, alpha=.1)

            print("NMF_fit_all_incl_holdout_and_test: fit and transform body")
            t0 = time.time()
            nfm.fit_transform(X_all)
            print("done in %0.3fs." % (time.time() - t0))

            with open(features_dir + "/" + filename + ".pkl", 'wb') as handle:
                joblib.dump(nfm, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            vocab = get_vocab(head_and_body)
            with open(features_dir + "/" + filename + ".pkl", 'rb') as handle:
                nfm = joblib.load(handle)


        vectorizer_head = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_body = vectorizer_body.fit_transform(bodies)

        print("NMF_fit_all_incl_holdout_and_test: transform head and body")
        # use the lda trained for body topcis on the headlines => if the headlines and bodies share topics
        # their vectors should be similar
        nfm_head_matrix = nfm.transform(X_train_head)
        nfm_body_matrix = nfm.transform(X_train_body)

        print('NMF_fit_all_incl_holdout_and_test: calculating cosine distance between head and body')
        # calculate cosine distance between the body and head
        X = []
        for i in range(len(nfm_head_matrix)):
            X_head_vector = np.array(nfm_head_matrix[i]).reshape((1, -1)) #1d array is deprecated
            X_body_vector = np.array(nfm_body_matrix[i]).reshape((1, -1))
            cos_dist = cosine_distances(X_head_vector, X_body_vector).flatten()
            X.append(cos_dist.tolist())
        print(len(nfm_head_matrix))
        return X

    h, b = get_head_body_tuples(corpus_name)
    h_test, b_test = get_head_body_tuples_test(corpus_name)
    h.extend(h_test)
    b.extend(b_test)
    head_and_body = combine_head_and_body(h,b)
    X = get_features(head_and_body)

    return X

def NMF_fit_all_concat_300_and_test(headlines,bodies,corpus_name):
    #http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-topics-extraction-with-nmf-lda-py
    # https://pypi.python.org/pypi/lda on bottom see suggestions like MALLET, hca
    # https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
    # https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text
    def combine_head_and_body(headlines, bodies):
        head_and_body = [str(headline) + " " + str(body) for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]

        return head_and_body

    def get_all_data(head_and_body):
        features_dir = f"{path}/{corpus_name}"
        filename = "NMF_fit_all_concat_300_and_test"
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1,1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            print("X_all_length (w Holdout round 50k): " + str(len(head_and_body)))
            vocab = vectorizer_all.vocabulary_
            print("NMF_fit_all_concat_300_and_test: complete vocabulary length=" + str(len(list(vocab.keys()))))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return X_all, vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                vocab = pickle.load(handle)
            vectorizer_all = TfidfVectorizer(vocabulary=vocab, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            return X_all, vectorizer_all.vocabulary_

    def get_vocab(head_and_body):
        features_dir = f"{path}/{corpus_name}"
        filename = "NMF_fit_all_concat_300_and_test"
        if not (os.path.exists(features_dir + "/" + filename + ".vocab")):
            vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', use_idf=True, norm='l2')
            X_all = vectorizer_all.fit_transform(head_and_body)
            vocab = vectorizer_all.vocabulary_
            print("NMF_fit_all_concat_300_and_test: complete vocabulary length=" + str(len(X_all[0])))

            with open(features_dir + "/" + filename + ".vocab", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return vocab
        else:
            with open(features_dir + "/" + filename + ".vocab", 'rb') as handle:
                return pickle.load(handle)


    def get_features(head_and_body):
        features_dir = f"{path}/{corpus_name}"
        filename = "NMF_fit_all_concat_300_and_test"
        if not (os.path.exists(features_dir + "/" + filename + ".pkl")):
            X_all, vocab = get_all_data(head_and_body)

            # calculates n most important topics of the bodies. Each topic contains all words but ordered by importance. The
            # more important topic words a body contains of a certain topic, the higher its value for this topic
            nfm = NMF(n_components=300, random_state=1, alpha=.1)

            print("NMF_fit_all_concat_300_and_test: fit NMF to all data")
            t0 = time.time()
            nfm.fit_transform(X_all)
            print("done in %0.3fs." % (time.time() - t0))

            with open(features_dir + "/" + filename + ".pkl", 'wb') as handle:
                joblib.dump(nfm, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            vocab = get_vocab(head_and_body)
            with open(features_dir + "/" + filename + ".pkl", 'rb') as handle:
                nfm = joblib.load(handle)


        vectorizer_head = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, norm='l2')
        X_train_body = vectorizer_body.fit_transform(bodies)

        print("NMF_fit_all_concat_300_and_test: transform head and body")
        # use the lda trained for body topcis on the headlines => if the headlines and bodies share topics
        # their vectors should be similar
        nfm_head_matrix = nfm.transform(X_train_head)
        nfm_body_matrix = nfm.transform(X_train_body)

        print('NMF_fit_all_concat_300_and_test: concat head and body')
        # calculate cosine distance between the body and head
        return np.concatenate([nfm_head_matrix, nfm_body_matrix], axis=1)

    h, b = get_head_body_tuples(corpus_name)
    h_test, b_test = get_head_body_tuples_test(corpus_name)
    h.extend(h_test)
    b.extend(b_test)
    head_and_body = combine_head_and_body(h, b)

    X = get_features(head_and_body)

    return X

def latent_semantic_indexing_gensim_holdout_and_test(headlines,bodies,corpus_name):
    """
    Takes all the data (holdout+test+train) and interpretes the headlines and bodies as different
    documents. Instead of combining them, they are appended. Then it tokenizes these ~50k headline-docs and ~50k body-docs,
    builds a Tfidf-Matrix out of them and creates a LSI-Model out of it. In the next step the headlines and
    bodies for the feature generation are also treated as different documents and merely appended. Also, they are tokenized and
    a Tfifd-Matrix is built. This matix is passed to the learned LSI-Model and a Matrix is being returned.
    In this matrix, each document is represented as a vector with length(topics) of (topic-id, distance of this doc to the topic).
    The probabilities are then taken as a feature vector for the document. The first half of the matrix represent the headline docs,
    the latter half represent the body docs. In the end, the feature vectors of the headlines get concatenated with its body feature vector.
    The differences to the latent_semantic_indexing_gensim are:
        - holdout data is also used
        - a Tfidf matrix is built and used to create the LSI model and also to retrieve the features instead of just a corpus to build the LSI model and
            passing each headline and body separately into the LSI model to retrieve its features (does it make a difference, since dictionary already takes
            tfidf into account?)
        - the vectors are taken fully and not just the cosinus distance between them
    """
    from gensim import corpora, models

    def combine_and_tokenize_head_and_body(headlines, bodies, file_path=None):
        all_text = []
        all_text.extend(headlines)
        all_text.extend(bodies)
        if file_path != None and (os.path.exists(file_path)):
            with open(file_path, 'rb') as handle:
                return pickle.load(handle)

        print("head+body appended size should be around 100k and 19/8k: " + str(len(bodies)))
        head_and_body_tokens = [nltk.word_tokenize(line) for line in all_text]

        if file_path != None:
            with open(file_path, 'wb') as handle:
                pickle.dump(head_and_body_tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return head_and_body_tokens

    def get_features(n_topics):
        features_dir = f"{path}/{corpus_name}"
        filename = "lsi_gensim_test_" + str(n_topics) + "topics_and_test"

        h, b = get_head_body_tuples(corpus_name)
        h_test, b_test = get_head_body_tuples_test(corpus_name)
        h.extend(h_test)
        b.extend(b_test)
        head_and_body = combine_and_tokenize_head_and_body(h,b,file_path=features_dir + "/" + "lsi_gensim_h_b_tokenized_and_test" + ".pkl")

        if (os.path.exists(features_dir + "/" + "lsi_gensim_holdout_and_test" + ".dict")):
            print("dict found and load")
            dictionary = corpora.Dictionary.load(features_dir + "/" + "lsi_gensim_all_and_test" + ".dict")
        else:
            print("create new dict")
            dictionary = corpora.Dictionary(head_and_body)
            dictionary.save(features_dir + "/" + "lsi_gensim_all_and_test" + ".dict")

        if (os.path.exists(features_dir + "/" + filename + ".lsi")):
            print("found lsi model")
            lsi = models.LsiModel.load(features_dir + "/" + filename + ".lsi")
        else:
            print("build corpus and tfidf corpus")
            corpus = [dictionary.doc2bow(text) for text in head_and_body]
            tfidf = models.TfidfModel(corpus)
            corpus_tfidf = tfidf[corpus]

            print("create new lsi model")
            lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics)
            lsi.save(features_dir + "/" + filename + ".lsi")

        # get tfidf corpus of head and body
        corpus_train = [dictionary.doc2bow(text) for text in combine_and_tokenize_head_and_body(headlines,bodies)]
        tfidf_train = models.TfidfModel(corpus_train)
        corpus_train_tfidf = tfidf_train[corpus_train]

        corpus_lsi = lsi[corpus_train_tfidf]

        X_head = []
        X_body = []
        i = 0
        for doc in corpus_lsi:
            if i < int(len(corpus_lsi) / 2):
                X_head_vector_filled = np.zeros(n_topics, dtype=np.float64)
                for id, prob in doc:
                    X_head_vector_filled[id] = prob
                X_head.append(X_head_vector_filled)
            else:
                X_body_vector_filled = np.zeros(n_topics, dtype=np.float64)
                for id, prob in doc:
                    X_body_vector_filled[id] = prob
                X_body.append(X_body_vector_filled)
            i += 1

        X = np.concatenate([X_head, X_body], axis=1)
        print("the lens:",len(X))
        return X

    n_topics = 300
    X = get_features(n_topics)

    return X

def latent_dirichlet_allocation_incl_holdout_and_test(headlines,bodies,corpus_name):
    # https://pypi.python.org/pypi/lda on bottom see suggestions like MALLET, hca
    # https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
    # https://www.quora.com/What-are-the-best-features-to-put-into-Latent-Dirichlet-Allocation-LDA-for-topic-modeling-of-short-text

    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            print(", ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

    def combine_head_and_body(headlines, bodies):
        head_and_body = [str(headline) + " " + str(body) for i, (headline, body) in
                         enumerate(zip(headlines, bodies))]
        return head_and_body

    def get_features(vocab):
        vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=False, norm='l2')
        X_train_head = vectorizer_head.fit_transform(headlines)

        vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=False, norm='l2')
        X_train_body = vectorizer_body.fit_transform(bodies)
        # calculates n most important topics of the bodies. Each topic contains all words but ordered by importance. The
        # more important topic words a body contains of a certain topic, the higher its value for this topic
        lda_body = LatentDirichletAllocation(n_components=100, learning_method='online', random_state=0, n_jobs=3)


        print("latent_dirichlet_allocation_incl_holdout_and_test: fit and transform body")
        t0 = time.time()
        lda_body_matrix = lda_body.fit_transform(X_train_body)
        print("done in %0.3fs." % (time.time() - t0))

        print("latent_dirichlet_allocation_incl_holdout_and_test: transform head")
        # use the lda trained for body topcis on the headlines => if the headlines and bodies share topics
        # their vectors should be similar
        lda_head_matrix = lda_body.transform(X_train_head)

        #print_top_words(lda_body, vectorizer_body.get_feature_names(), 100)

        print('latent_dirichlet_allocation_incl_holdout_and_test: calculating cosine distance between head and body')
        # calculate cosine distance between the body and head
        X = []
        for i in range(len(lda_head_matrix)):
            X_head_vector = np.array(lda_head_matrix[i]).reshape((1, -1)) #1d array is deprecated
            X_body_vector = np.array(lda_body_matrix[i]).reshape((1, -1))
            cos_dist = cosine_distances(X_head_vector, X_body_vector).flatten()
            X.append(cos_dist.tolist())
        return X


    h, b = get_head_body_tuples(corpus_name)

    h_test, b_test = get_head_body_tuples_test(corpus_name)

    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of heads: " + str(len(h)))
    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of bodies: " + str(len(b)))
    h.extend(h_test)
    b.extend(b_test)
    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of heads after ext: " + str(len(h)))
    print("word_ngrams_concat_tf5000_l2_w_holdout_and_test length of bodies after ext: " + str(len(b)))

    tfidf = TfidfVectorizer(ngram_range=(1,1), stop_words='english', max_features=5000, use_idf=False,
                            norm='l2')
    tfidf.fit_transform(combine_head_and_body(h,b))
    vocab = tfidf.vocabulary_

    X = get_features(vocab)
    return X

def read_data(corpus_name,dataset_name):
    base_path = f'F:/Individual_Project/{corpus_name}/dataset'
    # Extracting data
    if "fnc" in corpus_name:
      data = pd.read_csv(f"{base_path}/{dataset_name}.csv",header=None)
      data.columns = ["Body","Headline","Stance"]
      stance = data["Stance"].values.tolist()
      headline = data["Headline"].values.tolist()
      body = data["Body"].values.tolist()
      stance = np.asarray(stance, dtype = np.int64)
      return headline,body,stance
    if "snli" in corpus_name:
      data = pd.read_csv(f"{base_path}/{dataset_name}.csv",header=None)
      data.columns = ["Premise","Hypothesis","Stance"]
      stance = data["Stance"].values.tolist()
      premise = data["Premise"].values.tolist()
      hypothesis = data["Hypothesis"].values.tolist()
      stance = np.asarray(stance, dtype = np.int64)
      return premise,hypothesis,stance



mode = "train" #test
dataset_name = f"{mode}_dataset"
corpus_name = "fnc-1"
h,b,s = read_data(corpus_name,dataset_name)
filepath = f"F:/Individual_Project/{corpus_name}"


Topic_features1 = NMF_fit_all_incl_holdout_and_test(h,b,corpus_name)
with open(f"{filepath}/{mode}_NMF_fit_all_incl_holdout_and_test.pkl","wb") as handle:
    pickle.dump(Topic_features1,handle)
    handle.close()

Topic_features2 = NMF_fit_all_concat_300_and_test(h,b,corpus_name)
with open(f"{filepath}/{mode}_NMF_fit_all_concat_300_and_test.pkl","wb") as handle:
    pickle.dump(Topic_features2,handle)
    handle.close()

Topic_features3 = latent_semantic_indexing_gensim_holdout_and_test(h,b,corpus_name)
with open(f"{filepath}/{mode}_latent_semantic_indexing_gensim_holdout_and_test.pkl","wb") as handle:
    pickle.dump(Topic_features3,handle)
    handle.close()

Topic_features4 = latent_dirichlet_allocation_incl_holdout_and_test(h,b,corpus_name)
with open(f"{filepath}/{mode}_latent_dirichlet_allocation_incl_holdout_and_test.pkl","wb") as handle:
    pickle.dump(Topic_features4,handle)
    handle.close()