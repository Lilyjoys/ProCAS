import re
from collections import Counter
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans
from Resample.ExtremeBias_cluster import ExtremeBias_cluster
from Feature_extraction_algorithms.esm2 import get_esm1b_features, get_mistral_peptide_features, get_protbert_features, get_prott5_features
import pickle


def extract_file_number(filename):

    match = re.search(r'\((\d+)\)', filename)
    if match:
        return int(match.group(1))
    else:
        return None


def load_data(path):
    print("[Original Data Loading]")
    seq = []
    Y = []
    files = os.listdir(path)
    for file in files:
        y = extract_file_number(file)
        with open(os.path.join(path, file), 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    continue
                seq.append(line)
                Y.append(y)
    return seq, Y


def feature_extraction(seq, test_PSTAAP=False):

    from Feature_extraction_algorithms.MLPSTAAP import PSTAAP_feature
    from Feature_extraction_algorithms.esm2 import esm2
    from Feature_extraction_algorithms.CTD_feature import extract_ctd_features
    N = len(seq)
    empty_list_array = [[] for _ in range(N)]
    data = np.array(empty_list_array, dtype=object)
    feature = PSTAAP_feature(seq, test_PSTAAP)
    data = np.hstack((data, feature))
    data2 = esm2(seq)
    ctd_feature = extract_ctd_features(seq)

    return data.astype(np.float32), data2.astype(np.float32), ctd_feature


def make_ylabel(y):
    # transform y into multi-label
    Y = np.zeros((y.shape[0], 4))
    for i in range(y.shape[0]):
        if y[i] == 1:
            Y[i] = np.array([1, 0, 0, 0])  # a
        elif y[i] == 2:
            Y[i] = np.array([0, 1, 0, 0])  # c
        elif y[i] == 3:
            Y[i] = np.array([0, 0, 1, 0])  # m
        elif y[i] == 4:
            Y[i] = np.array([0, 0, 0, 1])  # s
        elif y[i] == 5:
            Y[i] = np.array([1, 1, 0, 0])  # ac
        elif y[i] == 6:
            Y[i] = np.array([1, 0, 1, 0])  # am
        elif y[i] == 7:
            Y[i] = np.array([1, 0, 0, 1])  # as
        elif y[i] == 8:
            Y[i] = np.array([0, 1, 1, 0])  # cm
        elif y[i] == 9:
            Y[i] = np.array([1, 1, 1, 0])  # acm
        elif y[i] == 10:
            Y[i] = np.array([1, 1, 0, 1])  # acs
        elif y[i] == 11:
            Y[i] = np.array([1, 1, 1, 1])  # acms
    print("[INFO]\tmulti-label Y:", Y.shape)
    return Y


def process_data(train_path=None, test_path=None):

    X_train, X_train2, X_test, X_test2, Y_train, Y_test = None, None, None, None, None, None
    label_train, label_test = None, None
    ctd_train, ctd_test = None, None
    esm1b_feat, prott5_feat, bert_feat, mistral_feat, tesm1b_feat, tprott5_feat, tbert_feat, tmistral_feat = None, None, None, None, None, None, None, None
    if train_path:
        seq_train, Y_train = load_data(train_path)
        X_train, X_train2, ctd_train = feature_extraction(seq_train)

        esm1b_feat = get_esm1b_features(seq_train, batch_size=64)
        prott5_feat = get_prott5_features(seq_train, batch_size=64)
        bert_feat = get_protbert_features(seq_train, batch_size=64)
        mistral_feat = get_mistral_peptide_features(seq_train, batch_size=64)


        Y_train = np.array(Y_train)
        label_train = make_ylabel(Y_train)

    if test_path:
        seq_test, Y_test = load_data(test_path)
        X_test, X_test2, ctd_test = feature_extraction(seq_test, test_PSTAAP=True)

        tesm1b_feat = get_esm1b_features(seq_test, batch_size=64)
        tprott5_feat = get_prott5_features(seq_test, batch_size=64)
        tbert_feat = get_protbert_features(seq_test, batch_size=64)
        tmistral_feat = get_mistral_peptide_features(seq_test, batch_size=64)

        Y_test = np.array(Y_test)
        label_test = make_ylabel(Y_test)

    with open("dataset/protein_features.pkl", "wb") as f:
        pickle.dump((
            esm1b_feat,
            prott5_feat,
            bert_feat,
            mistral_feat,
            tesm1b_feat,
            tprott5_feat,
            tbert_feat,
            tmistral_feat
        ), f)

    if train_path:
        return X_train, X_train2, Y_train, label_train, X_test, X_test2, Y_test, label_test, ctd_train, ctd_test
    else:
        return X_test, Y_test, label_test

