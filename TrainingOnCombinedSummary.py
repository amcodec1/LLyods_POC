# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:47:23 2018

@author: kagar3
"""

# import cPickle as pickle
import ast
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

"""Read the labelled file"""

TransactionDf = pd.read_csv('../../resources/CombinedSummary.csv',  encoding="utf-8")

TransactionDf['AmountInGBP'] = TransactionDf['Amount'].map(lambda x: ast.literal_eval(x)['Amount']).astype(np.float)
TransactionDf['Type'] = TransactionDf['BankTransactionCode'].map(
    lambda x: ast.literal_eval(x[:-1] + "\"\"" + '}')['Code'])


def BinningData(TransactionDf, bins):
    min_val = min((TransactionDf['AmountInGBP']) - 1)
    max_val = max(TransactionDf['AmountInGBP'])
    # print(min_val, max_val)
    custom_bucket_array = np.linspace(min_val, max_val, bins)
    cut_points = list(custom_bucket_array)
    group_name = ["low", "medium", "high"]
    TransactionDf["AmountInGBP_bin"] = pd.cut(TransactionDf["AmountInGBP"], cut_points, labels=group_name)
    return TransactionDf


BinningData(TransactionDf, 4)


# Get Merchant info from Description
def getMerchfromDesc(tranType, tranDesc):
    if tranType == 'POS':
        parsed = tranDesc.split(', ')
        lenPar = len(parsed)
        if parsed[lenPar - 1] == 'REFUND':
            merch = parsed[1:-2]
            return ('').join(merch) + 'REFUND'
        else:
            merch = parsed[1:-1]
            return ('').join(merch)
    else:
        return tranDesc


# Clean Merchant info
def cleanMerchant(txt):
    txt = re.sub(r'[^a-zA-Z\b]', ' ', str(txt).upper())
    return " ".join(txt.split())


TransactionDf['Merchant'] = TransactionDf.apply(
    lambda row: getMerchfromDesc(row['Type'], row['TransactionInformation']), axis=1)
TransactionDf['cleanedMerchant'] = TransactionDf['Merchant'].map(lambda x: cleanMerchant(x))

TransactionDf['ConcatCategory'] = TransactionDf['category'] + ', ' + TransactionDf['Detailed Category']
# len(TransactionDf['ConcatCategory'].unique())

TransactionDf['Predictorcolumn'] = TransactionDf['CreditDebitIndicator'] + ' ' + TransactionDf[
    'cleanedMerchant'] + ' ' + TransactionDf['Type'] + ' ' + TransactionDf['AmountInGBP_bin'].astype(str)

vectorizer = TfidfVectorizer(use_idf=True, stop_words='english', ngram_range=(1, 2), min_df=1)

vectors = vectorizer.fit_transform(TransactionDf['Predictorcolumn'])

with open('../../target/model/vectorizer.pkl', 'wb') as fin:
    pickle.dump(vectorizer, fin)

# clf = RandomForestClassifier(n_estimators=50, max_depth=None,min_samples_split=2, random_state=0)
# scores = cross_val_score(clf, vectors, TransactionDf['ConcatCategory'], cv=3)
# np.mean(scores)

# Training on complete data
RandomForest_model = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)
RandomForest_model.fit(vectors, TransactionDf['ConcatCategory'])

# Dump the trained decision tree classifier with Pickle
RandomForest_pkl_filename = '../../target/model/RandomForest_classifier.pkl'
# Open the file to save as pkl file
RandomForest_model_pkl = open(RandomForest_pkl_filename, 'wb')
pickle.dump(RandomForest_model, RandomForest_model_pkl)
# Close the pickle instances
RandomForest_model_pkl.close()