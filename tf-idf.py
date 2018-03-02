import argparse
from os.path import dirname, abspath, join, exists
import os
from datetime import datetime

import torch
from torch.optim import Adagrad, lr_scheduler
from torch import nn
import numpy as np

from preprocessors import DATASET_TO_PREPROCESSOR
import dictionaries
from dataloaders import TextDataset, TextDataLoader
from trainers import Trainer
from evaluators import Evaluator

DATASETS = ['amazon_review_full',
 'yelp_review_polarity',
 'yahoo_answers',
 'dbpedia',
 'ag_news',
 'yelp_review_full',
 'amazon_review_polarity',
 'sogou_news',
 'rt-polarity']

parser = argparse.ArgumentParser(description="TF-IDF Model for Baseline")
parser.add_argument('--dataset', type=str, default='ag_news', choices=DATASETS)
parser.set_defaults(preprocess_level='char')

args = parser.parse_args()

from random import seed
seed(0)
np.random.seed(0)
torch.manual_seed(0)

Preprocessor = DATASET_TO_PREPROCESSOR[args.dataset]
preprocessor = Preprocessor(args.dataset)
train_data, val_data, test_data = preprocessor.preprocess(level=args.preprocess_level)

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegressionCV

tfidf = TfidfVectorizer(ngram_range=(1,5), max_features=500000)
model = LogisticRegressionCV()

data = train_data + val_data
x = [text for text, label in data]
y = [label for text, label in data]
x_transformed = tfidf.fit_transform(x)
model.fit(x_transformed, y)

x_test = [text for text, label in test_data]
y_test = [label for text, label in test_data]
x_test_transformed = tfidf.transform(x_test)

train_score = model.score(x_transformed, y)
test_score = model.score(x_test_transformed, y_test)

result_base = "Train Accuracy: {train_acc:<.1%}  Test Accuracy: {test_acc:<.1%}"
result = result_base.format(train_acc=train_score, test_acc=test_score)
print(result)