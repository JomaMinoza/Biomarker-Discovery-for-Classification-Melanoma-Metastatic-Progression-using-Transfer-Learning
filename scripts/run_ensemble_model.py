import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

#Visualization Tools

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# AI Workflow Module
from model_classifier import ModelClassifier, CustomClassifier

# Models
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.pipeline import make_pipeline

from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.feature_selection import ColumnSelector

# Use for saving model 
import joblib

# Loading of data
train_df = pd.read_csv('../data/TCGA-SKCM_train_unresampled_v0.csv',index_col=0)
test_df = pd.read_csv('../data/TCGA-SKCM_test_unresampled_v0.csv',index_col=0)


# Get the genomic data
rnaseq_df = pd.read_csv('../data/RNASeq/SKCM_DATA_RNASeq.csv', index_col = 0)

# Preprocess genomic data
X = rnaseq_df.T
X.index = [index[0:12] for index in X.index.values]
X.index.names = ['submitter_id']
X = X[~X.index.duplicated(keep='first')]
print(X.shape)

# Get RF genes
rf_features_df = pd.read_csv('../data/Melanoma_RF_weights_all_genomic_data.csv',index_col=0)

top10_rf_features = rf_features_df.weights.head(10).index.values
top20_rf_features = rf_features_df.weights.head(20).index.values
top30_rf_features = rf_features_df.weights.head(30).index.values

# Get PPI genes
ppi_features_df = pd.read_csv('../data/skcm_ppi_betweenness_centrality.csv', index_col=0)

top10_ppi_features = ppi_features_df.betweenness_centrality.head(10).index.values
top20_ppi_features = ppi_features_df.betweenness_centrality.head(20).index.values


# Load the models

rf_lr_10_model       = joblib.load('../models/melanoma_rf_lr_10_genes_classifier.pkl')
rf_rf_20_model       = joblib.load('../models/melanoma_rf_rf_20_genes_classifier.pkl')
ppi_svm_sig_10_model      = joblib.load('../models/melanoma_ppi_svm_sig_10_genes_classifier.pkl')

# Setup pipeline

rf_lr_10_model_pipe       = make_pipeline(
                               ColumnSelector(cols=tuple(top10_rf_features)),
                               rf_lr_10_model
                            )
rf_rf_20_model_pipe      = make_pipeline(
                            ColumnSelector(cols=tuple(top20_rf_features)),
                            rf_rf_20_model
                         )

ppi_svm_sig_10_model_pipe = make_pipeline(
                            ColumnSelector(cols=tuple(top10_ppi_features)),
                            ppi_svm_sig_10_model
                         )

significant_model_pipelines = [
        rf_lr_10_model_pipe,
        rf_rf_20_model_pipe,
        ppi_svm_sig_10_model_pipe
]

# Model Development for Ensemble Soft Voting

model_classifier = ModelClassifier(
    train = train_df, 
    validation = test_df, 
    label = 'sample_type', 
    label_values = ['Primary Tumor', 'Metastatic'],
    features = features, 
    label_binarizer = False)
    
ensemble_sig_clf = EnsembleVoteClassifier(clfs=significant_model_pipelines, voting='soft')
ensemble_sig_clf.fit(model_classifier.train_features, model_classifier.train_labels)

# Evaluation Metrics
model_classifier.classifier_metrics(ensemble_sig_clf, confusion_matrix = True)

joblib.dump(ensemble_sig_clf, '../models/melanoma_ensemble_classifier.pkl', compress=9)

