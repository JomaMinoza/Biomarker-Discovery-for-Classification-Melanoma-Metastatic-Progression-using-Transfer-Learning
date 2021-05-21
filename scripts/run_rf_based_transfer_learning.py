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

# Use for saving model 
import joblib

# Loading of data

train_df = pd.read_csv('../data/TCGA-SKCM_train_unresampled_v0.csv',index_col=0)
test_df = pd.read_csv('../data/TCGA-SKCM_test_unresampled_v0.csv',index_col=0)

# Get genes in train data
genes = set(X.columns.values).intersection(set(train_df.columns.values))

features = list(genes)

# Model Development

model_classifier = ModelClassifier(
    train = train_df, 
    validation = test_df, 
    label = 'sample_type', 
    label_values = ['Primary Tumor', 'Metastatic'],
    features = features, 
    label_binarizer = False)
    
# Hyperparameter Tuning
    
rf_pipeline_params = [
    {   
        'clf__estimator': [RandomForestClassifier(random_state = 1, bootstrap = True)], 
        'clf__estimator__n_estimators': [30, 60, 70, 80, 90],
        'clf__estimator__max_features': [0.6, 0.65, 0.7, 0.75, 0.8],
        'clf__estimator__min_samples_leaf': [6, 8, 10, 12, 14],
        'clf__estimator__min_samples_split':  [2, 3, 5, 7]  
    },
]

grid_search_parameters = {
    "n_jobs": 4, 
    "cv": 5, 
    "return_train_score": False, 
    "verbose": 3, 
    "scoring": ["accuracy", "f1"],
    "refit": "f1"
}

model = model_classifier.classifier(rf_pipeline_params, grid_search_parameters, pca = False)

# Confusion Matrix of the Random Forest Based Feature Selection (Biomarker Discovery)

model_classifier.classifier_metrics(model, confusion_matrix = True)


# Plot of Random Forest Estimators

from subprocess import check_call
from sklearn.tree import export_graphviz

# Get the final model
final_model = model[1].estimator

for index in range(0, final_model.n_estimators):
    export_graphviz(final_model.estimators_[index],
                     out_file="../data/rf_plot/rf_{}.dot".format(index),
                     feature_names = model_classifier.train_features.columns, 
                     class_names = model_classifier.label_values,
                     filled = True)


for index in range(0, final_model.n_estimators):
    check_call(['dot','-Tpng',"../data/rf_plot/rf_{}.dot".format(index),'-o',"../data/rf_plot/rf_{}.png".format(index)])
    
    
# Save Feature Importance of the RF

rf_weights = pd.DataFrame(final_model.feature_importances_)
rf_weights.index = model_classifier.train_features.columns
rf_weights.columns = ["weights"]
rf_weights.sort_values(by=['weights'], inplace=True, ascending=False)
rf_weights.to_csv('../data/Melanoma_RF_weights_all_genomic_data.csv')

# Note: Only 139 Genes have weights



