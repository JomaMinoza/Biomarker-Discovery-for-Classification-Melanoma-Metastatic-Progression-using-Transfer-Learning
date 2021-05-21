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

## Get RF genes
features_df = pd.read_csv('../data/Melanoma_RF_weights_all_genomic_data.csv',index_col=0)

model_prefixes = [
    "melanoma_rf_lr_",
    "melanoma_rf_svm_",
    "melanoma_rf_svm_poly_",
    "melanoma_rf_svm_radial_",
    "melanoma_rf_svm_sig_",
    "melanoma_rf_nb_",
    "melanoma_rf_rf_"
]

model_names = [
    "Logistic Regression",
    "Support Vector Machines (Linear Kernel)",
    "Support Vector Machines (Polynomial Kernel)",
    "Support Vector Machines (Radial Basis Kernel)",
    "Support Vector Machines (Sigmoid Kernel)",
    "Gaussian Naive Bayes",
    "Random Forest"
]

# Model Setup

model_prefixes = [
    "melanoma_ppi_lr_",
    "melanoma_ppi_svm_",
    "melanoma_ppi_svm_poly_",
    "melanoma_ppi_svm_radial_",
    "melanoma_ppi_svm_sig_",
    "melanoma_ppi_nb_",
    "melanoma_ppi_rf_"
]

model_names = [
    "Logistic Regression",
    "Support Vector Machines (Linear Kernel)",
    "Support Vector Machines (Polynomial Kernel)",
    "Support Vector Machines (Radial Basis Kernel)",
    "Support Vector Machines (Sigmoid Kernel)",
    "Gaussian Naive Bayes",
    "Random Forest"
]

### Metrics

train_eval_scores_df = pd.DataFrame(
    columns = [
            "auc_score",
            "logloss",
            "accuracy",
            "f1_macro",
            "f1_micro",
            "f1_weighted",
            "precision_macro",
            "precision_micro",
            "precision_weighted",
            "recall_macro",
            "recall_micro",
            "recall_weighted"
    ]
)
validation_eval_scores_df = train_eval_scores_df.copy()

### Pipelines

# Logistic Regression

lr_pipeline_params = [
    {
        'clf__estimator': [LogisticRegression(
                              solver='liblinear', 
                              max_iter=10000, 
                              tol=0.0001, 
                              fit_intercept=True)], 
        'clf__estimator__C': [1e-03, 1e-2, 1e-1, 1, 10, 100],
        'clf__estimator__penalty': ['l1', 'l2']    
    }
]

# Support Vector Machines - Linear Kernel

svm_linear_pipeline_params = [
    {   
        'clf__estimator': [SVC(probability=True, kernel = "linear")], 
        'clf__estimator__gamma': ['scale', 0.1,  0.01, 0.001, 0.00001, 1, 10],
        'clf__estimator__decision_function_shape': ["ovo", "ovr"]   
    }
]

# Support Vector Machines - Polynomial Kernel

svm_poly_pipeline_params = [
    {   
        'clf__estimator': [SVC(probability=True, kernel = "poly")], 
        'clf__estimator__gamma': ['scale', 0.1, 0.01, 0.001, 0.00001, 1, 10],
        'clf__estimator__decision_function_shape': ["ovo", "ovr"]   
    },
]

# Support Vector Machines - Radial Basis Kernel


svm_rbf_pipeline_params = [
    {   
        'clf__estimator': [SVC(probability=True, kernel = "rbf")], 
        'clf__estimator__gamma': ['scale', 0.1,  0.01, 0.001, 0.00001, 1, 10],
        'clf__estimator__decision_function_shape': ["ovo", "ovr"]   
    },
]

# Support Vector Machines - Sigmoid Kernel


svm_sigmoid_pipeline_params = [
    {   
        'clf__estimator': [SVC(probability=True, kernel = "sigmoid")], 
        'clf__estimator__gamma': ['scale', 0.1,  0.01, 0.001, 0.00001, 1, 10],
        'clf__estimator__decision_function_shape': ["ovo", "ovr"]   
    },
]

# Naive Bayes

gnb_pipeline_params = [
    {   
        'clf__estimator': [GaussianNB(priors = None)], 
    },
]

# Random Forest

rf_pipeline_params = [
    {   
        'clf__estimator': [RandomForestClassifier(random_state = 1, bootstrap = True)], 
        'clf__estimator__n_estimators': [30, 60, 70, 80, 90],
        'clf__estimator__max_features': [0.6, 0.65, 0.7, 0.75, 0.8],
        'clf__estimator__min_samples_leaf': [6, 8, 10, 12, 14],
        'clf__estimator__min_samples_split':  [2, 3, 5, 7]  
    },
]


model_pipeline_params = [
    lr_pipeline_params,
    svm_linear_pipeline_params,
    svm_poly_pipeline_params,
    svm_rbf_pipeline_params,
    svm_sigmoid_pipeline_params,
    gnb_pipeline_params,
    rf_pipeline_params
]

# Training and Evaluation of Models

for model_idx, model_prefix in enumerate(model_prefixes):
    for n_gene in range(10,150,10):
        if n_gene == 140:
            n_gene = 139
                
        model_classifier = ModelClassifier(
            train = train_df, 
            validation = test_df, 
            label = 'sample_type', 
            label_values = ['Primary Tumor', 'Metastatic'],
            features = features_df.weights.head(n_gene).index.values, 
            label_binarizer = False)
        
        grid_search_parameters = {
            "n_jobs": 4, 
            "cv": 5, 
            "return_train_score": False, 
            "verbose": 3, 
            "scoring": ["f1","accuracy"],
            "refit": "f1"
        }

        model = model_classifier.classifier(model_pipeline_params[model_idx], grid_search_parameters, pca = False)
        
        # Evaluation Scores
        
        train_scores, validation_scores = model_classifier.all_evaluation_scores(model)
        
        train_scores["model"]      = "{0} - Top {1} Genes".format(model_names[model_idx],n_gene)
        validation_scores["model"] = "{0} - Top {1} Genes".format(model_names[model_idx],n_gene)
        
        train_eval_scores_df      = train_eval_scores_df.append(train_scores, ignore_index = True)
        validation_eval_scores_df = validation_eval_scores_df.append(validation_scores, ignore_index = True)
        
        # Save model
                
        joblib.dump(model, "../models/{0}{1}_genes_classifier.pkl".format(model_prefix, n_gene), compress=9)
        
        
train_eval_scores_df.set_index('model', inplace=True)
validation_eval_scores_df.set_index('model', inplace=True)

train_eval_scores_df.to_csv("../data/train_eval_scores.csv")
validation_eval_scores_df.to_csv("../data/validation_eval_scores.csv")