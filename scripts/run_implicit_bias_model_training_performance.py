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

clinical_df = pd.read_csv('../../data/Clinical/SKCM_DATA_Clinical.csv', index_col = 0)
clinical_df.set_index('submitter_id',inplace=True)
clinical_df = clinical_df[~clinical_df.index.duplicated(keep='first')]
clinical_df = clinical_df[clinical_df.sample_type != "Solid Tissue Normal"]

# Get RF genes
rf_features_df = pd.read_csv('../data/Melanoma_RF_weights_all_genomic_data.csv',index_col=0)

# Get PPI genes
ppi_features_df = pd.read_csv('../data/skcm_ppi_betweenness_centrality.csv', index_col=0)



# Models

model_prefixes = [
    "melanoma_rf_lr_",         #10
    "melanoma_rf_svm_",        #10
    "melanoma_rf_svm_radial_", #10
    "melanoma_rf_svm_sig_",    #30
    "melanoma_rf_nb_",         #20
    "melanoma_rf_rf_",         #20
    
    "melanoma_ppi_lr_",       # 20
    "melanoma_ppi_svm_",      # 20
    "melanoma_ppi_svm_sig_",  # 10
    "melanoma_ppi_nb_",       # 20
    "melanoma_ensemble"
]

model_n_genes = [
    10, 10, 10, 30, 20, 20,
    10, 20, 10, 20, 0
]

rf_techniques = [
    True, True, True, True, True, True,
    False, False, False, False, False
]

model_names = [
    "[RF] Logistic Regression",
    "[RF] Support Vector Machines (Linear Kernel)",
    "[RF] Support Vector Machines (Radial Basis Kernel)",
    "[RF] Support Vector Machines (Sigmoid Kernel)",
    "[RF] Gaussian Naive Bayes",
    "[RF] Random Forest",
    "[PPI] Logistic Regression",
    "[PPI] Support Vector Machines (Linear Kernel)",
    "[PPI] Support Vector Machines (Sigmoid Kernel)",
    "[PPI] Gaussian Naive Bayes",
    "Ensemble Model"
]

# Implicit Bias

bias_df = pd.DataFrame(
    columns = model_names,
    index   = ["sample_type_PT","sample_type_M",
               "age_0","age_20", "age_40", "age_60", "age_80", 
               "gender_male","gender_female",
               "bmi_normal","bmi_overweight","bmi_obese",
               "race_white","race_asian","race_not_reported",
               "ethnicity_hispanic","ethnicity_not_hispanic","ethnicity_not_reported"
              ]
)

model_classifier = ModelClassifier(
    train = train_df, 
    validation = test_df, 
    label = 'sample_type', 
    label_values = ['Primary Tumor', 'Metastatic'],
    features = train_df.columns.values, 
    label_binarizer = False)
    
bias_check_df = train_df[[
        'vital_status_Dead', 
        'ethnicity_not hispanic or latino',
        'ethnicity_not reported', 
        'age_at_index', 
        'bmi']]

bias_check_df.insert(loc=5,  column='sample_type',  value = model_classifier.train_labels)

bias_check_df.insert(loc=6, column='sample_type_PT',   value=[int(clinical_df.loc[bias_index,"sample_type"] == 'Primary Tumor') for bias_index in bias_check_df.index.values])
bias_check_df.insert(loc=7, column='sample_type_M',   value=[int(clinical_df.loc[bias_index,"sample_type"] == 'Metastatic') for bias_index in bias_check_df.index.values])

bias_check_df.insert(loc=8, column='age_0',      value=[int(age<20) for age in bias_check_df['age_at_index']])
bias_check_df.insert(loc=9, column='age_20',     value=[int(age>=20 and age<40) for age in bias_check_df['age_at_index']])
bias_check_df.insert(loc=10, column='age_40',     value=[int(age>=40 and age<60) for age in bias_check_df['age_at_index']])
bias_check_df.insert(loc=11, column='age_60',     value=[int(age>=60 and age<80) for age in bias_check_df['age_at_index']])
bias_check_df.insert(loc=12, column='age_80',    value=[int(age>=80) for age in bias_check_df['age_at_index']])

bias_check_df.insert(loc=13, column='gender_male',     value=[int(clinical_df.loc[bias_index,"gender"] == 'male') for bias_index in bias_check_df.index.values])
bias_check_df.insert(loc=14, column='gender_female',   value=[int(clinical_df.loc[bias_index,"gender"] == 'female') for bias_index in bias_check_df.index.values])

bias_check_df.insert(loc=15, column='ethnicity_hispanic',     value=[int(clinical_df.loc[bias_index,"ethnicity"] == 'hispanic or latino') for bias_index in bias_check_df.index.values])
bias_check_df.insert(loc=16, column='ethnicity_not_hispanic', value=[int(clinical_df.loc[bias_index,"ethnicity"] == 'not hispanic or latino') for bias_index in bias_check_df.index.values])
bias_check_df.insert(loc=17, column='ethnicity_not_reported', value=[int(clinical_df.loc[bias_index,"ethnicity"] == 'not reported') for bias_index in bias_check_df.index.values])

bias_check_df.insert(loc=18, column='race_white', value=[int(clinical_df.loc[bias_index,"race"] == 'white') for bias_index in bias_check_df.index.values])
bias_check_df.insert(loc=19, column='race_asian', value=[int(clinical_df.loc[bias_index,"race"] == 'asian') for bias_index in bias_check_df.index.values])
bias_check_df.insert(loc=20, column='race_not_reported', value=[int(clinical_df.loc[bias_index,"race"] == 'not reported') for bias_index in bias_check_df.index.values])

bias_check_df.insert(loc=21, column='bmi_normal',     value=[int(bmi>=18.5 and bmi<25) for bmi in bias_check_df['bmi']])
bias_check_df.insert(loc=22, column='bmi_overweight', value=[int(bmi>=25 and bmi<30) for bmi in bias_check_df['bmi']])
bias_check_df.insert(loc=23, column='bmi_obese',      value=[int(bmi>=30) for bmi in bias_check_df['bmi']])


for idx, model_prefix in enumerate(model_prefixes):
    n_gene = model_n_genes[idx]
    column_name = model_names[idx]
    
    if rf_techniques[idx]:
        features = rf_features_df.weights.head(n_gene).index.values
    else:
        if n_gene == 0:
            features = train_df.columns.values
        else:
            features = ppi_features_df.betweenness_centrality.head(n_gene).index.values
        
    model_classifier = ModelClassifier(
        train = train_df, 
        validation = test_df, 
        label = 'sample_type', 
        label_values = ['Primary Tumor', 'Metastatic'],
        features = features, 
        label_binarizer = False,
        log_transform = True,
        log_transform_features = features
    )
    
    if n_gene == 0:
        model =  joblib.load("../models/melanoma_ensemble_classifier.pkl")
    else:    
        model =  joblib.load("../models/{0}{1}_genes_classifier.pkl".format(model_prefix, n_gene))
    
    bias_check_df.insert(loc=23 + (idx + 1), 
                             column = column_name,     
                             value  = model.predict(model_classifier.train_features))
                             
                             
def set_accuracy(df, biased_field, model_field, percentage = False):
    tp = sum((df['sample_type'] == 1) & (df[model_field] == 1) & (df[biased_field] == 1))
    tn = sum((df['sample_type'] == 0) & (df[model_field] == 0) & (df[biased_field] == 1))
    N =  sum(df[biased_field] == 1)
    
    if percentage == False:
        return (tp + tn)
    else:
        return (tp + tn)/N
        
def set_N(df, biased_field):
    N =  sum(df[biased_field] == 1)
    return N
    
for model in bias_df.columns.values:
    for field in bias_df.index.values:
        bias_df.loc[field, model] = set_accuracy(bias_check_df, field, model, True)
        

bias_N_df = bias_df.copy()

for model in bias_df.columns.values:
    for field in bias_df.index.values:
        bias_N_df.loc[field, model] = set_N(bias_check_df, field)
        
        
bias_df.to_csv("../data/implicit_bias_training.csv")
bias_N_df.to_csv("../data/implicit_bias_N_training.csv")
