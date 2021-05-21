import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.stats import ttest_ind

# Get the data
rnaseq_df = pd.read_csv('../data/RNASeq/SKCM_DATA_RNASeq.csv', index_col = 0)

# Preprocess genomic data
X = rnaseq_df.T
X.index = [index[0:12] for index in X.index.values]
X.index.names = ['submitter_id']
X = X[~X.index.duplicated(keep='first')]
print(X.shape)

# Preprocess clinical data
clinical_df = pd.read_csv('../data/Clinical/SKCM_DATA_Clinical.csv', index_col = 0)
clinical_df.set_index('submitter_id',inplace=True)
print(clinical_df.shape)

# Get genes in train data
genes = set(X.columns.values).intersection(set(train_df.columns.values))
features = list(genes)

# Get RF genes
rf_features_df = pd.read_csv('../data/Melanoma_RF_weights_all_genomic_data.csv',index_col=0)

# Combined data
Y = clinical_df['sample_type'].to_frame()
Y = Y.replace({'Primary Tumor':0,'Metastatic':1, 'Additional Metastatic': 1})
Y = Y[Y.sample_type != 'Solid Tissue Normal']
X = X[rf_features_df.head(139).index.values].copy()
print(X.shape)
print(Y.shape)

genes_df = Y.merge(X, left_index=True, right_index=True)
print(genes_df.shape)

# Welch's T Test

genes_level_exp_significance_df = pd.DataFrame(columns = ["t_stat","p_value"], index = rf_features_df.head(139).index.values)

for gene in rf_features_df.head(139).index.values:
    pt_df = genes_df[genes_df['sample_type'] == 0][gene]
    m_df  = genes_df[genes_df['sample_type'] == 1][gene]
    genes_level_exp_significance_df.loc[gene] = ttest_ind(pt_df, m_df, equal_var = False)   
    
print("Significant genes {0}".format(genes_level_exp_significance_df[genes_level_exp_significance_df.p_value < 0.05].index.values))

genes_level_exp_significance_df.to_csv("../data/genes_level_exp_significance.csv")
