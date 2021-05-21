import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm

import seaborn as sns

import networkx as nx
import requests

# Get RF genes
features_df = pd.read_csv('../data/Melanoma_RF_weights_all_genomic_data.csv',index_col=0)

# Only 139 genes have weights
protein_list = features_df.head(139).index.values
proteins = '%0d'.join(protein_list)

# Retrieve protein-protein interaction network from STRINGDB
url = 'https://string-db.org/api/tsv/network?identifiers=' + proteins + '&species=9606'

headers = {
   'Content-Type': "application/x-www-form-urlencoded; charset=UTF-8"
}

params = {'identifiers': proteins, 'species': 9606, 'caller_identity': 'https://www.usacfi.net/'} 
r = requests.get(url, data = params)

lines = r.text.split('\n')                # pull the text from the response object and split based on new lines
data = [l.split('\t') for l in lines]     # split each line into its components based on tabs

# Convert to dataframe using the first row as the column names; drop empty, final row
df = pd.DataFrame(data[1:-1], columns = data[0]) 

# Dataframe with the preferred names of the two proteins and the score of the interaction
interactions = df[['preferredName_A', 'preferredName_B', 'score']] 

G=nx.Graph(name='Protein Interaction Graph')
interactions = np.array(interactions)
for i in range(len(interactions)):
    interaction = interactions[i]
    a = interaction[0] # protein a node
    b = interaction[1] # protein b node
    w = float(interaction[2]) # score as weighted edge where high scores = low weight
    G.add_weighted_edges_from([(a,b,w)]) 
    
# Get betweeness centrality score

bc = nx.betweenness_centrality(G)

betweenness_centrality_df = pd.DataFrame(data = bc.values())
betweenness_centrality_df.index = bc.keys()
betweenness_centrality_df.columns = ["betweenness_centrality"]
betweenness_centrality_df.sort_values(by=['betweenness_centrality'], inplace=True, ascending=False)

# Save results
betweenness_centrality_df.to_csv("../data/skcm_ppi_betweenness_centrality.csv")
