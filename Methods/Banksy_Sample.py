import os, csv, re
import random
import sys
import time
import tracemalloc
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics import adjusted_rand_score
import matplotlib.colors as clr
import matplotlib.pyplot as plt
from banksy.initialize_banksy import initialize_banksy
from banksy.run_banksy import run_banksy_multiparam
from banksy_utils.color_lists import spagcn_color

tracemalloc.start()
t1 = time.time()

adata = sc.read_visium("./DLPFC/151673", count_file="filtered_feature_bc_matrix.h5", load_images=True)
adata.var_names_make_unique()
df_meta = pd.read_csv("./DLPFC/151673/metadata.tsv", sep='\t')
adata.obs["ground_truth"] = df_meta.loc[adata.obs_names, "layer_guess"]
adata = adata[~pd.isnull(adata.obs["ground_truth"])]
n_cluster = len(set(adata.obs["ground_truth"]))

adata.layers['counts'] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=5000, inplace = True)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True, layer='counts', flavor='seurat_v3')

coord_keys = ('x_pixel', 'y_pixel', 'spatial')
x_coord, y_coord, xy_coord = coord_keys[0], coord_keys[1], coord_keys[2]
adata.obs[['x_pixel', 'y_pixel']] = adata.obsm['spatial']

output_folder = "./..."
resolutions = [0.6] # clustering resolution for Leiden clustering
pca_dims = [20] # number of dimensions to keep after PCA
lambda_list = [0.2] # lambda λ = 0.2 for cell typing; λ = 0.8 for domain segmentation
k_geom = 18 # 15 spatial neighbours
max_m = 1 # use AGF
nbr_weight_decay = 'scaled_gaussian' # can also be "reciprocal", "uniform" or "ranked"
nclust = 0
annotation_key = 'ground_truth'
cluster_algorithm = 'mclust'

banksy_dict = initialize_banksy(
    adata,
    coord_keys,
    k_geom,
    nbr_weight_decay=nbr_weight_decay,
    max_m=max_m,
    plt_edge_hist=False,
    plt_nbr_weights=False,
    plt_agf_angles=False,
    plt_theta=False)

results_df = run_banksy_multiparam(
    adata,
    banksy_dict,
    lambda_list,
    resolutions,
    max_m = max_m,
    color_list = spagcn_color,
    filepath = output_folder,
    key = coord_keys,
    pca_dims = pca_dims,
    annotation_key = annotation_key,
    max_labels = n_cluster,
    cluster_algorithm = cluster_algorithm,
    match_labels = False,
    savefig = False,
    add_nonspatial = False,
    variance_balance = False,
)
adata_res = results_df.loc[results_df.index[0],'adata']
adata_res.obs['Banksy'] = adata_res.obs[f'{results_df.index[0]}'].astype('category')
t2 = time.time()
t = t2 - t1
size, peak = tracemalloc.get_traced_memory()
memory = peak / 1024 / 1024
tracemalloc.stop()   

ARI = adjusted_rand_score(adata_res.obs['Banksy'], adata_res.obs['ground_truth'])
print(ARI)

adata.write_h5ad("./Banksy_results.h5ad")