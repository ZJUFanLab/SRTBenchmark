import stlearn as st
from pathlib import Path
st.settings.set_figure_params(dpi=180)
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
import time
import tracemalloc

tracemalloc.start()
t1 = time.time()

adata = sc.read_visium("./DLPFC/151673", count_file="filtered_feature_bc_matrix.h5", load_images=True)
adata.var_names_make_unique()
df_meta = pd.read_csv("./DLPFC/151673/metadata.tsv", sep='\t')
adata.obs["ground_truth"] = df_meta.loc[adata.obs_names, "layer_guess"]
adata = adata[~pd.isnull(adata.obs["ground_truth"])]
n_cluster = len(set(adata.obs["ground_truth"]))

st.pp.filter_genes(adata,min_cells=1)
st.pp.normalize_total(adata)
st.pp.log1p(adata)

st.em.run_pca(adata,n_comps=15)

TILE_PATH = "./TILE/"
st.pp.tiling(adata, TILE_PATH)
st.pp.extract_feature(adata)

# stSME
st.spatial.SME.SME_normalize(data, use_data="raw", weights="weights_matrix_pd_md") #
data_ = adata.copy()
data_.X = data_.obsm['raw_SME_normalized']

st.pp.scale(data_)
st.em.run_pca(data_,n_comps=15) 

st.tl.clustering.kmeans(data_, n_clusters=n_cluster, use_data="X_pca", key_added="X_pca_kmeans")
st.pl.cluster_plot(data_, use_label="X_pca_kmeans")

"""
# for datasets without image
st.pp.filter_genes(data,min_cells=3)
st.pp.normalize_total(data)
st.pp.log1p(data)
st.pp.scale(data)
st.em.run_pca(data,n_comps=50,random_state=0)
st.tl.clustering.kmeans(adata, n_clusters=n_cluster, use_data="X_pca", key_added="kmeans")
"""
t2 = time.time()
t = t2 - t1
size, peak = tracemalloc.get_traced_memory()
memory = peak / 1024 / 1024
tracemalloc.stop()

ARI = metrics.adjusted_rand_score(adata.obs["X_pca_kmeans"], adata.obs["ground_truth"])
print(ARI)

adata.write_h5ad("./stLearn_results.h5ad")

