import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
import time
import tracemalloc
import STAGATE

tracemalloc.start()
t1 = time.time()

adata = sc.read_visium("./DLPFC/151673", count_file="filtered_feature_bc_matrix.h5", load_images=True)
adata.var_names_make_unique()
df_meta = pd.read_csv("./DLPFC/151673/metadata.tsv", sep='\t')
adata.obs["ground_truth"] = df_meta.loc[adata.obs_names, "layer_guess"]
adata = adata[~pd.isnull(adata.obs["ground_truth"])]
n_cluster = len(set(adata.obs["ground_truth"]))

sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150) # Set "rad_cutoff" in range (50,950) based on the number of neighbors per spot
STAGATE.Stats_Spatial_Net(adata)

adata = STAGATE.train_STAGATE(adata, alpha=0.5, pre_resolution=0.2, n_epochs=1000, save_attention=False) # for ST and 10x Visium and "adata = STAGATE.train_STAGATE(adata, alpha=0)" for other datasets
sc.pp.neighbors(adata, use_rep="STAGATE")
sc.tl.umap(adata)
adata = STAGATE.mclust_R(adata, used_obsm="STAGATE", num_cluster=n_cluster)

t2 = time.time()
t = t2 - t1
size, peak = tracemalloc.get_traced_memory()
memory = peak / 1024 / 1024
tracemalloc.stop()

ARI = metrics.adjusted_rand_score(adata.obs["mclust"], adata.obs["ground_truth"])
print(ARI)

adata.write_h5ad("./STAGATE_results.h5ad")

