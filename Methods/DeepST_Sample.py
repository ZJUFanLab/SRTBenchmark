import warnings
warnings.filterwarnings("ignore")
import os
import time
import tracemalloc
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from DeepST import run
from pathlib import Path
from sklearn import metrics

tracemalloc.start()
t1 = time.time()

save_path = "" # Customize the save path
deepen = run(save_path = save_path,
             task = "Identify_Domain", 
             pre_epochs = 800,
             epochs = 1000, 
             use_gpu = True)

adata = sc.read_visium("./DLPFC/151673", count_file="filtered_feature_bc_matrix.h5", load_images=True)
adata.var_names_make_unique()
df_meta = pd.read_csv("./DLPFC/151673/metadata.tsv", sep='\t')
adata.obs["ground_truth"] = df_meta.loc[adata.obs_names, "layer_guess"]
adata = adata[~pd.isnull(adata.obs["ground_truth"])]
n_cluster = len(set(adata.obs["ground_truth"]))

adata = deepen._get_image_crop(adata, data_name="151673")
adata = deepen._get_augment(adata, spatial_type="LinearRegress", use_morphological=True) # "LinearRegress" is applicable to 10x visium and the remaining omics selects "BallTree"
graph_dict = deepen._get_graph(adata.obsm["spatial"], distType="BallTree") 
data = deepen._data_process(adata, pca_n_comps=200) # 200 reduced to min(200, adata.shape[0]-1, adata.shape[1]-1)
deepst_embed = deepen._fit(data=data, graph_dict=graph_dict,)
adata.obsm["DeepST_embed"] = deepst_embed
adata = deepen._get_cluster_data(adata, n_domains=n_cluster, priori=True)

t2 = time.time()
t = t2 - t1
size, peak = tracemalloc.get_traced_memory()
memory = peak / 1024 / 1024
tracemalloc.stop()

ARI = metrics.adjusted_rand_score(adata.obs["DeepST_refine_domain"], adata.obs["ground_truth"])
print(ARI)

adata.write_h5ad(save_path + "/DeepST_results.h5ad")
