import os, time, tracemalloc
import anndata as ad
import squidpy as sq
import cellcharter as cc
import pandas as pd
import scanpy as sc
import scvi
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch import seed_everything
seed_everything(12345)
scvi.settings.seed = 12345

tracemalloc.start()
t1 = time.time()

adata = sc.read_visium("./DLPFC/151673", count_file="filtered_feature_bc_matrix.h5", load_images=True)
adata.var_names_make_unique()
df_meta = pd.read_csv("./DLPFC/151673/metadata.tsv", sep='\t')
adata.obs["ground_truth"] = df_meta.loc[adata.obs_names, "layer_guess"]
adata = adata[~pd.isnull(adata.obs["ground_truth"])]
n_cluster = len(set(adata.obs["ground_truth"]))

sc.pp.filter_cells(adata, min_counts=1)
adata.layers['counts'] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True, layer='counts', flavor='seurat_v3')
adata = adata[:, adata.var.highly_variable]
adata = adata.copy()
scvi.model.SCVI.setup_anndata(adata, layer='counts')
model = scvi.model.SCVI(adata, n_latent=5)
model.train(early_stopping=True, plan_kwargs=dict(optimizer='AdamW', reduce_lr_on_plateau=True))
adata.obsm['X_scVI'] = model.get_latent_representation(adata).astype(np.float32)

sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True, spatial_key='spatial', percentile=99)
cc.gr.aggregate_neighbors(adata, n_layers=3, use_rep='X_scVI', out_key='X_cellcharter')
cls = cc.tl.Cluster(n_cluster, random_state=12345)
cls.fit(adata, use_rep='X_cellcharter')
adata.obs['cluster_cellcharter'] = cls.predict(adata, use_rep='X_cellcharter')

t2 = time.time()
t = t2 - t1
size, peak = tracemalloc.get_traced_memory()
memory = peak / 1024 / 1024
tracemalloc.stop()

ARI = metrics.adjusted_rand_score(adata.obs["cluster_cellcharter"], adata.obs["ground_truth"])
print(ARI)

adata.write_h5ad("./CellCharter_results.h5ad")