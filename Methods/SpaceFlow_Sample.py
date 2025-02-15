import squidpy as sq
import scanpy as sc
from SpaceFlow import SpaceFlow
import pandas as pd
import os
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


sf = SpaceFlow.SpaceFlow(adata=adata)
sf.preprocessing_data(n_top_genes=3000)
sf.train(spatial_regularization_strength=0.1, z_dim=50, lr=1e-3, epochs=1000, max_patience=50, min_stop=100, 
         random_seed=42, gpu=0, regularization_acceleration=True, edge_subset_sz=1000000)

res = 0.7
step = 0.1
target_num = n_cluster
sf.segmentation(domain_label_save_filepath="./domains.tsv", n_neighbors=50, resolution=res)
pred_clusters = np.array(sf.domains).astype(int)
uniq_pred = np.unique(pred_clusters)
old_num=len(uniq_pred)
print("Res = ", res, "Num of clusters = ", old_num)
run=0
max_run = 50
while old_num!=target_num:
    old_sign=1 if (old_num<target_num) else -1
    sf.segmentation(domain_label_save_filepath="./domains.tsv", n_neighbors=50, resolution=res+step*old_sign)
    pred_clusters = np.array(sf.domains).astype(int)
    uniq_pred = np.unique(pred_clusters)
    new_num=len(uniq_pred)
    print("Res = ", res+step*old_sign, "Num of clusters = ", new_num)
    if new_num==target_num:
        print("recommended res = ", str(res+step*old_sign))
        break
    new_sign=1 if (new_num<target_num) else -1
    if new_sign==old_sign:
        res=res+step*old_sign
        print("Res changed to", res)
        old_num=new_num
    else:
        step=step/2
        print("Step changed to", step)
    if run > max_run:
        print("Exact resolution not found")
        print("Recommended res = ", str(res))
        break
    run+=1

adata.obs["SpaceFlow"] = sf.domains.values
adata.obs["SpaceFlow"] = adata.obs["SpaceFlow"].astype("category")

t2 = time.time()
t = t2 - t1
size, peak = tracemalloc.get_traced_memory()
memory = peak / 1024 / 1024
tracemalloc.stop()

ARI = metrics.adjusted_rand_score(adata.obs["SpaceFlow"], adata.obs["ground_truth"])
print(ARI)

adata.write_h5ad("./SpaceFlow_results.h5ad")