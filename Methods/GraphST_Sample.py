import warnings
warnings.filterwarnings("ignore")
import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
import multiprocessing as mp
from sklearn import metrics
from GraphST import GraphST



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
os.environ['R_HOME'] = './...' # the location of R, which is necessary for mclust algorithm

tracemalloc.start()
t1 = time.time()

adata = sc.read_visium("./DLPFC/151673", count_file="filtered_feature_bc_matrix.h5", load_images=True)
adata.var_names_make_unique()
df_meta = pd.read_csv("./DLPFC/151673/metadata.tsv", sep='\t')
adata.obs["ground_truth"] = df_meta.loc[adata.obs_names, "layer_guess"]
adata = adata[~pd.isnull(adata.obs["ground_truth"])]
n_cluster = len(set(adata.obs["ground_truth"]))

# define model
model = GraphST.GraphST(adata, device=device)

# train model
adata = model.train()

# clustering
from GraphST.utils import clustering
radius = 50
tool = 'mclust' # set 'mclust' as default
if tool == 'mclust':
    clustering(adata, n_cluster, radius=radius, method=tool, refinement=True) # For DLPFC dataset, we use optional refinement step.
elif tool in ['leiden', 'louvain']:
    clustering(adata, n_cluster, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01, refinement=False)

t2 = time.time()
t = t2 - t1
size, peak = tracemalloc.get_traced_memory()
memory = peak / 1024 / 1024
tracemalloc.stop()

ARI = metrics.adjusted_rand_score(adata.obs["domain"], adata.obs["ground_truth"])
print(ARI)

adata.write_h5ad("./GraphST_results.h5ad")
