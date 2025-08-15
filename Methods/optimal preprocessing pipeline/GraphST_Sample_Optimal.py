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

def preprocess_pipeline(
    adata_raw,
    do_normalize=True,
    do_log=True,
    method=None,
    n_top_gene=2000,
    do_scale=True,
    do_pca=True,
    n_comps=200
):
    adata = adata_raw.copy()
    if method is not None:
        if method == 'hvg':
            sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=n_top_gene)
            adata = adata[:, adata.var.highly_variable]
        elif method == 'svg':
            svg = pd.read_csv(f"{sample}_3000_svg.csv")['gene'].tolist()
            adata = adata[:, adata.var_names.isin(svg)]
        else:
            raise ValueError("Invalid method: choose 'hvg' or 'svg'")
    if do_normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
    if do_log:
        sc.pp.log1p(adata)
    if do_scale:
        sc.pp.scale(adata, max_value=10)
    if do_pca:
        sc.tl.pca(adata, n_comps=n_comps)
    return adata

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

adata_raw = adata.copy()
adata_processed = preprocess_pipeline(
    adata_raw,
    do_normalize=True,
    do_log=True,
    method='hvg',
    n_top_gene=2000,
    do_scale=False,
    do_pca=False,
    n_comps=0
)
adata_processed.var['highly_variable'] = True
# define model
model = GraphST.GraphST(adata_processed, device=device)

# train model
adata_processed = model.train()

# clustering
from GraphST.utils import clustering
radius = 50
tool = 'mclust' # set 'mclust' as default
if tool == 'mclust':
    clustering(adata_processed, n_cluster, radius=radius, method=tool, refinement=True) # For DLPFC dataset, we use optional refinement step.
elif tool in ['leiden', 'louvain']:
    clustering(adata_processed, n_cluster, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01, refinement=False)

t2 = time.time()
t = t2 - t1
size, peak = tracemalloc.get_traced_memory()
memory = peak / 1024 / 1024
tracemalloc.stop()

ARI = metrics.adjusted_rand_score(adata_processed.obs["domain"], adata_processed.obs["ground_truth"])
print(ARI)

adata_processed.write_h5ad("./GraphST_results.h5ad")
