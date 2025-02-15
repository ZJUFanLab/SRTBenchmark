warnings.filterwarnings("ignore")
import os
import random
import sys
import time
import tracemalloc
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch.optim as optim
from models import Spatial_MGCN
from pathlib import Path
from utils import features_construct_graph, spatial_construct_graph1
from utils import *

def normalize(adata, highly_genes=3000):
    print("start select HVGs")
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata
def train():
    model.train()
    optimizer.zero_grad()
    com1, com2, emb, pi, disp, mean = model(features, sadj, fadj)
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)
    con_loss = consistency_loss(com1, com2)
    total_loss = alpha * zinb_loss + beta * con_loss + gamma * reg_loss
    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values
    total_loss.backward()
    optimizer.step()
    return emb, mean, zinb_loss, reg_loss, con_loss, total_loss

tracemalloc.start()
t1 = time.time()

adata = sc.read_visium("./DLPFC/151673", count_file="filtered_feature_bc_matrix.h5", load_images=True)
adata.var_names_make_unique()
df_meta = pd.read_csv("./DLPFC/151673/metadata.tsv", sep='\t')
adata.obs["ground_truth"] = df_meta.loc[adata.obs_names, "layer_guess"]
adata = adata[~pd.isnull(adata.obs["ground_truth"])]
n_cluster = len(set(adata.obs["ground_truth"]))

epochs = 200
lr = 0.001
weight_decay = 5e-4
k = 14
radius = 560
nhid1 = 128
nhid2 = 64
dropout = 0
alpha = 1
beta = 10
gamma = 0.1
no_cuda = False
no_seed = False
seed = 100
fdim = 3000

adata = normalize(adata, highly_genes=3000)
fadj = features_construct_graph(adata.X, k=k)
sadj, graph_nei, graph_neg = spatial_construct_graph1(adata, radius=radius)
adata.obsm["fadj"] = fadj
adata.obsm["sadj"] = sadj
adata.obsm["graph_nei"] = graph_nei.numpy()
adata.obsm["graph_neg"] = graph_neg.numpy()
features = torch.FloatTensor(adata.X.todense())
fadj = adata.obsm["fadj"]
sadj = adata.obsm["sadj"]
nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
graph_nei = torch.LongTensor(adata.obsm["graph_nei"])
graph_neg = torch.LongTensor(adata.obsm["graph_neg"])

cuda = not no_cuda and torch.cuda.is_available()
use_seed = not no_seed
if cuda:
    features = features.cuda()
    sadj = sadj.cuda()
    fadj = fadj.cuda()
    graph_nei = graph_nei.cuda()
    graph_neg = graph_neg.cuda()

np.random.seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(10seed0)
if not no_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
print(dataset, ' ', lr, ' ', alpha, ' ', beta, ' ', gamma)
model = Spatial_MGCN(nfeat=fdim,nhid1=nhid1,nhid2=hid2,dropout=dropout)
if cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
epoch_max = 0
ari_max = 0
idx_max = []
mean_max = []
emb_max = []

for epoch in range(epochs):
    emb, mean, zinb_loss, reg_loss, con_loss, total_loss = train()
    print(dataset, ' epoch: ', epoch, ' zinb_loss = {:.2f}'.format(zinb_loss),
          ' reg_loss = {:.2f}'.format(reg_loss), ' con_loss = {:.2f}'.format(con_loss),
          ' total_loss = {:.2f}'.format(total_loss))
    kmeans = KMeans(n_clusters=n_cluster).fit(emb)
    idx = kmeans.labels_
    ari_res = metrics.adjusted_rand_score(labels, idx)
    if ari_res > ari_max:
        ari_max = ari_res
        epoch_max = epoch
        idx_max = idx
        mean_max = mean
        emb_max = emb  

adata.obs["idx"] = idx_max.astype(str)
adata.obsm["emb"] = emb_max
adata.obsm["mean"] = mean_max
adata.layers["X"] = adata.X
adata.layers["mean"] = mean_max
pd.DataFrame(emb_max).to_csv("./Spatial_MGCN_emb.csv")
pd.DataFrame(idx_max).to_csv("./Spatial_MGCN_idx.csv")

t2 = time.time()
t = t2 - t1
size, peak = tracemalloc.get_traced_memory()
memory = peak / 1024 / 1024
tracemalloc.stop()

ARI = metrics.adjusted_rand_score(adata.obs["idx"], adata.obs["ground_truth"])
print(ARI)

adata.write_h5ad("./SpatialMGCN_results.h5ad")