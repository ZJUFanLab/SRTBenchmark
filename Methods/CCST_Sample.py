import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time
import tracemalloc
import numpy as np
import pandas as pd
import scanpy as sc
import pickle

from h5py import Dataset, Group
from sklearn import metrics
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv, DeepGraphInfomax, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from datetime import datetime 

def adata_preprocess(i_adata, min_cells=3, pca_n_comps=300):
    print('===== Preprocessing Data ')
    sc.pp.filter_genes(i_adata, min_cells=min_cells)
    adata_X = sc.pp.normalize_total(i_adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    adata_X = sc.pp.scale(adata_X)
    adata_X = sc.pp.pca(adata_X, n_comps=pca_n_comps)
    return adata_X

def get_adj(generated_data_fold, threshold):
    coordinates = np.load(generated_data_fold + 'coordinates.npy')
    if not os.path.exists(generated_data_fold):
        os.makedirs(generated_data_fold) 
    ############# get batch adjacent matrix
    cell_num = len(coordinates)

    ############ the distribution of distance 
    if 1:#not os.path.exists(generated_data_fold + 'distance_array.npy'):
        distance_list = []
        print ('calculating distance matrix, it takes a while')
        
        distance_list = []
        for j in range(cell_num):
            for i in range (cell_num):
                if i!=j:
                    distance_list.append(np.linalg.norm(coordinates[j]-coordinates[i]))

        distance_array = np.array(distance_list)
        #np.save(generated_data_fold + 'distance_array.npy', distance_array)
    else:
        distance_array = np.load(generated_data_fold + 'distance_array.npy')

    ###try different distance threshold, so that on average, each cell has x neighbor cells, see Tab. S1 for results
    from scipy import sparse
    import pickle
    import scipy.linalg

    for threshold in [threshold]:
        num_big = np.where(distance_array<threshold)[0].shape[0]
        print (threshold,num_big,str(num_big/(cell_num*2))) #300 22064 2.9046866771985256
        from sklearn.metrics.pairwise import euclidean_distances

        distance_matrix = euclidean_distances(coordinates, coordinates)
        distance_matrix_threshold_I = np.zeros(distance_matrix.shape)
        distance_matrix_threshold_W = np.zeros(distance_matrix.shape)
        for i in range(distance_matrix_threshold_I.shape[0]):
            for j in range(distance_matrix_threshold_I.shape[1]):
                if distance_matrix[i,j] <= threshold and distance_matrix[i,j] > 0:
                    distance_matrix_threshold_I[i,j] = 1
                    distance_matrix_threshold_W[i,j] = distance_matrix[i,j]
            
        ############### get normalized sparse adjacent matrix
        distance_matrix_threshold_I_N = np.float32(distance_matrix_threshold_I) ## do not normalize adjcent matrix
        distance_matrix_threshold_I_N_crs = sparse.csr_matrix(distance_matrix_threshold_I_N)
        with open(generated_data_fold + 'Adjacent', 'wb') as fp:
            pickle.dump(distance_matrix_threshold_I_N_crs, fp)

def get_type(cell_types, generated_data_fold):
    types_dic = []
    types_idx = []
    for t in cell_types:
        if not t in types_dic:
            types_dic.append(t) 
        id = types_dic.index(t)
        types_idx.append(id)

    n_types = max(types_idx) + 1 # start from 0
    np.save(generated_data_fold+'cell_types.npy', np.array(cell_types))
    np.savetxt(generated_data_fold+'types_dic.txt', np.array(types_dic), fmt='%s', delimiter='\t')
    
def get_data(data_path):
    data_file = data_path +'/'
    with open(data_file + 'Adjacent', 'rb') as fp:
        adj_0 = pickle.load(fp)
    X_data = np.load(data_file + 'features.npy')

    num_points = X_data.shape[0]
    adj_I = np.eye(num_points)
    adj_I = sparse.csr_matrix(adj_I)
    adj = (1-lambda_I)*adj_0 + lambda_I*adj_I

    cell_type_indeces = np.load(data_file + 'cell_types.npy', allow_pickle=True)
    
    return adj_0, adj, X_data, cell_type_indeces
def get_graph(adj, X):
    # create sparse matrix
    row_col = []
    edge_weight = []
    rows, cols = adj.nonzero()
    edge_nums = adj.getnnz() 
    for i in range(edge_nums):
        row_col.append([rows[i], cols[i]])
        edge_weight.append(adj.data[i])
    edge_index = torch.tensor(np.array(row_col), dtype=torch.long).T
    edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)

    graph_bags = []
    graph = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)  
    graph_bags.append(graph)
    return graph_bags
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.conv_2 = GCNConv(hidden_channels, hidden_channels)
        self.conv_3 = GCNConv(hidden_channels, hidden_channels)
        self.conv_4 = GCNConv(hidden_channels, hidden_channels)
        
        self.prelu = nn.PReLU(hidden_channels)
        
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv(x, edge_index, edge_weight=edge_weight)
        x = self.conv_2(x, edge_index, edge_weight=edge_weight)
        x = self.conv_3(x, edge_index, edge_weight=edge_weight)
        x = self.conv_4(x, edge_index, edge_weight=edge_weight)
        x = self.prelu(x)

        return x
class my_data():
    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr

def corruption(data):
    x = data.x[torch.randperm(data.x.size(0))]
    return my_data(x, data.edge_index, data.edge_attr)

def train_DGI(data_loader, in_channels):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    DGI_model = DeepGraphInfomax(
        hidden_channels=256,
        encoder=Encoder(in_channels=in_channels, hidden_channels=256),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)
    DGI_optimizer = torch.optim.Adam(DGI_model.parameters(), lr=1e-6)
    
    torch.backends.cudnn.enabled = False
    
    num_epoch = 5000
    import datetime
    start_time = datetime.datetime.now()
    for epoch in range(num_epoch):
        DGI_model.train()
        DGI_optimizer.zero_grad()
        
        DGI_all_loss = []
            
        for data in data_loader:
            data = data.to(device)
            pos_z, neg_z, summary = DGI_model(data=data)

            DGI_loss = DGI_model.loss(pos_z, neg_z, summary)
            DGI_loss.backward()
            DGI_all_loss.append(DGI_loss.item())
            DGI_optimizer.step()

        if ((epoch+1)%100) == 0:
            print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch+1, np.mean(DGI_all_loss)))

    end_time = datetime.datetime.now()
    DGI_filename =  f"{OUTPUT_PATH}/" +'DGI_lambdaI_' + str(lambda_I) + '_epoch' + str(num_epoch) + '.pth.tar'
    torch.save(DGI_model.state_dict(), DGI_filename)
    print('Training time in seconds: ', (end_time-start_time).seconds)
    return DGI_model
def PCA_process(X, nps):
    from sklearn.decomposition import PCA
    print('Shape of data to PCA:', X.shape)
    pca = PCA(n_components=nps)
    X_PC = pca.fit_transform(X)     #等价于pca.fit(X) pca.transform(X)
    print('Shape of data output by PCA:', X_PC.shape)
    print('PCA recover:', pca.explained_variance_ratio_.sum())
    return X_PC
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
def Kmeans_cluster(X_embedding, n_clusters, merge=False):
    cluster_model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=100, max_iter=1000, tol=1e-6)
    cluster_labels = cluster_model.fit_predict(X_embedding)

    # merge clusters with less than 3 cells
    if merge:
        cluster_labels = merge_cluser(X_embedding, cluster_labels)

    score = metrics.silhouette_score(X_embedding, cluster_labels, metric='euclidean')
    
    return cluster_labels, score
def res_search_fixed_clus(cluster_type, adata, fixed_clus_count, increment=0.02):
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]
        
        return:
            resolution[int]
    '''
    if cluster_type == 'leiden':
        for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            if count_unique_leiden == fixed_clus_count:
                break
    elif cluster_type == 'louvain':
        for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique_louvain = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            if count_unique_louvain == fixed_clus_count:
                break
    return res

tracemalloc.start()
t1 = time.time()
adata = sc.read_visium("./DLPFC/151673", count_file="filtered_feature_bc_matrix.h5", load_images=True)
adata.var_names_make_unique()
df_meta = pd.read_csv("./DLPFC/151673/metadata.tsv", sep='\t')
adata.obs["ground_truth"] = df_meta.loc[adata.obs_names, "layer_guess"]
adata = adata[~pd.isnull(adata.obs["ground_truth"])]
n_cluster = len(set(adata.obs["ground_truth"]))

features = adata_preprocess(adata, min_cells=5, pca_n_comps=200)
gene_ids = adata.var["gene_ids"]
coordinates = adata.obsm["spatial"]

generated_data_fold = "" # Customize the save path
np.save(generated_data_fold + 'features.npy', features)
np.save(generated_data_fold + 'coordinates.npy', np.array(coordinates))

get_adj(generated_data_fold, threshold=200) # Set parameter threshold in range (50,950) for other datasets based on the number of adjacent spots for each spot
cell_types = adata.obs["ground_truth"]
get_type(cell_types, generated_data_fold)

lambda_I = 0.3 # Set parameter lambda_I to 0.3 for 10x and ST, 0.8 for others
batch_size = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dj_0, adj, X_data, cell_type_indeces = get_data(generated_data_fold)
num_cell = X_data.shape[0]
num_feature = X_data.shape[1]
print("Adj:", adj.shape, "Edges:", len(adj.data))
print("X:", X_data.shape)    
num_epoch = 5000
print("-----------Deep Graph Infomax-------------")
data_list = get_graph(adj, X_data)
data_loader = DataLoader(data_list, batch_size=batch_size)
DGI_model = train_DGI(data_loader=data_loader, in_channels=num_feature)

for data in data_loader:
    data.to(device)
    X_embedding, _, _ = DGI_model(data)
    X_embedding = X_embedding.cpu().detach().numpy()
    X_embedding_filename = generated_data_fold + "lambdaI" + str(lambda_I) + "_epoch" + str(num_epoch) + "_Embed_X.npy"
    np.save(X_embedding_filename, X_embedding)

cluster_type = "kmeans"
print("-----------Clustering-------------")
X_embedding = np.load(X_embedding_filename)
if cluster_type == "kmeans":             
    X_embedding = PCA_process(X_embedding, nps=30)
    print("Shape of data to cluster:", X_embedding.shape)
    cluster_labels, score = Kmeans_cluster(X_embedding, n_cluster)
    adata.obs["kmeans"] = cluster_labels
else:
    adata_ = ad.AnnData(X_embedding)
    sc.tl.pca(adata_, n_comps=50, svd_solver="arpack")
    sc.pp.neighbors(adata_, n_neighbors=20, n_pcs=50)
    eval_resolution = res_search_fixed_clus(cluster_type, adata_, n_clusters)
    if cluster_type == "leiden":
        sc.tl.leiden(adata_, key_added="CCST_leiden", resolution=eval_resolution)
        cluster_labels = np.array(adata_.obs["leiden"])
        adata.obs["leiden"] = cluster_labels
        adata.obs["leiden"] = adata.obs["leiden"].astype("category")
        results_df = calculate_clustering_matrix(adata.obs["leiden"], adata.obs["ground_truth"], sample, "CCST_leiden", "t")
        df = df.append(results_df, ignore_index=True)

    if cluster_type == "louvain":
        sc.tl.louvain(adata_, key_added="CCST_louvain", resolution=eval_resolution)
        cluster_labels = np.array(adata_.obs["louvain"])
        cluster_labels = [ int(x) for x in cluster_labels]
        adata.obs["louvain"] = cluster_labels
        adata.obs["louvain"] = adata.obs["louvain"].astype("category")
        score = False
        results_df = calculate_clustering_matrix(adata.obs["louvain"], adata.obs["ground_truth"], sample, "CCST_louvain", "t")
        df = df.append(results_df, ignore_index=True)    

adata.obs["CCST"] = cluster_labels
adata.obs["CCST"] = adata.obs["CCST"].astype("category")

t2 = time.time()
t = t2 - t1
size, peak = tracemalloc.get_traced_memory()
memory = peak / 1024 / 1024
tracemalloc.stop()

ARI = metrics.adjusted_rand_score(adata.obs["CCST"], adata.obs["ground_truth"])
print(ARI)

adata.write_h5ad(generated_data_fold + "/CCST_results.h5ad")
