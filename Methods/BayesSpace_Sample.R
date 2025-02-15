library(BayesSpace)
library(dplyr)
library(Matrix)

gc(reset = T)
start_time <- Sys.time()

sce <- readRDS("./dlpfc.rds")

## Processing the data
set.seed(102)
sce <- spatialPreprocess(sce, platform = 'Visium', n.PCs = 15, n.HVGs = 2000) #choose platform = c("Visium", "ST")
q <- 7  # set to the number of clusters
d <- 15  # Number of PCs

## Run BayesSpace clustering
set.seed(104)
dlpfc <- spatialCluster(dlpfc, q=q, d=d, platform='Visium', nrep=50000, gamma=3, save.chain=TRUE) #set the smoothing parameter gamma=3 for Visium and gamma=2 for ST

ari <- mclust::adjustedRandIndex(dlpfc$spatial.cluster, dlpfc$'ground_truth') 
print(ari)      

saveRDS(dlpfc, file = "./BayesSpace_result.rds")  
