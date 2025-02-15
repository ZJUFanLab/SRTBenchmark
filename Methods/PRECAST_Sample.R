library(PRECAST)
library(Seurat)

gc(reset = T)
start_time <- Sys.time()

sce <- readRDS("./dlpfc.rds")
meta_data <- sce@meta.data
all(c("row", "col") %in% colnames(meta_data)) 

# Prepare the PRECASTObject
set.seed(2023)
preobj <- CreatePRECASTObject(seuList = list(sce), selectGenesMethod = "HVGs", gene.number = 2000)
# Add the model setting
PRECASTObj <- AddAdjList(preobj, platform = "Visium") # optional "Visuim", "ST" and "Other_SRT"
PRECASTObj <- AddParSetting(PRECASTObj, Sigma_equal = FALSE, coreNum = 1, maxIter = 30, verbose = TRUE)    
# Fit PRECAST
K <- length(unique(sce$"ground_truth"))
PRECASTObj <- PRECAST(PRECASTObj, K = K)
resList <- PRECASTObj@resList
PRECASTObj <- SelectModel(PRECASTObj)

# Put the reults into a Seurat object seuInt.
seuInt <- PRECASTObj@seulist[[1]]
seuInt@meta.data$cluster <- factor(unlist(PRECASTObj@resList$cluster))
seuInt@meta.data$batch <- 1
seuInt <- Add_embed(PRECASTObj@resList$hZ[[1]], seuInt, embed_name = "PRECAST")
posList <- lapply(PRECASTObj@seulist, function(x) cbind(x$row, x$col))
seuInt <- Add_embed(posList[[1]], seuInt, embed_name = "position")
Idents(seuInt) <- factor(seuInt@meta.data$cluster)
                  
end_time <- Sys.time()
runtime <- end_time - start_time
gc()
memInfo1 <- gc()
memInfo1[11] 
memInfo1[12] 
gc(reset=TRUE)
memInfo2 <- gc()
memInfo2[11] 
memInfo2[12] 
peak<-memInfo1[12]-memInfo2[12] 
                  
saveRDS(seuInt, file = "./PRECAST_result.rds")  