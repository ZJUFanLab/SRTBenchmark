library(BASS)
library(Matrix)
library(dplyr)
library(SingleCellExperiment)

gc(reset = T)
start_time <- Sys.time()

sce <- readRDS("./dlpfc.rds")
counts <- assays(sce)$counts
meta <- colData(sce)
cntm <- list()
cntm[[1]] <- counts
xym <- list()
xym[[1]] <- meta[, c("row", "col")]  

# Set hyper-parameters
C <- 20 # set to the number of cell types, default 20
R <- 7 # set to the number of spatial domains

set.seed(0)
# Set up BASS object
BASS <- createBASSObject(cntm, xym, C = C, R = R,
                         beta_method = "SW", init_method = "mclust", 
                         nsample = 10000)
# Data pre-processing
BASS <- BASS.preprocess(BASS, doLogNormalize = TRUE,
                        geneSelect = "sparkx", nSE = 3000, doPCA = TRUE, 
                        scaleFeature = FALSE, doBatchCorrect = FALSE, nPC = 20)
# Run BASS algorithm
BASS <- BASS.run(BASS)
# post-process posterior samples
BASS <- BASS.postprocess(BASS)

zlabels <- BASS@results$z # predicted spatial domain labels
clabels <- BASS@results$c # predicted cell type labels
meta$"BASS" <- zlabels[[1]]   

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
used_memory <- memInfo1[12] - memInfo2[12] 

ari <- mclust::adjustedRandIndex(meta$"BASS", meta$"ground_truth") 
print(ari)    

saveRDS(BASS, file = "./BASS_result.rds")  
