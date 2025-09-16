# SRTBenchmark
**Benchmarking spatial clustering methods for spatially resolved transcriptomics**

<img src="./figures/Graphical Abstract.png">

- We performed a comprehensive benchmarking analysis of 14 spatial clustering methods using ~600 datasets across ten technologies and eight organs.

- We provided practical recommendations for method selection for spatially resolved transcriptomics across technologies, organs, and biological replicates, involving either cell type clustering or spatial domain identification.

- We systematically assessed the influence of data characteristics and spatial patterns on clustering accuracy and offered the optimal preprocessing pipeline covering normalization, log transformation, gene selection, standardization, and dimension reduction steps for spatial clustering methods.

## Datasets and Methods
Please refer to **[Dataset](./Dataset)** for dataset details and the **[Methods](./Methods)** folder for example code of each clustering method.

## Method Recommendations
We conducted a comprehensive benchmark of **14 spatial clustering methods** across multiple **technologies, organs, and biological replicates**, and provided **method recommendations** tailored to different application scenarios. We provide a comprehensive summary of benchmarking results based on resolution, spatial continuity, and biological replicates.

| Technology     | Organ                             | Replicates | Recommendation (top 5)                        |
|----------------|-----------------------------------|------------|-----------------------------------------------|
| High Resolution| Brain 	                           | No	        | [BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py), [SpaGCN](./Methods/SpaGCN_Sample.py), [STAGATE](./Methods/STAGATE_Sample.py)|
| Low Resolution | Brain	                            | No	        | [STAGATE](./Methods/STAGATE_Sample.py), [GraphST](./Methods/GraphST_Sample.py), [SEDR](./Methods/SEDR_Sample.py), [BASS](./Methods/BASS_Sample.R), [BayesSpace](./Methods/BayesSpace_Sample.R)|
| 10× Visium	    | High continuity (Brain、Intestine)| No	        | [STAGATE](./Methods/STAGATE_Sample.py), [DeepST](./Methods/DeepST_Sample.py), [SEDR](./Methods/SEDR_Sample.py), [CCST](./Methods/CCST_Sample.py), [GraphST](./Methods/GraphST_Sample.py)|
| 10× Visium     | Low continuity  (Liver、Lung)     | No	        | [PRECAST](./Methods/PRECAST_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [STAGATE](./Methods/STAGATE_Sample.py), [SpaGCN](./Methods/SpaGCN_Sample.py), [BayesSpace](./Methods/BayesSpace_Sample.R)|
| 10× Visium	    | Brain	                            | Yes	       | [DeepST](./Methods/DeepST_Sample.py), [Banksy](./Methods/Banksy_Sample.py), [SEDR](./Methods/SEDR_Sample.py), [GraphST](./Methods/GraphST_Sample.py), [STAGATE](./Methods/STAGATE_Sample.py)|
| MERFISH	       | Brain	                            | Yes        | [BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [SpaGCN](./Methods/SpaGCN_Sample.py), [PRECAST](./Methods/PRECAST_Sample.R), [Banksy](./Methods/Banksy_Sample.py)|

For each technology and organ, we recommend methods that showed the most consistent performance.

1.Real datasets are highlighted in `bold`.

2.Datasets with biological replicates are marked with `<sup>#</sup>`.

3.For application not explicitly covered, recommendations are extended from the summarized benchmarking insights.

### **10× Visium**
 
- **Brain**：STAGATE, GraphST, SEDR<sup>#</sup>, Banksy<sup>#</sup>, DeepST<sup>#</sup>

- **Breast**：stLearn, PRECAST, BayesSpace
  
- **Heart**：stLearn, BASS, PRECAST
  
- **Intestine**：CCST, DeepST, CellCharter

- **Kidney**：PRECAST, BayesSpace, SpaGCN
  
- **Liver**：STAGATE, PRECAST, stLearn
  
- **Lung**：PRECAST, stLearn, GraphST

- **Skin**：BayesSpace, STAGATE, BASS

### **ST**

- **Brain**：BASS, BayesSpace, PRECAST

- Breast：PRECAST, stLearn, STAGATE
  
- Heart：PRECAST, stLearn, STAGATE
  
- Intestine：BASS, BayesSpace, PRECAST

- Kidney：PRECAST, stLearn, STAGATE
  
- Liver：PRECAST, stLearn, STAGATE
  
- Lung：PRECAST, stLearn, STAGATE

- Skin：PRECAST, stLearn, STAGATE

### **Slide-seq**

- **Brain**：STAGATE, SpaGCN, BASS

- Breast：BASS, stLearn, Banksy
  
- Heart：BASS, stLearn, Banksy
  
- **Kidney**：BASS, stLearn, Banksy
  
- Liver：BASS, stLearn, Banksy
  
- **Lung**：BASS, stLearn, Banksy

### **Stereo-seq**

- **Brain**：BASS, SpaGCN, stLearn

- Breast：BASS, stLearn, Banksy
  
- Heart：BASS, stLearn, Banksy
  
- Kidney：BASS, stLearn, Banksy
  
- Liver：BASS, stLearn, Banksy
  
- Lung：BASS, stLearn, Banksy

### **seqFISH+**

- **Brain**：PRECAST, BASS, stLearn
  
- Breast：PRECAST, BASS, stLearn
  
- Heart：PRECAST, BASS, stLearn
  
- Kidney：PRECAST, BASS, stLearn
  
- Liver：PRECAST, BASS, stLearn
  
- Lung：PRECAST, BASS, stLearn

### **STARmap**

- **Brain**：BASS, stLearn, Banksy
 
- Breast：BASS, stLearn, Banksy
  
- Heart：BASS, stLearn, Banksy
  
- Kidney：BASS, stLearn, Banksy
  
- Liver：BASS, stLearn, Banksy
  
- Lung：BASS, stLearn, Banksy

### **MERFISH**

- **Brain**：BASS<sup>#</sup>, stLearn<sup>#</sup>, SpatialMGCN, PRECAST, Banksy, SpaGCN<sup>#</sup>

- Breast：BASS, stLearn, SpaGCN
  
- Heart：BASS, stLearn, SpaGCN
  
- Intestine：BASS, stLearn, SpaGCN
  
- Liver：BASS, stLearn, SpaGCN
  
- Lung：BASS, stLearn, SpaGCN

### **CosMx**

- **Brain**：BASS, Banksy, SEDR

- Breast：Banksy, BASS, stLearn
  
- Heart：Banksy, BASS, stLearn
  
- Intestine：BASS, Banksy, SEDR
  
- Liver：Banksy, BASS, stLearn
  
- **Lung**：Banksy, BASS, stLearn

### **Xenium**

- **Brain**：GraphST, SpaceFlow, BASS
  
- **Breast**：Banksy, stLearn, BASS
  
- Heart：Banksy, stLearn, BASS
  
- Intestine：GraphST, SpaceFlow, BASS
  
- Liver：Banksy, stLearn, BASS
  
- Lung：Banksy, stLearn, BASS

### **Visium HD**

- **Brain**：stLearn, SpaGCN, PRECAST

- Breast：stLearn, SpaGCN, Banksy
  
- Heart：stLearn, SpaGCN, Banksy
  
- **Intestine**：stLearn, Banksy, SEDR
  
- Liver：stLearn, SpaGCN, Banksy
  
- Lung：stLearn, SpaGCN, Banksy
  

## Optimized Preprocessing Pipelines
We tested our optimized preprocessing pipeline on the **10x Visium DLPFC** dataset to improve clustering accuracy.

We show default pipelines and **optimized pipelines (marked in bold)**.

Parameters that differ from the default settings are indicated with **<sup>#</sup>**, whereas parameters marked with **<sup>*</sup>** denote those for which modifications are expected to exert a substantial influence on performance.

To facilitate use of this optimized pipeline, we also provide versions of the methods using the optimized pipelines in the **[Methods](./Methods)** folder.

| Method        | Normalization | Log Transformation | Genes Selection | Standardization | Dimension Reduction |
|---------------|---------------|-----------------|----------------|----------------|------------------|
| **[BASS](./Methods/BASS_Sample_Optimized.R)**                           | Yes            | Yes             | 3000 SVGs            | No<sup>*</sup> | 20 PCs             |
| [Banksy](./Methods/Banksy_Sample.py)                                    | Yes<sup>*</sup>| No<sup>*</sup>  | 2000 HVGs<sup>*</sup>| No             | 20 PCs<sup>*</sup> |
| **[Banksy (optimized)](./Methods/Banksy_Sample_Optimized.py)**          | Yes            | Yes<sup>#</sup> | 3000 SVGs<sup>#</sup>| Yes<sup>#</sup>| 15 PCs             |
| [BayesSpace](./Methods/BayesSpace_Sample.R)                             | Yes            | Yes             | 2000 HVGs            | No             | 15 PCs             |
| **[BayesSpace(optimized)](./Methods/BayesSpace_Sample_Optimized.R)**    | Yes            | Yes             | 5000 HVGs<sup>#</sup>| No             | 20 PCs<sup>#</sup> |
| [CCST](./Methods/CCST_Sample.py)                                        | Yes<sup>*</sup>| No<sup>*</sup>  | All Genes<sup>*</sup>| Yes<sup>*</sup>| 200 PCs            |
| **[CCST(optimized)](./Methods/CCST_Sample_Optimized.py)**               | Yes            | No              | 2000 HVGs<sup>#</sup>| Yes            | 50 PCs<sup>#</sup> |
| [CellCharter](./Methods/CellCharter_Sample.py)                          | Yes            | Yes             | 5000 HVGs<sup>*</sup>| No             | No                 |
| **[CellCharter(optimized)](./Methods/CellCharter_Sample_Optimized.py)** | Yes            | No<sup>#</sup>  | 2000 HVGs<sup>#</sup>| No             | No                 |
| [DeepST](./Methods/DeepST_Sample.py)                                    | Yes            | Yes<sup>*</sup> | All Genes<sup>*</sup>| Yes<sup>*</sup>| 200 PCs<sup>*</sup>|
| **[DeepST(optimized)](./Methods/DeepST_Sample_Optimized.py)**           | Yes            | Yes             | 3000 SVGs<sup>#</sup>| Yes            | 50 PCs<sup>#</sup> |
| [GraphST](./Methods/GraphST_Sample.py)                                  | Yes<sup>*</sup>| Yes             | 3000 HVGs            | Yes            | No                 |
| **[GraphST(optimized)](./Methods/GraphST_Sample_Optimized.py)**         | Yes            | Yes             | 2000 HVGs<sup>#</sup>| No<sup>#</sup> | No                 |
| [PRECAST](./Methods/PRECAST_Sample.R)                                   | Yes            | No              | 2000 HVGs            | No             | 15 PCs             |
| **[PRECAST(optimized)](./Methods/PRECAST_Sample_Optimized.R)**          | Yes            | No              | 5000 HVGs<sup>#</sup>| No             | 20 PCs<sup>#</sup> |
| [SEDR](./Methods/SEDR_Sample.py)                                        | Yes            | No<sup>*</sup>  | 2000 HVGs            | Yes<sup>*</sup>| 200 PCs            |
| **[SEDR(optimized)](./Methods/SEDR_Sample_Optimized.py)**               | Yes            | No              | 3000 SVGs<sup>#</sup>| Yes            | 50 PCs<sup>#</sup> |
| **[STAGATE](./Methods/STAGATE_Sample_Optimized.py)**                    | Yes<sup>*</sup>| Yes<sup>*</sup> | 3000 HVGs<sup>*</sup>| No<sup>*</sup> | No                 |
| [SpaGCN](./Methods/SpaGCN_Sample.py)                                    | Yes<sup>*</sup>| Yes             | All Genes            | No             | No                 |
| **[SpaGCN(optimized)](./Methods/SpaGCN_Sample_Optimized.py)**           | Yes            | Yes             | All Genes            | Yes<sup>#</sup>| No                 |
| [SpaceFlow](./Methods/SpaceFlow_Sample.py)                              | Yes            | Yes             | 3000 HVGs            | No             | No                 |
| **[SpaceFlow(optimized)](./Methods/SpaceFlow_Sample_Optimized.py)**     | Yes            | No<sup>#</sup>  | 3000 SVGs<sup>#</sup>| Yes<sup>#</sup>| No                 |
| [SpatialMGCN](./Methods/SpatialMGCN_Sample.py)                          | Yes            | No<sup>*</sup>  | 3000 HVGs            | Yes            | No                 |
| **[SpatialMGCN(optimized)](./Methods/SpatialMGCN_Sample_Optimized.py)** | Yes            | No              | 3000 SVGs<sup>#</sup>| No<sup>#</sup> | No                 |
| [stLearn](./Methods/stLearn_Sample.py)                                  | Yes<sup>*</sup>| Yes             | All Genes<sup>*</sup>| No<sup>*</sup> | 50 PCs             |
| **[stLearn(optimized)](./Methods/stLearn_Sample_Optimized.py)**         | Yes            | No<sup>#</sup>  | 3000 SVGs<sup>#</sup>| No             | 20 PCs<sup>#</sup> |

## About
