# SRTBenchmark
**Benchmarking spatial clustering methods for spatially resolved transcriptomics**

<img src="./figures/Graphical Abstract.png">

· We performed a comprehensive benchmarking analysis of 14 spatial clustering methods using ~600 datasets across ten technologies and eight organs.

· We provided practical recommendations for method selection for spatially resolved transcriptomics across technologies, organs, and biological replicates, involving either cell type clustering or spatial domain identification.

· We systematically assessed the influence of data characteristics and spatial patterns on clustering accuracy and offered the optimal preprocessing pipeline covering normalization, log transformation, gene selection, standardization, and dimension reduction steps for spatial clustering methods.

## Datasets and Methods
Please refer to **[Table1](./Table1_Dataset.xlsx)** for dataset details and the **[Methods](./Methods)** folder for example code of each clustering method.

## Overall Performance
We conducted a comprehensive benchmark of **14 spatial clustering methods** across multiple **technologies, organs, and biological replicates**, and provided **method recommendations** tailored to different application scenarios.

| Technology     | Organ          | Replicates | Recommendation (top 5)                        |
|----------------|----------------|------------|-----------------------------------------------|
| High Resolution| Brain 	        | No	        | BASS, stLearn, Banksy, SpaGCN, STAGATE        |
| Low Resolution | Brain	         | No	        | STAGATE, GraphST, SEDR, BASS, BayesSpace      |
| 10× Visium	    | High continuity| No	        | STAGATE, DeepST, SEDR, CCST, GraphST          |
| 10× Visium     | Low continuity | No	        | PRECAST, stLearn, STAGATE, SpaGCN, BayesSpace |
| 10× Visium	    | Brain	         | Yes	       | DeepST, Banksy, SEDR, GraphST, STAGATE        |
| MERFISH	       | Brain	         | Yes        | BASS, stLearn, SpaGCN, PRECAST, Banksy        |

### **10× Visium**
 
- **Brain**：STAGATE, GraphST, SEDR<sup>#</sup>, Banksy<sup>#</sup>, DeepST<sup>#</sup>

- **Breast**：stLearn, PRECAST, BayesSpace, SpaGCN, STAGATE
  
- **Heart**：stLearn, BASS, PRECAST, GraphST, STAGATE
  
- **Intestine**：CCST, DeepST, CellCharter, stLearn, STAGATE
  
- **Liver**：STAGATE, PRECAST, stLearn, SpatialMGCN, DeepST
  
- **Lung**：PRECAST, stLearn, GraphST, STAGATE, BayesSpace

### **ST**

- **Brain**：BASS, BayesSpace, PRECAST, CCST, stLearn

- Breast：PRECAST, stLearn, STAGATE, SpaGCN, BayesSpace
  
- Heart：PRECAST, stLearn, STAGATE, SpaGCN, BayesSpace
  
- Intestine：STAGATE, DeepST, SEDR, CCST, GraphST
  
- Liver：PRECAST, stLearn, STAGATE, SpaGCN, BayesSpace
  
- Lung：PRECAST, stLearn, STAGATE, SpaGCN, BayesSpace

### **Slide-seq**

- **Brain**：STAGATE, SpaGCN, BASS, CCST, SpaceFlow

### **Stereo-seq**

- **Brain**：BASS, SpaGCN, stLearn, STAGATE, SpatialMGCN

### **seqFISH+**

- **Brain**：PRECAST, BASS, stLearn, CellCharter, SpaGCN

### **STARmap**

- **Brain**：BASS, stLearn, Banksy, PRECAST, CellCharter

### **MERFISH**

- **Brain**：BASS<sup>#</sup>, stLearn<sup>#</sup>, SpatialMGCN, PRECAST, Banksy, SpaGCn<sup>#</sup>

### **CosMx**

- **Brain**：BASS, Banksy, SEDR, DeepST, CellCharter
  
- **Lung**：Banksy, BASS, stLearn, DeepST, SpatialMGCN

### **Xenium**

- **Brain**：GraphST, SpaceFlow, BASS, STAGATE, Banksy
  
- **Breast**：Banksy, stLearn, BASS, SpatialMGCN, SEDR

### **Visium HD**

- **Intestine**：stLearn, Banksy, SEDR, SpaGCN, SpatialMGCN 
  

## Optimal Preprocessing Pipelines
We tested our optimized preprocessing pipeline on the **10x Visium DLPFC** dataset to improve clustering accuracy.  
To facilitate use of this optimized pipeline, we also provide versions of the methods using the optimized pipelines in the **Methods** folder.

| Method        | Normalization | Log Transformation | Genes Selection | Standardization | Dimension Reduction |
|---------------|---------------|-----------------|----------------|----------------|------------------|
| BASS          | Yes           | Yes             | 3000 SVGs      | No             | 20 PCs           |
| Banksy        | Yes           | Yes             | 3000 SVGs      | Yes            | 15 PCs           |
| BayesSpace    | Yes           | Yes             | 5000 HVGs      | No             | 20 PCs           |
| CCST          | Yes           | No              | 2000 HVGs      | Yes            | 50 PCs           |
| CellCharter   | Yes           | No              | 2000 HVGs      | No             | No               |
| DeepST        | Yes           | Yes             | 3000 SVGs      | Yes            | 50 PCs           |
| GraphST       | Yes           | Yes             | 2000 HVGs      | No             | No               |
| PRECAST       | Yes           | No              | 5000 HVGs      | No             | 20 PCs           |
| SEDR          | Yes           | No              | 3000 SVGs      | Yes            | 50 PCs           |
| STAGATE       | Yes           | Yes             | 3000 HVGs      | No             | No               |
| SpaGCN        | Yes           | Yes             | All Genes      | Yes            | No               |
| SpaceFlow     | Yes           | No              | 3000 SVGs      | Yes            | No               |
| SpatialMGCN   | Yes           | No              | 3000 SVGs      | No             | No               |
| stLearn       | Yes           | No              | 3000 SVGs      | No             | 20 PCs           |

