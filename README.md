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

In detail, we provide recommendations of the most accurate methods tailored to different spatial transcriptomics technologies and organ types.

1.For application scenarios where real-world datasets are available, the recommended methods are indicated in `bold`.

2.For application scenarios that include biological replicates, the recommended methods are marked with <sup>`#`</sup>.

3.When real datasets were not included in our analyses, recommendations are inferred from the summarized benchmarking results.

### **10× Visium** [Low Resolution：55 μm]
 
- **`Brain`** (High continuity)：[STAGATE](./Methods/STAGATE_Sample.py), [GraphST](./Methods/GraphST_Sample.py), [SEDR](./Methods/SEDR_Sample.py)<sup>#</sup>, [Banksy](./Methods/Banksy_Sample.py)<sup>#</sup>, [DeepST](./Methods/DeepST_Sample.py)<sup>#</sup>

- **`Breast`** (Low continuity)：[stLearn](./Methods/stLearn_Sample.py), [PRECAST](./Methods/PRECAST_Sample.R), [BayesSpace](./Methods/BayesSpace_Sample.R)
  
- **`Heart`** (Low continuity)：[stLearn](./Methods/stLearn_Sample.py), [BASS](./Methods/BASS_Sample.R), [PRECAST](./Methods/PRECAST_Sample.R)
  
- **`Intestine`** (High continuity)：[CCST](./Methods/CCST_Sample.py), [DeepST](./Methods/DeepST_Sample.py), [CellCharter](./Methods/CellCharter_Sample.py)

- **`Kidney`** (Low continuity)：[PRECAST](./Methods/PRECAST_Sample.R), [BayesSpace](./Methods/BayesSpace_Sample.R), [SpaGCN](./Methods/SpaGCN_Sample.py)
  
- **`Liver`** (Low continuity)：[STAGATE](./Methods/STAGATE_Sample.py), [PRECAST](./Methods/PRECAST_Sample.R), [stLearn](./Methods/stLearn_Sample.py)
  
- **`Lung`** (Low continuity)：[PRECAST](./Methods/PRECAST_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [GraphST](./Methods/GraphST_Sample.py)

- **`Skin`** (Low continuity)：[BayesSpace](./Methods/BayesSpace_Sample.R)|, [STAGATE](./Methods/STAGATE_Sample.py), [BASS](./Methods/BASS_Sample.R)

### **Spatial Transcriptomic (ST)** [Low Resolution：100 μm]

- **`Brain`** (High continuity)：[BASS](./Methods/BASS_Sample.R), [BayesSpace](./Methods/BayesSpace_Sample.R), [PRECAST](./Methods/PRECAST_Sample.R)

- Breast (Low continuity)：[PRECAST](./Methods/PRECAST_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [STAGATE](./Methods/STAGATE_Sample.py)
  
- Heart (Low continuity)：[PRECAST](./Methods/PRECAST_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [STAGATE](./Methods/STAGATE_Sample.py)
  
- Intestine (High continuity)：[BASS](./Methods/BASS_Sample.R), [BayesSpace](./Methods/BayesSpace_Sample.R), [PRECAST](./Methods/PRECAST_Sample.R)

- Kidney (Low continuity)：[PRECAST](./Methods/PRECAST_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [STAGATE](./Methods/STAGATE_Sample.py)
  
- Liver (Low continuity)：[PRECAST](./Methods/PRECAST_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [STAGATE](./Methods/STAGATE_Sample.py)
  
- Lung (Low continuity)：[PRECAST](./Methods/PRECAST_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [STAGATE](./Methods/STAGATE_Sample.py)

- Skin (Low continuity)：[PRECAST](./Methods/PRECAST_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [STAGATE](./Methods/STAGATE_Sample.py)

### **Slide-seq** [High Resolution：10 μm]

- **`Brain`** (High continuity)：[STAGATE](./Methods/STAGATE_Sample.py), [SpaGCN](./Methods/SpaGCN_Sample.py), [BASS](./Methods/BASS_Sample.R)

- Breast (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)
  
- Heart (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)

- Intestine (High continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)
    
- **`Kidney`** (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)
  
- Liver (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)
  
- **`Lung`** (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)

- Skin (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)

### **Spatial enhanced resolution omics-sequencing (Stereo-seq)** [High Resolution：0.22 μm]

- **`Brain`** (High continuity)：[BASS](./Methods/BASS_Sample.R), [SpaGCN](./Methods/SpaGCN_Sample.py), [stLearn](./Methods/stLearn_Sample.py)

- Breast (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)
  
- Heart (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)

- Intestine (High continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)
  
- Kidney (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)
  
- Liver (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)
  
- Lung (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)

- Skin (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)

### **Sequential fluorescence in situ hybridization (seqFISH+)** [High Resolution：≦ 0.1 μm]

- **`Brain`** (High continuity)：[PRECAST](./Methods/PRECAST_Sample.R), [BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py)
  
- Breast (Low continuity)：[PRECAST](./Methods/PRECAST_Sample.R), [BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py)
  
- Heart (Low continuity)：[PRECAST](./Methods/PRECAST_Sample.R), [BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py)
  
- Intestine (High continuity)：[PRECAST](./Methods/PRECAST_Sample.R), [BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py)

- Kidney (Low continuity)：[PRECAST](./Methods/PRECAST_Sample.R), [BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py)
  
- Liver (Low continuity)：[PRECAST](./Methods/PRECAST_Sample.R), [BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py)
  
- Lung (Low continuity)：[PRECAST](./Methods/PRECAST_Sample.R), [BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py)

- Skin (Low continuity)：[PRECAST](./Methods/PRECAST_Sample.R), [BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py)

### **Spatially-resolved transcript amplicon readout mapping (STARmap)** [High Resolution：0.2～0.3 μm]

- **`Brain`** (High continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)
 
- Breast (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)
  
- Heart (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)
  
- Intestine (High continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)

- Kidney (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)
  
- Liver (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)
  
- Lung (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)

- Skin (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py)

### **Multiplexed error-robust fluorescence in situ hybridization (MERFISH)** [High Resolution：≦ 0.1 μm]

- **`Brain`** (High continuity)：[BASS](./Methods/BASS_Sample.R)<sup>#</sup>, [stLearn](./Methods/stLearn_Sample.py)<sup>#</sup>, [SpatialMGCN](./Methods/SpatialMGCN_Sample.py), [PRECAST](./Methods/PRECAST_Sample.R), [Banksy](./Methods/Banksy_Sample.py), [SpaGCN](./Methods/SpaGCN_Sample.py)<sup>#</sup>

- Breast (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [SpaGCN](./Methods/SpaGCN_Sample.py)
  
- Heart (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [SpaGCN](./Methods/SpaGCN_Sample.py)
  
- Intestine (High continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [SpaGCN](./Methods/SpaGCN_Sample.py)

- Kidney (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [SpaGCN](./Methods/SpaGCN_Sample.py)
  
- Liver (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [SpaGCN](./Methods/SpaGCN_Sample.py)
  
- Lung (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [SpaGCN](./Methods/SpaGCN_Sample.py)

- Skin (Low continuity)：[BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py), [SpaGCN](./Methods/SpaGCN_Sample.py)

### **CosMx™ Spatial Molecular Imager (CosMx)** [High Resolution：≦ 0.1 μm]

- **`Brain`** (High continuity)：[BASS](./Methods/BASS_Sample.R), [Banksy](./Methods/Banksy_Sample.py), [SEDR](./Methods/SEDR_Sample.py)

- Breast (Low continuity)：[Banksy](./Methods/Banksy_Sample.py), [BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py)
  
- Heart (Low continuity)：[Banksy](./Methods/Banksy_Sample.py), [BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py)
  
- Intestine (High continuity)：[BASS](./Methods/BASS_Sample.R), [Banksy](./Methods/Banksy_Sample.py), [SEDR](./Methods/SEDR_Sample.py)

- Kidney (Low continuity)：[Banksy](./Methods/Banksy_Sample.py), [BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py)

- Liver (Low continuity)：[Banksy](./Methods/Banksy_Sample.py), [BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py)
  
- **`Lung`** (Low continuity)：[Banksy](./Methods/Banksy_Sample.py), [BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py)

- Skin (Low continuity)：[Banksy](./Methods/Banksy_Sample.py), [BASS](./Methods/BASS_Sample.R), [stLearn](./Methods/stLearn_Sample.py)

### **10x Xenium in situ (Xenium)** [High Resolution：≦ 0.1 μm]

- **`Brain`** (High continuity)：[GraphST](./Methods/GraphST_Sample.py), [SpaceFlow](./Methods/SpaceFlow_Sample.py), [BASS](./Methods/BASS_Sample.R)
  
- **`Breast`** (Low continuity)：[Banksy](./Methods/Banksy_Sample.py), [stLearn](./Methods/stLearn_Sample.py), [BASS](./Methods/BASS_Sample.R)
  
- Heart (Low continuity)：[Banksy](./Methods/Banksy_Sample.py), [stLearn](./Methods/stLearn_Sample.py), [BASS](./Methods/BASS_Sample.R)
  
- Intestine (High continuity)：[GraphST](./Methods/GraphST_Sample.py), [SpaceFlow](./Methods/SpaceFlow_Sample.py), [BASS](./Methods/BASS_Sample.R)

- Kidney (Low continuity)：[Banksy](./Methods/Banksy_Sample.py), [stLearn](./Methods/stLearn_Sample.py), [BASS](./Methods/BASS_Sample.R)
  
- Liver (Low continuity)：[Banksy](./Methods/Banksy_Sample.py), [stLearn](./Methods/stLearn_Sample.py), [BASS](./Methods/BASS_Sample.R)
  
- Lung (Low continuity)：[Banksy](./Methods/Banksy_Sample.py), [stLearn](./Methods/stLearn_Sample.py), [BASS](./Methods/BASS_Sample.R)

- Skin (Low continuity)：[Banksy](./Methods/Banksy_Sample.py), [stLearn](./Methods/stLearn_Sample.py), [BASS](./Methods/BASS_Sample.R)

### **10x Visium HD (Visium HD)** [High Resolution：2 μm]

- **`Brain`** (High continuity)：[stLearn](./Methods/stLearn_Sample.py), [SpaGCN](./Methods/SpaGCN_Sample.py), [PRECAST](./Methods/PRECAST_Sample.R)

- Breast (Low continuity)：[stLearn](./Methods/stLearn_Sample.py), [SpaGCN](./Methods/SpaGCN_Sample.py), [Banksy](./Methods/Banksy_Sample.py)
  
- Heart (Low continuity)：[stLearn](./Methods/stLearn_Sample.py), [SpaGCN](./Methods/SpaGCN_Sample.py), [Banksy](./Methods/Banksy_Sample.py)
  
- **`Intestine`** (High continuity)：[stLearn](./Methods/stLearn_Sample.py), [Banksy](./Methods/Banksy_Sample.py), [SEDR](./Methods/SEDR_Sample.py)

- Kidney (Low continuity)：[stLearn](./Methods/stLearn_Sample.py), [SpaGCN](./Methods/SpaGCN_Sample.py), [Banksy](./Methods/Banksy_Sample.py)
  
- Liver (Low continuity)：[stLearn](./Methods/stLearn_Sample.py), [SpaGCN](./Methods/SpaGCN_Sample.py), [Banksy](./Methods/Banksy_Sample.py)
  
- Lung (Low continuity)：[stLearn](./Methods/stLearn_Sample.py), [SpaGCN](./Methods/SpaGCN_Sample.py), [Banksy](./Methods/Banksy_Sample.py)

- Skin (Low continuity)：[stLearn](./Methods/stLearn_Sample.py), [SpaGCN](./Methods/SpaGCN_Sample.py), [Banksy](./Methods/Banksy_Sample.py)
  

## Optimized Preprocessing Pipelines
We tested our optimized preprocessing pipeline on the **10x Visium DLPFC** dataset to improve clustering accuracy.

We show default pipelines and **optimized pipelines** (marked in `bold`).

Parameters that differ from the default settings are indicated with <sup>`#`</sup>.

To facilitate use of this optimized pipeline, we also provide versions of the methods using the optimized pipelines in the **[Methods](./Methods)** folder.

| Method        | Normalization | Log Transformation | Genes Selection | Standardization | Dimension Reduction |
|---------------|---------------|-----------------|----------------|----------------|------------------|
| [BASS (default)](./Methods/BASS_Sample.R)                               | Yes            | Yes             | 3000 SVGs            | No             | 20 PCs             |
| **[BASS (optimized)](./Methods/BASS_Sample_Optimized.R)**               | Yes            | Yes             | 3000 SVGs            | No             | 20 PCs             |
| [Banksy (default)](./Methods/Banksy_Sample.py)                          | Yes            | No              | 2000 HVGs            | No             | 20 PCs             |
| **[Banksy (optimized)](./Methods/Banksy_Sample_Optimized.py)**          | Yes            | Yes<sup>#</sup> | 3000 SVGs<sup>#</sup>| Yes<sup>#</sup>| 15 PCs             |
| [BayesSpace (default)](./Methods/BayesSpace_Sample.R)                   | Yes            | Yes             | 2000 HVGs            | No             | 15 PCs             |
| **[BayesSpace (optimized)](./Methods/BayesSpace_Sample_Optimized.R)**   | Yes            | Yes             | 5000 HVGs<sup>#</sup>| No             | 20 PCs<sup>#</sup> |
| [CCST (default)](./Methods/CCST_Sample.py)                              | Yes            | No              | All Genes            | Yes            | 200 PCs            |
| **[CCST (optimized)](./Methods/CCST_Sample_Optimized.py)**              | Yes            | No              | 2000 HVGs<sup>#</sup>| Yes            | 50 PCs<sup>#</sup> |
| [CellCharter (default)](./Methods/CellCharter_Sample.py)                | Yes            | Yes             | 5000 HVGs            | No             | No                 |
| **[CellCharter (optimized)](./Methods/CellCharter_Sample_Optimized.py)**| Yes            | No<sup>#</sup>  | 2000 HVGs<sup>#</sup>| No             | No                 |
| [DeepST (default)](./Methods/DeepST_Sample.py)                          | Yes            | Yes             | All Genes            | Yes            | 200 PCs            |
| **[DeepST (optimized)](./Methods/DeepST_Sample_Optimized.py)**          | Yes            | Yes             | 3000 SVGs<sup>#</sup>| Yes            | 50 PCs<sup>#</sup> |
| [GraphST (default)](./Methods/GraphST_Sample.py)                        | Yes            | Yes             | 3000 HVGs            | Yes            | No                 |
| **[GraphST (optimized)](./Methods/GraphST_Sample_Optimized.py)**        | Yes            | Yes             | 2000 HVGs<sup>#</sup>| No<sup>#</sup> | No                 |
| [PRECAST (default)](./Methods/PRECAST_Sample.R)                         | Yes            | No              | 2000 HVGs            | No             | 15 PCs             |
| **[PRECAST (optimized)](./Methods/PRECAST_Sample_Optimized.R)**         | Yes            | No              | 5000 HVGs<sup>#</sup>| No             | 20 PCs<sup>#</sup> |
| [SEDR](./Methods/SEDR_Sample.py)                                        | Yes            | No              | 2000 HVGs            | Yes            | 200 PCs            |
| **[SEDR (optimized)](./Methods/SEDR_Sample_Optimized.py)**              | Yes            | No              | 3000 SVGs<sup>#</sup>| Yes            | 50 PCs<sup>#</sup> |
| [STAGATE (default)](./Methods/STAGATE_Sample.py)                        | Yes            | Yes             | 3000 HVGs            | No             | No                 |
| **[STAGATE (optimized)](./Methods/STAGATE_Sample_Optimized.py)**        | Yes            | Yes             | 3000 HVGs            | No             | No                 |
| [SpaGCN (default)](./Methods/SpaGCN_Sample.py)                          | Yes            | Yes             | All Genes            | No             | No                 |
| **[SpaGCN (optimized)](./Methods/SpaGCN_Sample_Optimized.py)**          | Yes            | Yes             | All Genes            | Yes<sup>#</sup>| No                 |
| [SpaceFlow (default)](./Methods/SpaceFlow_Sample.py)                    | Yes            | Yes             | 3000 HVGs            | No             | No                 |
| **[SpaceFlow (optimized)](./Methods/SpaceFlow_Sample_Optimized.py)**    | Yes            | No<sup>#</sup>  | 3000 SVGs<sup>#</sup>| Yes<sup>#</sup>| No                 |
| [SpatialMGCN (default)](./Methods/SpatialMGCN_Sample.py)                | Yes            | No              | 3000 HVGs            | Yes            | No                 |
| **[SpatialMGCN (optimized)](./Methods/SpatialMGCN_Sample_Optimized.py)**| Yes            | No              | 3000 SVGs<sup>#</sup>| No<sup>#</sup> | No                 |
| [stLearn (default)](./Methods/stLearn_Sample.py)                        | Yes            | Yes             | All Genes            | No             | 50 PCs             |
| **[stLearn (optimized)](./Methods/stLearn_Sample_Optimized.py)**        | Yes            | No<sup>#</sup>  | 3000 SVGs<sup>#</sup>| No             | 20 PCs<sup>#</sup> |

## Citation
If this benchamrking is useful for your research, please cite the following paper:
```
Renjie Chen, Yue Yao, Jingyang Qian, Xin Peng, Xin Shao, Xiaohui Fan. 2025.
A comprehensive benchmarking for spatially resolved transcriptomics clustering methods across variable technologies, organs, and replicates.
iMeta 4: e70084. https://doi.org/10.1002/imt2.70084.
```
