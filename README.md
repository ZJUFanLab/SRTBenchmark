# SRTBenchmark
**Benchmarking spatial clustering methods for spatially resolved transcriptomics**
## Overall Performance
We conducted a comprehensive benchmark of **14 spatial clustering methods** across multiple **technologies, organs, and biological replicates**, and provided **method recommendations** tailored to different application scenarios.

<img src="./figures/Overall_accuracy" width="300">

| Technology      | Organ          | Replicates | Recommendation (top 5)                         |
|-----------------|----------------|------------|------------------------------------------------|
| High Resolution | Brain          | No         | BASS, stLearn, Banksy, SpaGCN, STAGATE       |
| Low Resolution  | Brain          | No         | STAGATE, GraphST, SEDR, BASS, BayesSpace     |
| 10x Visium      | High continuity| No         | STAGATE, DeepST, SEDR, CCST, GraphST         |
| 10x Visium      | Low continuity | No         | PRECAST, stLearn, STAGATE, SpaGCN, BayesSpace|
| 10x Visium      | Brain          | Yes        | DeepST, Banksy, SEDR, GraphST, STAGATE       |
| MERFISH         | Brain          | Yes        | BASS, stLearn, SpaGCN, PRECAST, Banksy       |
