import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path

# Figure 2A

width_inch = 18 / 2.54
height_inch = 3 / 2.54
fig, axes = plt.subplots(1, 10, figsize=(width_inch, height_inch), sharey=False)
data_dic = {'10x Visium': 'Mouse_Brain_Section_Coronal_data',
            'ST': 'processed/Mouse Brain_20A_data',
            'Slideseq': 'process/Brain/Mouse_Cortex_data_processed',
            'VisiumHD': 'Mouse_Brain/Mouse_Brain_FixedForzen_raw',
            'Stereoseq': 'processed/Mouse Hemi-Brain_Cell_Bin_Data_Whole_data',
            'seqFISH': 'precessed/OB_View0',
            'STARmap': 'processed/Mouse Visual Cortex_20180410_BY3_1kgenes_data',
            'MERFISH':'processed/Hypothalamus_Animal1_Bregma-0.04_domain',
            'CosMx':'Mouse_Brain/Mouse_Brain_Hemisphere_full',
            'Xenium':'Mouse_Brain/Mouse_Brain_MultiSection1_annotation',
           }
row = 0
col = 0
for col, (technology, sample) in enumerate(data_dic.items()):
    h5ad_path = f"dataset/{technology}/{sample}.h5ad"
    adata = sc.read(h5ad_path)
    if technology == '10x Visium':
        sc.pl.spatial(adata, color=['ground_truth'], show=False, ax=axes[col], legend_loc=None)
    elif technology == 'ST':
        sc.pl.spatial(adata, color=['ground_truth'], show=False, ax=axes[col], legend_loc=None, size=5)
    elif technology == 'STARmap':
        sc.pl.embedding(adata, basis='spatial', color=['celltype'], show=False, ax=axes[col], legend_loc=None, size=5)    
    elif technology == 'seqFISH':
        sc.pl.embedding(adata, basis='spatial', color=['ground_truth'], show=False, ax=axes[col], legend_loc=None, size=5)    
    elif technology == 'MERFISH':
        sc.pl.embedding(adata, basis='spatial', color=['ground_truth'], show=False, ax=axes[col], legend_loc=None, size=1)    
    else:
        sc.pl.embedding(adata, basis='spatial', color=['ground_truth'], show=False, ax=axes[col], legend_loc=None, size=0.2)
    axes[col].set_title(technology, fontsize=9, pad=3)
    axes[col].set_box_aspect(1)
    axes[col].set_xlabel('') 
    axes[col].set_ylabel('') 
    col += 1
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)
plt.savefig("plots/Figure2A.pdf", dpi=300)
plt.show()