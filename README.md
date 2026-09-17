# ISOGAM
An Adaptive Graph Autoencoder for Multimodal Spatial Omics Integration

ISOGAM is an enhanced multimodal spatial omics integration framework. It introduces two key innovations:

1. Graph Autoencoder (GE): Explicitly models spatial topology via graph convolution on the reconstruction path, while keeping the clustering embedding free from spatial smoothing.

2. Learnable Modality Weights (LW): Adaptively learns the importance of each modality (RNA, protein, image) and fuses them for clustering.

ISOGAM is designed for spatial domain identification and has been evaluated on five spatial omics datasets. 

## Requirements

Python 3.7

Dependencies are listed in requirements.txt

Installation:

```bash
conda create -n isogam_env python=3.7
conda activate isogam_env
pip install -r requirements.txt
```
Note: torch installed via pip is the CPU version by default. For GPU support, please install the CUDA version from the PyTorch official website. The code automatically uses GPU if available, otherwise falls back to CPU.


### Datasets

Dataset              Platform	        Modalities	       Download Link
Human Tonsil	     Visium CytAssist	RNA+Protein+Image  https://www.10xgenomics.com/datasets/gene-protein-expression-library-of-human-tonsil-cytassist-ffpe-2-standard

Human Tonsil Add-on	 Visium CytAssist	RNA+Protein+Image  https://www.10xgenomics.com/datasets/visium-cytassist-gene-and-protein-expression-library-of-human-tonsil-with-add-on-antibodies-h-e-6-5-mm-ffpe-2-standard

Human Glioblastoma	 Visium CytAssist	RNA+Protein+Image  https://www.10xgenomics.com/datasets/gene-and-protein-expression-library-of-human-glioblastoma-cytassist-ffpe-2-standard

Human Breast Cancer	 Visium CytAssist	RNA+Protein+Image  https://www.10xgenomics.com/datasets/gene-and-protein-expression-library-of-human-breast-cancer-cytassist-ffpe-2-standard

MMTV-PyMT (Mouse)	 Visium (SPOTS)	    RNA                https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE198353

Note: This repository only includes a small example dataset (tutorial/tonsil_data/) for quick demonstration. For other datasets, please download them from the links above and follow the pipeline in example.ipynb; simply modify the data paths.


## Quick Start

The example data is already placed under tutorial/tonsil_data/. Run:
```bash
cd tutorial
jupyter notebook
```
Open example.ipynb and run all cells. The notebook will:

1、Extract image features from the H&E image.

2、Load and preprocess RNA and protein data.

3、Build and train the ISOGAM model (or load cached embeddings).

4、Perform clustering and compute ICC / Moran's I.

5、Generate spatial domain plots and an ICC boxplot.

All output figures are saved under tutorial/tonsil_data/.

No installation is required for the code itself. The notebook automatically adds the miso module to sys.path.


## Ablation Study

We provide an ablation notebook ablation_experiment.ipynb, which compares three configurations:

The baseline (Autoencoder+ Equal weight for each modality);  GraphEncoder (Graphencoder+ Equal weight for each modality); ISOGAM ((Autoencoder+ learnable weight for each modality)

Note: The ablation notebook contains hard-coded paths. You may need to adjust them to your local data location.


### License

Part of the ISOGAM code is referenced from MISO, which was developed by a group from the University of Pennsylvania and permitted for non-profit research or academic purposes only.

## Citation
If you use ISOGAM in your research, please cite our paper.
