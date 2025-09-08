[![DOI](https://zenodo.org/badge/684201494.svg)](https://zenodo.org/doi/10.5281/zenodo.12735192)

# scPair is a computational framework for boosting single cell multimodal analysis by leveraging implicit feature selection and single cell atlases 

scPair, a deep learning framework for computational analysis of single cell multimodal data. scPair performs automatic, implicit feature selection to infer the features of each data modality that yield optimal mappings of cell states between data modalities. Our training procedure of scPair also addresses challenges of shallow sequencing of multimodal datasets by using higher depth unimodal data to learn robust covariance structure in each data modality, and in turn relies on multimodal data primarily for feature selection and integration of cell states between data modalities. We demonstrate that these two properties of scPair enable it to outperform existing methods on multimodal data analysis tasks such as cell state mapping and feature prediction, as well as simultaneous trajectory inference in both RNA and ATAC data components to identify time point-specific feature activity during cellular differentiation.
![alt text](https://github.com/quon-titative-biology/scPair/blob/main/img/scPair_Fig_1.png)


---
### Package installation

#### Option 1: Using uv (Recommended)
First, install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already:

Then clone this repository and set up the environment:
```bash
git clone https://github.com/quon-titative-biology/scPair
cd scPair
uv sync
```

To activate the virtual environment:
```bash
source .venv/bin/activate
```

#### Option 2: Using conda
Users can choose to create the environment provided under this repository [(env file)](https://github.com/quon-titative-biology/scPair/blob/main/scpair.yml):
```bash
conda env create --file=scpair.yml
```

PyPI installation will be released soon.

---
### Package requirements
scPair is implemented using `PyTorch 2.0.1`, `anndata 0.10.6`, and `scanpy 1.10.0` under `Python 3.10.x`. 

We have tested it on Azure.

---
### [Tutorials](https://github.com/quon-titative-biology/scPair/blob/main/tutorials/README.md)

* This repository is being updated periodically. For questions, please email hrhu@ucdavis.edu or create new issues under this repository.

---
### Updates
Paper published in Nature Communications (Nov 15, 2024)

Data access (July 1, 2024)

Tutorial updates (July 1, 2024; Mar 6, 2024; Apr 25, 2024)

First public release (Mar 1, 2024)

Repo created (Aug 28, 2023)

---

If it is helpful in your research, please consider citing it:
https://www.nature.com/articles/s41467-024-53971-2

```
@software{scPair,
  author = {Hu, Hongru and Quon, Gerald},
  title = {scPair: boosting single cell multimodal analysis by leveraging implicit feature selection and single cell atlases },
  DOI = {https://zenodo.org/doi/10.5281/zenodo.12735192}
  url = {https://github.com/quon-titative-biology/scPair/},
  version = {0.1},
  month = {7},
  year = {2024}
}
```
