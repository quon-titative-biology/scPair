# scPair is a computational framework for boosting single cell multimodal analysis by leveraging single cell atlases and implicit feature selection

scPair, a deep learning framework for computational analysis of single cell multimodal data. scPair performs automatic, implicit feature selection to infer the features of each data modality that yield optimal mappings of cell states between data modalities. Our training procedure of scPair also addresses challenges of shallow sequencing of multimodal datasets by using higher depth unimodal data to learn robust covariance structure in each data modality, and in turn relies on multimodal data primarily for feature selection and integration of cell states between data modalities. We demonstrate that these two properties of scPair enable it to outperform existing methods on multimodal data analysis tasks such as cell state mapping and feature prediction, as well as simultaneous trajectory inference in both RNA and ATAC data components to identify time point-specific feature activity during cellular differentiation.
![alt text](https://github.com/quon-titative-biology/scPair/blob/main/fig/scPair_Fig_1.png)


---
### Package installation
Please clone this repository:
```command line
git clone https://github.com/quon-titative-biology/scPair
cd scPair
```
---
### Package requirements
scPair reqiues Python 3.7.12 or higher and it is implemented using PyTorch 1.12.0.
You can choose run the following command line:

```command line
conda create -y --name scpair -c conda-forge -c bioconda python=3.7.12
conda activate scpair
conda install pytorch cudatoolkit=11.6 -c pytorch -c conda-forge
```

Or alternatively, you can choose to use the [environment file](https://github.com/quon-titative-biology/scPair/blob/main/scpair.yml) provided under this repository:
```command line
conda env create --file=scpair.yml
```

---
### [Tutorials](https://github.com/quon-titative-biology/scPair/blob/main/tutorials/README.md)

* This repository is being updated periodically. For questions, please email hrhu@ucdavis.edu

---
### Updates

Tutorial update (Mar 6, 2024)

First public release (Mar 1, 2024)

Repo created (Aug 28, 2023)

---

If it is helpful in your research, please consider citing it:

```
@software{scPair,
  author = {Hu, Hongru and Quon, Gerald},
  title = {Boosting single cell multimodal analysis by leveraging single cell atlases and implicit feature selection},
  url = {https://github.com/quon-titative-biology/scPair/},
  version = {1.0.0},
  month = {3},
  year = {2023}
}
```
