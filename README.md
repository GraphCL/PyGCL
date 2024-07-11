[pygcl-url]: https://github.com/GraphCL/PyGCL
[pypi-img]: https://img.shields.io/pypi/v/PyGCL
[pypi-url]: https://pypi.org/project/PyGCL
[python-img]: https://img.shields.io/github/languages/top/GraphCL/PyGCL?color=lightgrey
[stars-img]: https://img.shields.io/github/stars/GraphCL/PyGCL?color=yellow
[stars-url]: https://github.com/GraphCL/PyGCL/stargazers
[fork-img]: https://img.shields.io/github/forks/GraphCL/PyGCL?color=lightblue&label=fork
[fork-url]: https://github.com/GraphCL/PyGCL/network/members
[doc-img]: https://readthedocs.org/projects/pygcl/badge/?version=latest
[doc-url]: https://pygcl.readthedocs.io/en/latest/?badge=latest
[lines-img]: https://img.shields.io/tokei/lines/github/GraphCL/PyGCL?color=red
[visitors-img]: https://visitor-badge.glitch.me/badge?page_id=GraphCL.PyGCL

<p align="center">
<img src="logo.png" width="75%" class="center" alt="logo"/>
</p>

PyGCL is a [PyTorch](https://pytorch.org)-based open-source Graph Contrastive Learning (GCL) library, which features modularized GCL components from published papers, standardized evaluation, and experiment management.

[![Made with Python][python-img]][pygcl-url]
[![PyPI version][pypi-img]][pypi-url]
[![Documentation Status][doc-img]][doc-url]
[![GitHub stars][stars-img]][stars-url]
[![GitHub forks][fork-img]][fork-url]
[![Total lines][lines-img]][pygcl-url]
[![visitors][visitors-img]][pygcl-url]

---

# What is Graph Contrastive Learning?

Graph Contrastive Learning (GCL) establishes a new paradigm for learning graph representations without human annotations. A typical GCL algorithm firstly constructs multiple graph views via stochastic augmentation of the input and then learns representations by contrasting positive samples against negative ones.

👉 For a general introduction of GCL, please refer to our [paper](https://arxiv.org/abs/2109.01116) and [blog](https://sxkdz.github.io/research/GraphCL/). Also, this [repo](https://github.com/SXKDZ/awesome-self-supervised-learning-for-graphs) tracks newly published GCL papers.

# Install

## Prerequisites

PyGCL needs the following packages to be installed beforehand:

* Python 3.8+
* PyTorch 1.9+
* PyTorch-Geometric 1.7+
* Scikit-learn 0.24+
* Numpy
* tqdm
* NetworkX

## Installation via PyPI

Before install PyGCL, please ensure [pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) is installed.

To install PyGCL with [`pip`](https://pip.pypa.io/en/stable/), simply run:

```
pip install PyGCL
```

Then, you can import `GCL` from your current environment.

> **A note regarding DGL**
>
> Currently, PyGCL only relies on `dgl` in the `utils.build_dgl_graph` function, and this method may no longer be maintained in future versions. Therefore, in future versions of PyGCL, it may no longer depend on `dgl`. So please install `dgl` as needed.
>
> Currently the DGL team maintains two versions, `dgl` for CPU support and `dgl-cu***` for CUDA support. Since `pip` treats them as different packages, it is hard for PyGCL to check for the version requirement of `dgl`. We have removed such dependency checks for `dgl` in our setup configuration and require the users to [install a proper version](https://www.dgl.ai/pages/start.html) by themselves.

# Package Overview

Our PyGCL implements four main components of graph contrastive learning algorithms:

* Graph augmentation: transforms input graphs into congruent graph views.
* Contrasting architectures and modes: generate positive and negative pairs according to node and graph embeddings.
* Contrastive objectives: computes the likelihood score for positive and negative pairs.
* Negative mining strategies: improves the negative sample set by considering the relative similarity (the hardness) of negative sample.

We also implement utilities for training models, evaluating model performance, and managing experiments.

# Implementations and Examples

For a quick start, please check out the `examples` folder. We currently implemented the following methods:

* **DGI** (P. Veličković et al., Deep Graph Infomax, ICLR, 2019) [[Example1](examples/DGI_transductive.py), [Example2](examples/DGI_inductive.py)]
* **InfoGraph** (F.-Y. Sun et al., InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization, ICLR, 2020) [[Example](examples/InfoGraph.py)]
* **MVGRL** (K. Hassani et al., Contrastive Multi-View Representation Learning on Graphs, ICML, 2020) [[Example1](examples/MVGRL_node.py), [Example2](examples/MVGRL_graph.py)]
* **GRACE** (Y. Zhu et al., Deep Graph Contrastive Representation Learning, GRL+@ICML, 2020) [[Example](examples/GRACE.py)]
* **GraphCL** (Y. You et al., Graph Contrastive Learning with Augmentations, NeurIPS, 2020) [[Example](examples/GraphCL.py)]
* **SupCon** (P. Khosla et al., Supervised Contrastive Learning, NeurIPS, 2020) [[Example](examples/GRACE_SupCon.py)]
* **HardMixing** (Y. Kalantidis et al., Hard Negative Mixing for Contrastive Learning, NeurIPS, 2020)
* **DCL** (C.-Y. Chuang et al., Debiased Contrastive Learning, NeurIPS, 2020)
* **HCL** (J. Robinson et al., Contrastive Learning with Hard Negative Samples, ICLR, 2021)
* **Ring** (M. Wu et al., Conditional Negative Sampling for Contrastive Learning of Visual Representations, ICLR, 2021)
* **Exemplar** (N. Zhao et al., What Makes Instance Discrimination Good for Transfer Learning?, ICLR, 2021)
* **BGRL** (S. Thakoor et al., Bootstrapped Representation Learning on Graphs, arXiv, 2021) [[Example1](examples/BGRL_L2L.py), [Example2](examples/BGRL_G2L.py)]
* **G-BT** (P. Bielak et al., Graph Barlow Twins: A Self-Supervised Representation Learning Framework for Graphs, arXiv, 2021) [[Example](examples/GBT.py)]
* **VICReg** (A. Bardes et al., VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning, arXiv, 2021)

# Building Your Own GCL Algorithms

Besides try the above examples for node and graph classification tasks, you can also build your own graph contrastive learning algorithms straightforwardly.

## Graph Augmentation

In `GCL.augmentors`, PyGCL provides the `Augmentor` base class, which offers a universal interface for graph augmentation functions. Specifically, PyGCL implements the following augmentation functions:

| Augmentation                            | Class name        |
| --------------------------------------- | ----------------- |
| Edge Adding (EA)                        | `EdgeAdding`      |
| Edge Removing (ER)                      | `EdgeRemoving`    |
| Feature Masking (FM)                    | `FeatureMasking`  |
| Feature Dropout (FD)                    | `FeatureDropout`  |
| Edge Attribute Masking (EAR)            | `EdgeAttrMasking` |
| Personalized PageRank (PPR)             | `PPRDiffusion`    |
| Markov Diffusion Kernel (MDK)           | `MarkovDiffusion` |
| Node Dropping (ND)                      | `NodeDropping`    |
| Node Shuffling (NS)                     | `NodeShuffling`   |
| Subgraphs induced by Random Walks (RWS) | `RWSampling`      |
| Ego-net Sampling (ES)                   | `Identity`        |

Call these augmentation functions by feeding with a `Graph` in a tuple form of node features, edge index, and edge features `(x, edge_index, edge_attrs)` will produce corresponding augmented graphs.

### Composite Augmentations
PyGCL supports composing arbitrary numbers of augmentations together. To compose a list of augmentation instances `augmentors`, you need to use the `Compose` class:

```python
import GCL.augmentors as A

aug = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
```

You can also use the `RandomChoice` class to randomly draw a few augmentations each time:

```python
import GCL.augmentors as A

aug = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                      A.NodeDropping(pn=0.1),
                      A.FeatureMasking(pf=0.1),
                      A.EdgeRemoving(pe=0.1)],
                     num_choices=1)
```

### Customizing Your Own Augmentation

You can write your own augmentation functions by inheriting the base `Augmentor` class and defining the `augment` function.

## Contrasting Architectures and Modes

Existing GCL architectures could be grouped into two lines: negative-sample-based methods and negative-sample-free ones.

* Negative-sample-based approaches can either have one single branch or two branches. In single-branch contrasting, we only need to construct one graph view and perform contrastive learning within this view. In dual-branch models, we generate two graph views and perform contrastive learning within and across views.
* Negative-sample-free approaches eschew the need of explicit negative samples. Currently, PyGCL supports the bootstrap-style contrastive learning as well contrastive learning within embeddings (such as Barlow Twins and VICReg).

| Contrastive architectures    | Supported contrastive modes | Need negative samples | Class name             | Examples                                                     |
| ---------------------------- | --------------------------- | :-------------------: | ---------------------- | ------------------------------------------------------------ |
| Single-branch contrasting    | G2L only                    |           ✅           | `SingleBranchContrast` | [DGI](examples/DGI_transductive.py), [InfoGraph](examples/InfoGraph.py) |
| Dual-branch contrasting      | L2L, G2G, and G2L           |           ✅           | `DualBranchContrast`   | [GRACE](examples/GRACE.py)                                   |
| Bootstrapped contrasting     | L2L, G2G, and G2L           |           ❎           | `BootstrapContrast`    | [BGRL](examples/BGRL.py)                                     |
| Within-embedding contrasting | L2L and G2G                 |           ❎           | `WithinEmbedContrast`  | [GBT](examples/GBT.py)                                       |

Moreover, you can use `add_extra_mask` if you want to add positives or remove negatives. This function performs bitwise ADD to extra positive masks specified by `extra_pos_mask` and bitwise OR to extra negative masks specified by `extra_neg_mask`. It is helpful, for example, when you have supervision signals from labels and want to train the model in a semi-supervised manner.

Internally, PyGCL calls `Sampler` classes in `GCL.models` that receive embeddings and produce positive/negative masks. PyGCL implements three contrasting modes: (a) Local-Local (L2L), (b) Global-Global (G2G), and (c) Global-Local (G2L) modes. L2L and G2G modes contrast embeddings at the same scale and the latter G2L one performs cross-scale contrasting. To implement your own GCL model, you may also use these provided sampler models:

| Contrastive modes                    | Class name          |
| ------------------------------------ | ------------------- |
| Same-scale contrasting (L2L and G2G) | `SameScaleSampler`  |
| Cross-scale contrasting (G2L)        | `CrossScaleSampler` |

* For L2L and G2G, embedding pairs of the same node/graph in different views constitute positive pairs. You can refer to [GRACE](examples/GRACE.py) and [GraphCL](examples/GraphCL.py) for examples.
* For G2L, node-graph embedding pairs form positives. Note that for single-graph datasets, the G2L mode requires explicit negative sampling (otherwise no negatives for contrasting). You can refer to [DGI](examples/DGI_transductive.py) for an example.
* Some models (e.g., GRACE) add extra intra-view negative samples. You may manually call `sampler.add_intraview_negs` to enlarge the negative sample set.
* Note that the bootstrapping latent model involves some special model design (asymmetric online/offline encoders and momentum weight updates). You may refer to [BGRL](examples/BGRL.py) for details.

## Contrastive Objectives

In `GCL.losses`, PyGCL implements the following contrastive objectives:

| Contrastive objectives               | Class name        |
| ------------------------------------ | ----------------- |
| InfoNCE loss                         | `InfoNCE`         |
| Jensen-Shannon Divergence (JSD) loss | `JSD`             |
| Triplet Margin (TM) loss             | `Triplet`         |
| Bootstrapping Latent (BL) loss       | `BootstrapLatent` |
| Barlow Twins (BT) loss               | `BarlowTwins`     |
| VICReg loss                          | `VICReg`          |

All these objectives are able to contrast any arbitrary positive and negative pairs, except for Barlow Twins and VICReg losses that perform contrastive learning within embeddings. Moreover, for InfoNCE and Triplet losses, we further provide `SP` variants that computes contrastive objectives given only one positive pair per sample to speed up computation and avoid excessive memory consumption. 

## Negative Sampling Strategies

PyGCL further implements several negative sampling strategies:

| Negative sampling strategies      | Class name                                              |
| --------------------------------- | ------------------------------------------------------- |
| Subsampling                       | `GCL.models.SubSampler`                                 |
| Hard negative mixing              | `GCL.models.HardMixing`                                 |
| Conditional negative sampling     | `GCL.models.Ring`                                       |
| Debiased contrastive objective    | `GCL.losses.DebiasedInfoNCE `, `GCL.losses.DebiasedJSD` |
| Hardness-biased negative sampling | `GCL.losses.HardnessInfoNCE`, `GCL.losses.HardnessJSD`  |

The former three models serve as an additional sampling step similar to existing `Sampler ` ones and can be used in conjunction with any objectives. The last two objectives are only for InfoNCE and JSD losses.

## Utilities

PyGCL provides a variety of evaluator functions to evaluate the embedding quality:

| Evaluator              | Class name     |
| ---------------------- | -------------- |
| Logistic regression    | `LREvaluator`  |
| Support vector machine | `SVMEvaluator` |
| Random forest          | `RFEvaluator`  |

To use these evaluators, you first need to generate dataset splits by `get_split` (random split) or by `from_predefined_split` (according to preset splits).

# Contribution

Feel free to open an [issue](issues/new) should you find anything unexpected or [create pull requests](pulls) to add your own work! We are motivated to continuously make PyGCL even better.

# Citation

Please cite [our paper](https://arxiv.org/abs/2109.01116) if you use this code in your own work:

```
@article{Zhu:2021tu,
author = {Zhu, Yanqiao and Xu, Yichen and Liu, Qiang and Wu, Shu},
title = {{An Empirical Study of Graph Contrastive Learning}},
journal = {arXiv.org},
year = {2021},
eprint = {2109.01116v1},
eprinttype = {arxiv},
eprintclass = {cs.LG},
month = sep,
}
```

