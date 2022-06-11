# How Powerful are Spectral Graph Neural Networks

This repository is the official implementation of the model in the [following paper](https://arxiv.org/abs/2205.11172v1):

Xiyuan Wang, Muhan Zhang: How Powerful are Spectral Graph Neural Networks. ICML 2022

```{bibtex}
@article{JacobiConv,
  author    = {Xiyuan Wang and
               Muhan Zhang},
  title     = {How Powerful are Spectral Graph Neural Networks},
  journal   = {ICML},
  year      = {2022}
}
```

#### Requirements
Tested combination: Python 3.9.6 + [PyTorch 1.9.0](https://pytorch.org/get-started/previous-versions/) + [PyTorch_Geometric 2.0.3](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) + [PyTorch Sparse 0.6.12](https://github.com/rusty1s/pytorch_sparse)

Other required python libraries include: numpy, scikit-learn, optuna, seaborn etc.

### Reproduce Our Results

#### Image Filter Tasks

To reproduce results of JacobiConv on image datasets:
```
python ImgFilter.py --test --repeat 1 --dataset $dataset --fixalpha
```
where $dataset is selected from low, high, rejection, band, and comb. 

To reproduce results of linear GNN with other bases:
```
python ImgFilter.py --test --$basis --repeat 1 --dataset $dataset --fixalpha
```
where $basis is selected from cheby, power, and bern. 


We use optuna to select hyperparameters.
```
python ImgFilter.py --optruns 100  --dataset $dataset  --path $dir --name $dataset
```
The record file of optuna will be put in directory $dir.

#### Real-World Tasks

To reproduce results of JacobiConv on real-world datasets:
```
python RealWorld.py --test --repeat 10 --dataset $dataset --split dense
```
where $dataset is selected from pubmed, computers, squirrel, photo, chameleon, film, cora, citeseer, texas, cornell. 

To reproduce results of linear GNN with other bases:
```
python RealWorld.py --test --$basis  --fixalpha --repeat 10 --dataset $dataset --split dense
```
where $basis is selected from cheby, power, and bern. 

To reproduce other ablation studies:

Unifilter
```
python RealWorld.py --test --repeat 10 --dataset $dataset --split dense --sole
```
No-PCD
```
python RealWorld.py --test --repeat 10 --dataset $dataset --split dense --fixalpha
```
NL-RES 
```
python RealWorld.py --test --repeat 10 --dataset $dataset --split dense --resmultilayer
```
NL
```
python RealWorld.py --test --repeat 10 --dataset $dataset --split dense --multilayer
```

To select hyperparameters:
```
python RealWorld.py --repeat 3 --optruns 400 --split dense --dataset $dataset  --path $dir --name $dataset
```
The record file of optuna will be put in directory $dir.

