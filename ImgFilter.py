import torch
import numpy as np
from impl import models, PolyConv, GDataset, utils
import datasets
from torch.optim import Adam
import optuna
import torch.nn as nn


def split():
    '''
    Following `"BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation",
    we remove edge pixels.
    '''
    global masked_dataset
    masked_dataset = GDataset.GDataset(*baseG.get_split("valid"))


def buildModel(conv_layer, aggr, alpha, image_idx=0, **kwargs):
    emb = models.Seq([models.TensorMod(baseG.x[:, image_idx].reshape(-1, 1))])
    if args.power:
        conv_fn = PolyConv.PowerConv
    elif args.legendre:
        conv_fn = PolyConv.LegendreConv
    elif args.cheby:
        conv_fn = PolyConv.ChebyshevConv
    else:
        from functools import partial
        conv_fn = partial(PolyConv.JacobiConv, **kwargs)
    if args.fixalpha:
        from bestHyperparams import image_filter_alpha
        alpha = image_filter_alpha["power" if args.power else
                                   ("cheby" if args.cheby else "jacobi")][args.dataset]
    conv = PolyConv.PolyConvFrame(conv_fn,
                                  depth=conv_layer,
                                  aggr=aggr,
                                  alpha=alpha,
                                  fixed=args.fixalpha)
    comb = models.Combination(1, conv_layer + 1, args.sole)
    if args.bern:
        conv = PolyConv.Bern_prop(conv_layer)
    gnn = models.Gmodel(emb, conv, comb).to(device)
    return gnn


def search_hyper_params(trial):
    conv_layer = 10
    aggr = "gcn"
    lr1 = trial.suggest_categorical("lr1", [0.001, 0.005, 0.01, 0.05])
    lr2 = trial.suggest_categorical("lr2", [0.001, 0.005, 0.01, 0.05])
    lr3 = trial.suggest_categorical("lr3", [0.001, 0.005, 0.01, 0.05])
    wd1 = trial.suggest_categorical("wd1", [0.0, 1e-4, 5e-4, 1e-3])
    wd2 = trial.suggest_categorical("wd2", [0.0, 1e-4, 5e-4, 1e-3])
    wd3 = trial.suggest_categorical("wd3", [0.0, 1e-4, 5e-4, 1e-3])
    alpha = trial.suggest_float('alpha', 0.5, 2.0, step=0.5)
    a = trial.suggest_float('a', -1.1, -0.0, step=0.05)
    b = trial.suggest_float('b', -0.2, 3.0, step=0.05)
    return work(conv_layer,
                aggr,
                alpha,
                lr1,
                lr2,
                lr3,
                wd1,
                wd2,
                wd3,
                a=a,
                b=b)


def work(conv_layer: int = 10,
         aggr: str = "gcn",
         alpha: float = 1.0,
         lr1: float = 1e-2,
         lr2: float = 1e-2,
         lr3: float = 1e-2,
         wd1: float = 0,
         wd2: float = 0,
         wd3: float = 0,
         **kwargs):

    out_loss = []
    for rep in range(args.repeat):
        out_loss.append([])
        utils.set_seed(rep)
        for idx in range(50):
            y = masked_dataset.y[:, idx].reshape(-1, 1)
            gnn = buildModel(conv_layer, aggr, alpha, idx, **kwargs)
            optimizer = Adam([{
                'params': gnn.emb.parameters(),
                'weight_decay': wd1,
                'lr': lr1
            }, {
                'params': gnn.conv.parameters(),
                'weight_decay': wd2,
                'lr': lr2
            }, {
                'params': gnn.comb.parameters(),
                'weight_decay': wd3,
                'lr': lr3
            }])
            best_loss = np.inf
            early_stop = 0
            gnn.train()
            for i in range(1000):
                optimizer.zero_grad()
                pred = gnn(masked_dataset.edge_index, masked_dataset.edge_attr,
                           masked_dataset.mask)
                loss = torch.square(pred - y).sum()
                loss.backward()
                optimizer.step()
                loss = loss.item()
                if loss < best_loss:
                    best_loss = loss
                    early_stop = 0
                early_stop += 1
                if early_stop > 200:
                    break
            out_loss[-1].append(best_loss)
    print(
        f"end loss {np.average(out_loss):.6e}"
    )
    return np.average(out_loss)


if __name__ == '__main__':
    args = utils.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseG = datasets.load_dataset(args.dataset, args.split)
    baseG.to(device)
    masked_dataset = None
    output_channels = 1
    split()

    if args.test:
        from bestHyperparams import img_params
        print(work(**(img_params[args.dataset])))
    else:
        study = optuna.create_study(direction="minimize",
                                    storage="sqlite:///" + args.path +
                                    args.name + ".db",
                                    study_name=args.name,
                                    load_if_exists=True)
        study.optimize(search_hyper_params, n_trials=args.optruns)
        print("best params ", study.best_params)
        print("best valf1 ", study.best_value)