from impl import metrics, PolyConv, models, GDataset, utils
import datasets
import torch
from torch.optim import Adam
import optuna
import torch.nn as nn
import numpy as np
import seaborn as sns


def split():
    global baseG, trn_dataset, val_dataset, tst_dataset
    baseG.mask = datasets.split(baseG, split=args.split)
    trn_dataset = GDataset.GDataset(*baseG.get_split("train"))
    val_dataset = GDataset.GDataset(*baseG.get_split("valid"))
    tst_dataset = GDataset.GDataset(*baseG.get_split("test"))


def buildModel(conv_layer: int = 10,
               aggr: str = "gcn",
               alpha: float = 0.2,
               dpb: float = 0.0,
               dpt: float = 0.0,
               **kwargs):
    if args.multilayer:
        emb = models.Seq([
            models.TensorMod(baseG.x),
            nn.Dropout(p=dpb),
            nn.Sequential(nn.Linear(baseG.x.shape[1], output_channels),
                          nn.ReLU(inplace=True),
                          nn.Linear(output_channels, output_channels)),
            nn.Dropout(dpt, inplace=True)
        ])
    elif args.resmultilayer:
        emb = models.Seq([
            models.TensorMod(baseG.x),
            nn.Dropout(p=dpb),
            nn.Linear(baseG.x.shape[1], output_channels),
            models.ResBlock(
                nn.Sequential(nn.ReLU(inplace=True),
                              nn.Linear(output_channels, output_channels))),
            nn.Dropout(dpt, inplace=True)
        ])
    else:
        emb = models.Seq([
            models.TensorMod(baseG.x),
            nn.Dropout(p=dpb),
            nn.Linear(baseG.x.shape[1], output_channels),
            nn.Dropout(dpt, inplace=True)
        ])

    from functools import partial

    frame_fn = PolyConv.PolyConvFrame
    conv_fn = partial(PolyConv.JacobiConv, **kwargs)
    if args.power:
        conv_fn = PolyConv.PowerConv
    if args.legendre:
        conv_fn = PolyConv.LegendreConv
    if args.cheby:
        conv_fn = PolyConv.ChebyshevConv

    if args.bern:
        conv = PolyConv.Bern_prop(conv_layer)
    else:
        if args.fixalpha:
            from bestHyperparams import fixalpha_alpha
            alpha = fixalpha_alpha[args.dataset]["power" if args.power else (
                "cheby" if args.cheby else "jacobi")]
        conv = frame_fn(conv_fn,
                        depth=conv_layer,
                        aggr=aggr,
                        alpha=alpha,
                        fixed=args.fixalpha)
    comb = models.Combination(output_channels, conv_layer + 1, sole=args.sole)
    gnn = models.Gmodel(emb, conv, comb).to(device)
    return gnn


def work(conv_layer: int = 10,
         aggr: str = "gcn",
         alpha: float = 0.2,
         lr1: float = 1e-3,
         lr2: float = 1e-3,
         lr3: float = 1e-3,
         wd1: float = 0,
         wd2: float = 0,
         wd3: float = 0,
         dpb=0.0,
         dpt=0.0,
         **kwargs):
    outs = []
    for rep in range(args.repeat):
        utils.set_seed(rep)
        split()
        gnn = buildModel(conv_layer, aggr, alpha, dpb, dpt, **kwargs)
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
        val_score = 0
        early_stop = 0
        for i in range(1000):
            utils.train(optimizer, gnn, trn_dataset, loss_fn)
            score, _ = utils.test(gnn, val_dataset, score_fn, loss_fn=loss_fn)
            if score >= val_score:
                early_stop = 0
                val_score = score
            else:
                early_stop += 1
            if early_stop > 200:
                break
        outs.append(val_score)
    return np.average(outs)


def search_hyper_params(trial: optuna.Trial):
    conv_layer = 10
    aggr = "gcn"
    lr1 = trial.suggest_categorical("lr1", [0.0005, 0.001, 0.005, 0.01, 0.05])
    lr2 = trial.suggest_categorical("lr2", [0.0005, 0.001, 0.005, 0.01, 0.05])
    lr3 = trial.suggest_categorical("lr3", [0.0005, 0.001, 0.005, 0.01, 0.05])
    wd1 = trial.suggest_categorical("wd1", [0.0, 5e-5, 1e-4, 5e-4, 1e-3])
    wd2 = trial.suggest_categorical("wd2", [0.0, 5e-5, 1e-4, 5e-4, 1e-3])
    wd3 = trial.suggest_categorical("wd3", [0.0, 5e-5, 1e-4, 5e-4, 1e-3])
    alpha = trial.suggest_float('alpha', 0.5, 2.0, step=0.5)
    a = trial.suggest_float('a', -1.0, 2.0, step=0.25)
    b = trial.suggest_float('b', -0.5, 2.0, step=0.25)
    dpb = trial.suggest_float("dpb", 0.0, 0.9, step=0.1)
    dpt = trial.suggest_float("dpt", 0.0, 0.9, step=0.1)
    return work(conv_layer,
                aggr,
                alpha,
                lr1,
                lr2,
                lr3,
                wd1,
                wd2,
                wd3,
                dpb,
                dpt,
                a=a,
                b=b)


def test(conv_layer=10,
         aggr="gcn",
         alpha=1.0,
         lr1=1e-2,
         lr2=1e-2,
         lr3=1e-2,
         wd1=0.0,
         wd2=0.0,
         wd3=0.0,
         dpb=0.0,
         dpt=0.0,
         **kwargs):
    outs = []
    vals = []
    for rep in range(args.repeat):
        print("repeat ", rep)
        utils.set_seed(rep)
        split()
        gnn = buildModel(conv_layer, aggr, alpha, dpb, dpt, **kwargs)
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
        val_score = 0
        tst_score = 0
        early_stop = 0
        for i in range(1000):
            utils.train(optimizer, gnn, trn_dataset, loss_fn)
            score, _ = utils.test(gnn, val_dataset, score_fn, loss_fn=loss_fn)
            if score >= val_score:
                early_stop = 0
                val_score = score
                if args.savemodel:
                    torch.save(gnn.state_dict(), f"{args.dataset}_{rep}.pt")
                tst_score, _ = utils.test(gnn,
                                          tst_dataset,
                                          score_fn,
                                          loss_fn=loss_fn)
            else:
                early_stop += 1
            if early_stop > 200:
                break
        vals.append(val_score)
        outs.append(tst_score)
    outs = np.array(outs)
    print(
        f"avg {np.average(outs):.4f} error {np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(outs,func=np.mean,n_boot=1000),95)-outs.mean())):.4f}"
    )
    return np.average(outs)


if __name__ == '__main__':
    args = utils.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseG = datasets.load_dataset(args.dataset, args.split)
    baseG.to(device)
    trn_dataset, val_dataset, tst_dataset = None, None, None
    output_channels = baseG.y.unique().shape[0]

    loss_fn = nn.CrossEntropyLoss()
    score_fn = metrics.multiclass_accuracy
    split()

    if args.test:
        from bestHyperparams import realworld_params
        best_hyperparams = realworld_params
        print(test(**(best_hyperparams[args.dataset])))
    else:
        study = optuna.create_study(direction="maximize",
                                    storage="sqlite:///" + args.path +
                                    args.name + ".db",
                                    study_name=args.name,
                                    load_if_exists=True)
        study.optimize(search_hyper_params, n_trials=args.optruns)
        print("best params ", study.best_params)
        print("best valf1 ", study.best_value)
