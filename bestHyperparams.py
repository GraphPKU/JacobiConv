img_params = {
    "band": {
        'a': -0.75,
        'alpha': 1.5,
        'b': 0.75,
        'lr1': 0.001,
        'lr2': 0.001,
        'lr3': 0.05,
        'wd1': 0.0,
        'wd2': 0.0,
        'wd3': 0.0001
    },
    "comb": {
        'a': -0.75,
        'alpha': 2.0,
        'b': 0.75,
        'lr1': 0.05,
        'lr2': 0.001,
        'lr3': 0.05,
        'wd1': 0.001,
        'wd2': 0.0001,
        'wd3': 0.0
    },
    "high": {
        'lr1': 0.001,
        'lr2': 0.001,
        'lr3': 0.05,
        'wd1': 0.0005,
        'wd2': 0.0,
        'wd3': 0.0001,
        'alpha': 2.0,
        'a': -0.85,
        'b': 0.95
    },
    "low": {
        'a': -0.75,
        'alpha': 1.5,
        'b': 0.75,
        'lr1': 0.05,
        'lr2': 0.001,
        'lr3': 0.05,
        'wd1': 0.0005,
        'wd2': 0.0005,
        'wd3': 0.0001
    },
    "rejection": {
        'a': -0.75,
        'alpha': 2.0,
        'b': 0.5,
        'lr1': 0.05,
        'lr2': 0.001,
        'lr3': 0.05,
        'wd1': 0.001,
        'wd2': 0.0001,
        'wd3': 0.001
    }
}

image_filter_alpha = {
    "power": {
        'low': 0.5,
        'high': 0.5,
        'band': 1.0,
        'rejection': 1.0,
        'comb': 1.5
    },
    "cheby": {
        'low': 1.0,
        'high': 0.5,
        'band': 0.5,
        'rejection': 0.5,
        'comb': 0.5
    },
    "jacobi": {
        'low': 1.0,
        'high': 2.0,
        'band': 2.0,
        'rejection': 2.0,
        'comb': 2.0
    }
}

realworld_params = {
    'cora': {
        'a': 2.0,
        'alpha': 0.5,
        'b': -0.25,
        'dpb': 0.5,
        'dpt': 0.7,
        'lr1': 0.05,
        'lr2': 0.01,
        'lr3': 0.01,
        'wd1': 0.001,
        'wd2': 0.0001,
        'wd3': 5e-05
    },
    'citeseer': {
        'a': -0.5,
        'alpha': 0.5,
        'b': -0.5,
        'dpb': 0.9,
        'dpt': 0.8,
        'lr1': 0.05,
        'lr2': 0.001,
        'lr3': 0.01,
        'wd1': 5e-05,
        'wd2': 0.0,
        'wd3': 0.001
    },
    'pubmed': {
        'a': 1.5,
        'alpha': 0.5,
        'b': 0.25,
        'dpb': 0.0,
        'dpt': 0.5,
        'lr1': 0.05,
        'lr2': 0.05,
        'lr3': 0.05,
        'wd1': 0.0005,
        'wd2': 0.0005,
        'wd3': 0.0
    },
    'computers': {
        'a': 1.75,
        'alpha': 1.5,
        'b': -0.5,
        'dpb': 0.8,
        'dpt': 0.2,
        'lr1': 0.05,
        'lr2': 0.05,
        'lr3': 0.05,
        'wd1': 0.0001,
        'wd2': 0.0,
        'wd3': 0.0
    },
    'photo': {
        'a': 1.0,
        'alpha': 1.5,
        'b': 0.25,
        'dpb': 0.3,
        'dpt': 0.3,
        'lr1': 0.05,
        'lr2': 0.0005,
        'lr3': 0.05,
        'wd1': 5e-05,
        'wd2': 0.0,
        'wd3': 0.0
    },
    'chameleon': {
        'a': 0.0,
        'alpha': 2.0,
        'b': 0.0,
        'dpb': 0.6,
        'dpt': 0.5,
        'lr1': 0.05,
        'lr2': 0.01,
        'lr3': 0.05,
        'wd1': 0.0,
        'wd2': 0.0001,
        'wd3': 0.0005
    },
    'film': {
        'a': -1.0,
        'alpha': 1.0,
        'b': 0.5,
        'dpb': 0.9,
        'dpt': 0.7,
        'lr1': 0.05,
        'lr2': 0.05,
        'lr3': 0.01,
        'wd1': 0.001,
        'wd2': 0.0005,
        'wd3': 0.001
    },
    'squirrel': {
        'a': 0.5,
        'alpha': 2.0,
        'b': 0.25,
        'dpb': 0.4,
        'dpt': 0.1,
        'lr1': 0.01,
        'lr2': 0.01,
        'lr3': 0.05,
        'wd1': 5e-05,
        'wd2': 0.0,
        'wd3': 0.0
    },
    "texas": {
        'a': -0.5,
        'alpha': 0.5,
        'b': 0.0,
        'dpb': 0.8,
        'dpt': 0.7,
        'lr1': 0.05,
        'lr2': 0.005,
        'lr3': 0.01,
        'wd1': 0.001,
        'wd2': 0.0005,
        'wd3': 0.0005
    },
    "cornell": {
        'a': -0.75,
        'alpha': 0.5,
        'b': 0.25,
        'dpb': 0.4,
        'dpt': 0.7,
        'lr1': 0.05,
        'lr2': 0.005,
        'lr3': 0.001,
        'wd1': 0.0005,
        'wd2': 0.0005,
        'wd3': 0.0001
    }
}

fixalpha_alpha = {
    "cora": {
        "power": 1.0,
        "cheby": 0.5,
        "jacobi": 1.0
    },
    "citeseer": {
        "power": 0.5,
        "cheby": 0.5,
        "jacobi": 0.5
    },
    "pubmed": {
        "power": 1.0,
        "cheby": 1.0,
        "jacobi": 1.0
    },
    "computers": {
        "power": 2.0,
        "cheby": 1.5,
        "jacobi": 1.5
    },
    "photo": {
        "power": 2.0,
        "cheby": 1.0,
        "jacobi": 1.5
    },
    "chameleon": {
        "power": 2.0,
        "cheby": 2.0,
        "jacobi": 2.0
    },
    "film": {
        "power": 0.5,
        "cheby": 1.0,
        "jacobi": 0.5
    },
    "squirrel": {
        "power": 2.0,
        "cheby": 2.0,
        "jacobi": 2.0
    },
    "texas": {
        "power": 0.5,
        "cheby": 0.5,
        "jacobi": 1.0
    },
    "cornell": {
        "power": 0.5,
        "cheby": 0.5,
        "jacobi": 0.5
    },
}
