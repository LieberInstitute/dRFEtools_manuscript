# RFE simulations for regression (100 steps)
"""
This script runs RFE with cross-validation for the 10 traits.
Regression model. This uses the same CV and models as the dRFE.
"""

import os,errno
import dRFEtools
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_regression


def mkdir_p(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_data(simu):
    state = 13 + simu
    X, y = make_regression(
        n_samples=500, n_features=20000, n_informative=200, bias=0.2,
        n_targets=1, noise=10, random_state=state
    )
    return X, y


def rfecv_run(simu, cv, estimator, outdir, step):
    # Instantiate RFECV visualizer with a random forest regression
    X, y = load_data(simu)
    selector = RFECV(estimator, cv=cv, step=step, n_jobs=-1)
    start = time()
    selector = selector.fit(X, y)
    end = time()
    pd.DataFrame({"Simulation": simu,
                  "Feature":X.columns,
                  "Rank": selector.ranking_,
                  "Predictive": selector.support_,
                  "CPU Time": end - start,
                  "n features": selector.n_features_})\
      .to_csv("%s/RFECV_%d_predictions.txt" % (outdir, step),
              sep='\t', mode='a', index=False,
              header=True if simu == 0 else False)


def main():
    ## Generate 10-fold cross-validation
    step_size = 100; seed = 13
    cv = KFold(n_splits=10, shuffle=True, random_state=seed)
    ## Ridge
    outdir = 'ridge/'
    mkdir_p(outdir)
    regr = dRFEtools.Ridge(random_state=seed)
    for simu in range(10):
        print(simu)
        rfecv_run(simu, cv, regr, outdir, step_size)
    ## Elastic net
    outdir = 'enet/'
    mkdir_p(outdir)
    regr = dRFEtools.ElasticNet(alpha=0.01, random_state=seed)
    for simu in range(10):
        print(simu)
        rfecv_run(simu, cv, regr, outdir, step_size)
    ## SVR linear kernel
    outdir = 'svr/'
    mkdir_p(outdir)
    regr = dRFEtools.LinearSVR(random_state=seed, max_iter=10000)
    for simu in range(10):
        print(simu)
        rfecv_run(simu, cv, regr, outdir, step_size)
    ## Random forest
    outdir = 'rf/'
    mkdir_p(outdir)
    regr = dRFEtools.RandomForestRegressor(n_estimators=100, oob_score=True,
                                           n_jobs=-1, random_state=seed)
    for simu in range(10):
        print(simu)
        rfecv_run(simu, cv, regr, outdir, step_size)


if __name__ == '__main__':
    main()
