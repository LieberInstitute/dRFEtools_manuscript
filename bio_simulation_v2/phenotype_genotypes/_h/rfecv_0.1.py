# RFE simulations with phenotype ~ genotypes (10%)
"""
This script runs RFE with cross-validation for the 15 traits.
Using feature elmination to impute trait from genotypes. This
uses the same CV and models as the dRFE version as well as the
100 and 10 steps variation.
"""

import os,errno
import functools
import dRFEtools
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV

@functools.lru_cache()
def get_y_var():
    # Correlated component
    Ycorr = pd.read_csv("../../_m/genotype_simulation/Y_correlatedBg_genotype_simulation.csv", index_col=0)
    # Genetic component
    YgenBg = pd.read_csv("../../_m/genotype_simulation/Y_genBg_genotype_simulation.csv", index_col=0)
    YgenFixed = pd.read_csv("../../_m/genotype_simulation/Y_genFixed_genotype_simulation.csv", index_col=0)
    # Noise component
    YnoiseBg = pd.read_csv("../../_m/genotype_simulation/Y_noiseBg_genotype_simulation.csv", index_col=0)
    YnoiseFixed = pd.read_csv("../../_m/genotype_simulation/Y_noiseFixed_genotype_simulation.csv", index_col=0)
    # Combine
    Y = Ycorr + YgenBg + YgenFixed + YnoiseBg + YnoiseFixed
    return Y


@functools.lru_cache()
def get_X_var():
    snp_df = pd.read_csv("../../_m/genotype_simulation/Genotypes_genotype_simulation.csv",
                         index_col=0).T
    r = pd.get_dummies(snp_df, columns=snp_df.columns, dummy_na=True)
    r.columns = r.columns.str.replace('\.\d+', '', regex=True)
    return r


def mkdir_p(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_data(simu):
    X = get_X_var()
    Y = get_y_var().iloc[:, simu]
    return X,Y


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
      .to_csv("%s/RFECV_%.2f_predictions.txt" % (outdir, step),
              sep='\t', mode='a', index=False,
              header=True if simu == 0 else False)


def main():
    ## Generate 10-fold cross-validation
    step_size = 0.1; seed = 13
    cv = KFold(n_splits=10, shuffle=True, random_state=seed)
    ## Ridge
    outdir = 'ridge/'
    mkdir_p(outdir)
    regr = dRFEtools.Ridge(random_state=seed)
    for simu in range(15):
        print(simu)
        rfecv_run(simu, cv, regr, outdir, step_size)
    ## Elastic net
    outdir = 'enet/'
    mkdir_p(outdir)
    regr = dRFEtools.ElasticNet(alpha=0.01, random_state=seed)
    for simu in range(15):
        print(simu)
        rfecv_run(simu, cv, regr, outdir, step_size)
    ## SVR linear kernel
    outdir = 'svr/'
    mkdir_p(outdir)
    regr = dRFEtools.LinearSVR(random_state=seed, max_iter=10000)
    for simu in range(15):
        print(simu)
        rfecv_run(simu, cv, regr, outdir, step_size)
    ## Random forest
    outdir = 'rf/'
    mkdir_p(outdir)
    regr = dRFEtools.RandomForestRegressor(n_estimators=100, oob_score=True,
                                           n_jobs=-1, random_state=seed)
    for simu in range(15):
        print(simu)
        rfecv_run(simu, cv, regr, outdir, step_size)


if __name__ == '__main__':
    main()
