# RFE simulations with phenotype ~ genotypes (100 step)
"""
This script runs RFE with cross-validation for the 10 traits.
Using feature elmination to impute trait from genotypes. This
uses the same CV and models.
"""

import os,errno
import functools
import dRFEtools
import numpy as np
import pandas as pd
from time import time
from pandas_plink import read_plink
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV


@functools.lru_cache()
def cached_read_plink(file_prefix, verbose=True):
    """
    Load plink files
    """
    return read_plink(file_prefix, verbose)


@functools.lru_cache()
def get_plink(simu):
    """
    Simulation file prefix for plink files
    """
    file_prefix = "../../_m/genotype_simulation/simulation_%d" % simu
    return cached_read_plink(file_prefix)


def select_snps_from_plink_to_df(simu):
    """
    Given a <bim, fam, bed> triple of files and returns a pandas DataFrame
    with the SNP genotype values (0,1,2), where the rows are family IDs
    and columns are SNP IDs.
    """
    (bim, fam, bed) = get_plink(simu)
    dx = pd.DataFrame(bed.compute().transpose())
    dx.index = fam['fid']
    dx.columns = bim.snp
    dy = fam.loc[:, ["fid","trait"]].set_index("fid")
    dy.trait = dy.trait.astype("float")
    return dx, dy


def one_hot_encode_snp_df(snp_df):
    """
    Given a snp_df returned by select_snps_from_plink_to_df function,
    returns a one-hot-encoded version of it
    """
    dx = pd.get_dummies(snp_df, columns=snp_df.columns, dummy_na=True)
    dx.columns = dx.columns.str.replace('\.\d+', '', regex=True)
    return dx


def mkdir_p(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_data(simu):
    snp_df, Y = select_snps_from_plink_to_df(simu)
    X = one_hot_encode_snp_df(snp_df)
    return snp_df, Y


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


def permutation_run(estimator, cv, outdir, step_size):
    for simu in range(10):
        print(simu)
        rfecv_run(simu, cv, estimator, outdir, step_size)


def main():
    ## Generate 10-fold cross-validation
    step_size = 100; seed = 13
    cv = KFold(n_splits=10, shuffle=True, random_state=seed)
    ## Ridge
    outdir = 'ridge/'
    mkdir_p(outdir)
    regr = dRFEtools.Ridge(random_state=seed)
    permutation_run(regr, cv, outdir, step_size)
    ## SVR linear kernel
    outdir = 'svr/'
    mkdir_p(outdir)
    regr = dRFEtools.LinearSVR(random_state=seed, max_iter=10000)
    permutation_run(regr, cv, outdir, step_size)
    ## Elastic net
    outdir = 'enet/'
    mkdir_p(outdir)
    regr = dRFEtools.ElasticNet(alpha=0.01, random_state=seed)
    permutation_run(regr, cv, outdir, step_size)
    ## Random forest
    outdir = 'rf/'
    mkdir_p(outdir)
    regr = dRFEtools.RandomForestRegressor(n_estimators=100, oob_score=True,
                                           n_jobs=-1, random_state=seed)
    permutation_run(regr, cv, outdir, step_size)


if __name__ == '__main__':
    main()
