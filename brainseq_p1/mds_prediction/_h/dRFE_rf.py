"""
This script performs dRFEtools on genotypes (0, 1, 2) to predict
MDS of snpPCs.

It examines dimensional reduction and which SNPs contribute the
most to variance explained.

This is LD SNPs from TOPMed for BrainSEQ phase 1 individuals.
"""

import numpy as np
import pandas as pd
from time import time
from pandas_plink import read_plink
import os, errno, dRFEtools, functools
from sklearn.model_selection import KFold
from rpy2.robjects import r, pandas2ri, globalenv
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import explained_variance_score as evar


def mkdir_p(directory):
    """
    This function attempts to create a new directory if one
    does not already exist.
    """
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


@functools.lru_cache()
def get_phenotypes():
    """
    Load phenotype data
    """
    return pd.read_csv("../../_m/phenotypes_bsp1.tsv", sep='\t', index_col=0)


@functools.lru_cache()
def cached_read_plink(file_prefix, verbose=True):
    """
    Load plink files
    """
    return read_plink(file_prefix, verbose)


@functools.lru_cache()
def get_plink():
    """
    TopMed genotypes in plink format
    """
    file_prefix = "/ceph/projects/brainseq/genotype/download/topmed/"+\
        "imputation_filter/convert2plink/filter_maf_01/mds/_m/LIBD_Brain_TopMed"
    return cached_read_plink(file_prefix)


@functools.lru_cache()
def load_mds():
    """
    Loads MDS file
    """
    fn = "/ceph/projects/brainseq/genotype/download/topmed/imputation_filter/"+\
        "convert2plink/filter_maf_01/mds/python_version_mds/_m/"+\
        "LIBD_Brain_TopMed.mds"
    return pd.read_csv(fn, sep='\t', index_col=0).drop(["IID","SOL"], axis=1)


@functools.lru_cache()
def select_snps_from_plink_to_df():
    """
    Given a <bim, fam, bed> triple of files and returns a pandas DataFrame
    with the SNP genotype values (0,1,2), where the rows are family IDs
    and columns are SNP IDs.
    """
    (bim, fam, bed) = get_plink()
    dx = pd.DataFrame(bed.compute().transpose())
    dx.index = fam['fid']
    dx.columns = bim.snp
    return dx


def one_hot_encode_snp_df(snp_df):
    """
    Given a snp_df returned by select_snps_from_plink_to_df function,
    returns a one-hot-encoded version of it
    """
    dx = pd.get_dummies(snp_df, columns=snp_df.columns, dummy_na=True)
    dx.columns = dx.columns.str.replace('\.\d+', '', regex=True)
    return dx


@functools.lru_cache()
def load_data(pc):
    """
    Given PC (integer), this function:
    1) loads SNP dataframe
    2) converts SNP dataframe to onehot encoded version
    3) loads phenotypes from MDS dataframe and extracts specific PC
    """
    samples = list(set(load_mds().index) & set(get_phenotypes().BrNum))
    snp_df = select_snps_from_plink_to_df()\
        .loc[samples,:]\
        .reset_index()\
        .drop_duplicates(subset="fid")\
        .set_index("fid")
    X = one_hot_encode_snp_df(snp_df)
    Y = load_mds().loc[samples, :]\
                  .reset_index()\
                  .drop_duplicates(subset="FID")\
                  .set_index("FID")
    return X, Y["snpPC%d"%pc]


def rf_run(estimator, x_train, x_test, y_train, y_test, fold, outdir,
           frac, step, elim_rate, pc):
    features = x_train.columns
    d, pfirst = dRFEtools.rf_rfe(estimator, x_train, y_train, features,
                                 fold, outdir, elimination_rate=elim_rate,
                                 RANK=True)
    df_elim = pd.DataFrame([{'fold':fold, 'pc':pc, 'n features':k,
                             'R2 score':d[k][1], 'Mean Square Error':d[k][2],
                             'Explained Variance':d[k][3]} for k in d.keys()])
    n_features_max = max(d, key=lambda x: d[x][1])
    try:
        ## Max features from lowess curve
        n_features, _ = dRFEtools.extract_max_lowess(d, frac=frac, multi=False)
        n_redundant, _ = dRFEtools.extract_redundant_lowess(d, frac=frac,
                                                            step_size=step,
                                                            multi=False)
        if elim_rate == 0.1:
            dRFEtools.plot_with_lowess_vline(d, fold, outdir, frac=frac,
                                             step_size=step, classify=False)
    except ValueError:
        ## For errors in lowess estimate
        n_features = n_features_max
        n_redundant = n_features
    ## Fit model
    estimator.fit(x_train, y_train)
    all_fts = estimator.predict(x_test)
    estimator.fit(x_train.iloc[:, d[n_redundant][4]], y_train)
    labels_pred_redundant = estimator.predict(x_test.iloc[:, d[n_redundant][4]])
    estimator.fit(x_train.iloc[:,d[n_features][4]], y_train)
    labels_pred = estimator.predict(x_test.iloc[:, d[n_features][4]])
    ## Output test predictions
    pd.DataFrame({'fold': fold, 'pc': pc, 'real': y_test,
                  'predict_all':all_fts, 'predict_max': labels_pred,
                  'predict_redundant': labels_pred_redundant})\
      .to_csv("%s/test_predictions.txt" % outdir, sep='\t', mode='a',
              index=True, header=True if fold == 0 else False)
    output = dict()
    output['fold'] = fold; output['pc'] = pc
    output['n_features'] = n_features
    output['n_redundant'] = n_redundant
    output['n_max'] = n_features_max
    output['train_r2'] = dRFEtools.oob_score_r2(estimator, y_train)
    output['train_mse'] = dRFEtools.oob_score_mse(estimator, y_train)
    output['train_evar'] = dRFEtools.oob_score_evar(estimator, y_train)
    output['test_r2'] = r2_score(y_test, labels_pred)
    output['test_mse'] = mean_squared_error(y_test, labels_pred)
    output['test_evar'] = evar(y_test, labels_pred,
                               multioutput="uniform_average")
    metrics_df = pd.DataFrame.from_records(output, index=[pc])\
                             .reset_index().drop('index', axis=1)
    return df_elim, metrics_df


def dRFE_run(estimator, outdir, elim_rate, cv, pc):
    X, y = load_data(pc)
    pc_out = "%s/snpPC_%d" % (outdir, pc)
    mkdir_p(pc_out)
    ## default parameters
    frac = 0.3; step=0.04; fold = 0
    df_dict = pd.DataFrame(); output = pd.DataFrame()
    start = time()
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        df_elim, metrics_df = rf_run(estimator, X_train, X_test, y_train,y_test,
                                     fold, pc_out, frac, step, elim_rate, pc)
        df_dict = pd.concat([df_dict, df_elim], axis=0)
        output = pd.concat([output, metrics_df], axis=0)
        fold += 1
    end = time()
    df_dict.to_csv("%s/dRFE_snpPC_elimination.txt" % outdir,
                   sep='\t', mode='a', index=False,
                   header=True if pc == 0 else False)
    output.to_csv("%s/dRFE_snpPC_metrics.txt" % outdir,
                  sep='\t', mode='a', index=False,
                  header=True if pc == 0 else False)
    return end - start


def permutation_run(estimator, outdir, elim_rate, cv):
    cpu_lt = []; pc_lt = []
    for pc in range(1, 11):
        cpu = dRFE_run(estimator, outdir, elim_rate, cv, pc)
        pc_lt.append(pc)
        cpu_lt.append(cpu)
    pd.DataFrame({"snpPC": pc_lt, "CPU Time": cpu_lt})\
      .to_csv("%s/snpPC_time.csv" % (outdir), index=False)


def main():
    ## Generate 10-fold cross-validation
    seed = 13; elim_rate = 0.2
    cv = KFold(n_splits=10, shuffle=True, random_state=seed)
    ## Random forest
    outdir = './'
    regr = dRFEtools.RandomForestRegressor(n_estimators=100, oob_score=True,
                                           n_jobs=-1, random_state=seed)
    permutation_run(regr, outdir, elim_rate, cv)


if __name__ == '__main__':
    main()
