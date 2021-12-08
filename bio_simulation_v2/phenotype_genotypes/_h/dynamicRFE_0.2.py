# dynamic RFE simulations with phenotype ~ genotypes (20%)
"""
This script runs dynamic RFE with cross-validation for the 15
traits. Using feature elmination to impute trait from genotypes.
"""

import numpy as np
import pandas as pd
from time import time
import os,errno,dRFEtools,functools
from sklearn.model_selection import KFold
from rpy2.robjects import r, pandas2ri, globalenv
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import explained_variance_score as evar


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


def run_oob(estimator, x_train, x_test, y_train, y_test, fold, outdir,
            frac, step, simu, elim_rate):
    features = x_train.columns
    d, pfirst = dRFEtools.rf_rfe(estimator, x_train, y_train, features,
                                 fold, outdir, elimination_rate=elim_rate,
                                 RANK=True)
    df_elim = pd.DataFrame([{'fold':fold, "simulation": simu,
                             'elimination': elim_rate, 'n features':k,
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
    pd.DataFrame({'fold': fold, "simulation": simu, "elimination": elim_rate,
                  'real': y_test, 'predict_all': all_fts,
                  'predict_max': labels_pred,
                  'predict_redundant': labels_pred_redundant})\
      .to_csv("%s/test_predictions.txt" % outdir, sep='\t', mode='a',
              index=True, header=True if fold == 0 else False)
    output = dict()
    output['fold'] = fold
    output['simulation'] = simu
    output['elimination'] = elim_rate
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
    metrics_df = pd.DataFrame.from_records(output, index=[simu])\
                             .reset_index().drop('index', axis=1)
    return df_elim, metrics_df


def run_dev(estimator, x_train, x_test, y_train, y_test, fold, outdir,
                 frac, step, simu, elim_rate):
    features = x_train.columns
    d, pfirst = dRFEtools.dev_rfe(estimator, x_train, y_train, features,
                                  fold, outdir, elimination_rate=elim_rate,
                                  RANK=True)
    df_elim = pd.DataFrame([{'fold':fold, "simulation": simu,
                             'elimination': elim_rate, 'n features':k,
                             'R2 score':d[k][1], 'Mean Square Error':d[k][2],
                             'Explained Variance':d[k][3]} for k in d.keys()])
    n_features_max = max(d, key=lambda x: d[x][1])
    try:
        ## Max features from lowess curve
        ### multiple classification is False by default
        n_features, _ = dRFEtools.extract_max_lowess(d, frac=frac)
        n_redundant, _ = dRFEtools.extract_redundant_lowess(d, frac=frac,
                                                            step_size=step)
        if elim_rate == 0.1:
            dRFEtools.plot_with_lowess_vline(d, fold, outdir, frac=frac,
                                             step_size=step, classify=False)
    except ValueError:
        ## For errors in lowess estimate
        n_features = n_features_max
        n_redundant = n_features
    ## Fit model
    #x_dev, x_test, y_dev, y_test = train_test_split(x_train, y_train)
    estimator.fit(x_train, y_train)
    all_fts = estimator.predict(x_test)
    estimator.fit(x_train.iloc[:, d[n_redundant][4]], y_train)
    labels_pred_redundant = estimator.predict(x_test.iloc[:, d[n_redundant][4]])
    estimator.fit(x_train.iloc[:,d[n_features][4]], y_train)
    labels_pred = estimator.predict(x_test.iloc[:, d[n_features][4]])
    ## Output test predictions
    pd.DataFrame({'fold': fold, "simulation": simu, "elimination": elim_rate,
                  'real': y_test, 'predict_all': all_fts,
                  'predict_max': labels_pred,
                  'predict_redundant': labels_pred_redundant})\
      .to_csv("%s/test_predictions.txt" % outdir, sep='\t', mode='a',
              index=True, header=True if fold == 0 else False)
    output = dict()
    output['fold'] = fold
    output['simulation'] = simu
    output['elimination'] = elim_rate
    output['n_features'] = n_features
    output['n_redundant'] = n_redundant
    output['n_max'] = n_features_max
    output['train_r2'] = dRFEtools.dev_score_r2(estimator,
                                                x_train.iloc[:,d[n_features][4]],
                                                y_train)
    output['train_mse'] = dRFEtools.dev_score_mse(estimator,
                                                  x_train.iloc[:,d[n_features][4]],
                                                  y_train)
    output['train_evar'] = dRFEtools.dev_score_evar(estimator,
                                                    x_train.iloc[:,d[n_features][4]],
                                                    y_train)
    output['test_r2'] = r2_score(y_test, labels_pred)
    output['test_mse'] = mean_squared_error(y_test, labels_pred)
    output['test_evar'] = evar(y_test, labels_pred,
                               multioutput="uniform_average")
    metrics_df = pd.DataFrame.from_records(output, index=[simu])\
                             .reset_index().drop('index', axis=1)
    return df_elim, metrics_df


def dRFE_run(estimator, outdir, simu, elim_rate, cv, frac, step):
    X, y = load_data(simu)
    simu_out = "%s/simulate_%d" % (outdir, simu)
    mkdir_p(simu_out)
    ## default parameters
    fold = 0
    df_dict = pd.DataFrame(); output = pd.DataFrame()
    start = time()
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        df_elim, metrics_df = run_dev(estimator, X_train, X_test, y_train,
                                      y_test, fold, simu_out, frac, step,
                                      simu, elim_rate)
        df_dict = pd.concat([df_dict, df_elim], axis=0)
        output = pd.concat([output, metrics_df], axis=0)
        fold += 1
    end = time()
    df_dict.to_csv("%s/dRFE_simulation_elimination.txt" % outdir,
                   sep='\t', mode='a', index=False,
                   header=True if simu == 0 else False)
    output.to_csv("%s/dRFE_simulation_metrics.txt" % outdir,
                  sep='\t', mode='a', index=False,
                  header=True if simu == 0 else False)
    return end - start


def permutation_run(estimator, outdir, elim_rate, cv, frac, step):
    cpu_lt = []; simu_lt = []
    for simu in range(15):
        cpu = dRFE_run(estimator, outdir, simu, elim_rate, cv, frac, step)
        simu_lt.append(simu)
        cpu_lt.append(cpu)
    pd.DataFrame({"Simulation": simu_lt, "CPU Time": cpu_lt})\
      .to_csv("%s/simulation_time_%.2f.csv" % (outdir, elim_rate), index=False)


def main():
    ## Generate 10-fold cross-validation
    seed = 13; elim_rate = 0.2
    cv = KFold(n_splits=10, shuffle=True, random_state=seed)
    ## Ridge
    outdir = 'ridge/'
    mkdir_p(outdir)
    regr = dRFEtools.Ridge(random_state=seed)
    permutation_run(regr, outdir, elim_rate, cv, 0.30, 0.04)
    ## Elastic Net
    outdir = 'enet/'
    mkdir_p(outdir)
    regr = dRFEtools.ElasticNet(random_state=seed, alpha=0.01)
    permutation_run(regr, outdir, elim_rate, cv, 0.30, 0.01)
    ## SVR linear kernel
    outdir = 'svr/'
    mkdir_p(outdir)
    regr = dRFEtools.LinearSVR(random_state=seed, max_iter=10000)
    permutation_run(regr, outdir, elim_rate, cv, 0.20, 0.03)
    ## Random forest
    outdir = 'rf/'
    mkdir_p(outdir)
    regr = dRFEtools.RandomForestRegressor(n_estimators=100, oob_score=True,
                                           n_jobs=-1, random_state=seed)
    permutation_run(regr, outdir, elim_rate, cv, 0.30, 0.04)


if __name__ == '__main__':
    main()