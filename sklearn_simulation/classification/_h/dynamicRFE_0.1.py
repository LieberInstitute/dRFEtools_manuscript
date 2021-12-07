# dynamic RFE simulations for binary classification (10%)
"""
This script runs RFE with cross-validation for the 10 binary classification
simulation.
"""

import numpy as np
import pandas as pd
from time import time
import os,errno,dRFEtools
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import normalized_mutual_info_score as nmi

def mkdir_p(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_data(simu):
    state = 13 + simu
    X, y = make_classification(
        n_samples=500, n_features=20000, n_informative=100, n_redundant=300,
        n_repeated=0, n_classes=2, n_clusters_per_class=1, random_state=state,
        shuffle=False,
    )
    return X, y


def run_oob(estimator, x_train, x_test, y_train, y_test, fold, outdir,
            frac, step, simu, elim_rate):
    features = ["feature_%d" % x for x in range(x_train.shape[1])]
    d, pfirst = dRFEtools.rf_rfe(estimator, x_train, y_train, np.array(features),
                                 fold, outdir, elimination_rate=elim_rate,
                                 RANK=True)
    df_elim = pd.DataFrame([{'fold':fold, "simulation": simu,
                             'elimination': elim_rate, 'n features':k,
                             'NMI score':d[k][1], 'Accuracy score':d[k][2],
                             'ROC AUC score':d[k][3]} for k in d.keys()])
    n_features_max = max(d, key=lambda x: d[x][1])
    try:
        ## Max features from lowess curve
        n_features, _ = dRFEtools.extract_max_lowess(d, frac=frac, multi=False)
        n_redundant, _ = dRFEtools.extract_redundant_lowess(d, frac=frac,
                                                            step_size=step,
                                                            multi=False)
        if elim_rate == 0.1:
            dRFEtools.plot_with_lowess_vline(d, fold, outdir, frac=frac,
                                             step_size=step, multi=False)
    except ValueError:
        ## For errors in lowess estimate
        n_features = n_features_max
        n_redundant = n_features
    ## Fit model
    estimator.fit(x_train, y_train)
    all_fts = estimator.predict(x_test)
    estimator.fit(x_train[:, d[n_redundant][4]], y_train)
    labels_pred_redundant = estimator.predict(x_test[:, d[n_redundant][4]])
    estimator.fit(x_train[:,d[n_features][4]], y_train)
    labels_pred = estimator.predict(x_test[:, d[n_features][4]])
    ## Output test predictions
    kwargs = {"average": "weighted"}
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
    output['train_nmi'] = dRFEtools.oob_score_nmi(estimator, y_train)
    output['train_acc'] = dRFEtools.oob_score_accuracy(estimator, y_train)
    output['train_roc'] = dRFEtools.oob_score_roc(estimator, y_train)
    output['test_nmi'] = nmi(y_test, labels_pred, average_method="arithmetic")
    output['test_acc'] = accuracy_score(y_test, labels_pred)
    output['test_roc'] = roc_auc_score(y_test, labels_pred, **kwargs)
    metrics_df = pd.DataFrame.from_records(output, index=[simu])\
                             .reset_index().drop('index', axis=1)
    return df_elim, metrics_df


def run_dev(estimator, x_train, x_test, y_train, y_test, fold, outdir,
            frac, step, simu, elim_rate):
    features = ["feature_%d" % x for x in range(x_train.shape[1])]
    d, pfirst = dRFEtools.dev_rfe(estimator, x_train, y_train, np.array(features),
                                  fold, outdir, elimination_rate=elim_rate,
                                  RANK=True)
    df_elim = pd.DataFrame([{'fold':fold, "simulation": simu,
                             'elimination': elim_rate, 'n features':k,
                             'NMI score':d[k][1], 'Accuracy score':d[k][2],
                             'ROC AUC score':d[k][3]} for k in d.keys()])
    n_features_max = max(d, key=lambda x: d[x][1])
    try:
        ## Max features from lowess curve
        ### multiple classification is False by default
        n_features, _ = dRFEtools.extract_max_lowess(d, frac=frac)
        n_redundant, _ = dRFEtools.extract_redundant_lowess(d, frac=frac,
                                                            step_size=step)
        if elim_rate == 0.1:
            dRFEtools.plot_with_lowess_vline(d, fold, outdir, frac=frac,
                                             step_size=step, multi=False)
    except ValueError:
        ## For errors in lowess estimate
        n_features = n_features_max
        n_redundant = n_features
    ## Fit model
    #x_dev, x_test, y_dev, y_test = train_test_split(x_train, y_train)
    estimator.fit(x_train, y_train)
    all_fts = estimator.predict(x_test)
    estimator.fit(x_train[:, d[n_redundant][4]], y_train)
    labels_pred_redundant = estimator.predict(x_test[:, d[n_redundant][4]])
    estimator.fit(x_train[:,d[n_features][4]], y_train)
    labels_pred = estimator.predict(x_test[:, d[n_features][4]])
    ## Output test predictions
    kwargs = {"average": "weighted"}
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
    output['train_nmi'] = dRFEtools.dev_score_nmi(estimator,
                                                  x_train[:,d[n_features][4]],
                                                  y_train)
    output['train_acc'] = dRFEtools.dev_score_accuracy(estimator,
                                                       x_train[:,d[n_features][4]],
                                                       y_train)
    output['train_roc'] = dRFEtools.dev_score_roc(estimator,
                                                  x_train[:,d[n_features][4]],
                                                  y_train)
    output['test_nmi'] = nmi(y_test, labels_pred, average_method="arithmetic")
    output['test_acc'] = accuracy_score(y_test, labels_pred)
    output['test_roc'] = roc_auc_score(y_test, labels_pred, **kwargs)
    metrics_df = pd.DataFrame.from_records(output, index=[simu])\
                             .reset_index().drop('index', axis=1)
    return df_elim, metrics_df


def dRFE_run(estimator, outdir, simu, elim_rate, cv, run_fnc):
    X, y = load_data(simu)
    simu_out = "%s/simulate_%d" % (outdir, simu)
    mkdir_p(simu_out)
    ## default parameters
    frac = 0.3; step=0.05; fold = 0
    df_dict = pd.DataFrame(); output = pd.DataFrame()
    start = time()
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        df_elim, metrics_df = run_fnc(estimator, X_train, X_test, y_train,
                                      y_test, fold, simu_out, frac, step, simu,
                                      elim_rate)
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


def permutation_run(estimator, outdir, elim_rate, cv, run_fnc):
    cpu_lt = []; simu_lt = []
    for simu in range(10):
        cpu = dRFE_run(estimator, outdir, simu, elim_rate, cv, run_fnc)
        simu_lt.append(simu)
        cpu_lt.append(cpu)
    pd.DataFrame({"Simulation": simu_lt, "CPU Time": cpu_lt})\
      .to_csv("%s/simulation_time_%.2f.csv" % (outdir, elim_rate), index=False)


def main():
    ## Generate 10-fold cross-validation
    seed = 13; elim_rate = 0.1
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=20211005)
    ## Logistic regression
    outdir = 'lr/'
    mkdir_p(outdir)
    cla = dRFEtools.LogisticRegression(n_jobs=-1, random_state=seed,
                                       max_iter=1000, penalty="l2")
    permutation_run(cla, outdir, elim_rate, cv, run_dev)
    ## SGD classifier
    outdir = 'sgd/'
    mkdir_p(outdir)
    cla = dRFEtools.SGDClassifier(random_state=seed, n_jobs=-1,
                                  loss="perceptron")
    permutation_run(cla, outdir, elim_rate, cv, run_dev)
    ## SVC linear kernel
    outdir = 'svc/'
    mkdir_p(outdir)
    cla = dRFEtools.LinearSVC(random_state=seed, max_iter=10000)
    permutation_run(cla, outdir, elim_rate, cv, run_dev)
    ## Random forest
    outdir = 'rf/'
    mkdir_p(outdir)
    cla = dRFEtools.RandomForestClassifier(n_estimators=100, oob_score=True,
                                           n_jobs=-1, random_state=seed)
    permutation_run(cla, outdir, elim_rate, cv, run_oob)


if __name__ == '__main__':
    main()
