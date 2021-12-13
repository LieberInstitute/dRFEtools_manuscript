# RFE simulations for bulk RNA-sequencing (10%)
"""
This script runs RFE with cross-validation for the 10 simulation
to classify tumor versus control lung tissues (0.6, 0.4). This
uses the same CV and models as the dRFE version.
"""

import numpy as np
import pandas as pd
from time import time
import os,errno,dRFEtools
from sklearn.feature_selection import RFECV
from rpy2.robjects import r, pandas2ri, globalenv
from sklearn.model_selection import StratifiedKFold


def mkdir_p(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_data(simu):
    pandas2ri.activate()
    globalenv["simu"] = simu+1
    r('''
    suppressPackageStartupMessages(library(dplyr))
    counts <- data.table::fread(paste0("../../_m/bulk_data/simulated_counts_",
                                       simu,".tsv.gz")) %>%
        tibble::column_to_rownames("V1") %>% as.matrix
    phenotypes <- data.table::fread(paste0("../../_m/bulk_data/simulated_sampleInfo_",
                                           simu, ".tsv")) %>%
        tibble::column_to_rownames("V1") %>% select("Group")
    x <- edgeR::DGEList(counts=counts, samples=phenotypes)
    x <- edgeR::calcNormFactors(x, method="TMM")
    Z <- edgeR::cpm(x, log=TRUE) %>% as.data.frame
    ''')
    return r['Z'].T, r["phenotypes"]


def rfecv_run(simu, cv, estimator, outdir, step):
    # Instantiate RFECV visualizer with a random forest regression
    X, y = load_data(simu)
    y.loc[:, "Group"] = y.Group.astype("category").cat.codes
    selector = RFECV(estimator, cv=cv, step=step, n_jobs=-1)
    start = time()
    selector = selector.fit(X, np.ravel(y))
    end = time()
    pd.DataFrame({"Simulation": simu,
                  "Feature":np.array(X.columns),
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
    step_size = 0.1; seed = 13
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=20211005)
    ## SGD classifier
    outdir = 'sgd/'
    mkdir_p(outdir)
    cla = dRFEtools.SGDClassifier(random_state=seed, n_jobs=-1,
                                  loss="perceptron")
    permutation_run(cla, cv, outdir, step_size)
    ## SVC linear kernel
    outdir = 'svc/'
    mkdir_p(outdir)
    cla = dRFEtools.LinearSVC(random_state=seed, max_iter=10000)
    permutation_run(cla, cv, outdir, step_size)
    ## Logistic regression
    outdir = 'lr/'
    mkdir_p(outdir)
    cla = dRFEtools.LogisticRegression(n_jobs=-1, random_state=seed,
                                       max_iter=1000, penalty="l2")
    permutation_run(cla, cv, outdir, step_size)
    ## Random forest
    outdir = 'rf/'
    mkdir_p(outdir)
    cla = dRFEtools.RandomForestClassifier(n_estimators=100, oob_score=True,
                                           n_jobs=-1, random_state=seed)
    permutation_run(cla, cv, outdir, step_size)


if __name__ == '__main__':
    main()
