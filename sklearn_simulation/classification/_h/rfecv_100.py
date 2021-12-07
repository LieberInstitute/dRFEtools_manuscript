# RFE simulations for binary classification (100)
"""
This script runs RFE with cross-validation for the 10 binary classification
simulation. This uses the same CV and models as the dRFE version.
"""

import numpy as np
import pandas as pd
from time import time
import os,errno,dRFEtools
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold


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


def rfecv_run(simu, cv, estimator, outdir, step):
    # Instantiate RFECV visualizer with a random forest regression
    X, y = load_data(simu)
    features = ["feature_%d" % x for x in range(X.shape[1])]
    selector = RFECV(estimator, cv=cv, step=step, n_jobs=-1)
    start = time()
    selector = selector.fit(X, y)
    end = time()
    pd.DataFrame({"Simulation": simu,
                  "Feature":features,
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
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=20211005)
    ## Logistic regression
    outdir = 'lr/'
    mkdir_p(outdir)
    cla = dRFEtools.LogisticRegression(n_jobs=-1, random_state=seed,
                                       max_iter=1000, penalty="l2")
    permutation_run(cla, cv, outdir, step_size)
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
    ## Random forest
    outdir = 'rf/'
    mkdir_p(outdir)
    cla = dRFEtools.RandomForestClassifier(n_estimators=100, oob_score=True,
                                           n_jobs=-1, random_state=seed)
    permutation_run(cla, cv, outdir, step_size)


if __name__ == '__main__':
    main()
