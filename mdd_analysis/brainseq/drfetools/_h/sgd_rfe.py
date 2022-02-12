"""
This script uses dRFEtools to classify MDD versus control
in BrainSeq Phase 1. This is a poly-A dataset.
"""
import dRFEtools
import numpy as np
import pandas as pd
from gtfparse import read_gtf
from matplotlib import rcParams
import matplotlib.pyplot as plt
import errno, os, functools, argparse
from rpy2.robjects import r, pandas2ri, globalenv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import normalized_mutual_info_score as nmi

@functools.lru_cache()
def get_gtf(gtf_file):
    return read_gtf(gtf_file)


def gene_annotation(gtf_file, feature="gene"):
    gtf0 = get_gtf(gtf_file)
    gtf = gtf0[gtf0["feature"] == feature]
    return gtf[["gene_id", "gene_name", "gene_type",
                "seqname", "start", "end", "strand"]]


def mkdir_p(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_data(fn):
    pandas2ri.activate()
    globalenv['fn'] = fn
    r('''
    suppressMessages({library(tidyverse)})
    load(fn)
    expr = as.data.frame(v$E)
    model = data.frame(v$design)
    genes = rownames(expr)
    weights = as.data.frame(v$weights) %>% mutate(rowname=rownames(v$E)) %>%
      column_to_rownames(var="rowname")
    colnames(weights) = colnames(v$E)
    ''')
    return r['expr'].T, r['model'].MDD.astype("int64")


def extract_feature_annotation(pred_feat, path, fold, gtf_file):
    # Get important features
    dft = pd.DataFrame.from_records(pred_feat,
                                    columns=['feature_importance',
                                             'Geneid'])
    dft['Fold'] = fold
    # Get gene annotation
    annot = gene_annotation(gtf_file)
    annot = annot.loc[:, ['gene_id', 'gene_name', 'gene_type', 'seqname']]\
                 .rename(columns={'seqname': 'chrom'})
    annot['ensemblID'] = annot.gene_id.str.replace("\\..*", "", regex=True)
    pred_df = dft.merge(annot, how='left', left_on='Geneid', right_on='gene_id')
    pred_df.to_csv('%s/important_features.txt' % path,
                   sep='\t', mode='a', index=False,
                   header=True if fold == 0 else False)


def R_function():
    pandas2ri.activate()
    r('''
    ml_residuals <- function(train_indices, test_indices)
    {
                    # Subset for training data
    expression_train = expr[, train_indices]
                    # Null model
    null_model_train = v$design[train_indices, ] %>% as.data.frame %>%
        select(-c(starts_with("MDD"))) %>% as.matrix
                    # Fit model
    fit_train = limma::lmFit(expression_train, design=null_model_train)
                    # Calculate residuals from training data
    residuals_train = expression_train - (fit_train$coefficients %*% t(null_model_train))
    residuals_train_sd = apply(residuals_train, 1, sd)
    residuals_train_mean = apply(residuals_train, 1, mean)
                    # Normalize residuals
    residuals_train_norm = (residuals_train - residuals_train_mean) / residuals_train_sd
                    # Subset for test data
    expression_test = expr[, test_indices]
                    # Null model
    null_model_test = v$design[test_indices, ] %>% as.data.frame %>%
        select(-c(starts_with("MDD"))) %>% as.matrix
                    # Apply training to test data and normalize
    residuals_test = expression_test - (fit_train$coefficients %*% t(null_model_test))
    residuals_test_norm = (residuals_test - residuals_train_mean) / residuals_train_sd
    return (list(residuals_train_norm = residuals_train_norm,
                 residuals_test_norm = residuals_test_norm))
    }
    ''')


def residualize(train_index, test_index):
    pandas2ri.activate()
    R_function()
    globalenv['train_index'] = train_index+1 # Adjust for python
    globalenv['test_index'] = test_index+1 # Adjust for python
    r('''
    res_mx <- ml_residuals(train_index, test_index)
    X_test = data.frame(t(res_mx$residuals_test_norm),
                        row.names=row.names(v$targets[test_index,]))
    colnames(X_test) <- row.names(v$genes)
    X_train = data.frame(t(res_mx$residuals_train_norm),
                        row.names=row.names(v$targets[train_index,]))
    colnames(X_train) <- row.names(v$genes)
    ''')
    return r['X_train'], r['X_test']


def optimize_dev(X, Y, estimator, cv, outdir):
    fold_num = 0
    for train_index, test_index in cv.split(X, Y):
        Y_train, Y_test = Y[train_index], Y[test_index]
        fold_num += 1
    fold_num -= 1
    X_train, X_test = residualize(train_index, test_index)
    features = X_train.columns
    d, pfirst = dRFEtools.dev_rfe(estimator, X_train.values, Y_train,
                                  features, fold_num, outdir,
                                  elimination_rate=0.1, RANK=False)
    for frac in [0.2, 0.25, 0.3, 0.35]:
        plt.clf()
        dRFEtools.optimize_lowess_plot(d, fold_num, outdir, frac=frac,
                                       step_size=0.05,classify=True,
                                       save_plot=True)
    ## Default 0.35 looks good for frac
    for step in [0.01, 0.02, 0.03, 0.04]:
        plt.clf()
        dRFEtools.optimize_lowess_plot(d, fold_num, outdir, frac=0.35,
                                       step_size=step,classify=True,
                                       save_plot=True)


def dev_run(x_train, x_test, y_train, y_test, fold, outdir,
            gtf_file, estimator, frac, step):
    features = x_train.columns
    d, pfirst = dRFEtools.dev_rfe(estimator, x_train.values, y_train.values,
                                  features, fold, outdir, elimination_rate=0.1)
    df_elim = pd.DataFrame([{'fold':fold, 'n features':k,
                             'NMI score':d[k][1], 'Accuracy score':d[k][2],
                             'ROC AUC score':d[k][3]} for k in d.keys()])
    n_features_max = max(d, key=lambda x: d[x][1])
    try:
        ## Max features from lowess curve
        ### multiple classification is False by default
        n_features, _ = dRFEtools.extract_max_lowess(d, frac=frac)
        n_redundant, _ = dRFEtools.extract_redundant_lowess(d, frac=frac,
                                                            step_size=step)
        dRFEtools.plot_with_lowess_vline(d, fold, outdir, frac=frac,
                                         step_size=step, multi=False,
                                         classify=True)
    except ValueError:
        ## For errors in lowess estimate
        n_features = n_features_max
        n_redundant = n_features
    ## Fit model
    estimator.fit(x_train, y_train)
    all_fts = estimator.predict(x_test)
    estimator.fit(x_train.values[:, d[n_redundant][4]], y_train)
    labels_pred_redundant = estimator.predict(x_test.values[:, d[n_redundant][4]])
    estimator.fit(x_train.values[:,d[n_features][4]], y_train)
    labels_pred = estimator.predict(x_test.values[:, d[n_features][4]])
    ## Output test predictions
    kwargs = {"average": "weighted"}
    pd.DataFrame({'fold': fold, 'real': y_test,
                  'predict_all': all_fts, 'predict_max': labels_pred,
                  'predict_redundant': labels_pred_redundant},
                 index=x_test.index)\
      .to_csv("%s/test_predictions.txt" % outdir, sep='\t', mode='a',
              index=True, header=True if fold == 0 else False)
    # Annotate features
    pred_features = sorted(list(zip(estimator.feature_importances_,
                                    x_train.columns[d[n_features][4]])),
                           reverse=True)
    extract_feature_annotation(pred_features, outdir, fold, gtf_file)
    # Save output data
    output = dict()
    output['n_max'] = n_features_max
    output['n_features'] = n_features
    output['n_redundant'] = n_redundant
    output['n_features_all_features'] = pfirst[0]
    output['train_nmi_all_features'] = pfirst[1]
    output['train_acc_all_features'] = pfirst[2]
    output['train_roc_all_features'] = pfirst[3]
    output['train_nmi'] = dRFEtools.dev_score_nmi(estimator,
                                                  x_train.values[:,d[n_features][4]],
                                                  y_train)
    output['train_acc'] = dRFEtools.dev_score_accuracy(estimator,
                                                       x_train.values[:,d[n_features][4]],
                                                       y_train)
    output['train_roc'] = dRFEtools.dev_score_roc(estimator,
                                                  x_train.values[:,d[n_features][4]],
                                                  y_train)
    output['test_nmi'] = nmi(y_test, labels_pred, average_method="arithmetic")
    output['test_acc'] = accuracy_score(y_test, labels_pred)
    output['test_roc'] = roc_auc_score(y_test, labels_pred, **kwargs)
    output['test_acc_redundant'] = accuracy_score(y_test, labels_pred_redundant)
    output['test_roc_redundant'] = roc_auc_score(y_test, labels_pred_redundant,
                                                 **kwargs)
    output['test_nmi_redundant'] = nmi(y_test, labels_pred_redundant,
                                       average_method="arithmetic")
    return output, df_elim


def main_loop(args):
    outdir = "sgd"
    mkdir_p(outdir)
    ## Prep model
    cla = dRFEtools.SGDClassifier(n_jobs=-1, loss="perceptron",
                                  random_state=20211229)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=20211004)
    ## Load data
    X, Y = load_data(args.voom_file)
    skf.get_n_splits(X, Y)
    ## Main analysis
    frac = 0.30; step_size = 0.04; fold = 0
    fields = ['n_features_all_features', 'train_acc_all_features',
              'train_nmi_all_features', 'train_roc_all_features', 'n_max',
              'n_features', 'train_acc', 'train_nmi', 'train_roc', 'test_acc',
              'test_nmi', 'test_roc', 'n_redundant', 'test_acc_redundant',
              'test_nmi_redundant', 'test_roc_redundant']
    df_dict = pd.DataFrame()
    with open("%s/dRFEtools_10folds.txt" % (outdir), "w") as f:
        print("\t".join(["fold"] + fields), file=f, flush=True)
        for train_index, test_index in skf.split(X, Y):
            X_train, X_test = residualize(train_index, test_index)
            Y_train, Y_test = Y[train_index], Y[test_index]
            o, df_elim = dev_run(X_train, X_test, Y_train, Y_test, fold, outdir,
                                 args.gtf_file, cla, frac, step_size)
            df_dict = pd.concat([df_dict, df_elim], axis=0)
            print("\t".join([str(fold)] + [str(o[x]) for x in fields]),
                  flush=True, file=f)
            fold += 1
        df_dict.to_csv("%s/feature_elimination_allFolds_metrics.txt" % outdir,
                       sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gtf_file', type=str)
    parser.add_argument('--voom_file', type=str)
    args=parser.parse_args()
    os.environ['NUMEXPR_MAX_THREADS'] = '32'
    rcParams.update({'figure.max_open_warning': 0})
    main_loop(args)


if __name__ == '__main__':
    main()
