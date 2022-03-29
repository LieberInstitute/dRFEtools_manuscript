# Examine enrichment in psychiatric disorders TWAS and DEGs
import numpy as np
import pandas as pd
from os import environ
from functools import lru_cache
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

environ['NUMEXPR_MAX_THREADS'] = '32'

@lru_cache()
def get_degs(dx):
    config = {"MDD": "MDD", "Schizo": "SZ", "Bipolar": "BD"}
    return pd.read_csv("../../../differential_analysis/_m/%s/diffExpr_CTLvs%s_full.txt" %
                       (dx.lower(), config[dx]), sep='\t', index_col=0)


@lru_cache()
def get_predictive(dx):
    df = pd.read_csv("../../metrics_summary/_m/dRFE_predictive_features.txt.gz",
                     sep='\t')
    return df[(df["Diagnosis"] == dx)].copy()


@lru_cache()
def get_redundant(dx):
    df = pd.read_csv("../../metrics_summary/_m/dRFE_redundant_features.txt.gz",
                     sep='\t')
    return df[(df["Diagnosis"] == dx)].copy()


def fet(dx, fnc):
    """
    Calculates Fisher's Exact test (fet) with sets a and b in universe u.
    Inputs are sets.
    """
    u = set(get_degs(dx).gencodeID)
    a = set(get_degs(dx)[(get_degs(dx)["adj.P.Val"] < 0.05)].gencodeID)
    b = set(get_degs(dx).merge(fnc(dx), left_index=True,
                               right_on="Geneid").gencodeID)
    yes_a = u.intersection(a)
    yes_b = u.intersection(b)
    no_a = u - a
    no_b = u - b
    m = [[len(yes_a.intersection(yes_b)), len(no_a.intersection(yes_b))],
         [len(yes_a.intersection(no_b)), len(no_a.intersection(no_b))]]
    #print(m)
    return fisher_exact(m)


def enrichment_loop(fnc, label):
    or_lt = []; pval_lt = []; dx_lt = []; set_lt = [];
    for diagnosis in ["MDD", "Schizo"]:
        oddratio, pvals = fet(diagnosis, fnc)
        or_lt.append(oddratio); pval_lt.append(pvals);
        dx_lt.append(diagnosis); set_lt.append(label)
    return pd.DataFrame({"Diagnosis": dx_lt, "Type": set_lt,
                         "OR": or_lt, "P-value": pval_lt})


def main():
    ## Enrichment
    dt1 = enrichment_loop(get_predictive, "Minimum")
    dt2 = enrichment_loop(get_redundant, "Redundant")
    dt = pd.concat([dt1, dt2], axis=0)
    fdr = multipletests(dt.loc[:, "P-value"], method='fdr_bh')[1]
    dt["FDR"] = fdr
    ## Save enrichment
    dt.to_csv("enrichment_analysis_DEGs_vs_ML.tsv",
              sep='\t', index=False)


if __name__ == '__main__':
    main()
