"""
Gnerate the accuracy results for each fold, model, and algorithm.
"""
import pandas as pd
from functools import lru_cache

@lru_cache()
def get_test_results():
    return pd.read_csv("../../_m/rf/test_predictions.txt", sep='\t',index_col=0)


@lru_cache()
def subset_diagnosis(num, fold):
    true = get_test_results()[(get_test_results()["real"] == num) &
                              (get_test_results()["fold"] == fold)].copy()
    false = get_test_results()[(get_test_results()["real"] != num) &
                               (get_test_results()["fold"] == fold)].copy()
    return true, false


@lru_cache()
def drfe_accuracy(dx, fold):
    config = {"CTL": 0, "MDD": 1, "SZ": 2}
    true, false = subset_diagnosis(config[dx], fold)
    TP = sum(true.real == true.predict_max_cat)
    TN = sum(false.predict_max_cat != config[dx])
    FP = sum(false.predict_max_cat == config[dx])
    FN = sum(true.predict_max_cat != config[dx])
    return TP, FN, TN, FP


def get_metrics():
    tp_lt = []; fn_lt = []; tn_lt = []; fp_lt = []; fold_lt = []; dx_lt = []
    for fold in range(10):
        for dx in ["CTL", "MDD", "SZ"]:
            tp, fn, tn, fp = drfe_accuracy(dx, fold)
            fold_lt.append(fold); tp_lt.append(tp); fn_lt.append(fn)
            tn_lt.append(tn); fp_lt.append(fp); dx_lt.append(dx)
    return pd.DataFrame({"Diagnosis": dx_lt, "Fold": fold_lt, "TP": tp_lt,
                         "FN": fn_lt, "TN": tn_lt, "FP": fp_lt})


def main():
    get_metrics()\
        .to_csv("binary_test_accuracy_metrics.tsv", sep='\t', index=False)


if __name__ == "__main__":
    main()
