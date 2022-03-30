"""
Gnerate the accuracy results for each fold, model, and algorithm.
"""
import functools
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression


def load_data(simu):
    state = 13 + simu
    X, y = make_regression(
        n_samples=500, n_features=20000, n_informative=200, bias=0.2,
        n_targets=1, noise=0.5, random_state=state, shuffle=False,
    )
    return X, y


def get_features(simu):
    X, _ = load_data(simu)
    n_informative = 200
    features = ["feature_%d" % x for x in range(X.shape[1])]
    return features[:n_informative], features[n_informative:]


def rfe_roc_curve(simu, algorithm, elim):
    true, false = get_features(simu)
    if elim == 100:
        fn = "../../_m/%s/RFECV_%d_predictions.txt" % (algorithm, elim)
    else:
        fn = "../../_m/%s/RFECV_%.2f_predictions.txt" % (algorithm, elim)
    df0 = pd.read_csv(fn, sep='\t')
    df0 = df0[(df0["Simulation"] == simu)].copy()
    df1 = pd.DataFrame({"Feature": [x for x in df0.Feature],
                        "Y_predict": [int(x) for x in df0.Predictive]})
    df2 = pd.concat([pd.DataFrame({"Feature":true, "Y_test":1}),
                     pd.DataFrame({"Feature":false, "Y_test":0})], axis=0)
    return pd.merge(df2, df1, on="Feature")


def rfe_accuracy(simu, algorithm, elim):
    true, false = get_features(simu)
    if elim == 100:
        fn = "../../_m/%s/RFECV_%d_predictions.txt" % (algorithm, elim)
    else:
        fn = "../../_m/%s/RFECV_%.2f_predictions.txt" % (algorithm, elim)
    df = pd.read_csv(fn, sep='\t')
    TP = df[(df["Simulation"] == simu) & (df["Feature"].isin(true)) &
            (df["Predictive"])].shape[0]
    FN = df[(df["Simulation"] == simu) & (df["Feature"].isin(true)) &
            ~(df["Predictive"])].shape[0]
    TN = df[(df["Simulation"] == simu) & (df["Feature"].isin(false)) &
            ~(df["Predictive"])].shape[0]
    FP = df[(df["Simulation"] == simu) & (df["Feature"].isin(false)) &
            (df["Predictive"])].shape[0]
    CPU = df.loc[(df["Simulation"] == simu), :]\
            .groupby("Simulation").first().reset_index().loc[:, "CPU Time"]
    return TP, FN, TN, FP, CPU.values[0]


@functools.lru_cache()
def get_n_features(elim, simu, alg):
    fn = "../../_m/%s/dRFEtools_%.2f/dRFE_simulation_metrics.txt" % (alg, elim)
    df = pd.read_csv(fn, sep='\t')
    df = df[(df["simulation"]==simu)].copy()
    return df.median().n_features, df.median().n_max, df.median().n_redundant


@functools.lru_cache()
def get_ranked_features(elim, simu, alg):
    fn = "../../_m/%s/dRFEtools_%.2f/simulate_%d/" % (alg, elim, simu)+\
        "rank_features.txt"
    df = pd.read_csv(fn, sep='\t', header=None,
                     names=["Feature", "Fold", "Rank"])\
           .pivot_table(values="Rank", index="Feature", columns="Fold")\
           .median(axis=1).reset_index().rename(columns={0:"rank"})
    df["Rank"] = df["rank"].rank()
    return df.loc[:, ["Feature", "Rank"]].sort_values("Rank")


@functools.lru_cache()
def get_drfe_cpuTime(elim, simu, alg):
    fn = "../../_m/%s/dRFEtools_%.2f/simulation_time_%.2f.csv" % (alg, elim, elim)
    df = pd.read_csv(fn)
    return df[(df["Simulation"] == simu)].loc[:, "CPU Time"]


def drfe_roc_curve(simu, alg, elim):
    true, false = get_features(simu)
    n_features, n_max, n_redundant = get_n_features(elim, simu, alg)
    df0 = get_ranked_features(elim, simu, alg)
    pred_true = df0[(df0["Rank"] <= n_features)].Feature
    pred_false = df0[(df0["Rank"] > n_features)].Feature
    df1 = pd.concat([pd.DataFrame({"Feature":pred_true, "Y_predict":1}),
                     pd.DataFrame({"Feature":pred_false,"Y_predict":0})],axis=0)
    df2 = pd.concat([pd.DataFrame({"Feature":true, "Y_test":1}),
                     pd.DataFrame({"Feature":false, "Y_test":0})], axis=0)
    return pd.merge(df2, df1, on="Feature")


def cal_metrics(df):
    df["Y_sum"] = np.cumsum(df.Y_predict)
    df["N_sum"] = np.cumsum(1 - df.Y_predict)
    tpr = df.Y_sum / np.sum(df.Y_predict)
    fpr = df.N_sum / np.sum(df.Y_predict == 0)
    return tpr, fpr, df.Feature


def drfe_accuracy(simu, alg, elim):
    true, false = get_features(simu)
    n_features, n_max, n_redundant = get_n_features(elim, simu, alg)
    cpu = get_drfe_cpuTime(elim, simu, alg).values[0]
    df = get_ranked_features(elim, simu, alg)
    TP = df[(df["Feature"].isin(true)) & (df["Rank"] <= n_features)].shape[0]
    FN = df[(df["Feature"].isin(true)) & (df["Rank"] > n_features)].shape[0]
    TN = df[(df["Feature"].isin(false)) & (df["Rank"] > n_features)].shape[0]
    FP = df[(df["Feature"].isin(false)) & (df["Rank"] <= n_features)].shape[0]
    return TP, FN, TN, FP, cpu


def get_metrics(fnc1, fnc2, label, elim_set):
    alg = []; elim_lt = []; simu_lt = []; cpu_lt = [];
    tp_lt = []; fn_lt = []; tn_lt = []; fp_lt = []; dy = pd.DataFrame();
    for elim in elim_set:
        for simu in range(10):
            for algorithm in ["ridge", "enet", "svr", "rf"]:
                tp, fn, tn, fp, cpu = fnc1(simu, algorithm, elim)
                tpr, fpr, feature = cal_metrics(fnc2(simu, algorithm, elim))
                tmp_df = pd.DataFrame({"RFE_Method":label, "Elimination":elim,
                                       "Algorithm":algorithm, "Simulation":simu,
                                       "Feature":feature, "TPR":tpr, "FPR":fpr})
                dy = pd.concat([dy, tmp_df], axis=0)
                alg.append(algorithm); elim_lt.append(elim);
                simu_lt.append(simu); tp_lt.append(tp); cpu_lt.append(cpu)
                fn_lt.append(fn); tn_lt.append(tn); fp_lt.append(fp)
    dx = pd.DataFrame({"RFE_Method": label, "Elimination": elim_lt,
                       "Algorithm": alg, "Simulation": simu_lt, "TP": tp_lt,
                       "FN": fn_lt, "TN": tn_lt, "FP": fp_lt, "CPU": cpu_lt})
    return dx, dy


def main():
    dt1, dx1 = get_metrics(rfe_accuracy, rfe_roc_curve, "RFE", [0.1, 100])
    dt2, dx2 = get_metrics(drfe_accuracy, drfe_roc_curve, "dRFE", [0.1, 0.2])
    pd.concat([dt1, dt2], axis=0)\
      .to_csv("simulated_data_accuracy_metrics.tsv", sep='\t', index=False)
    pd.concat([dx1, dx2], axis=0)\
      .to_csv("simulated_data_roc_metrics.tsv.gz", sep='\t', index=False)


if __name__ == "__main__":
    main()
