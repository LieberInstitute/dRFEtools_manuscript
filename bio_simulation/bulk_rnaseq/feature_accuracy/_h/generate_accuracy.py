"""
Generate the accuracy results for each fold, model, and algorithm.
"""
import functools
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


@functools.lru_cache()
def load_data(simu):
    num = simu + 1
    fn = "../../../_m/bulk_data/simulated_geneInfo_%d.tsv.gz" % num
    return pd.read_csv(fn, sep='\t', index_col=0)


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
    fn = "../../_m/%s/dRFEtools_%.2f/simulation_time_%.2f.csv" % (alg,elim,elim)
    df = pd.read_csv(fn)
    return df[(df["Simulation"] == simu)].loc[:, "CPU Time"]


def get_features(simu):
    true = np.array(load_data(simu)[(load_data(simu)["DE.ind"])].index)
    false = np.array(load_data(simu)[~(load_data(simu)["DE.ind"])].index)
    return true, false


def rfe_accuracy(simu, algorithm, elim):
    true, false = get_features(simu)
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


def get_metrics(fnc, label, elim_set):
    alg = []; elim_lt = []; simu_lt = []; cpu_lt = [];
    tp_lt = []; fn_lt = []; tn_lt = []; fp_lt = [];
    for elim in elim_set:
        for simu in range(10):
            for algorithm in ["lr", "sgd", "svc", "rf"]:
                tp, fn, tn, fp, cpu = fnc(simu, algorithm, elim)
                alg.append(algorithm); elim_lt.append(elim);
                simu_lt.append(simu); tp_lt.append(tp); cpu_lt.append(cpu)
                fn_lt.append(fn), tn_lt.append(tn); fp_lt.append(fp)
    return pd.DataFrame({"RFE_Method": label, "Elimination": elim_lt,
                         "Algorithm": alg, "Simulation": simu_lt, "TP": tp_lt,
                         "FN": fn_lt, "TN": tn_lt, "FP": fp_lt, "CPU": cpu_lt})


def main():
    dt1 = get_metrics(rfe_accuracy, "RFE", [0.1, 100])
    dt2 = get_metrics(drfe_accuracy, "dRFE", [0.1, 0.2])
    pd.concat([dt1, dt2], axis=0)\
      .to_csv("simulated_data_accuracy_metrics.tsv", sep='\t', index=False)


if __name__ == "__main__":
    main()
