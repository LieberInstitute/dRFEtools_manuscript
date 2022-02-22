"""
This script run statistics for RFE and dRFE across all algorithms.
"""
import pandas as pd
import statsmodels.api as sm
from functools import lru_cache
from bioinfokit.analys import stat
from statsmodels.formula.api import ols

@lru_cache()
def load_model_results():
    '''
    This function loads the simulation results.
    '''
    fn = "../../feature_accuracy/_m/simulated_data_accuracy_metrics.tsv"
    return pd.read_csv(fn, sep='\t')


@lru_cache()
def add_metrics():
    '''
    This function adds accuracy and false discovery rate to RF metrics.
    '''
    df = load_model_results()
    df.loc[:, "Accuracy"] = (df.TP + df.TN) / (df.TP + df.TN + df.FN + df.FP)
    df.loc[:, "FDR"] = (1 - (df.TP / (df.TP + df.FP)))
    df.loc[:, "ID"] = df["RFE_Method"] + "_" + df["Elimination"].astype(str)
    return df


@lru_cache()
def fit_models():
    """
    This function fits an linear model for one-way ANOVA.
    """
    cpu_model = ols('CPU ~ C(ID)*C(Algorithm)', data=add_metrics()).fit()
    acc_model = ols('Accuracy ~ C(ID)*C(Algorithm)', data=add_metrics()).fit()
    fdr_model = ols('FDR ~ C(ID)*C(Algorithm)', data=add_metrics()).fit()
    return cpu_model, acc_model, fdr_model


def twoway_anova():
    cpu_model, acc_model, fdr_model = fit_models()
    with open("stats_output.log", "w") as f:
        print("Oneway ANOVA:")
        # CPU
        print("CPU comparison:", file=f)
        anova_table = sm.stats.anova_lm(cpu_model, typ=2) # Type 2 ANOVA
        print(anova_table, file=f)
        # Accuracy
        print("Accuracy comparison:", file=f)
        anova_table = sm.stats.anova_lm(acc_model, typ=2) # Type 2 ANOVA
        print(anova_table, file=f)
        # FDR
        print("FDR comparison:", file=f)
        anova_table = sm.stats.anova_lm(fdr_model, typ=2) # Type 2 ANOVA
        print(anova_table, file=f)


def tukey_analysis():
    res = stat()
    with open("stats_output.log", "a") as f:
        print("Tukey HSD:")
        # CPU
        res.tukey_hsd(df=add_metrics(), res_var="CPU",
                      xfac_var=["ID", "Algorithm"],
                      anova_model="CPU ~ C(ID)*C(Algorithm)")
        print("CPU comparison:", file=f)
        print(res.tukey_summary, file=f)
        # Accuracy
        res.tukey_hsd(df=add_metrics(), res_var="Accuracy",
                      xfac_var=["ID", "Algorithm"],
                      anova_model="Accuracy ~ C(ID)*C(Algorithm)")
        print("Accuracy comparison:", file=f)
        print(res.tukey_summary, file=f)
        # FDR
        res.tukey_hsd(df=add_metrics(), res_var="FDR",
                      xfac_var=["ID", "Algorithm"],
                      anova_model="FDR ~ C(ID)*C(Algorithm)")
        print("FDR comparison:", file=f)
        print(res.tukey_summary, file=f)


def main():
    twoway_anova()
    tukey_analysis()


if __name__ == "__main__":
    main()
